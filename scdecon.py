#!/usr/bin/python
'''
digital deconvolution of blood cell types: 441873
'''
### imports ###

# system
import subprocess
import warnings
import argparse
import logging
import time
import sys
import os
import warnings
import cPickle
import h5py
import numpy as np
import networkx as nx
import multiprocessing

# scikit
import sklearn.cluster
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn import feature_selection

# cvxpy
import cvxpy

# app
from utils.matops import *
#from utils.rfuncs import *
from utils.misc import *

# logging
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s', )

# hack to silence argparser.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

### configuration ###

### definitions ###

### classes ###

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                logging.info('%s: exiting' % proc_name)
                self.task_queue.task_done()
                break
            logging.info('%s: %s' % (proc_name, next_task))
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):

    def __init__(self, S, x, k, j):
        self.S = S
        self.x = x
        self.k = k
        self.j = j

    def __call__(self):
        c, o = _cqp(self.S, self.x, self.k)
        return c, self.j

    def __str__(self):
        return 'column: %i' % self.j

class TaskMissing(object):

    def __init__(self, X, S, Sbase, features, cmax, j, scorefn, dim):
        self.X = X
        self.S = S
        self.Sbase = Sbase
        self.features = features
        self.cmax = cmax
        self.j = j
        self.scorefn = scorefn
        self.dim = dim
        
    def __call__(self):
        s, c, score = _missing(self.X, self.S, self.Sbase, self.features, self.cmax, self.j, self.scorefn, self.dim)
        return s, c, score

    def __str__(self):
        return 'column: %i' % self.j
        
        


### internal functions ###
def _avg_S(Z, y):
    """ create S from known labels """
    S, cats = avg_cat(y, np.transpose(Z))
    return S


def _cqp(S, x, k, sum2=1.0):
    ''' solves using cvxpy '''

    # reshape x to matrix.
    z = np.zeros((x.shape[0],1))
    z[:,0] = x
    x = z

    # cast to object.
    S = cvxpy.matrix(S)
    x = cvxpy.matrix(x)

    # create variables.
    c = cvxpy.variable(k, 1, name='c')

    # create constraints.
    geqs = cvxpy.greater_equals(c,0.0)
    sum1 = cvxpy.equals(cvxpy.sum(c), sum2)
    constrs = [geqs, sum1]
    #constrs = []

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(S*c-x)), constraints=constrs)
    p.options['abstol'] = 1e-10
    p.options['reltol'] = 1e-9
    p.options['feastol'] = 1e-5
    p.options['maxiters'] = 500

    # solve the program.
    #p.solve(quiet=False)
    p.solve(quiet=True)

    # return results.
    return c.value, p.objective.value


def _sqp(x, Sm, C, c):
    ''' solves using cvxpy '''

    # simplify.
    m = Sm.shape[0]

    # subtract x from (Sm * c)
    T = np.dot(Sm, C) - x

    # cast to object.
    #c = cvxpy.matrix(c)
    #x = cvxpy.matrix(x)
    T = cvxpy.matrix(T)

    # create variables.
    S = cvxpy.variable(T.shape[0], 1, name='s')

    # create constraints.
    geqs = cvxpy.greater_equals(S,0.0)
    constrs = [geqs]

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(T+(c*S))), constraints=constrs)
    p.options['abstol'] = 1e-10
    p.options['reltol'] = 1e-9
    p.options['feastol'] = 1e-5
    p.options['maxiters'] = 500

    p.options['abstol'] = 1e-10
    p.options['reltol'] = 1e-9
    p.options['feastol'] = 1e-5
    p.options['maxiters'] = 500

    # solve the program.
    p.solve(quiet=True)

    # return results.
    return S.value, p.objective.value

def _cqp_c(S, x, k, sum2):
    ''' solves using cvxpy '''

    # reshape x to matrix.
    z = np.zeros((x.shape[0],1))
    z[:,0] = x
    x = z

    # cast to object.
    S = cvxpy.matrix(S)
    x = cvxpy.matrix(x)

    # create variables.
    c = cvxpy.variable(k, 1, name='c')

    # create constraints.
    geqs = cvxpy.greater_equals(c,0.0)
    sum1 = cvxpy.equals(cvxpy.sum(c), sum2)
    constrs = [geqs, sum1]
    #constrs = []

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(S*c-x)),constraints=constrs)
    p.options['abstol'] = 1e-10
    p.options['reltol'] = 1e-9
    p.options['feastol'] = 1e-5
    p.options['maxiters'] = 500

    # solve the program.
    #p.solve(quiet=False)
    p.solve(quiet=True)

    # return results.
    return c.value, p.objective.value

def _solve_C(X, S):
    """ solves using QP"""

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = S.shape[1]
    C = np.zeros((k,n), dtype=np.float)

    # solve each concentration independently.
    for j in range(n):

        # solve column.
        c, o = _cqp(S, X[:,j], k)

        # save it.
        for i in range(k):
            C[i,j] = c[i,0]

    # return it.
    return C

def _solve_C_parallel(X, S, num_threads):
    """ solves using QP and multiprocessing """


    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = S.shape[1]
    C = np.zeros((k,n), dtype=np.float)

    # establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    logging.info('creating %d consumers' % num_threads)
    consumers = [ Consumer(tasks, results) for i in xrange(num_threads) ]
    for w in consumers:
        w.start()

    # solve each concentration independently.
    num_jobs = 0
    for j in range(n):

        # create the job.
        tasks.put(Task(S, X[:,j], k, j))

        # track it.
        num_jobs += 1

    # add a poison pill for each consumer
    for i in xrange(num_threads):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # start printing results
    while num_jobs:

        # get vector and index.
        c, j = results.get()

        # save it.
        for i in range(k):
            C[i,j] = c[i,0]

        # track it.
        num_jobs -= 1

    # return the matrix.
    return C


### callable functions ###

def decon(args):
    """ main function for deconvolution"""

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    # sanity.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)

    # decide which method to build S.
    if args.y != None:
        S = _avg_S(Z, y)
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    # run deconvolution
    if args.num_threads == None:
        C = _solve_C(X, S)
    else:
        C = _solve_C_parallel(X, S, args.num_threads)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

    # note that we are done.
    logging.info("done")


def _estimate_missing(x, S, cmax, features, dims):
    """ performs fixed esimate of cm and s
        x = vector for mixture
        S = signature matrix for known cell types, i.e. (m,k-1) dim
        cmax = the total amount of mixtured explained by S
        (n,m,k) = dimensions
    """

    # simplify.
    n, m, k = dims

    # create the temporary matrix.
    Xtmp = np.asmatrix(x).transpose()

    # compute the concentration.
    CS, oc = _cqp_c(S, x, k-1, cmax)

    # extract the concentration of new cell-type.
    cm = 1.0 - np.sum(CS[:])

    # take subset of genes for computing profile.
    Xtmp = Xtmp[features,:]
    Stmp = S[features,:]
    m = len(features)

    # compute the short profile of the missing cell-type.
    Snew, os = _sqp(Xtmp, Stmp, CS, cm)

    # copy to vector.
    c = np.array([ CS[l,0] for l in range(k-1)] + [cm])
    s = np.array([ Snew[i,0] for i in range(m)])

    # return results
    return s, c


def _improve_missing(x, S, cmax, dims):
    """ performs fixed esimate of cm and s"""

    # simplify.
    n, m, k = dims

    # compute the concentration.
    C, oc = _cqp(S, x, k)

    # create a small matrix.
    CS = C[0:k-1,:]

    # extract the updated value of missing cell-type concentration
    cm = C[-1,0]

    # create the temporary matrix.
    Xtmp = np.asmatrix(x).transpose()
    Stmp = S[:,0:k-1]

    # compute the profile of the missing cell-type.
    Snew, os = _sqp(Xtmp, Stmp, CS, cm)

    # copy to vector.
    c = np.array([ CS[l,0] for l in range(k-1)] + [cm])
    s = np.array([ Snew[i,0] for i in range(m)])

    # return results
    return s, c

def debug_missing(args):
    """ development for missing data"""

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    n = X.shape[1]
    m = X.shape[0]
    k = np.unique(y).shape[0]

    # compute signature.
    S = _avg_S(Z, y)

    # compute our own C.
    C = np.zeros((k, n), dtype=np.float)
    for j in range(n):
        C[:,j] = 1.0 / float(k)

    # compute our own X.
    X = np.dot(S, C)

    # extract a subset to create a missing gene.
    Sm = S[:,[0,1,2,3]]

    # identify subset of genes for use in deconvolution.
    ytmp = np.where(y != 4)[0]
    Ztmp = Z[:, ytmp]

    clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=20)
    clf.fit(np.transpose(Ztmp), ytmp)
    features = np.where(clf.get_support() == True)[0]

    # copy the matrix.
    decon_missing(X.copy(), Sm.copy(), features.copy())

def _missing(X, S, Sbase, features, cmax, j, scorefn, dims):
    """
    determine the missing data
    """

    # simplify.
    n, m, k = dims

    # extract mixed vector.
    x = X[:,j]

    # create temporary signature.
    Stmp = S.copy()
    for i in range(m):
        for l in range(k-1):
            Stmp[i,l] = Sbase[i,l]
    Stmp = Stmp[features,:]

    # perform fixed estimate.
    s, c = _estimate_missing(x, Sbase, cmax, features, (n,m,k))

    # broadcast value into last column.
    Stmp[:,-1] = s
    xtmp = x[features]

    # score it.
    x2 = np.dot(Stmp, np.asmatrix(c).transpose())
    x2 = np.array([x2[ee,0] for ee in range(len(features)) ])
    score = scorefn(xtmp, x2)
    pscore = score

    # iterativly try to improve estimate.
    for qq in range(100):

        # improve it.
        s, c = _improve_missing(xtmp, Stmp, cmax, (n,len(features),k))

        # broadcast value into last column.
        Stmp[:,-1] = s

        # score it.
        x2 = np.dot(Stmp, np.asmatrix(c).transpose())
        x2 = np.array([x2[ee,0] for ee in range(len(features)) ])
        score = scorefn(xtmp, x2)

        # break if little improvement.
        if pscore <= score or np.abs(pscore - score) < 0.01:
            break
        
        # update score tracking.
        pscore = score

    # return the final.
    return s, c, score


def decon_missing(X, Sbase, features):
    """ main function for missing data deconvolution
        *** currently hacked for debugging purposes
    """

    # extract information.
    n = X.shape[1]
    m = X.shape[0]
    k = Sbase.shape[1] + 1

    # freeze both X and S.
    X.setflags(write=False)
    Sbase.setflags(write=False)

    # create final signature and concentraction matrix.
    C = np.zeros((k, n), dtype=np.float)
    S = np.zeros((m, k), dtype=np.float)

    # solve each concentration independently.
    for j in range(n):

        # try various concentrations for missing.
        #for cmax in [0.8]:
        for cmax in np.arange(0.7, 1.0, 0.01):

            # compute s, c
            s, c, score = _missing(X, S, Sbase, features, cmax, j, rmse_vector, (n, m, k))

            # yield it.
            yield j, cmax, s, c, score
            #print '%.2f' % cmax, '%.5f' % score, '%.5f' % rmse_vector(np.array([.2] * 5), c)







def decon_scale(args):
    """ main function for deconvolution"""

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    # sanity.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)

    # decide which method to build S.
    if args.y != None:
        S = _avg_S(Z, y)
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    # count zeros in Z
    k = len(np.unique(y))
    for l in range(k):
        SC_l = Z[:,np.where(y == l)[0]]

        for i in range(Z.shape[0]):
            SC_m = SC_l[i,:]

            # compute probability.
            f = 1.0 / float(len(np.where(SC_m == 0.0)[0])) / float(SC_m.shape[0])

    # run deconvolution
    C = _solve_C(X, S)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)

    # note that we are done.
    logging.info("done")

def pca_decon(args):
    """ main function for deconvolution"""

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    # sanity.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)


    # transform single-cell data.
    Z_t = np.transpose(Z)
    pca = PCA(n_components=3, whiten=False)
    Z_t = pca.fit(Z_t, y).transform(Z_t)
    Z = np.transpose(Z_t)

    # decide which method to build S.
    if args.y != None:
        S = _avg_S(Z, y)
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    # transoform mixture data.
    X_t = np.transpose(X)
    pca2 = PCA(n_components=3, whiten=False)
    X_t = pca2.fit(X_t).transform(X_t)
    X = np.transpose(X_t)

    # run deconvolution
    C = _solve_C(X, S)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)

    # note that we are done.
    logging.info("done")

### script ###

if __name__ == '__main__':

    # mode parser.
    main_p = argparse.ArgumentParser()
    subp = main_p.add_subparsers(help='sub-command help')

    #### testing functions ####

    # clustering single-cells
    subpp = subp.add_parser('decon', help='deconvolve the single-cells and mixtures using QP')
    subpp.add_argument('-X', dest='X', required=True, help='mixture: genes x samples')
    subpp.add_argument('-Z', dest='Z', required=True, help='single-cell matrix: genes x cells')
    subpp.add_argument('-y', dest='y', help='cell type labels for each cell')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-S', dest='S', help='output numpy matrix')
    subpp.add_argument('-p', dest='num_threads', type=int, help='number of threads for parallel solving')
    subpp.add_argument('-k', type=int, dest='num_cluster', help='number of clusters, must be supplied if z_lbls is not')
    subpp.set_defaults(func=decon)

    # clustering single-cells
    subpp = subp.add_parser('decon_missing', help='deconvolve the single-cells and mixtures using QP look for a missing cigy butt brain')
    subpp.add_argument('-X', dest='X', required=True, help='mixture: genes x samples')
    subpp.add_argument('-Z', dest='Z', required=True, help='single-cell matrix: genes x cells')
    subpp.add_argument('-y', dest='y', help='cell type labels for each cell')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-S', dest='S', help='output numpy matrix')
    subpp.add_argument('-p', dest='num_threads', type=int, help='number of threads for parallel solving')
    subpp.add_argument('-k', type=int, dest='num_cluster', help='number of clusters, must be supplied if z_lbls is not')
    subpp.set_defaults(func=debug_missing)

    subpp = subp.add_parser('deconscale', help='deconvolve the single-cells and mixtures using QP, scale X by 1/missing')
    subpp.add_argument('-X', dest='X', required=True, help='mixture: genes x samples')
    subpp.add_argument('-Z', dest='Z', required=True, help='single-cell matrix: genes x cells')
    subpp.add_argument('-y', dest='y', help='cell type labels for each cell')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-k', type=int, dest='num_cluster', help='number of clusters, must be supplied if z_lbls is not')
    subpp.set_defaults(func=decon_scale)

    subpp = subp.add_parser('pcacon', help='deconvolve the single-cells and mixtures using QP on the PCA transformed data')
    subpp.add_argument('-X', dest='X', required=True, help='mixture: genes x samples')
    subpp.add_argument('-Z', dest='Z', required=True, help='single-cell matrix: genes x cells')
    subpp.add_argument('-y', dest='y', help='cell type labels for each cell')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-k', type=int, dest='num_cluster', help='number of clusters, must be supplied if z_lbls is not')
    subpp.set_defaults(func=pca_decon)

    ### pipeline ###

    # parse args.
    args = main_p.parse_args()
    args.func(args)
