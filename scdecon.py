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
import operator
import itertools

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


def _cqp(S, X, sum2=1.0, scorefn=None):
    ''' solves using cvxpy '''

    # simplify.
    m = S.shape[0]
    k = S.shape[1]

    # cast to object.
    S = cvxpy.matrix(S)
    X = cvxpy.matrix(X)

    # create variables.
    C = cvxpy.variable(k, 1, name='c')

    # create constraints.
    geqs = cvxpy.greater_equals(C,0.0)
    sum1 = cvxpy.equals(cvxpy.sum(C), sum2)
    constrs = [geqs, sum1]

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(S*C-X)), constraints=constrs)
    #p.options['abstol'] = 1e-10
    #p.options['reltol'] = 1e-9
    #p.options['feastol'] = 1e-5
    #p.options['maxiters'] = 500

    # solve the program.
    p.solve(quiet=True)

    # compute x2
    C = C.value        
    X2 = np.dot(S, C)
        
    c = np.array([C[l,0] for l in range(k)])
    x = np.array([X[i,0] for i in range(m)])
    x2 = np.array([X2[i,0] for i in range(m)])

    # compute score.
    if scorefn == None:
        o = p.objective.value
    else:
        o = scorefn(x, x2)

    # return results.
    return c, o
    
def _shqp(X, Sbase, c, scorefn=None):
    ''' solves using cvxpy '''

    # simplify.
    m = Sbase.shape[0]
    k = Sbase.shape[1]

    # create shortened C matrix.
    Cbase = np.zeros((k,1))
    Cbase[0:k,0] = c[0:k]
    cm = c[-1]
    
    # cast to object.
    X = cvxpy.matrix(X)
    Sbase = cvxpy.matrix(Sbase)
    Cbase = cvxpy.matrix(Cbase)

    # create variables.
    SV = cvxpy.variable(m, 1, name='s')

    # create constraints.
    geqs = cvxpy.greater_equals(SV,0.0)
    constrs = [geqs]
    
    # set equation.
    eq = ((Sbase*Cbase) + (SV*cm)) - X

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(eq)), constraints=constrs)
    #p.options['abstol'] = 1e-10
    #p.options['reltol'] = 1e-9
    #p.options['feastol'] = 1e-5
    #p.options['maxiters'] = 500

    # solve the program.
    p.solve(quiet=True)

    # compute x2
    C = np.asmatrix(c).transpose()    
    S = np.zeros((m, k+1))
    for i, l in zip(range(m), range(k)): 
        S[i, l] = Sbase[i, l]
    for i in range(m):
        S[i,-1] = SV.value[i]
    
    X2 = np.dot(S, C)
    x = np.array([X[i,0] for i in range(m)])
    x2 = np.array([X2[i,0] for i in range(m)])

    # compute score.
    if scorefn == None:
        o = p.objective.value
    else:
        o = scorefn(x, x2)

    # simplify.
    s = np.array([SV.value[i,0] for i in range(m)])

    # return results.
    return s, o


def _sqp(X, Sbase, cm, scorefn=None):
    ''' solves using cvxpy '''

    # simplify.
    m = Sbase.shape[0]
    k = Sbase.shape[1] + 1
    
    # create the S variables.
    SV = cvxpy.variable(m, 1, name='s')
    
    # create the concentration variables.
    CV = cvxpy.variable(k - 1, 1, name='c')

    # case to matrix.
    X = cvxpy.matrix(X)
    Sbase = cvxpy.matrix(Sbase)

    # create constraints.
    sgeqs = cvxpy.greater_equals(SV,0.0)
    cgeqs = cvxpy.greater_equals(CV,0.0)
    sum1 = cvxpy.equals(cvxpy.sum(CV), 1.0 - cm)
    constrs = [sgeqs, cgeqs, sum1]

    # create the equation.
    eq = ((Sbase*CV)+(SV*cm)) - X

    # create the program.
    p = cvxpy.program(cvxpy.minimize(cvxpy.norm2(eq)), constraints=constrs)
    #p.options['abstol'] = 1e-10
    #p.options['reltol'] = 1e-9
    #p.options['feastol'] = 1e-5
    #p.options['maxiters'] = 500

    # solve the program.
    p.solve(quiet=True)

    # case to vectors.
    s = np.array([SV.value[i,0] for i in range(m)])
    c = np.array([CV.value[l,0] for l in range(k-1)] + [cm])

    # compute x2
    S = np.zeros((m, k))
    for i, l in zip(range(m), range(k-1)): 
        S[i, l] = Sbase[i, l]
    for i in range(m):
        S[i,-1] = SV.value[i]
    
    C = np.zeros((k,1))
    for l in range(k-1):
        C[l,0] = CV.value[l,0]
    C[-1,0] = cm
        
    X2 = np.dot(S, C)
        
    x = np.array([X[i,0] for i in range(m)])
    x2 = np.array([X2[i,0] for i in range(m)])

    # compute score.
    if scorefn == None:
        o = p.objective.value
    else:
        o = scorefn(x, x2)

    # return results.
    return s, c, o

def _solve_C(X, S, num_threads=1):
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

def _solve_missing(X, Sbase, features, scorefn, cheat_s=None, cheat_C=None):
    """ solves using QP"""

    scorefn = meanrel_vector
    scorefn = meanabs_vector

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = Sbase.shape[1] + 1
    C = np.zeros((k,n), dtype=np.float)

    # shrink it.
    Sbase = Sbase[features,:]
    X = X[features,:]

    # create array for new S.
    Snew = np.zeros((m, n), dtype=np.float)

    # solve each concentration independently.
    bysample = list()
    for j in range(n):

        if j > 3: break

        # freeze X.
        Xqp = X[:,[j]]
        Xqp.setflags(write=False)

        # try various concentrations for missing.
        byscore = list()
        for cm in np.arange(0.0, 1.0, 0.1):

            # compute s, c
            #s, c, o = _sqp(Xqp, Sbase, cm, scorefn)
            s, c, o = _missing(Xqp, Sbase, cm, scorefn)
            
            # debug mode.
            if cheat_s != None or cheat_C != None:
                
                # simplify cheat.
                cheat_c = cheat_C[:,j]
                
                # compute score.
                byscore.append((o, scorefn(cheat_c, c), scorefn(cheat_s, s)))
                #print '%.2f' % cm, '%.3f' % o, ' '.join(['%.3f' % x for x in c]), ' '.join(['%.3f' % x for x in s])
    
        # sort it.
        byscore = sorted(byscore, key=operator.itemgetter(0))
            
        # save it.
        bysample.append(byscore)
    
    
        
    # print it to the screen.
    for i in range(len(np.arange(0.0, 1.0, 0.1))):
        
        row = list()
        for s in bysample:
            row.append('%.5f' % s[i][1])
        
        print ' '.join([str(i)] + row)
        
        


    print "DEBUG DONE"
    sys.exit()

    # return it.
    return C, S

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


def _missing(Xqp, Sbase, cm, scorefn):
    """
    determine the missing data
    """
    
    # simplify.
    m = Sbase.shape[0]
    k = Sbase.shape[1] + 1

    # create big matrix.
    S = np.zeros((m, k))
    for i, l in itertools.product(range(m), range(k-1)): 
        S[i, l] = Sbase[i, l]

    # guess the firsty
    s, c, so = _sqp(Xqp, Sbase, cm, scorefn)

    # fill the signature.
    S[:,-1] = s

    # save each one.
    scores = list()

    # iterativly try to improve estimate.
    for qq in range(25):

        # estimate concentration.
        c, co = _cqp(S, Xqp, sum2=1.0, scorefn=scorefn)

        # estimate signature.
        s, so = _shqp(Xqp, Sbase, c, scorefn=scorefn)
        
        # update S.
        S[:,-1] = s

        # take the minimum.
        scores.append((co, so, c, s))
        
    # choose maximum.
    co, so, c, s = sorted(scores, key=operator.itemgetter(0,1))[0]
        
    # return the final.
    return s, c, so


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
        C = _solve_C_parallel(X, S, num_threads = args.num_threads)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

    # note that we are done.
    logging.info("done")
    

def decon_missing(args):
    """ main function for deconvolution with missing data"""

    # debug.
    '''
    Sfull = np.array([
        [0.1, 0.2, 0.7],
        [0.7, 0.5, 0.2],
        [0.3, 0.4, 0.1],
        [0.9, 0.1, 0.1],
        [0.4, 0.6, 0.7],
        [0.1, 0.2, 0.8]])
        
    Ctrue = np.array([[0.3], [0.5], [0.2]])
    X = np.dot(Sfull, Ctrue)
    
    # trim it.
    Sbase = Sfull[:,[0,1]]
    features = range(Sfull.shape[0])

    # load data.
    '''
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    # sanity.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)

    # extract informative markers.
    clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=10)
    clf.fit(np.transpose(Z), y)
    features = np.where(clf.get_support() == True)[0]
    #features = np.arange(X.shape[0])
    
    # decide which method to build S.
    if args.y != None:
        Sbase = _avg_S(Z, y)
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    
    # run deconvolution
    C, S = _solve_missing(X, Sbase, features, None)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

    # note that we are done.
    logging.info("done")

'''
def decon_missing(X, Sbase, features):
    """ main function for missing data deconvolution
        *** currently hacked for debugging purposes
    """

    clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=20)
    clf.fit(np.transpose(Ztmp), ytmp)
    features = np.where(clf.get_support() == True)[0]


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

'''




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
    subpp.set_defaults(func=decon_missing)

    ### pipeline ###

    # parse args.
    args = main_p.parse_args()
    args.func(args)
