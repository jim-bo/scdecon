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
from multiprocessing import Pool
import operator
import itertools
# hack to silence argparser.
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# scikit
import sklearn.cluster
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn import feature_selection

# cvxpy
import cvxopt
import cvxopt.modeling
import cvxpy

# lmfit.
from lmfit import minimize, Parameters, Parameter, report_fit

# app
from utils.matops import *
#from utils.rfuncs import *
from utils.misc import *

# logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )


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

    def __init__(self, S, x, j):
        self.S = S
        self.x = x
        self.j = j

    def __call__(self):
        c, o = _cqp(self.S, self.x)
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


def _x_S(x, S):
    ''' solves using cvxpy '''

    # simplify.
    n = x.shape[1]
    m = S.shape[0]
    k = S.shape[1]

    # sanity.
    assert x.shape[1] == 1

    # cast to object.
    x = cvxopt.matrix(x)
    S = cvxopt.matrix(S)

    # create variables.
    c = cvxpy.Variable(k, n, name='c')

    # create constraints.
    constrs = list()
    constrs.append(c >= 0.0)
    constrs.append(sum(c) == 1.0)

    # create the objective.
    #obj = cvxpy.Minimize(sum(cvxpy.square(S*c - x)))
    obj = cvxpy.Minimize(cvxpy.norm(S*c - x, p=2))

    # solve the program.
    p = cvxpy.Problem(obj, constrs)
    p.solve()

    # turn into array.
    c = np.squeeze(np.array(c.value))
    
    # compute score.
    o = p.objective.value

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

def _X_SM_C(X, SM, C):
    ''' solves for missing cell-type using complete concentrations
        X = (m x n)
        SM = (m x k-1)
        C = (k, n)
    '''

    # dimensions.
    n = X.shape[1]
    m = X.shape[0]
    k = C.shape[0]

    # create shortened C matrix.
    CM = np.zeros((k-1,n))
    CM[0:k-1,:] = C[0:k-1,:]
    
    # create the known vector.
    c = np.asmatrix(C[-1,:])
    
    # compute the dot product of SM, CM
    LH = np.dot(SM, CM)
    
    # cast to cvxopt.
    X = cvxopt.matrix(X)
    LH = cvxopt.matrix(LH)
    c = cvxopt.matrix(c)

    # create variables.
    SV = cvxpy.Variable(m, 1, name='s')

    # create constraints.
    constrs = list()
    constrs.append(sum(SV) == 0.0)

    # create the objective.
    #obj = cvxpy.Minimize(cvxpy.norm((LH + (SV*C)) - , p=2))

    # solve the program.
    #p = cvxpy.Problem(obj, constrs)
    p.solve()

    # turn into array.
    #c = np.squeeze(np.array(c.value))
    
    # compute score.
    #o = p.objective.value

    # set equation.
    eq1 = SV * c
    eq2 = LH + eq1
    eq = eq2 - X


    # return results.
    return s, o


def _x_SM_c(x, SM, cguess):
    ''' solves for missing cell-type using concentration guess
        X = (m x n)
        SM = (m x k-1)
        c = (k, n)
    '''

    # simplify.
    n = x.shape[1]
    m = x.shape[0]
    k = SM.shape[1] + 1

    # sanity check.
    assert n == 1
    assert isinstance(cguess, float)

    # cast matrix.
    x = cvxopt.matrix(x)
    SM = cvxopt.matrix(SM)

    # create variables.
    cv = cvxpy.Variable(k-1, 1, name='c')
    sv = cvxpy.Variable(m, 1, name='s')

    # create constraints.
    constrs = list()
    constrs.append(sv >= 0.0)
    constrs.append(sum(cv) == 1.0 - cguess)
    
    # create the objective.
    obj = cvxpy.Minimize(cvxpy.norm(((SM*cv) + (sv*cguess)) - x, p=2))

    # solve the program.
    p = cvxpy.Problem(obj, constrs)
    p.solve(solver=cvxpy.ECOS)
    
    # status check
    if p.status == "solver_error":
        p.solve(solver=cvxpy.CVXOPT)
    
    # final check.
    if p.status == "solver_error":
        logging.error("solver failed!")
        return None, None, None

    # turn into array.
    s = np.squeeze(np.array(sv.value))
    c = np.array(list(np.squeeze(np.array(cv.value))) + [cguess])
    
    # compute score.
    o = p.objective.value
    
    # return it
    return s, c, o

def _nllsq(X, SM):
    ''' solves for missing cell-type using concentration guess
        X = (m x n)
        SM = (m x k-1)
        C = (k, n)
    '''
    # simplify.
    n = X.shape[1]
    m = X.shape[0]
    k = SM.shape[1] + 1
    
    # create parameters.
    params = Parameters()
    
    # C: add one for k-1, j constraints
    for j in range(n):
        for l in range(k -1):
        
            # build the parameter.
            params.add('C_%i_%i' % (l, j), value=1.0 / float(k), min=0.0, max=1.0, vary=True)
        
        # add sum to 1 constraint.    
        constr = '1.0 - ' + '+'.join(['C_%i_%i' % (l, j) for l in range(0,k-1)])
        
        # add last variable.
        params.add('C_%i_%i' % (k-1, j), value=1.0 / float(k), min=0.0, max=1.0, vary=True, expr=constr)

    # S: add missing vars.
    for i in range(m):
        
        # initial value is average of existing.
        iv = np.average(SM[i,:])

        # build the parameter.
        params.add('s%i' % (i), value=iv, min=0.0, vary=True)

    # create user variables.
    XUD = np.zeros((m, n))      # dot product
    XUS = np.zeros((m, n))      # difference
    CU = np.zeros((k, n))
    SU = np.zeros((m, k))
    SU[:,0:k-1] = SM[:,:]

    # do fit, here with leastsq model
    try:
        result = minimize(_lstsq_sC, params, args=(X, SU, CU, XUD, XUS))
    except ZeroDivisionError as e:
        return None, "ZeroDivisionError", e
    
    # extract results.
    for l in range(k):
        for j in range(n):
             CU[l, j] = params['C_%i_%i' % (l, j)].value
    
    for i in range(m):
        SU[i, -1] = params['s%i' % i].value
    
    
    # compute the final model.
    np.dot(SU, CU, out=XUD)
    np.subtract(XUD, X, out=XUS)
    np.square(XUS, out=XUD)
    o = np.sqrt(np.sum(XUD))

    # return it all.
    return SU[:, -1], CU, o

def _lstsq_sC(params, XT, SU, CU, XUD, XUS):

    # dimension.
    n = XT.shape[1]
    m = XT.shape[0]
    k = SU.shape[1]

    # populate concentrations.
    for j in range(n):
        for l in range(k):
            CU[l, j] = params['C_%i_%i' % (l, j)].value
    
    # populate signatures.
    for i in range(m):
        SU[i, -1] = params['s%i' % i].value
    
    # compute dot product.
    np.dot(SU, CU, out=XUD)
    
    # compute the absolute difference.
    np.subtract(XUD, XT, out=XUS)
    
    # return the residual
    return XUS.flatten()
    
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
    debug = list()
    for j in range(n):

        # create the job.
        tasks.put(Task(S, X[:,j], j))
        #debug.append(Task(S, X[:,j], j)())

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
        #c, j = debug.pop()

        # save it.
        C[:, j] = c

        # track it.
        num_jobs -= 1

    # return the matrix.
    return C

def _solve_missing(X, Sbase, features, scorefn, S_cheat=None, C_cheat=None):
    """ solves using QP: iterative random guess."""

    # shrink it.
    Sbase = Sbase[features,:]
    X = X[features,:]

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = Sbase.shape[1] + 1
    C = np.zeros((k,n), dtype=np.float)

    # create array for new S.
    Snew = np.zeros((m, n), dtype=np.float)
    S = np.zeros((m, k), dtype=np.float)
    for i, l in itertools.product(range(m), range(k-1)):
        S[i, l] = Sbase[i, l]

    sys.exit()
    # random starting points.
    byscore = list()
    for zz in range(3):

        # take a random guess.
        #cguess = np.array([zz] * n)
        #cguess = np.random.rand(n)
        #cguess = cguess / np.sum(cguess)
        #cguess = Ctrue[-1,:]
        cguess = np.array([.2] * 5)

        # solve globally.
        s, C, so = _sqp(X, Sbase, cguess, scorefn)
        pso = sys.float_info.max

        # copy into full matrix.
        S[:,-1] = s

        # score it.
        #score_s = scorefn(cheat_s, s)
        #score_c = rmse_matrix(Ctrue, C)
        #print o, score_s, score_c, cguess

        # iterativly improve the guess.
        for qq in range(10):

            # re-estimate C given S.
            for j in range(n):

                # estimate.
                c, co = _cqp(S, X[:,j], sum2=1.0, scorefn=scorefn)

                # copy into full array.
                C[:, j] = c

            # fix the cguess.
            cguess = C[-1, :]

            # score it.
            #score_s = scorefn(cheat_s, s)
            #score_c = rmse_matrix(Ctrue, C)
            #print so, score_s, score_c, cguess
            #print so, cguess

            # solve globally.
            s, C, so = _sqp(X, Sbase, cguess, scorefn)

            # copy into full matrix.
            S[:,-1] = s

            # break if no change in global score.
            if so >= pso or abs(pso-so) < 0.00001:
                break
            pso = so

        # fix best.
        #score_s = scorefn(cheat_s, s)
        #score_c = rmse_matrix(Ctrue, C)
        #print so, score_s, score_c, ' '.join([str(x) for x in cguess])

        # score it.
        byscore.append((so, S, C))

    # sort it.
    byscore.sort(key=operator.itemgetter(0))

    # sanity check.
    for j in range(n):
        assert np.abs(1.0 - np.sum(C[:,j])) <= .001, 'concentration doesnt sum'

    # return it.
    so, S, C = byscore[0]
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

def _debug_x_S(dims, verbose=False):

    # extract.
    n, m, k = dims
    
    # create simulation.
    X, S, SM, C = _debug_setup(n, m, k)

    # track total error.
    avg_err = 0.0
    
    # solve each sample individually.
    if verbose: logging.debug("_x_S: x_S")
    for j in range(n):
        
        # create matrix.
        x = X[:,[j]]
        
        # solve it.
        c, o = _x_S(x, S)
    
        # score it,
        score = meanabs_vector(C[:,j], c)
        avg_err += score
    
        # compare it.
        if verbose: logging.debug("%i: %.5f %.5f" % (j, score, o))
    
    # note its ability.
    if verbose: logging.debug("_x_S: %.5f" % avg_err)   
    
    # return the results.
    return n, m, k, avg_err

def _debug_x_SM_c(dims, verbose=False):
    
    # extract.
    n, m, k = dims
    
    # create simulation.
    X, S, SM, C = _debug_setup(n, m, k)
    
    # track total error.
    err_s = 0.0
    err_c = 0.0
    
    # solve each sample individually.
    if verbose: logging.debug("_x_SM_c:")
    for j in range(n):
        
        # create matrix.
        x = X[:,[j]]
        
        # fix true guess.
        cguess = C[-1, j]
        
        # solve it.
        sp, cp, o = _x_SM_c(x, SM, cguess)
        
        # skip if None.
        if sp == None: continue
    
        # score it,
        score_s = meanabs_vector(S[:,-1], sp)
        score_c = meanabs_vector(C[:,j], cp)
        err_s += score_s
        err_c += score_c
    
        # compare it.
        if verbose: logging.debug("%i: %.5f %.5f %.5f" % (j, score_s, score_c, o))
    
    # take the mean.
    err_s /= n
    err_c /= n
    
    # note its ability.
    if verbose: logging.debug("_x_SM_c: %.5f %.5f" % (err_s, err_c))
    
    # return total error.
    return n, m, k, err_s, err_c

    
def _debug_nllsq(n, m, k, verbose=False):
        
    # create simulation.
    X, S, SM, C = _debug_setup(n, m, k)
    
    # track total error.
    err_s = 0.0
    err_c = 0.0
    
    # solve them straight up.
    sp, CP, o = _nllsq(X, SM)
    
    # error check.
    if sp == None:
        logging.error("nnlsq: %i %i %i:%s" % (n ,m, k, CP))
        return n, m, k, None, CP, o
    
    # score the s vector
    score_s = meanabs_vector(S[:,-1], sp)
    
    # take average absolute error.
    tmp = list()
    score_c = np.average(np.array([meanabs_vector(C[:,j], CP[:,j]) for j in range(n)]))
            
    # print it.
    if verbose: logging.debug("nnlsq: %i %i %i %.5f %.5f %.5f" % (n, m, k, score_s, score_c, o))
    
    # return total error.
    return n, m, k, score_s, score_c
    
def _debug_x_SM_c_noguess(dims, verbose=False):
    
    # extract.
    n, m, k = dims
    
    # create simulation.
    X, S, SM, C = _debug_setup(n, m, k)
    
    # track total error.
    err_s = 0.0
    err_c = 0.0
    
    # solve each sample individually.
    if verbose: logging.debug("_x_SM_c_noguess:")
    for j in range(n):
        
        # create matrix.
        x = X[:,[j]]
        
        # true guess.
        tguess = C[-1, j]
        
        # iterate over guess range.
        for cguess in np.arange(0.0, 1.0, 0.005):
        
            # solve it.
            sp, cp, o = _x_SM_c(x, SM, cguess)
            
            # skip if None.
            if sp == None: continue
        
            # score it,
            score_s = meanabs_vector(S[:,-1], sp)
            score_c = meanabs_vector(C[:,j], cp)
            err_s += score_s
            err_c += score_c
        
            # compare it.
            if verbose: logging.debug("%i: %.5f %.5f %.5f %.5f %.5f" % (j, tguess, cguess, score_s, score_c, o))
    
    # take the mean.
    err_s /= n
    err_c /= n
    
    # note its ability.
    if verbose: logging.debug("_x_SM_c_noguess: %.5f %.5f" % (err_s, err_c))
    
    # return total error.
    return n, m, k, err_s, err_c

def _debug_setup(n, m, k):
    
    # create signature.
    S = np.random.random_integers(0, 200, size=(m, k))
    
    # extract known result.
    s = S[:,-1]
    s.setflags(write=False)
    
    # create concentrations.
    C = np.random.rand(k, n)
    for j in range(n):
        C[:,j] = C[:,j] / np.sum(C[:,j])
    C = C.round(3) 
    
    # compute mixtures.
    X = np.dot(S, C)
    
    # freeze them all.
    X.setflags(write=False)
    S.setflags(write=False)
    C.setflags(write=False)
    
    # reduce signature.
    SM = S[:,0:k-1]
    SM.setflags(write=False)
    
    # return everything.
    return X, S, SM, C

def _debug_runit(nlist, mlist, klist, repeat, fn, parallel=0, verbose=False):

        
    # create parameter lists.
    qlist = np.arange(repeat)

    # create iterator.
    its = list()
    for q, n, m, k in itertools.product(qlist, nlist, mlist, klist):
        its.append((n, m, k))
        
    # not how many we are running.
    logging.info("starting %i jobs" % len(its))
        
    # determine run type
    if parallel == 0:

        # run it serial.
        jobs = list()
        for n, m, k in its:
            jobs.append(fn(n, m, k, verbose=verbose))

    else:
        
        # create the pool.
        pool = Pool(processes = parallel)
        
        # add jobs.
        ares = list()
        for n, m, k in its:
            ares.append(pool.apply_async(fn, (n, m, k), dict(verbose=verbose)))
            
        # close and get results.
        pool.close()
        pool.join()
        
        # get results.
        jobs = [a.get() for a in ares]      
        
    # return the results.
    return jobs

def _debug_extract_1(jobs):

    # loop over job with one result.
    result = dict()
    for n, m, k, co in jobs:
        
        # track key.
        key = (n,m,k)
        
        # save dictionary.
        if key not in result:
            result[key] = list()
        
        # save result.
        result[key].append(co)

    # return it.
    return result

def _debug_extract_2(jobs):

    # extract results.
    result = dict()
    for n, m, k, co, so in jobs:
        
        # track key.
        key = (n,m,k)
        
        # save dictionary.
        if key not in result:
            result[key] = [list(), list()]
        
        # skip if bad.
        if co == None:
            continue
        
        # save result.
        result[key][0].append(co)
        result[key][1].append(so)

    # return results.
    return result

def _debug_print_1(result):
    
    # print results.
    keys = sorted(result.keys(), key=operator.itemgetter(0,1,2))
    for key in keys:
        txt = '%i,%i,%i' % key
        txt += ',%.5f' % np.average(np.array(result[key]))
        print txt

def _debug_print_2(result):

    # print results.
    keys = sorted(result.keys(), key=operator.itemgetter(0,1,2))
    for key in keys:
        txt = '%i,%i,%i' % key
        txt += ',%.5f,%.5f' % (np.average(np.array(result[key][0])), np.average(np.array(result[key][1])))
        print txt

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
        C = _solve_C(X, S, num_threads = args.num_threads)

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

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)

    # sanity.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)

    # extract informative markers.
    #clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=10)
    #clf.fit(np.transpose(Z), y)
    #features = np.where(clf.get_support() == True)[0]
    features = np.arange(X.shape[0])

    # decide which method to build S.
    if args.y != None:
        Sbase = _avg_S(Z, y)
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    # lets cheat.
    if args.S_cheat != None:
        _master_cheat(X, Sbase, features, args)
        print "DONE CHEATING"
        sys.exit()

    # run deconvolution
    C, S = _solve_missing(X, Sbase, features, rmse_vector, S_cheat=S_cheat, C_cheat=C_cheat)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

    # note that we are done.
    logging.info("done")

def debug_master(args):
    ''' debug stuff '''
    
    # freeze randomness.
    np.random.seed(seed=1)
        
    #  solve c given S
    if 1 == 0:
        
        # create parameter lists.
        nlist = np.arange(20, 25, 5)
        mlist = np.arange(20, 25, 5)
        klist = np.arange(2, 30, 1)
        
        # run it
        jobs = _debug_runit(nlist, mlist, klist, 10, _debug_x_S, parallel=20)
        
        # extract results.
        results = _debug_extract_1(jobs)
        
        # print it.
        _debug_print_1(results)
            
        # finish.
        sys.exit()
         
    # solve s, c given x, cguess and SM
    if 1 == 0:
        
        # create parameter lists.
        nlist = np.arange(20, 25, 5)
        mlist = np.arange(20, 25, 5)
        klist = np.arange(3, 30, 1)

        # run it
        jobs = _debug_runit(nlist, mlist, klist, 10, _debug_x_SM_c, parallel=20)
        
        # extract results.
        results = _debug_extract_2(jobs)    

        # print it.
        _debug_print_2(results)        
        
        # leave it.
        sys.exit()

    # solve s, c given x, and SM (no cguess)
    if 1 == 0:
        
        # create parameter lists.
        nlist = np.arange(5, 10, 5)
        mlist = np.arange(5, 10, 5)
        klist = np.arange(2, 3, 1)
        
        # create parameter lists.
        nlist = np.arange(20, 25, 5)
        mlist = np.arange(20, 25, 5)
        klist = np.arange(3, 30, 1)

        # run it
        jobs = _debug_runit(nlist, mlist, klist, 10, _debug_x_SM_c_noguess, parallel=20)
        
        # extract results.
        results = _debug_extract_2(jobs)    

        # print it.
        _debug_print_2(results)        
        
        # leave it.
        sys.exit()
        
    # solve s, C given X, using NLLSQ
    if 1 == 1:
        
        # create parameter lists.
        nlist = np.arange(2, 24, 2)
        mlist = np.arange(2, 46, 2)
        klist = np.arange(2, 10, 1)

        # run it
        jobs = _debug_runit(nlist, mlist, klist, 10, _debug_nllsq, parallel=20, verbose=True)
        
        # extract results.
        results = _debug_extract_2(jobs)    

        # print it.
        _debug_print_2(results)        
        
        # leave it.
        sys.exit()
    

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
    subpp.add_argument('-S_cheat', dest='S_cheat', help='use only for debugging purposes!')
    subpp.add_argument('-C_cheat', dest='C_cheat', help='use only for debugging purposes!')
    subpp.add_argument('-H_cheat', dest='H_cheat', help='use only for debugging purposes!')
    subpp.add_argument('-fs_cheat', dest='fs_cheat', help='use only for debugging purposes!')
    subpp.set_defaults(func=decon_missing)

    # development and validation of method.
    subpp = subp.add_parser('debug', help='debugging')
    subpp.set_defaults(func=debug_master)

    ### pipeline ###

    # parse args.
    args = main_p.parse_args()
    args.func(args)
