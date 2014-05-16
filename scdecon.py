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
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s', )

### configuration ###

### definitions ###

### classes ###

### debugging functions ###

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

### private functions ###

def _qp_solve_c(x, S):
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

def _lmfit_sc(X, SM):
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
        constr = '1.0 - ' + '-'.join(['C_%i_%i' % (l, j) for l in range(0,k-1)])

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
        result = minimize(_obj_sc, params, args=(X, SU, CU, XUD, XUS))
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

def _obj_sc(params, XT, SU, CU, XUD, XUS):

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

def _lmfit_c(x, S):
    ''' solve for a missing concentration profile.
        input:
            x = (m,)
            S = (m x k)
        
        output:
            c = (k,)
    '''
    # simplify.
    m = x.shape[0]
    k = S.shape[1]

    # create parameters.
    params = Parameters()

    # c add one for k - 1 constraints
    for l in range(k - 1):

        # build the parameter.
        params.add('c_%i' % (l), value=1.0 / float(k), min=0.0, max=1.0, vary=True)

    # build sum to 1 constraint.
    constr = '1.0 - ' + '-'.join(['c_%i' % (l) for l in range(0,k - 1)])

    # add last variable.
    params.add('c_%i' % (k-1), value=1.0 / float(k), min=0.0, max=1.0, vary=True, expr=constr)

    # do fit, here with leastsq model
    try:
        result = minimize(_obj_c, params, args=(x, S))
    except ZeroDivisionError as e:
        return None, "ZeroDivisionError", e

    # extract results.
    c = np.array([params['c_%i' % l].value for l in range(k)])    

    # compute the final model.
    o = np.sqrt(np.sum(np.square(x - np.dot(S, c))))

    # return it all.
    return c, o

def _obj_c(params, x, S):
    """ objective for misisng concentrations only """

    # simplify.
    m = x.shape[0]
    k = S.shape[1]
    
    # build array.
    c = np.array([params['c_%i' % l].value for l in range(k)])
    
    # compute dot product.
    x2 = np.dot(S, c)
    
    # compute.
    return np.abs(x - x2)
    
### middle functions ###

def solve_SC(X, Z, y):
    """ solves using QP and multiprocessing """

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = len(np.unique(y)) + 1

    # create reduced S
    SM, t = avg_cat(y, np.transpose(Z))

    # solve them straight up.
    sp, C, o = _lmfit_sc(X, SM)

    # create new S.
    S = np.zeros((m,k))
    
    # copy known data.
    S[:,0:k-1] = SM[:,0:k-1]

    # copy new data.
    S[:,-1] = sp
            
    # return it.
    return S, C


def solve_C(X, Z, y, num_threads=1):
    """ solves using QP and multiprocessing """

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = len(np.unique(y))
    C = np.zeros((k,n), dtype=np.float)

    # create S
    S, t = avg_cat(y, np.transpose(Z))

    # create list of jobs.
    jobs = list()

    # execute the jobs.
    if num_threads == 1:

        # loop over each column.
        for j in range(n):

            # call the program.
            c, o = _qp_solve_c(X[:,[j]], S)
            
            # update stuff.
            C[:,j] = c
            
    # return it.
    return S, C

### callable functions ###

def decon(args):
    """ main function for deconvolution"""

    # load data.
    X = np.load(args.X)
    Z = np.load(args.Z)
    y = np.load(args.y)
    n = X.shape[1]
    m = X.shape[0]
    k = args.k

    # sanity checks.
    if args.y == None and args.k == None:
        logging.error("need to specify y or k")
        sys.exit(1)

    if Z.shape[0] != m:
        logging.error("dimensions don't add up")
        sys.exit(1)

    # run deconvolution
    num_threads = 1
    if args.num_threads != None:
        num_threads = args.num_threads

    # run it.
    S, C = solve_C(X, Z, y, num_threads = num_threads)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

    # note that we are done.
    logging.info("done")

def test_UCQP(args):
    """ verify deconvolution using missing data """

    # create the data.
    n = 5
    m = 10
    k = 3
    X, S, SM, C_true = _debug_setup(n, m, k)
    
    # solve each column.
    for j in range(n):
        
        # solve.
        c1, o1 = _lmfit_c(X[:,j], S)
        c2, o2 = _qp_solve_c(X[:,[j]], S)
    
        # score it.
        score1 = meanabs_vector(C_true[:,j], c1)
        score2 = meanabs_vector(C_true[:,j], c2)
        
        print '%.5f %.5f %.5f %.5f' % (score1, o1, score2, o2)
    

def test_UCQPM(args):
    """ verify deconvolution using missing data """

    # create the data.
    n = 10
    m = 5
    k = 3
    X, S, SM, C_true = _debug_setup(n, m, k)
    
    # run the function.
    s, C_pred, o = _lmfit_sc(X, SM)
    
    # score the concentrations.
    for j in range(n):
        score = meanabs_vector(C_true[:,j], C_pred[:,j])
        
        print 'c_%i %.5f' % (j, score)
    
    # score the signature.
    score = meanabs_vector(S[:,-1], s)
    print 's   %.5f %.5f' % (score, o)

### script ###

if __name__ == '__main__':

    # mode parser.
    main_p = argparse.ArgumentParser()
    subp = main_p.add_subparsers(help='sub-command help')

    #### actual functions ####

    # run basic least squares deconvolution
    subpp = subp.add_parser('decon', help='deconvolve the single-cells and mixtures using QP')
    subpp.add_argument('-X', dest='X', required=True, help='mixture: genes x samples')
    subpp.add_argument('-Z', dest='Z', required=True, help='single-cell matrix: genes x cells')
    subpp.add_argument('-y', dest='y', help='cell type labels for each cell')
    subpp.add_argument('-k', type=int, dest='k', help='number of clusters, must be supplied if z_lbls is not')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-S', dest='S', help='output numpy matrix')
    subpp.add_argument('-p', dest='num_threads', type=int, help='number of threads for parallel solving')
    subpp.set_defaults(func=decon)

    #### testing functions ####

    # verify basic deconvolution.
    subpp = subp.add_parser('test_UCQP', help='test the missing deconvolution')
    subpp.set_defaults(func=test_UCQP)

    # verify missing deconvolution.
    subpp = subp.add_parser('test_UCQPM', help='test the missing deconvolution')
    subpp.set_defaults(func=test_UCQPM)

    # parse args.
    args = main_p.parse_args()
    args.func(args)
