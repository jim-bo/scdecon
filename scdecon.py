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

### middle functions ###

def solve_C(X, S, num_threads=1):
    """ solves using QP and multiprocessing """

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = S.shape[1]
    C = np.zeros((k,n), dtype=np.float)

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
    return C

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

    # decide which method to build S.
    if args.y != None:
        S, t = avg_cat(y, np.transpose(Z))
    else:
        raise NotImplementedError, 'i pity the fool that needs this'

    # run deconvolution
    num_threads = 1
    if args.num_threads != None:
        num_threads = args.num_threads

    # run it.
    C = solve_C(X, S, num_threads = num_threads)

    # sanity check
    if np.isnan(C).any() == True:
        logging.warning("found nan")
        C = np.nan_to_num(C)

    # save it using numpy.
    np.save(args.C, C)
    np.save(args.S, S)

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
    subpp.add_argument('-k', type=int, dest='k', help='number of clusters, must be supplied if z_lbls is not')
    subpp.add_argument('-C', dest='C', help='output numpy matrix')
    subpp.add_argument('-S', dest='S', help='output numpy matrix')
    subpp.add_argument('-p', dest='num_threads', type=int, help='number of threads for parallel solving')
    subpp.set_defaults(func=decon)

    ### pipeline ###

    # parse args.
    args = main_p.parse_args()
    args.func(args)
