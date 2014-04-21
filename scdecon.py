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
import rpy2.robjects as R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# scikit
import sklearn.cluster
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.decomposition import PCA

# cvxpy
import cvxpy

# app
from utils.matops import *
from utils.rfuncs import *
from utils.misc import *

# logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )

# hack to silence argparser.
warnings.filterwarnings('ignore', category=DeprecationWarning)

### configuration ###

### definitions ###

### classes ###

###  ###

### internal functions ###
def _avg_S(Z, y):
    """ create S from known labels """
    S, cats = avg_cat(y, np.transpose(Z))
    return S


def _cqp(S, x, k):
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
    sum1 = cvxpy.equals(cvxpy.sum(c), 1.0)
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
    return c.value

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
        c = _cqp(S, X[:,j], k)

        # save it.
        for i in range(k):
            C[i,j] = c[i,0]

    # return it.
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
    subpp.add_argument('-k', type=int, dest='num_cluster', help='number of clusters, must be supplied if z_lbls is not')
    subpp.set_defaults(func=decon)

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
