#!/usr/bin/python
'''
prepares various data sets for running. This is not protable code.
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
import itertools
import numpy as np
import numpy.random
import random
import math
import operator
import StringIO
from multiprocessing import Pool
warnings.filterwarnings("ignore")
# statistics.
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import gmean
from sklearn import feature_selection
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.base import ClassifierMixin

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )

# app
from utils.matops import *
from utils.misc import *
from simulation import SimSingleCell
#from utils.plotting import *
#from utils.heirheat import *
#from utils.cluster import *
#from utils.rfuncs import *
from scdecon import solve_C, solve_SC

# hack to silence argparser.
warnings.filterwarnings('ignore', category=DeprecationWarning)

### configuration ###

### definitions ###
## dimensions
# n: # of mixed samples
# m: # of biomarkers
# k: # of celltypes
# l: # of single cell samples

## matrix
# X:    m * n
# S:    m * k
# C:    k * n
# SC:   m * l

## iterators.
# n: i
# m: j
# k: a
# l: b

INPUT_FILE = "%s/input.cpickle"
K_MAX = 5

#################################################
### classes ###
#################################################
class ScPredict(ClassifierMixin):

    def __init__(self, S):
        self.S = S

    def fit(self, X, y):

        # perform deconvolution.
        Xtmp = np.transpose(X)
        zz, C = solve_C(Xtmp, self.S, y, num_threads=1, avg_meth=99)

        print C[:,0]
        sys.exit()

        # reshape C.
        C = np.transpose(C)

        ## basic classifier.
        self.basic_cl = linear_model.LogisticRegression(C=1e5)
        self.basic_cl.fit(X, y)

        ## frequency classifiers.
        self.freqs_cl = linear_model.LogisticRegression(C=1e5)
        self.freqs_cl.fit(C, y)

    def predict(self, X):

        ## basic prediction.
        c1 = self.basic_cl.predict(X)

        ## frequency prediction.
        zz, C = solve_C(np.transpose(X), self.S, [], num_threads=1, avg_meth=99)
        C = np.transpose(C)
        c2 = self.freqs_cl.predict(C)

        # take vote.
        c = np.zeros(c1.shape[0], dtype=np.int)
        for i in range(c1.shape[0]):
            if c1[i] == c2[i]:
                c[i] = c1[i]
            else:
                c[i] = c2[i]

        # return it.
        return c

#################################################
### functions ###
#################################################

def create_exp(args):

    # make the important cell types.
    W1, y1 = make_classification(50, n_features=25, n_informative=5, n_classes=2)

    # make the noise cell types.
    W2, y2 = make_classification(50, n_features=25, n_informative=5, n_classes=2)

    # increment y2 to become noise.
    y2 = y2 + 2

    # create the reference profiles.
    W = np.vstack((W1, W2))
    r = np.concatenate((y1[y1 == 0], y2))

    # create single-cell simulator.
    sim_a = SimSingleCell(np.transpose(W), r)
    Z, y = sim_a.sample_1(100)

    # create the signature.
    H, hy = avg_cat(y, np.transpose(Z))

    # create the concentrations.
    C1 = np.zeros((H.shape[1], 50))
    C1[:, :] = .33
    C1[0, :] = 0.0

    C2 = np.zeros((H.shape[1], 50))
    C2[:, :] = .33
    C2[1, :] = 0.0

    # create the bulk samples.
    X1 = np.dot(H, C1)
    X2 = np.dot(H, C2)
    X = np.hstack((X1, X2))

    # set these labels into noise.
    y = np.array([0] * 50 + [1] * 50)

    # save the data.
    data = [X, y, H]
    save_pickle(INPUT_FILE % args.sim_dir, data)


def scpredict(args):

    # load the data.
    X, y, S = load_pickle(INPUT_FILE % args.sim_dir)

    # create model.
    log_cl = linear_model.LogisticRegression(C=1e5)
    sc_cl = ScPredict(S)

    # create the pipelines.
    bspipe = Pipeline([('cl', log_cl)])
    scpipe = Pipeline([('cl', sc_cl)])

    # compute the scores.
    with warnings.catch_warnings(record=True) as warns:

        # run the score.
        bs_scores = cross_validation.cross_val_score(bspipe, X, y, cv=K_MAX, scoring='f1')
        sc_scores = cross_validation.cross_val_score(scpipe, X, y, cv=K_MAX, scoring='f1')

        # calculate the average.
        bs_mean, bs_std = bs_scores.mean(), bs_scores.std() * 2
        cs_mean, cs_std = bs_scores.mean(), bs_scores.std() * 2

        print (bs_mean, bs_std), (cs_mean, cs_std)


#################################################
### script ###
#################################################
if __name__ == '__main__':

    # mode parser.
    main_p = argparse.ArgumentParser()
    subp = main_p.add_subparsers(help='sub-command help')

    ## create simulations ##

    # simulation
    subpp = subp.add_parser('create_exp', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=create_exp)

    # prediction.
    subpp = subp.add_parser('predict', help='prediction')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=scpredict)

    # parse args.
    args = main_p.parse_args()
    args.func(args)
