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

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )

# app
from utils.matops import *
from utils.misc import *
from utils.plotting import *
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
### classes ###

class SimSingleCell(object):
    """ simulates single cells """

    def __init__(self, SC, y, load=False):
        """ creates object and fills tables"""

        # save pointers.
        self.SC = SC
        self.y = y
        self.k = len(set(list(self.y)))
        self.m = self.SC.shape[0]

        # create random variables.
        if load == False:
            self._create_pa()
            self._create_exp()
        else:
            self.load(load)

    def sample_1(self, n_per_class):
        """ samples according to type 1 """

        # create array.
        Z = np.zeros((self.m, self.k * n_per_class), dtype=np.float)
        y = np.zeros(self.k * n_per_class, dtype=np.int)

        # assign labels to simulated data.
        l = -1
        for z in range(y.shape[0]):
            if z % (n_per_class) == 0:
                l += 1
            y[z] = l

        # loop over each gene.
        for i in range(self.m):

            # loop over each class.
            for l in range(0, y.shape[0], n_per_class):

                # broadcast assign generated values.
                a = self.rv_pa[(i,y[l])].rvs(size=n_per_class)
                b = self.rv_exp[(i,y[l])].rvs(size=n_per_class)
                Z[i,l:l+n_per_class] = a * b

        # fix so all values greater= 0
        idx = Z[Z < 0.0] = 0.0

        # return the sample and labels
        return Z, y

    def save(self, out_path):

        # pickle parameters.
        save_pickle(out_path, {
            'exp_loc':self.exp_loc,
            'exp_scale':self.exp_scale,
            'pa_p':self.pa_p,
        })

    def load(self, in_path):

        # load parameters.
        params = load_pickle(in_path)
        self.exp_loc = params['exp_loc']
        self.exp_scale = params['exp_scale']
        self.pa_p = params['pa_p']

        # create rv
        self.rv_pa = dict()
        self.rv_exp = dict()
        for l, i in itertools.product(range(self.k), range(self.m)):
            self.rv_pa[(i,l)] = bernoulli(self.pa_p[i,l])
            self.rv_exp[(i,l)] = norm(loc=self.exp_loc[i,l], scale=self.exp_scale[i,l])

    def _create_exp(self):
        """ creates expression from normal distrib rv"""

        # create parameter matrix.
        self.exp_loc = np.zeros((self.m,self.k), dtype=np.float)
        self.exp_scale = np.zeros((self.m,self.k), dtype=np.float)
        self.rv_exp = dict()

        # loop over each gene/cell type
        for l in range(self.k):
            SC_l = self.SC[:,np.where(self.y == l)[0]]

            for i in range(self.m):
                SC_m = SC_l[i,:]

                # remove zeros.
                a = SC_m.shape
                SC_m = SC_m[SC_m > 0.0]
                SC_m = SC_m[SC_m > 0.0]
                b = SC_m.shape

                # force zero if all zero.
                if b[0] == 0:
                    SC_m = np.array([0.0 for x in range(10)])

                # fit distribution.
                loc, scale = norm.fit(SC_m)
                self.exp_loc[i,l] = loc
                self.exp_scale[i,l] = scale

                # build rv
                self.rv_exp[(i,l)] = norm(loc=self.exp_loc[i,l], scale=self.exp_scale[i,l])


    def _create_pa(self):
        """ create pres/abs probably rv """

        # create p/a probability.
        self.pa_p = np.zeros((self.m,self.k), dtype=np.float)
        self.rv_pa = dict()

        for l in range(self.k):
            SC_l = self.SC[:,np.where(self.y == l)[0]]

            for i in range(self.m):
                SC_m = SC_l[i,:]

                # compute probability.
                self.pa_p[i,l] = 1.0 - float(len(np.where(SC_m == 0.0)[0])) / float(SC_m.shape[0])

                # build rv.
                self.rv_pa[(i,l)] = bernoulli(self.pa_p[i,l])


def _create_C(n, k, c_type):

    # create the object.
    C = np.zeros((k,n))

    # build each sample.
    for j in range(n):
        if c_type == 0:

            # uniform.
            C[:,j] = 1.0 / float(k)

        elif c_type == 1:

            # random
            C = np.random.rand(k, n)
            for j in range(n):
                C[:,j] = C[:,j] / np.sum(C[:,j])

        else:
            raise NotImplementedError
            
    # round them to 3 decimals.
    C = C.round(3)

    # return the matrix.
    return C

def _create_X(n, m, k, e, C, SAMPLES, ilu, klist):

    # sample from mixture to create X.
    X = np.zeros((m, n), dtype=np.float)

    # for each sample.
    for j in range(n):

        # choose e from each cell type.
        idxs = list()
        for l in range(len(klist)):

            # compute count.
            count = int(np.rint(e * C[l,j]) + 1)

            # choose index.
            idxs += list(np.random.choice(ilu[klist[l]], size = count))

        # build big slice.
        bs = SAMPLES[:,idxs]

        # reduce to active genes.
        bs = bs[0:m,:]

        # assign average to sample.
        for i in range(m):
            X[i,j] = np.average(bs[i,:])

    # return X
    return X

def _create_Z(t, SAMPLES, ilu, klist):

    # track columns and labels
    cols = list()
    lbls = list()

    # build them by cell-type.
    for l in range(len(klist)):

        # choose index.
        cols += list(np.random.choice(ilu[klist[l]], size = t))

        # add to labels.
        lbls += [l] * t

    # take subset sample.
    Z = SAMPLES[:,cols]
    lbls = np.array(lbls)

    # return it
    return Z, lbls


def _create_glist(g, Z, y):


    # score them.
    f, p = feature_selection.f_classif(Z, y)

    # rank them
    idx = np.argsort(f)[::-1]

    # skip nans.
    tmp = list()
    for i in range(len(idx)):
        if np.isnan(f[idx[i]]) == False:
            tmp.append(idx[i])
    idx = np.array(tmp)

    # return features.
    return idx

def _run_UCQP(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):

    # extend paths.
    X_path = '%s.npy' % X_path
    Z_path = '%s.npy' % Z_path
    y_path = '%s.npy' % y_path

    # load data.
    X = np.load(X_path)
    Z = np.load(Z_path)
    y = np.load(y_path)

    # call function.
    S, C = solve_C(X, Z, y, num_threads=1)

    # save them.
    np.save(S_path, S)
    np.save(C_path, C)

    # return the good info.
    return dkey, method_name, S_path, C_path

def _run_UCQPM(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):

    # extend paths.
    X_path = '%s.npy' % X_path
    Z_path = '%s.npy' % Z_path
    y_path = '%s.npy' % y_path

    # load data.
    X = np.load(X_path)
    Z = np.load(Z_path)
    y = np.load(y_path)

    # call function.
    S, C = solve_SC(X, Z, y)

    # sanity check.
    #assert S.shape[1] == len(np.unique(y)) + 1
    #for j in range(X.shape[1]):
    #    assert abs(1.0 - np.sum(C[:,j])) < 0.001

    # save them.
    np.save(S_path, S)
    np.save(C_path, C)

    # return the good info.
    return dkey, method_name, S_path, C_path


def _prep_R(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):

    # extend paths.
    X_tmp = '%s/X.tmp' % wdir
    Z_tmp = '%s/Z.tmp' % wdir

    # strip null filled.
    X = np.load(X_path)
    Z = np.load(Z_path)

    # strip nulls.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(Z[i,:]) > 0.0:
            if np.sum(X[i,:]) > 0.0:
                good_rows.append(i)
    Z = Z[good_rows,:]
    X = X[good_rows,:]

    # write again.
    with open(X_tmp, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    with open(Z_tmp, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

    # return the paths.
    return X_tmp, Z_tmp

def _save_R(dkey, method_name, status, Stmp, Ctmp, S_path, C_path):

    # load the txt and save as numpy.
    S = np.loadtxt(Stmp)
    C = np.loadtxt(Ctmp)
    np.save(S_path, S)
    np.save(C_path, C)

    # return the good info.
    if status == True:
        return dkey, method_name, S_path, C_path
    else:
        return dkey, method_name, None, None

def _run_DECONF(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):
    """ deconvolution """

    # extend paths.
    X_tmp = '%s/X.tmp' % wdir
    Z_tmp = '%s/Z.tmp' % wdir
    X_path = '%s.npy' % X_path
    Z_path = '%s.npy' % Z_path

    # prepare this.
    X_tmp, Z_tmp = _prep_R(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name)

    # run deconvolution.
    status, Stmp, Ctmp = R_DECONF(X_tmp, Z_tmp, y_path, k, S_path, C_path, wdir)

    # save and return.
    return _save_R(dkey, method_name, status, Stmp, Ctmp, S_path, C_path)


def _run_SSKL(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):
    """ deconvolution """

    # extend paths.
    X_tmp = '%s/X.tmp' % wdir
    Z_tmp = '%s/Z.tmp' % wdir
    X_path = '%s.npy' % X_path
    Z_path = '%s.npy' % Z_path
    y_path = '%s.npy' % y_path

    # prepare this.
    X_tmp, Z_tmp = _prep_R(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name)

    # run deconvolution.
    status, Stmp, Ctmp = R_SSKL(X_tmp, Z_tmp, y_path, k, S_path, C_path, wdir)

    # save and return.
    return _save_R(dkey, method_name, status, Stmp, Ctmp, S_path, C_path)


def _run_DSA(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name):
    """ deconvolution """

    # extend paths.
    X_tmp = '%s/X.tmp' % wdir
    Z_tmp = '%s/Z.tmp' % wdir
    X_path = '%s.npy' % X_path
    Z_path = '%s.npy' % Z_path
    y_path = '%s.npy' % y_path

    # prepare this.
    X_tmp, Z_tmp = _prep_R(X_path, Z_path, y_path, k, S_path, C_path, wdir, dkey, method_name)

    # run deconvolution.
    status, Stmp, Ctmp = R_DSA(X_tmp, Z_tmp, y_path, k, S_path, C_path, wdir)

    # save and return.
    return _save_R(dkey, method_name, status, Stmp, Ctmp, S_path, C_path)

def _evl_C(key, C_true, C_pred, scorefns, scorenms):

    # sanity check.
    if C_pred.shape != C_true.shape:
        return None, None

    # flatten them both.
    C_true = np.hstack(C_true.T)
    C_pred = np.hstack(C_pred.T)

    # compute average metric.
    scores = list()
    for scorefn, name in zip(scorefns, scorenms):

        # score it column based average.
        s = scorefn(C_true, C_pred)

        # sanity check.
        scores.append(s)

    # return the score.
    return key, scores

def _evl_s(key, s_true, s_pred, scorefns, scorenms):

    # sanity check.
    if s_pred.shape != s_true.shape:
        return None, None

    # compute average metric.
    scores = list()
    for scorefn, name in zip(scorefns, scorenms):

        # score it column based average.
        s = scorefn(s_true, s_pred)

        # sanity check.
        scores.append(s)

    # return the score.
    return key, scores

def _remap_missing(C_test, S_test, mj, k):
    
    # remap the row order for C
    Ctmp = C_test.copy()
    C_test[mj, :] = Ctmp[-1,:]
    for l in range(mj+1, k):
        C_test[l, :] = Ctmp[l-1, :]

    # remap the column order for H/S
    Stmp = S_test.copy()
    S_test[:, mj] = S_test[:,-1]
    for l in range(mj+1, k):
        S_test[:, l] = Stmp[:, l-1]

    # return it.
    return C_test, S_test

def _save_data(e_dir, key, names, matrs, master):
    
    # create the directory.
    if os.path.isdir(e_dir) == False:
        subprocess.call(["mkdir", "-p", e_dir])

    # save them.
    master[key] = dict()
    for l, r in zip(names, matrs):

        # create the base path.
        l_path = "%s/%s" % (e_dir, l)

        # save in numpy format.
        np.save('%s.npy' % l_path, r)

        # save some in R format.
        if l in set(['ZMG', 'XG']):
            write_r_mat('%s.txt' % l_path, r)

        # save to the master key.
        master[key][l] = l_path

### functions ###

def create_exp(args):
    ''' creates entire experiment '''

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    sim_obj = '%s/sim.cpickle' % sim_dir
    mas_obj = '%s/mas.cpickle' % sim_dir
    dep_obj = '%s/dep.cpickle' % sim_dir

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    kmax = len(np.unique(sc_lbls))

    # create gene list subset.
    glist = _create_glist(SC.shape[0], np.transpose(SC), sc_lbls)

    # always re-order SC.
    SC = SC[glist, :]

    # build the S simulation object.
    if os.path.isfile(sim_obj) == False:
        # create simulator.
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=sim_obj)

    # build the master object.
    if os.path.isfile(mas_obj) == False:
        master = dict()
    else:
        master = load_pickle(mas_obj)

    # create variable iterators.
    nmixs = np.arange(5, 55, 5)                             # mixtures
    mgene = np.array([4, 8, 16, 24, 48, 96, 128, 256])      # genes
    kcell = np.arange(3, 6, 1)                              # cell types
    enois = np.arange(5, 125, 25)                           # samples in mixture
    ctype = np.arange(2)                                    # concentration type.
    qtrys = np.arange(0, 10, 1)                             # repitions

    # simulate the single-cells.
    SAMPLES, samp_y = sim.sample_1(10000)
    SAMPLES.setflags(write=False)
    samp_y.setflags(write=False)

    # make index lookup.
    ilu = dict()
    for a in range(kmax):
        ilu[a] = np.where(samp_y == a)[0]

    # define mapping for cell-types less than given.
    kmap = dict()
    kmap[2] = [1,2]
    kmap[3] = [1,2,3]
    kmap[4] = [1,2,3,4]
    kmap[5] = [0,1,2,3,4]

    # iterate over all variables but the genes.
    for n, k, e, c, q in itertools.product(nmixs, kcell, enois, ctype, qtrys):

        # master key.
        key = [n, k, e, c, q]

        # note it.
        logging.info("building: %s" % ' '.join([str(v) for v in key]))

        # fix number of single-cells to k * n
        t = k * n

        # simulate the single-cells.
        Z, y = _create_Z(t, SAMPLES, ilu, kmap[k])

        # create the signature.
        H, hy = avg_cat(y, np.transpose(Z))

        # create the reduced signature.
        idx = np.where(y!=r)[0]
        ZM = Z[:, idx]
        ym = y[idx]
        HM, hy = avg_cat(ym, np.transpose(ZM))

        # build the concentration
        C = _create_C(n, k, c)

        # create the master X
        X = _create_X(n, m, k, e, C, SAMPLES, ilu, kmap[k])

        raise NotImplementedError

        # create the mixture.
        X = _create_X(n, m, k, e, C, SAMPLES, ilu, kmap[k])

        # create experiment paths.
        e_dir = '%s/master/%i_%i_%i_%i_%i' % (sim_dir, n, k, e, c, q)

        # create the directory.
        if os.path.isdir(e_dir) == False:
            subprocess.call(["mkdir", "-p", e_dir])

        # save them.
        names = "ZyHCX"
        matrs = [Z, y, H, C, X]
        master[key] = dict()
        for l, r in zip(names, matrs):

            # create the base path.
            l_path = "%s/%s" % (e_dir, l)

            # save in numpy format.
            np.save('%s.npy' % l_path, r)

            # save some in R format.
            if l == "Z" or l == "X":
                write_r_mat('%s.txt' % l_path, r)

            # save to the master key.
            master[key][l] = l_path

    # save the master.
    save_pickle(mas_obj, master)

    # create the derivative combinations.
    for n, k, e, c, q in itertools.product(nmixs, kcell, enois, ctype, qtrys):

        # record it.
        logging.info("loading: %i %i %i %i %i" % (n, k, e, c, q))

        # master key.
        mkey = (n, k, e, c, q)

        # load the master data.
        Z = np.load('%s.npy' % master[mkey]['Z'])
        y = np.load('%s.npy' % master[mkey]['y'])
        H = np.load('%s.npy' % master[mkey]['H'])
        C = np.load('%s.npy' % master[mkey]['C'])
        X = np.load('%s.npy' % master[mkey]['X'])

        # no funny business.
        for p in [Z, y, H, C, X]:
            p.setflags(write=False)

        # loop over derivatives.
        for m in mgene:

            # dependent key.
            dkey = (n, k, e, c, q, m)

            # overwrite mode.
            if dkey in dependent and args.overwrite == False:
                continue

            # note it.
            logging.info("building: %i" % (m))

            # create single-cell subset.
            Xs = X[range(m),:]
            Zs = Z[range(m),:]

            # create experiment paths.
            e_dir = '%s/dependent/%i_%i_%i_%i_%i/%i' % (sim_dir, n, k, e, c, q, m)

            # create the directory.
            if os.path.isdir(e_dir) == False:
                subprocess.call(["mkdir", "-p", e_dir])

            # save them.
            names = "XZ"
            matrs = [Xs, Zs]
            dependent[dkey] = dict()
            for l, r in zip(names, matrs):

                # create the base path.
                l_path = "%s/%s" % (e_dir, l)

                # save in numpy format.
                np.save('%s.npy' % l_path, r)

                # save some in R format.
                if l == "Z" or l == "X":
                    write_r_mat('%s.txt' % l_path, r)

                # save to the master key.
                dependent[dkey][l] = l_path

    # save the master.
    save_pickle(mas_obj, master)
    save_pickle(dep_obj, dependent)


def create_mis(args):
    ''' creates entire experiment '''

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    sim_obj = '%s/sim.cpickle' % sim_dir
    mas_obj = '%s/mas.cpickle' % sim_dir
    dep_obj = '%s/dep.cpickle' % sim_dir

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    kmax = len(np.unique(sc_lbls))

    # create gene list subset.
    glist = _create_glist(SC.shape[0], np.transpose(SC), sc_lbls)

    # always re-order SC.
    SC = SC[glist, :]

    # build the S simulation object.
    if os.path.isfile(sim_obj) == False:
        # create simulator.
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=sim_obj)

    # build the master object.
    if os.path.isfile(mas_obj) == False:
        master = dict()
    else:
        master = load_pickle(mas_obj)

    # create variable iterators.
    #nmixs = np.arange(5, 25, 5)                            # mixtures.
    nmixs = np.array([10])                                  # mixtures.
    mgene = np.array([16])                                   # genes.
    #mgene = np.array([4, 24])                              # genes.
    kcell = np.arange(4, 5, 1)                              # cell types.
    #enois = np.array([0, 2, 6, 16, 64, 128])                # samples in mixture.
    enois = np.array([0])                                   # samples in mixture.
    #ctype = np.arange(2)                                    # concentration type.
    ctype = np.array([1])                                    # concentration type.
    rmode = np.arange(0, 6, 1)                              # cell-type to remove
    qtrys = np.arange(0, 10, 1)                              # repitions

    # simulate the single-cells.
    SAMPLES, samp_y = sim.sample_1(10000)
    SAMPLES.setflags(write=False)
    samp_y.setflags(write=False)

    # make index lookup.
    ilu = dict()
    for a in range(kmax):
        ilu[a] = np.where(samp_y == a)[0]

    # define mapping for cell-types less than given.
    kmap = dict()
    kmap[2] = [1,2]
    kmap[3] = [1,2,3]
    kmap[4] = [1,2,3,4]
    kmap[5] = [0,1,2,3,4]

    # create the master matrix.
    for n, k, e, c, r, q in itertools.product(nmixs, kcell, enois, ctype, rmode, qtrys):
       
        # skip the crazy ones.
        if r >= k: continue
       
        # begin key.
        mkey = [n, k, e, c, r, q]

        # note it.
        logging.info("building: %s" % ' '.join([str(v) for v in mkey]))

        # fix number of single-cells to k * n
        t = k * n

        # simulate the single-cells.
        Z, y = _create_Z(t, SAMPLES, ilu, kmap[k])

        # create the signature.
        H, hy = avg_cat(y, np.transpose(Z))

        # create the reduced signature.
        idx = np.where(y!=r)[0]
        ZM = Z[:, idx]
        ym = y[idx]

        # renumber labeling.
        for l in range(r+1, k):
            ym[np.where(ym == l)[0]] -= 1
            
        # compute signature again.
        HM, hy = avg_cat(ym, np.transpose(ZM))

        # build the concentration
        C = _create_C(n, k, c)

        # create the master X
        if e != 0:
            # create using sampling.
            X = _create_X(n, m, k, e, C, SAMPLES, ilu, kmap[k])
        else:
            # create using dot product.
            X = np.dot(H, C)

        # save first masters.
        m_dir = '%s/true/%i_%i_%i/%i_%i_%i' % (sim_dir, n, k, e, c, r, q)
        names = ["Z", "y", "H", "C" , "X", "ZM", "ym", "HM"]
        matrs = [Z, y, H, C, X, ZM, ym, HM]
        _save_data(m_dir, tuple(mkey), names, matrs, master)

        # loop over the gene variable.
        for m in mgene:
            
            # skip the crazy ones.
            if n >= m: continue
            
            # strip X, ZM to right size.
            XG = X[0:m, :]
            ZMG = ZM[0:m, :]

            # update key.
            dkey = tuple(mkey + [m])
            
            # skip saving this guy.
            if dkey in master and args.overwrite == False:
                continue

            # create experiment paths.
            d_dir = '%s/input/%i_%i_%i_%i/%i_%i_%i' % (sim_dir, n, k, e, c, r, q, m)
            names = ["XG", "ZMG"]
            matrs = [XG, ZMG]
            _save_data(d_dir, dkey, names, matrs, master)

    # save the master.
    save_pickle(mas_obj, master)

def run_exp(args):
    """ runs the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    dep_obj = '%s/dep.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir

    # extract method info.
    method_name, method_fn = args.method_sig

    # load the simulation data stuff.
    master = load_pickle(mas_obj)
    dependent = load_pickle(dep_obj)

    # create results dictionary.
    if os.path.isfile(res_obj) == False:
        results = dict()
    else:
        results = load_pickle(res_obj)

    # track progress.
    total = len(dependent.keys())

    # load R libraries.
    load_R_libraries()

    # create the pool.
    pool = Pool(processes = args.num_cpu)

    # loop over each dependent.
    active = list()
    cnt = 0
    for dkey in dependent:

        # check if we can skip.
        if dkey in results and method_name in results[dkey]:
            if os.path.isfile(results[dkey][method_name]["S"]) and os.path.isfile(results[dkey][method_name]["C"]):
                continue

        # simplify.
        n, k, e, c, q, m = dkey

        # infer master key.
        mkey = (n, k, e, c, q)

        # extract paths.
        Z_path = dependent[dkey]['Z']
        X_path = dependent[dkey]['X']
        y_path = master[mkey]['y']

        # make a working directory.
        w_dir = '%s/working/%s/%i_%i_%i_%i_%i/%i' % (sim_dir, method_name, n, k, e, c, q, m)
        S_path = '%s/S.npy' % w_dir
        C_path = '%s/C.npy' % w_dir

        # create the working directory.
        if os.path.isdir(w_dir) == False:
            subprocess.call(['mkdir', '-p', w_dir])

        # call run method, return process.
        logging.info("running: %i of %i: %s %i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, q, m))
        active.append(pool.apply_async(method_fn, (X_path, Z_path, y_path, k, S_path, C_path, w_dir, dkey, method_name)))
        #x = method_fn(X_path, Z_path, y_path, k, S_path, C_path, w_dir, dkey, method_name)
        cnt += 1

        # flush pool occasionally.
        if cnt % 1000 == 0:

            # wait on all active processes.
            for a in active:

                # get info and add it.
                dkey, method_name, S_path, C_path = a.get()
                if S_path == None: continue
                if dkey not in results:
                    results[dkey] = dict()
                results[dkey][method_name] = {"S":S_path, "C":C_path}

            # clear it.
            active = list()

            # save the results.
            save_pickle(res_obj, results)

    # close pool and get results.
    pool.close()
    pool.join()

    # add to results.
    for a in active:

        # get info and add it.
        dkey, method_name, S_path, C_path = a.get()
        if S_path == None: continue
        if dkey not in results:
            results[dkey] = dict()
        results[dkey][method_name] = {"S":S_path, "C":C_path}

    # save the results.
    save_pickle(res_obj, results)


def run_mis(args):
    """ runs the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    dep_obj = '%s/dep.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir

    # extract method info.
    method_name, method_fn = args.method_sig

    # load the simulation data stuff.
    master = load_pickle(mas_obj)

    # create results dictionary.
    if os.path.isfile(res_obj) == False:
        results = dict()
    else:
        results = load_pickle(res_obj)

    # track progress.
    total = len(master.keys())

    # load R libraries.
    #load_R_libraries()

    # create the pool.
    if args.debug == False:
        pool = Pool(processes = args.num_cpu)

    # solve easiest first.
    keys = sorted(master.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # loop over each dependent.
    active = list()
    cnt = 0
    for dkey in keys:

        # skip short keys.
        if len(dkey) != 7: continue

        # check if we can skip.
        if dkey in results and method_name in results[dkey] and args.debug == False:
            if os.path.isfile(results[dkey][method_name]["S"]) and os.path.isfile(results[dkey][method_name]["C"]):
                continue

        # simplify.
        n, k, e, c, r, q, m = dkey
        mkey = n, k, e, c, r, q

        # extract paths.
        XG_path = master[dkey]['XG']
        ZMG_path = master[dkey]['ZMG']
        ym_path = master[mkey]['ym']

        # make a working directory.
        w_dir = '%s/working/%s/%i_%i_%i_%i/%i_%i_%i' % (sim_dir, method_name, n, k, e, c, r, q, m)
        S_path = '%s/S.npy' % w_dir
        C_path = '%s/C.npy' % w_dir

        # create the working directory.
        if os.path.isdir(w_dir) == False:
            subprocess.call(['mkdir', '-p', w_dir])

        # call run method, return process.
        fnargs = (XG_path, ZMG_path, ym_path, k, S_path, C_path, w_dir, dkey, method_name)
        if args.debug == False:

            # run it in multiprocessing pool
            logging.info("running: %i of %i: %s %i_%i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, r, q, m))
            active.append(pool.apply_async(method_fn, fnargs))
        else:

            # run it in this thread.
            logging.debug("debugging: %i of %i: %s %i_%i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, r, q, m))
            debug_result = method_fn(*fnargs)
            
            # load cheat data.
            S_true = np.load('%s.npy' % master[mkey]['H'])
            S_true  = S_true[0:m,:]
            C_true = np.load('%s.npy' % master[mkey]['C'])
            S_pred = np.load(debug_result[2])
            C_pred = np.load(debug_result[3])
            C_pred, S_pred = _remap_missing(C_pred, S_pred, r, k)
            
            # evaluate it directly.
            key, c_score = _evl_C(dkey, C_true, C_pred, [meanabs_vector], ["meanabs"])
            key, s_score = _evl_s(dkey, S_true[:,r], S_pred[:,r], [meanrel_vector], ["meanrel"])

            # report it.
            logging.debug("s_score: %.5f\tc_score: %.5f" % (s_score[0], c_score[0]))

        # progress tracking variable.
        cnt += 1

        # flush pool occasionally.
        if cnt % 500 == 0 and args.debug == False:

            # wait on all active processes.
            for a in active:

                # get info and add it.
                dkey, method_name, S_path, C_path = a.get()
                if S_path == None: continue
                if dkey not in results:
                    results[dkey] = dict()
                results[dkey][method_name] = {"S":S_path, "C":C_path}

            # clear it.
            active = list()

            # save the results.
            save_pickle(res_obj, results)

    # close pool and get results. 
    if args.debug == False:
        
        # close the worker pool.
        pool.close()
        pool.join()

        # add to results.
        for a in active:

            # get info and add it.
            dkey, method_name, S_path, C_path = a.get()
            if S_path == None: continue
            if dkey not in results:
                results[dkey] = dict()
            results[dkey][method_name] = {"S":S_path, "C":C_path}

        # save the results.
        save_pickle(res_obj, results)

def pred_mis(args):
    """ runs the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir

    # load the simulation data stuff.
    master = load_pickle(mas_obj)

    # solve easiest first.
    keys = sorted(master.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # loop over each dependent.
    for dkey in keys:

        # skip short keys.
        if len(dkey) != 7: continue

        # simplify.
        n, k, e, c, r, q, m = dkey
        mkey = n, k, e, c, r, q

        # extract paths.
        XG_path = master[dkey]['XG']
        ZMG_path = master[dkey]['ZMG']
        ym_path = master[mkey]['ym']

        # make a working directory.
        ucqp_dir = '%s/working/%s/%i_%i_%i_%i/%i_%i_%i' % (sim_dir, "UCQP", n, k, e, c, r, q, m)
        ucqpm_dir = '%s/working/%s/%i_%i_%i_%i/%i_%i_%i' % (sim_dir, "UCQPM", n, k, e, c, r, q, m)
        S_path = '%s/S.npy' % ucqp_dir
        S_path_m = '%s/S.npy' % ucqpm_dir
        C_path = '%s/C.npy' % ucqp_dir
        C_path_m = '%s/C.npy' % ucqpm_dir
        
        for d in [ucqp_dir, ucqpm_dir]:
            subprocess.call(["mkdir", "-p", d])
        
        # run with missing data.
        _run_UCQPM(XG_path, ZMG_path, ym_path, k, S_path_m, C_path_m, ucqpm_dir, dkey, "UCQPM")
        _run_UCQP(XG_path, ZMG_path, ym_path, k, S_path, C_path, ucqp_dir, dkey, "UCQP")
        
        # load cheat data.
        X = np.load('%s.npy' % XG_path)
        S_true = np.load('%s.npy' % master[mkey]['H'])
        S_true  = S_true[0:m,:]
        C_true = np.load('%s.npy' % master[mkey]['C'])
        
        # load missing prediction
        print S_path_m, S_path
        S_predm = np.load(S_path_m)
        C_predm = np.load(C_path_m)
        C_predm, S_predm = _remap_missing(C_predm, S_predm, r, k)
        
        # load the other prediction.
        S_pred = np.load(S_path)
        C_pred = np.load(C_path)
        
        # compute the residual.
        X1 = np.dot(S_pred, C_pred)
        X2 = np.dot(S_predm, C_predm)
        
        xl = np.hstack(X.T)
        x1l = np.hstack(X1.T)
        x2l = np.hstack(X2.T)
        
        r1 = np.sqrt(np.sum(np.square(xl - x1l)))
        r2 = np.sqrt(np.sum(np.square(xl - x2l)))
        
        print r1, r2

            
        # evaluate it directly.
        #key, c_score = _evl_C(dkey, C_true, C_pred, [meanabs_vector], ["meanabs"])
        #key, s_score = _evl_s(dkey, S_true[:,r], S_pred[:,r], [meanrel_vector], ["meanrel"])

        # report it.
        #logging.debug("s_score: %.5f\tc_score: %.5f" % (s_score[0], c_score[0]))



def evl_exp(args):
    """ evaluates the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    dep_obj = '%s/dep.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir

    # extract method info.
    method_name, method_fn = args.method_sig

    # load the simulation data stuff.
    master = load_pickle(mas_obj)
    dependent = load_pickle(dep_obj)
    results = load_pickle(res_obj)

    # track progress.
    total = len(dependent.keys())

    # sort the keys.
    keys = sorted(results.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # define the score functions.
    scorefns = [meanabs_vector, rmse_vector, meanrel_vector]
    scorenms = ["meanabs", "rmse", "meanrel"]

    # loop over each dependent.
    pkey = ""
    output = list()
    for dkey in keys:

        # expand the key.
        n, k, e, c, q, m = dkey
        mkey = (n, k, e, c, q)
        skey = n, k, e, c, m        # remove reference ot repeat variable

        # load the true concentrations.
        if mkey != pkey:
            C_true = np.load('%s.npy' % master[mkey]['C'])

        # skip if method not present.
        if method_name not in results[dkey]: continue

        # score it.
        output.append(_evl_C(skey, C_true, results[dkey][method_name]['C'], scorefns, scorenms))

    # compress by key
    compressed = dict()
    for key, scores in output:

        # skip if missing.
        if key == None: continue

        # boot the key.
        if key not in compressed:
            compressed[key] = dict()

        # loop over each score.
        for s, name in zip(scores, scorenms):

            # boot the scores.
            if name not in compressed[key]:
                compressed[key][name] = list()

            # add it.
            compressed[key][name].append(s)

    # sort, average and print.
    keys = sorted(compressed.keys(), key=operator.itemgetter(0,1,2,3,4))
    for key in keys:
        for name in scorenms:
            txt = '%i,%i,%i,%i,%i' % key
            txt = '%s,%s,%.5f' % (txt, name, np.average(np.array(compressed[key][name])))
            print txt



def evl_mis(args):
    """ evaluates the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir

    # extract method info.
    method_name = args.method_name
    assert method_name != "UCQP"
    
    # load the simulation data stuff.
    master = load_pickle(mas_obj)
    results = load_pickle(res_obj)

    # sort the keys.
    keys = sorted(results.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # define the score functions.
    scorefns = [meanabs_vector, rmse_vector, meanrel_vector]
    scorenms = ["meanabs", "rmse", "meanrel"]

    # loop over each dependent.
    output = list()
    for dkey in keys:

        # skip short keys.
        if len(dkey) != 7: continue

        # expand the key.
        n, k, e, c, r, q, m = dkey
        mkey = (n, k, e, c, r, q)
        skey = n, k, e, c, r, m        # remove reference ot repeat variable

        # load the true concentrations.
        S_true = np.load('%s.npy' % master[mkey]['H'])
        S_true = S_true[0:m,:]
        C_true = np.load('%s.npy' % master[mkey]['C'])
        
        S_pred = np.load(results[dkey][method_name]['S'])
        C_pred = np.load(results[dkey][method_name]['C'])
        C_pred, S_pred = _remap_missing(C_pred, S_pred, r, k)
            
        # evaluate it directly.
        key, c_score = _evl_C(dkey, C_true, C_pred, scorefns, scorenms)
        key, s_score = _evl_s(dkey, S_true[:,r], S_pred[:,r], scorefns, scorenms)
        
        # score it.
        for h in range(len(scorenms)):
            output.append((skey, scorenms[h], s_score[h], c_score[h]))

    # compress by key
    compressed = dict()
    for key, name, s_score, c_score in output:

        # skip if missing.
        if key == None: continue

        # boot the key.
        if key not in compressed:
            compressed[key] = dict()

        # boot the scores.
        if name not in compressed[key]:
            compressed[key][name] = dict()
            compressed[key][name]['S'] = list()
            compressed[key][name]['C'] = list()

        # add it.
        compressed[key][name]['S'].append(s_score)
        compressed[key][name]['C'].append(c_score)

    # sort, average and print.
    keys = sorted(compressed.keys(), key=operator.itemgetter(0,1,2,3,4))
    for key in keys:
        for name in scorenms:
            s = np.average(np.array(compressed[key][name]['S']))
            c = np.average(np.array(compressed[key][name]['C']))
            txt = '%i,%i,%i,%i,%i,%i' % key
            txt = '%s,%s,%.5f,%.5f' % (txt, name, s, c)
            print txt

### script ###

if __name__ == '__main__':

    # mode parser.
    main_p = argparse.ArgumentParser()
    subp = main_p.add_subparsers(help='sub-command help')

    ## create simulations ##

    # comprehensive simulation
    subpp = subp.add_parser('create_exp', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-o', dest='overwrite', action='store_true', help='overwrite existing, otherwise will only add new')
    subpp.set_defaults(func=create_exp)

    # missing simulation
    subpp = subp.add_parser('create_mis', help='creates experiment data with missing cell-type')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-o', dest='overwrite', action='store_true', help='overwrite existing, otherwise will only add new')
    subpp.set_defaults(func=create_mis)

    ## run the experiment ##

    # comprehensive simulation
    subpp = subp.add_parser('run_exp', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    me_g = subpp.add_mutually_exclusive_group(required=True)
    me_g.add_argument('-UCQP', dest='method_sig', action='store_const', const=('UCQP', _run_UCQP), help='UCQP')
    me_g.add_argument('-DECONF', dest='method_sig', action='store_const', const=('DECONF', _run_DECONF), help='DECONF')
    me_g.add_argument('-SSKL', dest='method_sig', action='store_const', const=('SSKL', _run_SSKL), help='SSKL')
    me_g.add_argument('-DSA', dest='method_sig', action='store_const', const=('DSA', _run_DSA), help='DSA')
    subpp.set_defaults(func=run_exp)

    # missing simulation
    subpp = subp.add_parser('run_mis', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    me_g = subpp.add_mutually_exclusive_group(required=True)
    me_g.add_argument('-UCQPM', dest='method_sig', action='store_const', const=('UCQPM', _run_UCQPM), help='UCQPM')
    me_g.add_argument('-UCQP', dest='method_sig', action='store_const', const=('UCQP', _run_UCQP), help='UCQP')
    subpp.add_argument('-d', dest='debug', action='store_true', help='use debug mode (i.e.) skip multiprocessing')
    subpp.set_defaults(func=run_mis)

    # prediction accuracy
    subpp = subp.add_parser('pred_mis', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    subpp.set_defaults(func=pred_mis)

    ## evaluate the results ##

    # comprehensive simulation
    subpp = subp.add_parser('evl_exp', help='evaluates experiment results')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    me_g = subpp.add_mutually_exclusive_group(required=True)
    me_g.add_argument('-UCQP', dest='method_sig', action='store_const', const=('UCQP', _run_UCQP), help='UCQP')
    me_g.add_argument('-DECONF', dest='method_sig', action='store_const', const=('DECONF', _run_DECONF), help='DECONF')
    me_g.add_argument('-SSKL', dest='method_sig', action='store_const', const=('SSKL', _run_SSKL), help='SSKL')
    me_g.add_argument('-DSA', dest='method_sig', action='store_const', const=('DSA', _run_DSA), help='DSA')
    subpp.set_defaults(func=evl_exp)

    # missing simulation
    subpp = subp.add_parser('evl_mis', help='evaluates experiment results')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    me_g = subpp.add_mutually_exclusive_group(required=True)
    me_g.add_argument('-UCQP', dest='method_name', action='store_const', const='UCQP', help='UCQP')
    me_g.add_argument('-UCQPM', dest='method_name', action='store_const', const='UCQPM', help='UCQPM')
    subpp.set_defaults(func=evl_mis)

    ## plotting ##

    # plot single-cell matrix using PCA.
    subpp = subp.add_parser('plot_pca_sc', help='plots the single cell values using PCA')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-fig', dest='fig_file', type=str, required=True, help='figure file')
    subpp.set_defaults(func=pca_sc)

    # parse args.
    args = main_p.parse_args()
    args.func(args)
