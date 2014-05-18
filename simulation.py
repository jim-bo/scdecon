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
from utils.rfuncs import *
#from scdecon import solve_C, solve_SC

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


def _create_C(n, k, c_type, r=None, q=None):

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

        elif c_type == 2:
            
            # uniform after forcing r to be something.
            for j in range(n):
                C[r, j] = q
                
            for j in range(n):
                for l in range(k):
                    if l != r:
                        C[l, j] = (1.0-q) / float(k-1)
            
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

def _evl_C(C_true, C_pred, scorefn):

    # sanity check.
    if C_pred.shape != C_true.shape:
        return None

    # flatten them both.
    C_true = np.hstack(C_true.T)
    C_pred = np.hstack(C_pred.T)


    # score it column based average.
    s = scorefn(C_true, C_pred)

    # return the score.
    return s

def _evl_s(s_true, s_pred, scorefn):

    # sanity check.
    if s_pred.shape != s_true.shape:
        return None
        
    # score it column based average.
    s = scorefn(s_true, s_pred)

    # return the score.
    return s

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

def _match_pred(C_pred, S_pred, C_true, S_true):
    
    # make the signatures.
    try:
        order = match_signatures(S_true, S_pred)
    except ValueError as e:
        return C_pred, S_pred
    
    # remap both S and C.
    S_pred = S_pred[:,order]
    C_pred = C_pred[order,:]
    
    # return them
    return C_pred, S_pred

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
   
    ## vary the noise ##
    nmixs = np.array([5, 10, 15, 20, 25, 30])
    mgene = np.array([8, 16, 32, 64, 128, 256])   
    kcell = np.array([5])
    enois = np.array([5, 20, 40, 100])
    ctype = np.array([0, 1])

    ## these guys stay the same.
    #rmode = np.arange(0, 6, 1)
    qtrys = np.arange(0, 10, 1)

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
    for n, k, e, c, q in itertools.product(nmixs, kcell, enois, ctype, qtrys):
       
        # begin key.
        mkey = [n, k, e, c, q]

        # fix number of single-cells to k * n
        t = k * n

        # simulate the single-cells.
        Z, y = _create_Z(t, SAMPLES, ilu, kmap[k])

        # create the signature.
        H, hy = avg_cat(y, np.transpose(Z))

        # build the concentration
        C = _create_C(n, k, c)

        # create the master X
        if e != 100:
            # create using sampling.
            X = _create_X(n, 256, k, e, C, SAMPLES, ilu, kmap[k])
        else:
            # create using dot product.
            X = np.dot(H, C)

        # save first masters.
        m_dir = '%s/true/%i_%i_%i/%i_%i' % (sim_dir, n, k, e, c, q)
        names = ["Z", "y", "H", "C" , "X"]
        matrs = [Z, y, H, C, X]
        _save_data(m_dir, tuple(mkey), names, matrs, master)

        # loop over the gene variable.
        for m in mgene:
            
            # skip the crazy ones.
            if n > m: continue
            
            # strip X, ZM to right size.
            XG = X[0:m, :]
            ZG = Z[0:m, :]

            # update key.
            dkey = tuple(mkey + [m])
            
            # skip saving this guy.
            if dkey in master and args.overwrite == False:
                continue

            # note it.
            logging.info("building: %s" % ' '.join([str(v) for v in dkey]))

            # create experiment paths.
            d_dir = '%s/input/%i_%i_%i_%i/%i_%i' % (sim_dir, n, k, e, c, q, m)
            names = ["XG", "ZG"]
            matrs = [XG, ZG]
            _save_data(d_dir, dkey, names, matrs, master)

    # save the master.
    save_pickle(mas_obj, master)


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

    ## vary number of mixes ##
    #nmixs = np.arange(5,55,5)
    #mgene = np.array([16])   
    #kcell = np.array([5])
    #enois = np.array([0])
    #ctype = np.array([1])
    
    ## vary number of genes ##
    #nmixs = np.array([8])
    #mgene = np.array([8, 16, 24, 48, 96, 128])   
    #kcell = np.array([5])
    #enois = np.array([0])
    #ctype = np.array([1])
    
    ## vary the noise ##
    #nmixs = np.array([8])
    #mgene = np.array([16])   
    #kcell = np.array([5])
    #enois = np.array([5, 10, 20, 40, 80, 100])
    #ctype = np.array([1])
    
    ## vary the noise ##
    nmixs = np.array([5, 10, 15, 20])
    mgene = np.array([8, 16, 32, 64])   
    kcell = np.array([5])
    enois = np.array([5, 20, 40, 100])
    ctype = np.array([1])

    ## these guys stay the same.
    rmode = np.arange(0, 6, 1)
    qtrys = np.arange(0, 1, 1)

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
        if e != 100:
            # create using sampling.
            X = _create_X(n, 256, k, e, C, SAMPLES, ilu, kmap[k])
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
            if n > m: continue
            
            # strip X, ZM to right size.
            XG = X[0:m, :]
            ZMG = ZM[0:m, :]

            # update key.
            dkey = tuple(mkey + [m])
            
            # skip saving this guy.
            if dkey in master and args.overwrite == False:
                continue

            # note it.
            logging.info("building: %s" % ' '.join([str(v) for v in dkey]))

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

    # load R libraries.
    load_R_libraries()

    # create the pool.
    if args.debug == False:
        pool = Pool(processes = args.num_cpu)

    # solve easiest first.
    keys = sorted(master.keys(), key=operator.itemgetter(0,1,2,3,4))

    # count total jobs.
    total = np.sum(np.array([len(dkey) for dkey in keys]) != 6)      

    # loop over each dependent.
    active = list()
    cnt = 0
    for dkey in keys:

        # skip short keys.
        if len(dkey) != 6: continue

        # check if we can skip.
        if dkey in results and method_name in results[dkey] and args.debug == False:
            if os.path.isfile(results[dkey][method_name]["S"]) and os.path.isfile(results[dkey][method_name]["C"]):
                continue

        # simplify.
        n, k, e, c, q, m = dkey
        mkey = n, k, e, c, q

        # extract paths.
        XG_path = master[dkey]['XG']
        ZG_path = master[dkey]['ZG']
        y_path = master[mkey]['y']

        # make a working directory.
        w_dir = '%s/working/%s/%i_%i_%i_%i/%i_%i' % (sim_dir, method_name, n, k, e, c, q, m)
        S_path = '%s/S.npy' % w_dir
        C_path = '%s/C.npy' % w_dir

        # create the working directory.
        if os.path.isdir(w_dir) == False:
            subprocess.call(['mkdir', '-p', w_dir])

        # call run method, return process.
        fnargs = (XG_path, ZG_path, y_path, k, S_path, C_path, w_dir, dkey, method_name)
        if args.debug == False:

            # run it in multiprocessing pool
            logging.info("running: %i of %i: %s %i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, q, m))
            active.append(pool.apply_async(method_fn, fnargs))
        else:

            # run it in this thread.
            logging.debug("debugging: %i of %i: %s %i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, q, m))
            debug_result = method_fn(*fnargs)
            
            # load cheat data.
            S_true = np.load('%s.npy' % master[mkey]['H'])
            S_true  = S_true[0:m,:]
            C_true = np.load('%s.npy' % master[mkey]['C'])
            S_pred = np.load(debug_result[2])
            C_pred = np.load(debug_result[3])
            
            # evaluate it directly.
            c_score = _evl_C(C_true, C_pred, meanabs_vector)
            
            if c_score == None:
                logging.debug("bad dimensions")
                continue

            # report it.
            logging.debug("c_score: %.5f" % (c_score))

        # progress tracking variable.
        cnt += 1

        # flush pool occasionally.
        if cnt % 50 == 0 and args.debug == False:

            # wait on all active processes.
            for a in active:

                # get info and add it.
                dkey, method_name, S_path, C_path = a.get()
                if S_path == None: continue
                if dkey not in results:
                    results[dkey] = dict()
                results[dkey][method_name] = {"S":S_path, "C":C_path}
                n, k, e, c, q, m = dkey
                logging.info("finished: %i of %i: %s %i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, q, m))

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

    # load R libraries.
    load_R_libraries()

    # create the pool.
    if args.debug == False:
        pool = Pool(processes = args.num_cpu)

    # solve easiest first.
    keys = sorted(master.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # count total jobs.
    total = np.sum(np.array([len(dkey) for dkey in keys]) != 7)      

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
            print "n=%i m=%i" % (n,m), m*k, n*k, n*m
            debug_result = method_fn(*fnargs)
            
            print debug_result
            
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

            if s_score == None or c_score == None:
                logging.debug("bad dimensions")
                continue

            # report it.
            logging.debug("s_score: %.5f\tc_score: %.5f" % (s_score[0], c_score[0]))

        # progress tracking variable.
        cnt += 1

        # flush pool occasionally.
        if cnt % 50 == 0 and args.debug == False:

            # wait on all active processes.
            for a in active:

                # get info and add it.
                dkey, method_name, S_path, C_path = a.get()
                if S_path == None: continue
                if dkey not in results:
                    results[dkey] = dict()
                results[dkey][method_name] = {"S":S_path, "C":C_path}
                n, k, e, c, r, q, m = dkey
                logging.info("finished: %i of %i: %s %i_%i_%i_%i_%i_%i %i" % (cnt, total, method_name, n, k, e, c, r, q, m))

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


def _bin_class_gen(nmixs, kcell, enois, ctype, rmode, qtrys, mgene, SAMPLES, ilu, kmap):
    """ generate 50/50 mix of missing and non """

    # create the master matrix.
    for n, k, e, c, r, q in itertools.product(nmixs, kcell, enois, ctype, rmode, qtrys):
       
        # skip the crazy ones.
        if r >= k: continue
       
        # begin key.
        mkey = [n, k, e, c, r, q]

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
        if e != 100:
            # create using sampling.
            X = _create_X(n, 256, k, e, C, SAMPLES, ilu, kmap[k])
        else:
            # create using dot product.
            X = np.dot(H, C)

        # loop over the gene variable.
        for m in mgene:
            
            # skip the crazy ones.
            if n > m: continue
            
            # strip X, ZM to right size.
            XG = X[0:m, :]
            ZMG = ZM[0:m, :]
            ZG = Z[0:m, :]
            
            # yield complete.
            yield XG, ZG, y, False
            
            # yield missing.
            yield XG, ZMG, ym, True


def pred_survey(args):
    ''' prediction survey '''

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

     
    ## vary the noise ##
    #nmixs = np.array([5, 10, 15, 20])
    nmixs = np.array([15])
    mgene = np.array([16])   
    kcell = np.array([5])
    enois = np.array([100])
    ctype = np.array([0])

    ## these guys stay the same.
    rmode = np.arange(0, 6, 1)
    qtrys = np.arange(0, 5, 1)

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
    for X, Z, y, truth in _bin_class_gen(nmixs, kcell, enois, ctype, rmode, qtrys, mgene, SAMPLES, ilu, kmap):
       
        # run both methods.
        S1, C1 = solve_SC(X, Z, y)
        S2, C2 = solve_C(X, Z, y)
            
        # compute residual.
        X1 = np.dot(S1, C1)
        X2 = np.dot(S2, C2)
        
        r1 = np.around(np.sqrt(np.sum(np.square((X - X1)))), 5)
        r2 = np.around(np.sqrt(np.sum(np.square((X - X2)))), 5)
            
        # classifier.
        ismissing = r1 < r2
        
        #
        print truth, ismissing
        
        #if truth == False:
            #print (X - X1)[:,0]
            #print (X - X2)[:,0]
            
        #    print r1, r2
        #    break
            

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
    results = load_pickle(res_obj)
    
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
    
        # load it.
        C_pred = np.load(results[dkey][method_name]['C'])

        c1 = C_true.flatten()
        c2 = C_pred.flatten()

        # looo pover each thang.
        for scorefn, sname in zip(scorefns, scorenms):      
        
            # scor eit.
            c_score = _evl_C(C_true, C_pred, scorefn)
        
            if c_score == None: continue
        
            # save it.
            output.append((skey, sname, c_score))
                
    sys.exit()
                
    # compress by key
    compressed = dict()
    for key, name, c_score in output:

        # skip if missing.
        if key == None: continue

        # boot the key.
        if key not in compressed:
            compressed[key] = dict()

        # boot the scores.
        if name not in compressed[key]:
            compressed[key][name] = dict()
            compressed[key][name]['C'] = list()

        # add it.
        compressed[key][name]['C'].append(c_score)

    # sort, average and print.
    keys = sorted(compressed.keys(), key=operator.itemgetter(0,1,2,3,4))
    for key in keys:
        for name in scorenms:
            c = np.average(np.array(compressed[key][name]['C']))
            txt = ','.join(['%i' % x for x in key])
            txt = '%s,%s,%.5f' % (txt, name, c)
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
        #skey = n, k, e, c, r, m        # remove reference ot repeat variable
        skey = n, k, e, c, m        # remove reference ot repeat variable and cell types

        # load the true concentrations.
        S_true = np.load('%s.npy' % master[mkey]['H'])
        S_true = S_true[0:m,:]
        C_true = np.load('%s.npy' % master[mkey]['C'])
        
        # load the predicted.
        S_pred = np.load(results[dkey][method_name]['S'])
        C_pred = np.load(results[dkey][method_name]['C'])
        

        # score for each cell-type.
        for scorefn, sname in zip(scorefns, scorenms):        
        
            # remap if its not DECONF
            if method_name != "DECONF":

                # remap to known order.
                C_pred, S_pred = _remap_missing(C_pred, S_pred, r, k)
                
                # evaluate it directly.
                c_score = _evl_C(C_true, C_pred, scorefn)
                s_score = _evl_s(S_true[:,r], S_pred[:,r], scorefn)
            else:
                
                # perform matching.
                C_pred, S_pred = _match_pred(C_pred, S_pred, C_true, S_true)

                
                # score C.
                c_score = _evl_C(C_true, C_pred, scorefn)            
                
                # average S over cell-type
                tmp = list()
                for l in range(k):
                    s = _evl_s(S_true[:,l], S_pred[:,l], scorefn)
                    if s == None: continue
                    tmp.append(s)
                if len(tmp) == 0: continue
                s_score = np.average(np.array(tmp))

                # skip bad guys.
                if c_score == None or s_score == None:
                    continue
            
                # evaluate it directly.
                c_score = _evl_C(C_true, C_pred, scorefn)
                s_score = _evl_s(S_true[:,r], S_pred[:,r], scorefn)
            
            # save it.
            output.append((skey, sname, s_score, c_score))
                

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
            #txt = '%i,%i,%i,%i,%i,%i' % key
            txt = ','.join(['%i' % x for x in key])
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
    subpp.add_argument('-d', dest='debug', action='store_true', help='use debug mode (i.e.) skip multiprocessing')
    subpp.set_defaults(func=run_exp)

    # missing simulation
    subpp = subp.add_parser('run_mis', help='creates experiment data')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    me_g = subpp.add_mutually_exclusive_group(required=True)
    me_g.add_argument('-UCQPM', dest='method_sig', action='store_const', const=('UCQPM', _run_UCQPM), help='UCQPM')
    me_g.add_argument('-UCQP', dest='method_sig', action='store_const', const=('UCQP', _run_UCQP), help='UCQP')
    me_g.add_argument('-DECONF', dest='method_sig', action='store_const', const=('DECONF', _run_DECONF), help='DECONF')
    subpp.add_argument('-d', dest='debug', action='store_true', help='use debug mode (i.e.) skip multiprocessing')
    subpp.set_defaults(func=run_mis)

    ## prediction ability ##

    # prediction accuracy
    subpp = subp.add_parser('pred_survey', help='prediction survey')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-p', type=int, dest='num_cpu', required=True, help='number of processors to use')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.set_defaults(func=pred_survey)

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
    me_g.add_argument('-DECONF', dest='method_name', action='store_const', const='DECONF', help='DECONF')
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
