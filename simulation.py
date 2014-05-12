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
warnings.filterwarnings("ignore")
# statistics.
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import gmean
from sklearn import feature_selection

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )

# app
from utils.matops import *
#from utils.rfuncs import *
from utils.misc import *
from utils.plotting import *
#from utils.heirheat import *
from utils.cluster import *

#from scdecon import decon_missing
#from scdecon import _solve_missing

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


### private functions ###

def _run_it_R(wdir, script_txt, n, m, k):

    # write it.
    script_file = '%s/script.R' % wdir
    with open(script_file, "wb") as fout:
        fout.write(script_txt)

    # run it.
    try:
        retval = subprocess.check_output(["Rscript", script_file], stderr=subprocess.STDOUT, env=os.environ)
    except subprocess.CalledProcessError as e:
        txt = "R-script failed: %s\n" % script_file
        txt += '%s\n' % e.output
        txt += '=======================\n'
        logging.error(txt)
        return None

    # load it.
    try:
        C = np.loadtxt('%s/C.txt' % wdir)
        S = np.loadtxt('%s/S.txt' % wdir)
    except:
        txt = "R-script failed: %s\n" % script_file
        txt += "couldn't find matrix\n"
        txt += '=======================\n'
        logging.error(txt)
        return None

    # sanity check.
    if C.shape != (k, n) or S.shape[1] != (k):
        txt = "R-script bad dim: %s\n" % script_file
        txt += "expected: C=(%i,%i), S=(%i,%i)\n" % (k,n,m,k)
        txt += "recieved: C=(%i,%i), S=(%i,%i)\n" % (C.shape[0], C.shape[1], S.shape[0], S.shape[1])
        txt += '=======================\n'
        logging.error(txt)
        return None

    # return results.
    return S, C


def _run_it_uconn(wdir, script_txt, n, m, k, missing=None):

    # write it.
    script_file = '%s/script.sh' % wdir
    with open(script_file, "wb") as fout:
        fout.write(script_txt)

    # run it.
    try:
        retval = subprocess.check_output(["sh", script_file], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        txt = "UCONN-script failed: %s\n" % script_file
        txt += '%s\n' % e.output
        txt += '=======================\n'
        logging.error(txt)
        return None

    # load it.
    try:
        C = np.load('%s/C.npy' % wdir)
        S = np.load('%s/S.npy' % wdir)
    except:
        txt = "UCONN-script failed: %s\n" % script_file
        txt += "couldn't find matrix\n"
        txt += '=======================\n'
        logging.error(txt)
        return None

    # sanity check.
    if missing == None:
        if C.shape != (k, n) or S.shape != (m,k):
            txt = "UCONN-script failed: %s\n" % script_file
            txt += "bad dim\n"
            txt += "expected: C=(%i,%i), S=(%i,%i)\n" % (k,n,m,k)
            txt += "recieved: C=(%i,%i), S=(%i,%i)\n" % (C.shape[0], C.shape[1], S.shape[0], S.shape[1])
            txt += '=======================\n'
            return None

    else:
        if C.shape != (k + 1, n) or S.shape != (m, k + 1):
            txt = "UCONN-script failed: %s\n" % script_file
            txt += "bad dim\n"
            txt += "expected: C=(%i,%i), S=(%i,%i)\n" % (k,n,m,k)
            txt += "recieved: C=(%i,%i), S=(%i,%i)\n" % (C.shape[0], C.shape[1], S.shape[0], S.shape[1])
            txt += '=======================\n'
            return None

    # return results.
    return S, C

def _avg_S(sc_lbls, SC):
    """ create S from known labels """
    S, cats = avg_cat(sc_lbls, np.transpose(SC))
    return S

def _log_noise(a, b, c):
    """ log noise matrix """

    # create random number
    N = np.random.uniform(size=(a,b))

    # multiple by ln2.
    N = np.exp(N * math.log(2) * c)

    return N


def _pick_housekeeping(mat, detectors, candidates, minH=1, maxH=10, thres=0.15):
    """
        Compute Internal control gene-stability measure M described in
        Vandesompele et al., 2002
        Consider only detectors with no missing values
        Inputs: mat: n x m matrix of Ct values, n is the number of samples,
               m is the number of detectors
               detectors: correspond to the columns of mat
               candidates: candidate housekeeping detectors
               thres: cut-off value of pairwise variation
        Output: a list of housekeeping genes
    """

    # candidates must be highly expression.
    candidates = list()
    for j in range(mat.shape[1]):
        if np.average(mat[:,j]) < 26:
            candidates.append(j)
    detectors = np.array(detectors)
    n, m = mat.shape
    noMissing = ~np.any(np.isnan(mat), axis=0)
    indices = np.arange(m)[noMissing]
    cIndices = []
    for i in indices: # Filter out non-candidates
        if detectors[i] in candidates: cIndices.append(i)
    cIndices = np.array(cIndices)
    h = len(cIndices)
    if h <= minH: return detectors[cIndices]
    M = []
    for i in cIndices: # candidate indices
        # log-transformed expression ratios
        tmp = mat[:, cIndices].T - mat[:, i] # transposed: columns are samples
        M.append( np.std(tmp, axis=1).sum() / (h-1) ) # std across samples
    M = np.array(M); indices = M.argsort()
    M = M[indices]; cIndices = cIndices[indices]
    # Old normalization factor. Sum of n Ct vectors. Needs to be divided by n.
    nfOld = mat[:, cIndices[:minH]].sum(axis=1)
    for i in xrange(minH, min(maxH, h)):
        nfNew = nfOld + mat[:, cIndices[i]]
        std = np.std(nfNew/(i + 1) - nfOld/i)
        if std < thres: break # No need to include cIndices[i]
        nfOld = nfNew
    return detectors[cIndices[:i]].tolist()



### callable functions ###

def load_mmc3_figure1b(args):
    ''' mmc3_figure1b '''

    # defined labels. PSC is missing
    labels = set(['HSC','NSC','MASC','ISC', 'PSC'])

    # open csv
    samples = dict()
    with open(args.input_file) as fin:

        # loop and parse.
        for line in fin:

            # parse.
            line = line.strip().split("\t")

            # look for hit.
            if line[0] in labels:

                # create first.
                if line[0] not in samples:
                    samples[line[0]] = [line[1::], list()]
                    continue

                # add hits.
                samples[line[0]][1].append( [float(x) for x in line[1::]] )



    # sanity check overlap in genes.
    keys = samples.keys()
    for i in range(1, len(keys)):
        key1 = keys[i-1]
        key2 = keys[i]
        if samples[key1][0] != samples[key2][0]:
            logging.error("genes don't match")
            sys.exit()
        biomarkers = samples[key1][0]

    # reformat.
    for key in keys:
        samples[key] = samples[key][1]

    # simplify sizes.
    nb = len(biomarkers)
    nc = len(samples)
    nsc = sum([len(samples[x]) for x in samples])

    # create gene labels.
    b_lbls = np.array(biomarkers)
    blu = dict()
    for i in range(len(b_lbls)):
        blu[b_lbls[i]] = i


    # create SC matrix.
    SC = np.zeros((nb, nsc), dtype=np.float)

    # label arrays.
    c_lbls = np.array(list(labels))
    clu = dict()
    k = 0
    for k in range(len(c_lbls)):
        clu[c_lbls[k]] = k

    sc_lbls = np.zeros(nsc, dtype=np.int)

    # load data into SC matrix.
    l = 0
    for c in samples:

        # process each sample.
        for s in samples[c]:

            # save data.
            for i in range(len(s)):
                SC[i,l] = s[i]

            # save label.
            sc_lbls[l] = clu[c]
            l += 1

    # put into linear form.
    SC = np.square(SC)

    # shink clu just to the unique and sorted variety.
    c_lbls = np.array([x[0] for x in sorted(clu.items(), key=operator.itemgetter(1))])

    # create input.
    np.save(args.SC, SC)
    np.save(args.sc_lbls, sc_lbls)
    np.save(args.c_lbls, c_lbls)
    np.save(args.b_lbls, b_lbls)


def _UCQP(X, Z, y, k, C_path, S_path, wdir, xtra=None):

    # save them to temporary file.
    subprocess.call(['mkdir', '-p', wdir])

    Xout = '%s/X.npy' % (wdir)
    Zout = '%s/Z.npy' % (wdir)
    yout = '%s/y.npy' % (wdir)
    np.save(Xout, X)
    np.save(Zout, Z)
    np.save(yout, y)

    # call out method.
    cmd = list()
    cmd.append('#!/bin/bash')
    cmd.append('# UCQP')
    cmd.append('')
    cmd.append('# run it.')
    tmp = list()
    tmp.append('python')
    tmp.append('/home/jrl03001/code/scdecon2/scdecon.py')
    tmp.append('decon')
    tmp.append('-X %s' % Xout)
    tmp.append('-Z %s' % Zout)
    tmp.append('-y %s' % yout)
    tmp.append('-C %s' % C_path)
    tmp.append('-S %s' % S_path)
    tmp.append('-p 5')
    cmd.append(' '.join(tmp))
    cmd.append('')
    cmd.append('')

    # run it.
    _run_it_uconn(wdir, '\n'.join(cmd), X.shape[1], X.shape[0], k)



def _UCQPM(X, Z, y, k, C_path, S_path, wdir, xtra=None):

    # save them to temporary file.
    subprocess.call(['mkdir', '-p', wdir])

    Xout = '%s/X.npy' % (wdir)
    Zout = '%s/Z.npy' % (wdir)
    yout = '%s/y.npy' % (wdir)
    np.save(Xout, X)
    np.save(Zout, Z)
    np.save(yout, y)

    if xtra != None:
        Cout = '%s/C_cheat.npy' % (wdir)
        Sout = '%s/S_cheat.npy' % (wdir)
        Hout = '%s/H_cheat.npy' % (wdir)
        fout = '%s/fs_cheat.npy' % (wdir)
        np.save(Sout, xtra['S'])
        np.save(Cout, xtra['C'])
        np.save(Hout, xtra['H'])
        np.save(fout, xtra['fs'])

    # call out method.
    cmd = list()
    cmd.append('#!/bin/bash')
    cmd.append('# UCQP')
    cmd.append('')
    cmd.append('# run it.')
    tmp = list()
    tmp.append('python')
    tmp.append('/home/jrl03001/code/scdecon2/scdecon.py')
    tmp.append('decon_missing')
    tmp.append('-X %s' % Xout)
    tmp.append('-Z %s' % Zout)
    tmp.append('-y %s' % yout)
    tmp.append('-C %s' % C_path)
    tmp.append('-S %s' % S_path)
    tmp.append('-p 5')
    if xtra != None:
        tmp.append('-S_cheat %s' % Sout)
        tmp.append('-C_cheat %s' % Cout)
        tmp.append('-H_cheat %s' % Hout)
        tmp.append('-fs_cheat %s' % fout)
    cmd.append(' '.join(tmp))
    cmd.append('')
    cmd.append('')

    # run it.
    _run_it_uconn(wdir, '\n'.join(cmd), X.shape[1], X.shape[0], k, missing=True)

def _UCQPS(X, Z, y, k, C_path, wdir, xtra=None):

    # save them to temporary file.
    subprocess.call(['mkdir', '-p', wdir])

    Xout = '%s/X.npy' % (wdir)
    Zout = '%s/Z.npy' % (wdir)
    yout = '%s/y.npy' % (wdir)
    np.save(Xout, X)
    np.save(Zout, Z)
    np.save(yout, y)

    # call out method.
    cmd = list()
    cmd.append('#!/bin/bash')
    cmd.append('# UCQPS')
    cmd.append('')
    cmd.append('# run it.')
    tmp = list()
    tmp.append('python')
    tmp.append('/home/jrl03001/code/scdecon2/scdecon.py')
    tmp.append('deconscale')
    tmp.append('-X %s' % Xout)
    tmp.append('-Z %s' % Zout)
    tmp.append('-y %s' % yout)
    tmp.append('-C %s' % C_path)
    cmd.append(' '.join(tmp))
    cmd.append('')
    cmd.append('')

    # open script file.
    run_sh = '%s/run.sh' % (wdir)
    with open(run_sh, "w") as fout:
        fout.write('\n'.join(cmd))

    # make executable and run.
    subprocess.call(["chmod", "u+x", run_sh])
    subprocess.call([run_sh])



def _RAND(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    ''' just randomly create '''

    # save them to temporary file.
    subprocess.call(['mkdir', '-p', wdir])

    Xout = '%s/X.npy' % (wdir)
    Zout = '%s/Z.npy' % (wdir)
    yout = '%s/y.npy' % (wdir)
    np.save(Xout, X)
    np.save(Zout, Z)
    np.save(yout, y)

    # call out method.
    cmd = list()
    cmd.append('#!/bin/bash')
    cmd.append('# UCQP')
    cmd.append('')
    cmd.append('# run it.')
    tmp = list()
    tmp.append('python')
    tmp.append('/home/jrl03001/code/scdecon2/scdecon.py')
    tmp.append('decon_missing')
    tmp.append('-X %s' % Xout)
    tmp.append('-Z %s' % Zout)
    tmp.append('-y %s' % yout)
    tmp.append('-C %s' % C_path)
    tmp.append('-S %s' % S_path)
    tmp.append('-p 5')
    cmd.append(' '.join(tmp))
    cmd.append('')
    cmd.append('')

    # run it.
    _run_it_uconn(wdir, '\n'.join(cmd), X.shape[1], X.shape[0], k, missing=True)


def _PCQP(X, Z, y, k, C_path, wdir, xtra=None):

    # save them to temporary file.
    subprocess.call(['mkdir', '-p', wdir])

    Xout = '%s/X.npy' % (wdir)
    Zout = '%s/Z.npy' % (wdir)
    yout = '%s/y.npy' % (wdir)
    np.save(Xout, X)
    np.save(Zout, Z)
    np.save(yout, y)

    # call out method.
    cmd = list()
    cmd.append('#!/bin/bash')
    cmd.append('# PCQP')
    cmd.append('')
    cmd.append('# run it.')
    tmp = list()
    tmp.append('python')
    tmp.append('/home/jrl03001/code/scdecon2/scdecon.py')
    tmp.append('pcacon')
    tmp.append('-X %s' % Xout)
    tmp.append('-Z %s' % Zout)
    tmp.append('-y %s' % yout)
    tmp.append('-C %s' % C_path)
    cmd.append(' '.join(tmp))
    cmd.append('')
    cmd.append('')

    # open script file.
    run_sh = '%s/run.sh' % (wdir)
    with open(run_sh, "w") as fout:
        fout.write('\n'.join(cmd))

    # make executable and run.
    subprocess.call(["chmod", "u+x", run_sh])
    subprocess.call([run_sh])

def _DECONF(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    """ deconvolution """

    # identify non-null rows.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(X[i,:]) > 0.0:
            good_rows.append(i)
    X = X[good_rows,:]

    # serialize stuff to matrix.
    tmp_mat = '%s/mat.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    # run deconvolution.
    txt = '''# load libraries.
suppressMessages(library(CellMix));
suppressMessages(library(GEOquery));

# load data.
exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);

# run deconvolution.
res <- ged(eset, {num}, method='deconf');

# write matrix.
write.table(coef(res), file="{path}/C.txt", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{path}/S.txt", row.names=FALSE, col.names=FALSE)
'''.format(path=wdir, num=k)

    # run it in R.
    out = _run_it_R(wdir, txt, X.shape[1], X.shape[0], k)

    # skip if bad.
    if out == None:
        return
    S, C = out

    # write it.
    np.save(C_path, C)
    np.save(S_path, S)


def _QPROG(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    """ deconvolution """

    # compute S.
    S = _avg_S(y, Z)

    # identify non-null rows.
    good_rows = list()
    for i in range(S.shape[0]):
        if np.sum(S[i,:]) > 0.0:
            good_rows.append(i)
    S = S[good_rows,:]

    # serialize stuff to matrix.
    tmp_mat = '%s/S.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(S.shape[1])]) + '\n')
        for i in range(S.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in S[i,:]]) + '\n')

    # identify non-null rows.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(X[i,:]) > 0.0:
            good_rows.append(i)
    X = X[good_rows,:]

    # serialize stuff to matrix.
    tmp_mat = '%s/mat.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    # create the R script.
    txt = '''# load libraries.
suppressMessages(library(CellMix));
suppressMessages(library(GEOquery));

# load data.
sigFile <- file.path("{path}", "S.dat");
S <- as.matrix(read.table(sigFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));

exprFile <- file.path("{path}", "mat.dat");
X <- as.matrix(read.table(exprFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));

# run decon.
res <- ged(X, S, method='qprog');

# write matrix.
write.table(coef(res), file="{path}/C.txt", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{path}/S.txt", row.names=FALSE, col.names=FALSE)
'''.format(path=wdir)

    # run it in R.
    out = _run_it_R(wdir, txt, X.shape[1], X.shape[0], k)

    # skip if bad.
    if out == None:
        return
    S, C = out

    # write it.
    np.save(C_path, C)
    np.save(S_path, S)


def _DSA(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    """ deconvolution """

    # write Z.
    good_rows = list()
    for i in range(Z.shape[0]):
        if np.sum(Z[i,:]) > 0.0:
            good_rows.append(i)
    Z = Z[good_rows,:]

    tmp_mat = '%s/Z.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

    # write X.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(X[i,:]) > 0.0:
            good_rows.append(i)
    X = X[good_rows,:]

    tmp_mat = '%s/mat.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    # simplify labels.
    lbls = ','.join(['%i' % x for x in y])

    # whut marker method.
    if xtra != None:
        if xtra not in set(['Abbas', 'maxcol', 'HSD']):
            logging.error("bad xtra parameter")
            sys.exit(1)
        xm = xtra
    else:
        xm = 'Abbas'

    # extract markers.
    txt = '''# load libraries.
suppressMessages(library(CellMix));
suppressMessages(library(GEOquery));

# load pure single-cells
sigFile <- file.path("{path}", "Z.dat");
Z <- as.matrix(read.table(sigFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
Z <- Z + 1

# labels
y <- c({y});

# perform extraction.
sml <- extractMarkers(Z, data=y, method='{xm}')

# load the mixture data.
exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);

# perform deconvolution.
res <- ged(eset, sml, 'DSA', verbose=TRUE)

# write matrix.
write.table(coef(res), file="{path}/C.txt", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{path}/S.txt", row.names=FALSE, col.names=FALSE)
'''.format(path=wdir, y=lbls, xm=xm)

    # run it in R.
    out = _run_it_R(wdir, txt, X.shape[1], X.shape[0], k)

    # skip if bad.
    if out == None:
        return
    S, C = out

    # write it.
    np.save(C_path, C)
    np.save(S_path, S)



def _FROB(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    """ deconvolution """

    # write Z.
    good_rows = list()
    for i in range(Z.shape[0]):
        if np.sum(Z[i,:]) > 0.0:
            good_rows.append(i)
    Z = Z[good_rows,:]

    tmp_mat = '%s/Z.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

    # write X.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(X[i,:]) > 0.0:
            good_rows.append(i)
    X = X[good_rows,:]

    tmp_mat = '%s/mat.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    # simplify labels.
    lbls = ','.join(['%i' % x for x in y])


    # whut marker method.
    if xtra != None:
        if xtra not in set(['Abbas', 'maxcol', 'HSD']):
            logging.error("bad xtra parameter")
            sys.exit(1)
        xm = xtra
    else:
        xm = 'Abbas'

    # extract markers.
    txt = '''# load libraries.
suppressMessages(library(CellMix));
suppressMessages(library(GEOquery));

# load pure single-cells
sigFile <- file.path("{path}", "Z.dat");
Z <- as.matrix(read.table(sigFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
Z <- Z + 1

# labels
y <- c({y});

# perform extraction.
sml <- extractMarkers(Z, data=y, method='{xm}')

# load the mixture data.
exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);

# perform deconvolution.
res <- ged(eset, sml, 'ssFrobenius')

# write matrix.
write.table(coef(res), file="{path}/C.txt", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{path}/S.txt", row.names=FALSE, col.names=FALSE)
'''.format(path=wdir, y=lbls, xm=xm)

    # run it in R.
    out = _run_it_R(wdir, txt, X.shape[1], X.shape[0], k)

    # skip if bad.
    if out == None:
        return
    S, C = out

    # write it.
    np.save(C_path, C)
    np.save(S_path, S)



def _SSKL(X, Z, y, k, C_path, S_path, wdir, xtra=None):
    """ deconvolution """

    # write Z.
    good_rows = list()
    for i in range(Z.shape[0]):
        if np.sum(Z[i,:]) > 0.0:
            good_rows.append(i)
    Z = Z[good_rows,:]

    tmp_mat = '%s/Z.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

    # write X.
    good_rows = list()
    for i in range(X.shape[0]):
        if np.sum(X[i,:]) > 0.0:
            good_rows.append(i)
    X = X[good_rows,:]

    tmp_mat = '%s/mat.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(X.shape[1])]) + '\n')
        for i in range(X.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in X[i,:]]) + '\n')

    # simplify labels.
    lbls = ','.join(['%i' % x for x in y])


    # whut marker method.
    if xtra != None:
        if xtra not in set(['Abbas', 'maxcol', 'HSD']):
            logging.error("bad xtra parameter")
            sys.exit(1)
        xm = xtra
    else:
        xm = 'Abbas'

    # extract markers.
    txt = '''# load libraries.
suppressMessages(library(CellMix));
suppressMessages(library(GEOquery));

# load pure single-cells
sigFile <- file.path("{path}", "Z.dat");
Z <- as.matrix(read.table(sigFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
Z <- Z + 1

# labels
y <- c({y});

# perform extraction.
sml <- extractMarkers(Z, data=y, method='{xm}')

# load the mixture data.
exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);

# perform deconvolution.
res <- ged(eset, sml, 'ssKL')

# write matrix.
write.table(coef(res), file="{path}/C.txt", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{path}/S.txt", row.names=FALSE, col.names=FALSE)
'''.format(path=wdir, y=lbls, xm=xm)

    # run it in R.
    out = _run_it_R(wdir, txt, X.shape[1], X.shape[0], k)

    # skip if bad.
    if out == None:
        return
    S, C = out

    # write it.
    np.save(C_path, C)
    np.save(S_path, S)

def _sim_gen(Xs, Zs, ys, method, work_dir):
    """
    generate test cases for process / review
    """

    # santiy.
    assert len(Xs) == len(Zs) and len(Zs) == len(ys), 'bad setup'

    # loop over each experiment.
    for e in range(len(Xs)):

        # get elements.
        X = Xs[e]
        Z = Zs[e]
        y = ys[e]

        # make paths.
        wdir = '%s/%d/%s' % (work_dir, e, method)

        # make common output file.
        C_path = "%s/C.npy" % (wdir)
        S_path = "%s/S.npy" % (wdir)

        # yield it.
        yield X, Z, y, wdir, C_path, S_path, e

def _sim_C(n, m, k, c_type):

    # simulate the concentrations.
    C = np.zeros((k, n), dtype=np.float)
    for j in range(n):
        if c_type == 1:

            # uniform.
            C[:,j] = 1.0 / float(k)

        elif c_type == 2:

            # arithmetic.
            x = [float(x) for x in range(1,k+1)]
            random.shuffle(x)
            x = np.array(x)
            x = x / np.sum(x)
            C[:,j] = x

        elif c_type == 3:

            # geometric.
            x = list(np.vander([k], k)[0])
            random.shuffle(x)
            x = [float(z) for z in x]
            x = np.array(x)
            x = x / np.sum(x)
            C[:,j] = x

        else:
            logging.error("unknown C type")
            sys.exit(1)

    return C

def _sim_X_sample(n, m, k, C, ilu, Xtmp):

    # sample from mixture to create X.
    X = np.zeros((m, n), dtype=np.float)

    # for each sample.
    for j in range(n):

        # choose 100 from each cell type.
        idxs = list()
        for a in range(k):

            # compute count.
            count = int(np.rint(w * C[a,j]) + 1)

            # choose index.
            idxs += list(np.random.choice(ilu[a], size = count))

        # build big slice.
        bs = Xtmp[:,idxs]

        # assign average to sample.
        for i in range(m):
            X[i,j] = np.average(bs[i,:])

    # return X.
    return X

### public functions ###

def create_exp1(args):
    """ creates S is calculated directly from average of SC """

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    c_lbls = np.load(args.c_lbls)
    b_lbls = np.load(args.b_lbls)

    # compute dimensions.
    n = args.n
    m = SC.shape[0]
    k = c_lbls.shape[0]
    l = SC.shape[1]
    t = args.t
    q = args.q
    e = args.e

    # build the S simulation object.
    if os.path.isfile(args.sim_obj) == False:
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(args.sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=args.sim_obj)

    # loop over the number of experiments.
    Xs = list()
    Cs = list()
    Ss = list()
    Zs = list()
    ys = list()

    # create master S.
    S = _avg_S(sc_lbls, SC)

    # loop over each experiment.
    for gg in range(q):

        # simulate the single-cells.
        Z, y = sim.sample_1(t)

        # create S-timate
        H = _avg_S(y, Z)

        # simulate the concentrations.
        C = np.zeros((k, n), dtype=np.float)
        for j in range(n):
            if args.c_type == 1:

                # uniform.
                C[:,j] = 1.0 / float(k)

            elif args.c_type == 2:

                # arithmetic.
                x = [float(x) for x in range(1,k+1)]
                random.shuffle(x)
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            elif args.c_type == 3:

                # geometric.
                x = list(np.vander([k], k)[0])
                random.shuffle(x)
                x = [float(z) for z in x]
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            else:
                logging.error("unknown C type")
                sys.exit(1)

        # compute mixtures directly from our model
        X2 = np.dot(H, C)

        # add noise.
        N = _log_noise(X2.shape[0], X2.shape[1], e)
        X2 = X2 * N

        # save to list.
        Xs.append(X2)
        Cs.append(C)
        #Ss.append(Snorm)
        Ss.append(S)
        Zs.append(Z)
        ys.append(y)

    # save experiment.
    save_pickle(args.ref_file, {'Xs':Xs, 'Ss':Ss, 'Cs':Cs, 'Zs':Zs, 'ys':ys})
    save_pickle(args.test_file, {'Xs':Xs, 'Zs':Zs, 'ys':ys})

    # save simulated single-cells for plotting.
    Z, y = sim.sample_1(50)
    np.save('%s/Z.npy' % args.sim_dir, Z)
    np.save('%s/z_lbls.npy' % args.sim_dir, y)


def create_exp2(args):
    """ creates S is calculated directly from average of SC """

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    c_lbls = np.load(args.c_lbls)
    b_lbls = np.load(args.b_lbls)

    # compute dimensions.
    n = args.n
    m = SC.shape[0]
    k = c_lbls.shape[0]
    l = SC.shape[1]
    t = args.t
    q = args.q
    e = args.e
    g = args.g

    # build the S simulation object.
    if os.path.isfile(args.sim_obj) == False:
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(args.sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=args.sim_obj)

    # loop over the number of experiments.
    Xs = list()
    Cs = list()
    Ss = list()
    Zs = list()
    ys = list()

    # create master S.
    S = _avg_S(sc_lbls, SC)

    # loop over each experiment.
    for gg in range(q):

        # simulate the single-cells.
        Z, y = sim.sample_1(t)

        # create S-timate
        H = _avg_S(y, Z)

        # simulate the concentrations.
        C = np.zeros((k, n), dtype=np.float)
        for j in range(n):
            if args.c_type == 1:

                # uniform.
                C[:,j] = 1.0 / float(k)

            elif args.c_type == 2:

                # arithmetic.
                x = [float(x) for x in range(1,k+1)]
                random.shuffle(x)
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            elif args.c_type == 3:

                # geometric.
                x = list(np.vander([k], k)[0])
                random.shuffle(x)
                x = [float(z) for z in x]
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            else:
                logging.error("unknown C type")
                sys.exit(1)

        # compute mixtures directly from our model
        X2 = np.dot(H, C)
        print X2[0:5,0], X2[0:5,1]
        # add noise.
        N = _log_noise(X2.shape[0], X2.shape[1], e)
        X2 = X2 * N

        # get teh informative features.
        if g != m:

            # identify non-null rows.
            good_rows = list()
            for i in range(Z.shape[0]):
                if np.sum(Z[i,:]) > 0.0:
                    good_rows.append(i)
            Z = Z[good_rows,:]

            # anova feature selection.
            clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=g)
            clf.fit(np.transpose(Z), y)
            features = np.where(clf.get_support() == True)[0]

            # chop the genes down.
            Z = Z[features,:]
            X2 = X2[features,:]



        #print Z[0,0], H[0,0], X2[0,0]
        #print Z.shape, H.shape, X2.shape

        # save to list.
        Xs.append(X2)
        Cs.append(C)
        #Ss.append(Snorm)
        Ss.append(S)
        Zs.append(Z)
        ys.append(y)

    print "CHANGED??"
    for X in Xs:
        print X[0:5,0], X[0:5,1]

    ## DEBUG
    Xfull = np.zeros((Xs[0].shape[0], Xs[0].shape[1]*len(Xs)), dtype=np.float)
    Zfull = np.zeros((Zs[0].shape[0], Zs[0].shape[1]*len(Zs)), dtype=np.float)
    yfull = np.zeros(ys[0].shape[0]*len(ys), dtype=np.int)

    # loop over each experiment.
    xj = 0
    zj = 0
    yi = 0
    for  X, Z, y in zip(Xs, Zs, ys):
        # copy into.
        for j in range(X.shape[1]):
            Xfull[:,xj] = X[:,j]
            xj += 1
        for j in range(Z.shape[1]):
            Zfull[:,zj] = Z[:,j]
            zj += 1
        for i in range(y.shape[0]):
            yfull[yi] = y[i]
            yi += 1

    # write to txt.
    with open("/home/jrl03001/figures/decon/exp3/X.csv", 'wb') as fout:
        for i in range(Xfull.shape[0]):
            fout.write(','.join(['%.3f' % x for x in Xfull[i,:]]) + '\n')
    with open("/home/jrl03001/figures/decon/exp3/Z.csv", 'wb') as fout:
        for i in range(Zfull.shape[0]):
            fout.write(','.join(['%.3f' % x for x in Zfull[i,:]]) + '\n')


    # save experiment.
    save_pickle(args.ref_file, {'Xs':Xs, 'Ss':Ss, 'Cs':Cs, 'Zs':Zs, 'ys':ys})
    save_pickle(args.test_file, {'Xs':Xs, 'Zs':Zs, 'ys':ys})

    # load it and print again.
    data = load_pickle(args.test_file)
    Xs = data['Xs']

    print "HEY DOOD??"
    for X in Xs:
        print X[0:5,0], X[0:5,1]


def create_exp3(args):
    """ creates S is calculated directly from average of SC """

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    c_lbls = np.load(args.c_lbls)
    b_lbls = np.load(args.b_lbls)

    # compute dimensions.
    n = args.n
    m = SC.shape[0]
    k = c_lbls.shape[0]
    l = SC.shape[1]
    t = args.t
    q = args.q
    g = args.g
    w = args.w
    e = args.e

    # build the S simulation object.
    if os.path.isfile(args.sim_obj) == False:
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(args.sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=args.sim_obj)

    # loop over the number of experiments.
    Xs = list()
    Cs = list()
    Ss = list()
    Zs = list()
    ys = list()

    # create master S.
    S = _avg_S(sc_lbls, SC)

    # simulate the single-cells.
    Xtmp, xy = sim.sample_1(1000)

    # make index lookup.
    ilu = dict()
    for a in range(k):
        ilu[a] = np.where(xy == a)[0]

    # loop over each experiment.
    for gg in range(q):

        # simulate the single-cells.
        Z, y = sim.sample_1(t)

        # create S-timate
        H = _avg_S(y, Z)

        # simulate the concentrations.
        C = np.zeros((k, n), dtype=np.float)
        for j in range(n):
            if args.c_type == 1:

                # uniform.
                C[:,j] = 1.0 / float(k)

            elif args.c_type == 2:

                # arithmetic.
                x = [float(x) for x in range(1,k+1)]
                random.shuffle(x)
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            elif args.c_type == 3:

                # geometric.
                x = list(np.vander([k], k)[0])
                random.shuffle(x)
                x = [float(z) for z in x]
                x = np.array(x)
                x = x / np.sum(x)
                C[:,j] = x

            else:
                logging.error("unknown C type")
                sys.exit(1)

        # sample from mixture to create X.
        X = np.zeros((m, n), dtype=np.float)

        # for each sample.
        for j in range(n):

            # choose 100 from each cell type.
            idxs = list()
            for a in range(k):

                # compute count.
                count = int(np.rint(w * C[a,j]) + 1)

                # choose index.
                idxs += list(np.random.choice(ilu[a], size = count))

            # build big slice.
            bs = Xtmp[:,idxs]

            # assign average to sample.
            for i in range(m):
                X[i,j] = np.average(bs[i,:])

        # anova feature selection.
        clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=g)
        clf.fit(np.transpose(Z), y)
        features = np.where(clf.get_support() == True)[0]

        # chop the genes down.
        Z = Z[features,:]
        X = X[features,:]

        # add noise.
        if e != None:
            N = _log_noise(X.shape[0], X.shape[1], e)
            X = X * N

        # save to list.
        Xs.append(X)
        Cs.append(C)
        Ss.append(S)
        Zs.append(Z)
        ys.append(y)

    # save experiment.
    save_pickle(args.ref_file, {'Xs':Xs, 'Ss':Ss, 'Cs':Cs, 'Zs':Zs, 'ys':ys})
    save_pickle(args.test_file, {'Xs':Xs, 'Zs':Zs, 'ys':ys})


def create_exp4(args):
    """ simulate a single missing cell type """

    # load data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    c_lbls = np.load(args.c_lbls)
    b_lbls = np.load(args.b_lbls)

    # compute dimensions.
    n = args.n
    m = SC.shape[0]
    k = c_lbls.shape[0]
    l = SC.shape[1]
    t = args.t
    q = args.q
    g = args.g
    w = args.w
    e = args.e
    r = args.r

    # build the S simulation object.
    if os.path.isfile(args.sim_obj) == False:
        sim = SimSingleCell(SC, sc_lbls)
        sim.save(args.sim_obj)
    else:
        sim = SimSingleCell(SC, sc_lbls, load=args.sim_obj)

    # loop over the number of experiments.
    # basic data.
    Xs = list() # mixture
    Ss = list() # mixture
    Cs = list() # concentrations
    Zs = list() # samples single-cells
    Hs = list() # signature from Z
    ys = list() # labels of Zs
    fs = list() # features used
    mjs = list()  # missing cell type index.

    # sampled data.
    XGs = list() # mixtures with genes removed.
    ZGs = list() # single-cells with genes removed

    # missing.
    ZMs = list() # single-cell with missing cell-type removed
    yms = list() # labels with missing cell-type removed

    # create master S.
    S = _avg_S(sc_lbls, SC)

    # simulate the single-cells.
    Xtmp, xy = sim.sample_1(1000)

    # make index lookup.
    ilu = dict()
    for a in range(k):
        ilu[a] = np.where(xy == a)[0]

    # loop over each experiment.
    for gg in range(q):

        # simulate the single-cells.
        Z, y = sim.sample_1(t)

        # create S-timate
        H = _avg_S(y, Z)

        # simulate C.
        C = _sim_C(n, m, k, args.c_type)

        # simulate X.
        #X = _sim_X_sample(n, m, k, C, ilu, Xtmp)
        X = np.dot(H, C)

        # add noise.
        if e != None:
            N = _log_noise(X.shape[0], X.shape[1], e)
            X = X * N

        # anova feature selection.
        clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=g)
        clf.fit(np.transpose(Z), y)
        features = np.where(clf.get_support() == True)[0]
        features.sort()

        # chop the genes down.
        XG = X[features,:]
        ZG = Z[features,:]

        # remove cell type.
        yidx = np.where(y != r)[0]
        ytmp = y[yidx]
        Ztmp = ZG[:, yidx]

        # renumber stuff.
        for ll in range(r+1, k):
            ytmp[np.where(ytmp == ll)[0]] -= 1

        # save basic to list.
        Xs.append(X)
        Ss.append(S)
        Cs.append(C)
        Zs.append(Z)
        Hs.append(H)
        ys.append(y)
        fs.append(features)
        mjs.append(r)

        # sampled data.
        XGs.append(XG)
        ZGs.append(ZG)

        # missing.
        ZMs.append(Ztmp)
        yms.append(ytmp)

    # dont save.
    if args.debug == True:
        return

    # save experiment.
    save_pickle(args.ref_file, {'Xs':Xs, 'Ss':Ss, 'Cs':Cs, 'Zs':Zs, 'ZGs':ZGs, 'Hs':Hs, 'ys':ys, 'fs':fs, 'mjs':mjs})
    save_pickle(args.test_file, {'XGs':XGs, 'ZMs':ZMs, 'yms':yms})


def run_sim(args):
    """ runs the simulation """

    # load testing data.
    data = load_pickle(args.test_file)
    Xs = data['Xs']
    Zs = data['Zs']
    ys = data['ys']
    k = args.k

    # method name
    method = args.method

    # loop over each experiment.
    for  X, Z, y, wdir, C_path, S_path, idx in _sim_gen(Xs, Zs, ys, method, args.work_dir):

        # check for skippability.
        if os.path.isfile('%s/C.npy' % wdir):
            continue

        # make directory.
        if os.path.isdir(wdir) == False:
            subprocess.call(['mkdir','-p',wdir])

        # make args list.
        alist = (X, Z, y, k, C_path, S_path, wdir)

        # method switch.
        if method == "UCQP":
            _UCQP(*alist, xtra=args.xtra_args)
        elif method == "UCQPM":
            _UCQPM(*alist, xtra=args.xtra_args)
        elif method == "QPROG":
            _QPROG(*alist, xtra=args.xtra_args)
        elif method == "DECONF":
            _DECONF(*alist, xtra=args.xtra_args)
        elif method == "DSA":
            _DSA(*alist, xtra=args.xtra_args)
        elif method == "FROB":
            _FROB(*alist, xtra=args.xtra_args)
        elif method == "SSKL":
            _SSKL(*alist, xtra=args.xtra_args)
        else:
            logging.error("unknown method: %s" % method)
            sys.exit(1)

def run_sim_missing(args):
    """ runs the simulation for missing data. setup to
    allow cheating"""

    # load testing data.
    data = load_pickle(args.test_file)
    ref = load_pickle(args.ref_file)
    XGs = data['XGs']
    ZMs = data['ZMs']
    yms = data['yms']
    k = args.k

    Ss_cheat = ref['Ss']
    Hs_cheat = ref['Hs']
    Cs_cheat = ref['Cs']
    fs_cheat = ref['fs']

    # method name
    method = args.method

    # loop over each experiment.
    for  X, Z, y, wdir, C_path, S_path, idx in _sim_gen(XGs, ZMs, yms, method, args.work_dir):

        # check for skippability.
        if os.path.isfile('%s/C.npy' % wdir):
            logging.info("skipping: %s" % '%s/C.npy' % wdir)
            continue

        # make directory.
        if os.path.isdir(wdir) == False:
            subprocess.call(['mkdir','-p',wdir])

        # cheat for UCQPC
        if method == "UCQPC":

            # get cheat elements.
            Z = ref['ZGs'][idx]
            y = ref['ys'][idx]

        # make args list.
        alist = (X, Z, y, k, C_path, S_path, wdir)

        # method switch.
        if method == "UCQPC":
            _UCQP(*alist, xtra=args.xtra_args)
        elif method == "UCQPM":
            _UCQPM(*alist, xtra={'S':Ss_cheat[idx], 'C':Cs_cheat[idx], 'H':Hs_cheat[idx], 'fs':fs_cheat[idx]})
        elif method == "DECONF":
            _DECONF(*alist, xtra=args.xtra_args)
        elif method == "RAND":
            _RAND(*alist, xtra=args.xtra_args)
        else:
            logging.error("unknown method: %s" % method)
            sys.exit(1)


def evl_sim(args):
    """ runs the simulation """

    # load data.
    test = load_pickle(args.test_file)
    ref = load_pickle(args.ref_file)

    # simplify.
    n = ref['Xs'][0].shape[1]
    m = ref['Xs'][0].shape[0]
    k = ref['Ss'][0].shape[1]
    q = len(ref['Xs'])
    method = args.method
    scorem = args.scorem

    # setup testers.
    if args.missing == False:
        Xs = test['Xs']
        Zs = test['Zs']
        ys = test['ys']
    else:
        Xs = test['XGs']
        Zs = test['ZMs']
        ys = test['yms']

    # loop over each experiment.
    for  X_test, Z_test, y_test, wdir, C_path, S_path, idx in _sim_gen(Xs, Zs, ys, method, args.work_dir):

        # get elements.
        S_ref = ref['Ss'][idx]
        X_ref = ref['Xs'][idx]
        Z_ref = ref['Zs'][idx]
        H_ref = ref['Hs'][idx]
        C_ref = ref['Cs'][idx]
        y_ref = ref['ys'][idx]
        features = ref['fs'][idx]

        # shrink things appropriatly.
        H_ref = H_ref[features,:]

        # load the test matrix.
        if os.path.isfile(C_path):
            C_test = np.load(C_path)
        else:
            # silenty skip missing.
            continue

        # load the test matrix.
        if os.path.isfile(S_path):
            S_test = np.load(S_path)
        else:
            # silenty skip missing.
            continue

        # round to 5 decimals.
        C_ref = np.round(C_ref, decimals=5)
        C_test = np.round(C_test, decimals=5)

        # missing.
        if args.missing == True:

            # grab missing specific.
            mj = ref['mjs'][idx]

            # insert new one.
            if args.method == "UCQPM":

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

            elif args.method == "DECONF":

                # find ordering by matching and pearson correlation.
                order = match_signatures(H_ref, S_test)

                # rorder S and C
                S_test = S_test[:,order]
                C_test = C_test[order,:]

            elif args.method == "UCQPC":
                pass

            else:
                raise NotImplementedError

        # set the scoring function.
        if scorem == 'rmse':
            metric = rmse_vector
        elif scorem == 'pearson':
            metric = pearson_vector
        elif scorem == 'nrmse':
            metric = nrmse_vector
        elif scorem == 'meanabs':
            metric = meanabs_vector
        elif scorem == 'maxabs':
            metric = maxabs_vector
        else:
            logging.error("unknown score method")
            sys.exit()

        # compute column wise average.
        vals = list()
        for j in range(C_ref.shape[1]):
            v = metric(C_ref[:,j], C_test[:,j])
            vals.append(v)
        total = np.average(np.array(vals))

        # compute cell type scores.
        scores = [total]
        for i in range(C_ref.shape[0]):
            scores.append(metric(C_ref[i,:], C_test[i,:]))

        # compute missing signature score.
        if args.missing == True or args.ucqpc == True:
            s_score = metric(H_ref[:, mj], S_test[:, mj])
            txt = ' '.join(['%.5f' % x for x in [s_score] + scores])
        else:
            # print the results.
            txt = ' '.join(['%.5f' % x for x in [s_score] + scores])

        print txt

### script ###

if __name__ == '__main__':

    # mode parser.
    main_p = argparse.ArgumentParser()
    subp = main_p.add_subparsers(help='sub-command help')

    ## load data from sources ##
    # mmc3_figure1b
    subpp = subp.add_parser('mmc3_figure1b', help='import example data')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-d', dest='input_file', required=True, help='input file')
    subpp.add_argument('-g', dest='guoji', action='store_true', default=False, help='guoji 9')
    subpp.set_defaults(func=load_mmc3_figure1b)

    ## create simulations ##
    # basic experiment.
    subpp = subp.add_parser('create_exp1', help='creates experiment 1 data')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-n', type=int, dest='n', required=True, help='number of mixed samples')
    subpp.add_argument('-q', type=int, dest='q', required=True, help='number of experiments')
    subpp.add_argument('-t', type=int, dest='t', required=True, help='number of single-cells')
    subpp.add_argument('-c', type=int, dest='c_type', required=True, help='concentration type')
    subpp.add_argument('-e', type=float, dest='e', required=True, help='error parameter: 0.0 - 0.05')
    subpp.add_argument('-so', dest='sim_obj', required=True, help='simulation object')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-td', dest='test_file', required=True, help='serialized experiment file')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=create_exp1)

    # vary the gene.
    subpp = subp.add_parser('create_exp2', help='creates experiment 1 data')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-n', type=int, dest='n', required=True, help='number of mixed samples')
    subpp.add_argument('-q', type=int, dest='q', required=True, help='number of experiments')
    subpp.add_argument('-t', type=int, dest='t', required=True, help='number of single-cells')
    subpp.add_argument('-c', type=int, dest='c_type', required=True, help='concentration type')
    subpp.add_argument('-e', type=float, dest='e', required=True, help='error parameter: 0.0 - 0.05')
    subpp.add_argument('-g', type=float, dest='g', required=True, help='number of genes to use')
    subpp.add_argument('-so', dest='sim_obj', required=True, help='simulation object')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-td', dest='test_file', required=True, help='serialized experiment file')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=create_exp2)

    # vary gene and sample mixtures directly.
    subpp = subp.add_parser('create_exp3', help='creates experiment 3 data')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-n', type=int, dest='n', required=True, help='number of mixed samples')
    subpp.add_argument('-q', type=int, dest='q', required=True, help='number of experiments')
    subpp.add_argument('-t', type=int, dest='t', required=True, help='number of single-cells')
    subpp.add_argument('-c', type=int, dest='c_type', required=True, help='concentration type')
    subpp.add_argument('-w', type=float, dest='w', required=True, help='error parameter: 1 - 1000')
    subpp.add_argument('-e', type=float, dest='e', required=True, help='noise parameter: 0.0 - 0.05')
    subpp.add_argument('-g', type=float, dest='g', required=True, help='number of genes to use')
    subpp.add_argument('-so', dest='sim_obj', required=True, help='simulation object')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-td', dest='test_file', required=True, help='serialized experiment file')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=create_exp3)

    subpp = subp.add_parser('create_exp4', help='creates missing data experiment: 4')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-n', type=int, dest='n', required=True, help='number of mixed samples')
    subpp.add_argument('-q', type=int, dest='q', required=True, help='number of experiments')
    subpp.add_argument('-t', type=int, dest='t', required=True, help='number of single-cells')
    subpp.add_argument('-r', type=int, dest='r', required=True, help='cell-type to exclude')
    subpp.add_argument('-c', type=int, dest='c_type', required=True, help='concentration type')
    subpp.add_argument('-w', type=float, dest='w', required=True, help='error parameter: 1 - 1000')
    subpp.add_argument('-e', type=float, dest='e', required=True, help='noise parameter: 0.0 - 0.05')
    subpp.add_argument('-g', type=float, dest='g', required=True, help='number of genes to use')
    subpp.add_argument('-so', dest='sim_obj', required=True, help='simulation object')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-td', dest='test_file', required=True, help='serialized experiment file')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.add_argument('-debug', dest='debug', action='store_true', help='debug mode: for developer')
    subpp.set_defaults(func=create_exp4)


    ## run simulations ##
    subpp = subp.add_parser('run_sim', help='run simulation 3')
    subpp.add_argument('-td', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for temporary files')
    subpp.add_argument('-k', dest='k', required=True, type=int, help='number of cell types')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    subpp.add_argument('-x', dest='xtra_args', help='optional arguments: comma sep list')
    subpp.set_defaults(func=run_sim)

    subpp = subp.add_parser('run_sim_missing', help='run simulation with missing data')
    subpp.add_argument('-td', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for temporary files')
    subpp.add_argument('-k', dest='k', required=True, type=int, help='number of cell types')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    subpp.add_argument('-x', dest='xtra_args', help='optional arguments: comma sep list')
    subpp.set_defaults(func=run_sim_missing)

    ## evaluate simulations ##
    subpp = subp.add_parser('evl_sim', help='evaluate simulation')
    subpp.add_argument('-tf', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-rf', dest='ref_file', required=True, help='the reference file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for simulation')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    subpp.add_argument('-s', dest='scorem', required=True, help='metric to use')
    #subpp.add_argument('-out', dest='out_file', required=True, help='output file')
    subpp.add_argument('-missing', dest='missing', action='store_true', help='adds missing signature info')
    subpp.set_defaults(func=evl_sim)

    ## testing and random stuff ##
    subpp = subp.add_parser('plot_sim', help='plot X and Z values')
    subpp.add_argument('-td', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for temporary files')
    subpp.add_argument('-k', dest='k', required=True, type=int, help='number of cell types')
    subpp.add_argument('-fig', dest='fig_file', type=str, required=True, help='figure file')
    subpp.set_defaults(func=plot_sim)

    subpp = subp.add_parser('compare_cluster_avg', help='visualizes kmeans')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-k', type=int, dest='k', required=True, help='number of clusters')
    subpp.set_defaults(func=compare_cluster_avg)

    subpp = subp.add_parser('compare_cluster_sample', help='visualizes kmeans')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-k', type=int, dest='k', required=True, help='number of clusters')
    subpp.set_defaults(func=compare_cluster_sample)

    subpp = subp.add_parser('plot_gene', help='plots single cell values')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-g', dest='gene_name', type=str, required=True, help='gene name')
    subpp.add_argument('-fdir', dest='fig_dir', type=str, required=True, help='figure directory')
    subpp.set_defaults(func=plot_gene)

    subpp = subp.add_parser('plot_singlecell', help='plots true/predicted concentrations')
    subpp.add_argument('-d', dest='base_dir', type=str, required=True, help='base directory')
    subpp.add_argument('-mlist', dest='mlist', type=str, required=True, help='methods')
    subpp.add_argument('-c', dest='c', type=int, required=True, help='concentration')
    subpp.add_argument('-q', dest='q', type=int, required=True, help='q')
    subpp.add_argument('-e', dest='e', type=int, required=True, help='e')
    subpp.add_argument('-tlist', dest='tlist', type=str, required=True, help='t')
    subpp.add_argument('-g', dest='g', type=int, required=True, help='g')
    subpp.set_defaults(func=plot_singlecell)

    subpp = subp.add_parser('plot_varygene', help='plots as a function of # genes')
    subpp.add_argument('-d', dest='base_dir', type=str, required=True, help='base directory')
    subpp.add_argument('-mlist', dest='mlist', type=str, required=True, help='methods')
    subpp.add_argument('-c', dest='c', type=int, required=True, help='concentration')
    subpp.add_argument('-q', dest='q', type=int, required=True, help='q')
    subpp.add_argument('-e', dest='e', type=int, required=True, help='e')
    subpp.add_argument('-glist', dest='glist', type=str, required=True, help='g')
    subpp.add_argument('-t', dest='t', type=int, required=True, help='t')
    subpp.set_defaults(func=plot_varygene)

    subpp = subp.add_parser('plot_Z', help='plots the single cell values')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-fig', dest='fig_file', type=str, required=True, help='figure file')
    subpp.set_defaults(func=pca_Z)

    subpp = subp.add_parser('plot_genes', help='plots top 20 single cell values')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-fdir', dest='fig_dir', type=str, required=True, help='figure directory')
    subpp.set_defaults(func=plot_genes)

    subpp = subp.add_parser('plot_scatter', help='plots true/predictced scatter')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for temporary files')
    subpp.add_argument('-tf', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-rf', dest='ref_file', required=True, help='the reference file')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    subpp.add_argument('-fig', dest='fig_file', type=str, required=True, help='figure file')
    subpp.set_defaults(func=plot_scatter)

    subpp = subp.add_parser('heatmap', help='heatmap/cluster single cell values')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-fig', dest='fig_path', type=str, required=True, help='figure path')
    subpp.set_defaults(func=heatmap)

    # parse args.
    args = main_p.parse_args()
    args.func(args)
