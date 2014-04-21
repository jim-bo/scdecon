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
import random
import math
import operator
import StringIO

# statistics.
from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import gmean
from sklearn import feature_selection

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s', )

# app
from utils.matops import *
from utils.rfuncs import *
from utils.misc import *
from utils.plotting import *
#from utils.heirheat import *
from scdecon.cluster import *


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


def _UCQP(X, Z, y, k, C_path, wdir):

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


def _PCQP(X, Z, y, k, C_path, wdir):

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

def _DECONF(X, Z, y, k, C_path, wdir):
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

    # load the R libraries.
    load_R_libraries()
    r_ged = R.r['ged']
    r_coef = R.r['coef']
    r_basis = R.r['basis']

    # run deconvolution.
    txt = '''exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);
res <- ged(eset, {num}, method='deconf');
'''.format(path=wdir, num=k)
    R.r(txt)
    res = R.r('res')

    # extract data.
    Stmp = r_basis(res)
    Ctmp = r_coef(res)

    # convert.
    S, rownames, colnames = r2npy(Stmp)
    C, rownames, colnames = r2npy(Ctmp)

    # write it.
    np.save(C_path, C)


def _QPROG(X, Z, y, k, C_path, wdir):
    """ deconvolution """

    # load the R libraries.
    load_R_libraries()
    r_ged = R.r['ged']
    r_coef = R.r['coef']
    r_basis = R.r['basis']

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


    # run deconvolution.
    txt = '''# load data.
sigFile <- file.path("{path}", "S.dat");
S <- as.matrix(read.table(sigFile, header=TRUE, sep="\t", row.names=1, as.is=TRUE));

exprFile <- file.path("{path}", "mat.dat");
X <- as.matrix(read.table(exprFile, header=TRUE, sep="\t", row.names=1, as.is=TRUE));

# run decon.
res <- ged(X, S, method='qprog');
'''.format(path=wdir)

    R.r(txt)
    res = R.r('res')

    # extract data.
    Stmp = r_basis(res)
    Ctmp = r_coef(res)

    # convert.
    S, rownames, colnames = r2npy(Stmp)
    C, rownames, colnames = r2npy(Ctmp)

    # write it.
    np.save(C_path, C)


def _DSA(X, Z, y, k, C_path, wdir):
    """ deconvolution """

    # load the R libraries.
    load_R_libraries()
    r_ged = R.r['ged']
    r_coef = R.r['coef']
    r_basis = R.r['basis']

    ## identify marker genes,
    # identify non-null rows.
    good_rows = list()
    for i in range(Z.shape[0]):
        if np.sum(Z[i,:]) > 0.0:
            good_rows.append(i)
    Z = Z[good_rows,:]

    # serialize single-cells to matrix.
    tmp_mat = '%s/Z.dat' % wdir
    with open(tmp_mat, "wb") as fout:
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

    # simplify labels.
    lbls = ','.join(['%i' % x for x in y])

    # extract markers.
    txt = '''# load the pure cells into expression set
sigFile <- file.path("{path}", "Z.dat");
Z <- as.matrix(read.table(sigFile, header=TRUE, sep="\t", row.names=1, as.is=TRUE));
Z <- Z + 1

# labels
y <- c({y});

# perform extraction.
sml <- extractMarkers(Z, data=y, method='Abbas')
'''.format(path=wdir, y=lbls)
    R.r(txt)

    ## run the deconvolution
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
    txt = '''exprsFile <- file.path("{path}", "mat.dat");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);
res <- ged(eset[sml,], sml, 'DSA', verbose=TRUE)
'''.format(path=wdir, num=k)
    R.r(txt)

    # extract data.
    res = R.r('res')
    Stmp = r_basis(res)
    Ctmp = r_coef(res)

    # convert.
    S, rownames, colnames = r2npy(Stmp)
    C, rownames, colnames = r2npy(Ctmp)

    # write it.
    np.save(C_path, C)


def _sim_gen(Xs, Zs, ys, method):
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
        wdir = '%s/%d/%s' % (args.work_dir, e, method)

        # make common output file.
        C_path = "%s/C.npy" % (wdir)

        # yield it.
        yield X, Z, y, wdir, C_path, e

### public functions ###


def plot_gene(args):
    """ plots expression values """

    # load the data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    b_lbls = np.load(args.b_lbls)
    c_lbls = np.load(args.c_lbls)

    # simulate single-cells.
    sim = SimSingleCell(SC, sc_lbls, load=False)
    TMP, we = sim.sample_1(1000)

    # set gene name.
    gene_name = args.gene_name

    # loop over each class.
    xlist = list()
    for c in range(len(c_lbls)):

        # extract subset of SC.
        SC_s = SC[:,np.where(sc_lbls == c)[0]]
        TMP_s = TMP[:,np.where(we == c)[0]]

        # extract subset of gene.
        SC_s = SC_s[np.where(b_lbls == gene_name)[0],:]
        TMP_s = TMP_s[np.where(b_lbls == gene_name)[0],:]

        # make sure is 1d (some duplicate genes measured)
        SC_s = np.ravel(SC_s)
        TMP_s = np.ravel(TMP_s)

        # store list.
        xlist.append((SC_s, "%s:%s" % (str(c_lbls[c]), "t")))
        xlist.append((TMP_s, "%s:%s" % (str(c_lbls[c]), "s")))

    # plot it.
    fname = '%s/%s.pdf' % (args.fig_dir, gene_name)
    gene_histo(xlist, fname, gene_name)

def plot_genes(args):
    """ plots expression values """

    # load the data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    b_lbls = np.load(args.b_lbls)
    c_lbls = np.load(args.c_lbls)

    # get teh informative features.
    clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=20)
    clf.fit(np.transpose(SC), sc_lbls)
    features = np.where(clf.get_support() == True)[0]

    # simulate single-cells.
    sim = SimSingleCell(SC, sc_lbls)
    TMP, we = sim.sample_1(1000)

    # loop over genes:
    for i in features:

        # set gene name.
        gene_name = b_lbls[i]

        # loop over each class.
        xlist = list()
        for c in range(len(c_lbls)):

            # extract subset of SC.
            SC_s = SC[:,np.where(sc_lbls == c)[0]]
            TMP_s = TMP[:,np.where(we == c)[0]]

            # extract subset of gene.
            SC_s = SC_s[np.where(b_lbls == gene_name)[0],:]
            TMP_s = TMP_s[np.where(b_lbls == gene_name)[0],:]

            # make sure is 1d (some duplicate genes measured)
            SC_s = np.ravel(SC_s)
            TMP_s = np.ravel(TMP_s)

            # store list.
            xlist.append((SC_s, "%s:%s" % (str(c_lbls[c]), "t")))
            xlist.append((TMP_s, "%s:%s" % (str(c_lbls[c]), "s")))

        # plot it.
        fname = '%s/%s.pdf' % (args.fig_dir, gene_name)
        gene_histo(xlist, fname, gene_name)

def heatmap(args):
    """ heatmap and clustering """

    # load the data.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)
    b_lbls = np.load(args.b_lbls)
    c_lbls = np.load(args.c_lbls)

    # get teh informative features.
    clf = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=20)
    clf.fit(np.transpose(SC), sc_lbls)
    features = np.where(clf.get_support() == True)[0]

    # extract subset.
    SC = SC[features,:]

    # create master S.
    #S = _avg_S(sc_lbls, SC)

    # make the heatmap.
    print c_lbls
    sys.exit()
    heirheatmap(SC, sc_lbls, args.fig_path)
    #graph = TestHeatmap()
    #graph.plot(args.fig_path, SC, b_lbls, [str(x) for x in sc_lbls])

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

    # build list of house keepers.
    '''
    tmp = ['ACTB', 'GAPDH']
    hkeepers = list()
    for h in tmp:
        hkeepers += list(np.where(b_lbls == h)[0])
    hkeepers = sorted(hkeepers)

    # normalize S/
    Snorm = S.copy()
    for a in range(k):
        Snorm[:,a] = S[:,a] / gmean(S[hkeepers,a])
    '''
    # simulate single-cells.
    #TMP, we = sim.sample_1(t * 1000)

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

        '''
        # sample to compute mixtures.
        cheat_cnt = 0
        X = np.zeros((m, n), dtype=np.float)
        for j in range(n):

            # loop over each class.
            for l in range(k):

                # decide how many samples.
                scnt = int(50 * C[l,j])

                print scnt

                # choose appropriate subset.
                sub = Z[:, y == l]

                # choose randomly among these subsets.
                idxs = np.random.choice(sub.shape[1], size=scnt, replace=True)
                cheat_cnt += len(idxs)

                # sum these up gene by gene.
                for i in range(m):
                    X[i, j] = np.sum(sub[i,idxs])



        print "wago"
        sys.exit()
        '''
        # add noise.
        N = _log_noise(X2.shape[0], X2.shape[1], e)
        X2 = X2 * N
        '''
        # normalize by geometric mean of housekeepers.
        for j in range(n):
            X2[:,j] = X2[:,j] / gmean(X2[hkeepers,j])

        for j in range(n):
            X[:,j] = X[:,j] / gmean(X[hkeepers,j])

        for b in range(l):
            Z[:,b] = Z[:,b] / gmean(Z[hkeepers,b])
        '''


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
    for  X, Z, y, wdir, C_path, idx in _sim_gen(Xs, Zs, ys, method):

        # make directory.
        if os.path.isdir(wdir) == False:
            subprocess.call(['mkdir','-p',wdir])

        # method switch.
        if method == "UCQP":
            _UCQP(X, Z, y, k, C_path, wdir)
        elif method == "QPROG":
            _QPROG(X, Z, y, k, C_path, wdir)
        elif method == "DECONF":
            _DECONF(X, Z, y, k, C_path, wdir)
        elif method == "DSA":
            _DSA(X, Z, y, k, C_path, wdir)
        elif method == "PCQP":
            _PCQP(X, Z, y, k, C_path, wdir)
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
    q = len(ref['Xs'])
    method = args.method

    # loop over each experiment.
    for  X_test, Z_test, y_test, wdir, C_path, idx in _sim_gen(test['Xs'], test['Zs'], test['ys'], method):

        # get elements.
        X_ref = ref['Xs'][idx]
        Z_ref = ref['Zs'][idx]
        C_ref = ref['Cs'][idx]
        y_ref = ref['ys'][idx]

        # load the test matrix.
        C_test = np.load(C_path)

        # compute the column score.
        total = rmse_cols(C_test, C_ref)

        # compute the cell-type score.
        scores = [total]
        for i in range(C_ref.shape[0]):
            scores.append(rmse_vector(C_test[i,:], C_ref[i,:]))

        # print the results.
        txt = ' '.join(['%.5f' % x for x in scores])
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

    # normalized data.
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
    subpp.add_argument('-so', dest='sim_obj', required=True, help='simulation object')
    subpp.add_argument('-rd', dest='ref_file', required=True, help='serialized experiment file')
    subpp.add_argument('-td', dest='test_file', required=True, help='serialized experiment file')
    subpp.add_argument('-sd', dest='sim_dir', required=True, help='simulation directory for validation')
    subpp.set_defaults(func=create_exp1)

    ## run simulations ##
    subpp = subp.add_parser('run_sim', help='run simulation')
    subpp.add_argument('-td', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for temporary files')
    subpp.add_argument('-k', dest='k', required=True, help='number of cell types')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    subpp.set_defaults(func=run_sim)

    ## evaluate simulations ##
    subpp = subp.add_parser('evl_sim', help='evaluate simulation')
    subpp.add_argument('-tf', dest='test_file', required=True, help='the test file')
    subpp.add_argument('-rf', dest='ref_file', required=True, help='the reference file')
    subpp.add_argument('-wd', dest='work_dir', required=True, help='working directory for simulation')
    subpp.add_argument('-m', dest='method', required=True, help='method to test')
    #subpp.add_argument('-out', dest='out_file', required=True, help='output file')
    subpp.set_defaults(func=evl_sim)

    ## testing and random stuff ##
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

    subpp = subp.add_parser('plot_genes', help='plots top 20 single cell values')
    subpp.add_argument('-SC', dest='SC', required=True, help='path for matrix')
    subpp.add_argument('-sc_lbls', dest='sc_lbls', required=True, help='path for matrix')
    subpp.add_argument('-b_lbls', dest='b_lbls', required=True, help='path for matrix')
    subpp.add_argument('-c_lbls', dest='c_lbls', required=True, help='path for matrix')
    subpp.add_argument('-fdir', dest='fig_dir', type=str, required=True, help='figure directory')
    subpp.set_defaults(func=plot_genes)

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
