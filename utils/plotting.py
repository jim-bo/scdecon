""" plotting """
## imports ##
import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
#import brewer2mpl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch

# violin plot
from scipy.stats import gaussian_kde
from numpy.random import normal
from numpy import arange

# scikit
from sklearn.decomposition import PCA

# application.
from utils.matops import *
from utils.misc import *
#from simulation import _sim_gen
from simulation import _remap_missing, _match_pred

## high-level functions ##
def pca_X_Z(X, Z, y, figure_path):
    """ plots the experiment """

    # create color map.
    unique_vals = sorted(list(np.unique(y)))
    num_colors = len(unique_vals)
    cmap = plt.get_cmap('gist_rainbow')
    cnorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    # do PCA
    pcaX = PCA(n_components=2)
    pcaZ = PCA(n_components=2)
    Xt = np.transpose(X)
    Zt = np.transpose(Z)
    Xp = pcaX.fit(Xt).transform(Xt)
    Zp = pcaZ.fit(Zt).transform(Zt)

    # plot pure.
    for i in unique_vals:

        # get color.
        color = cmap(1.*i/num_colors)
        label = str(i)

        # plot it.
        plt.scatter(Zp[y == i, 0], Zp[y == i, 1], c=color, label=label)

    # plot mixed.
    plt.scatter(Xp[:, 0], Zp[:, 1], c="black", label="mix")

    # add legend.
    plt.legend()
    plt.savefig(figure_path)

def pca_sc(args):
    """ plots the experiment """

    # simplify.
    Z = np.load(args.SC)
    y = np.load(args.sc_lbls)
    labels = np.load(args.c_lbls)
    figure_path = args.fig_file

    # create color map.
    unique_vals = sorted(list(np.unique(y)))
    num_colors = len(unique_vals)
    cmap = plt.get_cmap('gist_rainbow')
    cnorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    # do PCA
    pcaZ = PCA(n_components=2)
    Zt = np.transpose(Z)
    Zp = pcaZ.fit(Zt).transform(Zt)

    # plot pure.
    for i in unique_vals:

        # get color.
        color = cmap(1.*i/num_colors)
        label = labels[i]

        # plot it.
        plt.scatter(Zp[y == i, 0], Zp[y == i, 1], c=color, label=label)

    # add labels.
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    # add legend.
    plt.legend()
    plt.savefig(figure_path)


def plot_scatter(args):
    """ plots experiment as scatter plot """

    # load data.
    test = load_pickle(args.test_file)
    ref = load_pickle(args.ref_file)

    # simplify.
    n = ref['Xs'][0].shape[1]
    m = ref['Xs'][0].shape[0]
    q = len(ref['Xs'])
    method = args.method

    # create list by cell-type.
    bycell = dict()

    # loop over each experiment.
    for X_test, Z_test, y_test, wdir, C_path, S_path, idx in _sim_gen(test['Xs'], test['Zs'], test['ys'], method, args.work_dir):

        # get elements.
        X_ref = ref['Xs'][idx]
        Z_ref = ref['Zs'][idx]
        C_ref = ref['Cs'][idx]
        y_ref = ref['ys'][idx]

        # load the test matrix.
        if os.path.isfile(C_path):
            C_test = np.load(C_path)
        else:
            # silenty skip missing.
            continue

        # round to 5 decimals.
        C_ref = np.round(C_ref, decimals=5)
        C_test = np.round(C_test, decimals=5)

        # add by cell type.
        for j in range(C_ref.shape[1]):
            for l in range(C_ref.shape[0]):
                if l not in bycell:
                    bycell[l] = [list(), list()]

                bycell[l][0].append(C_ref[l,j])
                bycell[l][1].append(C_test[l,j])

    # create color map.
    unique_vals = sorted(bycell.keys())
    num_colors = len(unique_vals)
    cmap = plt.get_cmap('gist_rainbow')
    cnorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    # print them
    for l in bycell:

        # get data.
        x = np.array(bycell[l][0])
        y = np.array(bycell[l][1])

        # plot the regression.
        fit = plb.polyfit(x, y, 1)
        fit_fn = plb.poly1d(fit)

        # compute r^2
        yhat = fit_fn(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y-ybar)**2)
        r2 = ssreg / sstot

        # compute the color.
        color = cmap(1.*l/num_colors)

        # plot the points.
        plt.plot(x, y, '.', color=color, label='%i, r^2=%.2f' % (l,r2))

        # plot the regression.
        plt.plot(x, fit_fn(x), '--', color=color)

        # plot middle line.
        plt.plot(np.arange(0,1.1,.1), np.arange(0,1.1,.1), '-', color='black')

    # add legend.
    plt.legend()
    plt.savefig(args.fig_file)



def gene_histo(xlist, figure_path, title):
    """ plots histogram for gene """

    # extract data.
    data = [x[0] for x in xlist]
    lbls = [x[1] for x in xlist]

    # create figure.
    pos = range(len(xlist))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _violin_plot(ax, data, pos, bp=1)


    # finalize.
    plt.title(title)
    ax.set_xticks(pos)
    ax.set_xticklabels(lbls)
    ax.set_ylim([0,300])
    plt.savefig(figure_path)


def heirheatmap(M, row_labels, path):
    """ heirarchy heatmap """

    # transpose data so samples - rows, cols - genes
    M = np.transpose(M)

    # convert numpy to DF.
    testDF = pd.DataFrame(M)

    # plot it.
    axi = plt.imshow(testDF,interpolation='nearest',cmap=cm.RdBu)
    ax = axi.get_axes()
    _clean_axis(ax)

    # calculate pairwise distances for rows
    pairwise_dists = distance.squareform(distance.pdist(testDF))

    # cluster
    clusters = sch.linkage(pairwise_dists,method='complete')

    # make dendrogram.
    den = sch.dendrogram(clusters,color_threshold=np.inf)

    # rename row clusters
    row_clusters = clusters
    col_pairwise_dists = distance.squareform(distance.pdist(testDF.T))
    col_clusters = sch.linkage(col_pairwise_dists,method='complete')

    ## plot the heatmap and dendrogram ##
    # plot the results
    fig = plt.figure()
    heatmapGS = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0,width_ratios=[0.25,1],height_ratios=[0.25,1])

    ### col dendrogram ####
    col_denAX = fig.add_subplot(heatmapGS[0,1])
    col_denD = sch.dendrogram(col_clusters,color_threshold=np.inf)
    _clean_axis(col_denAX)

    ### row dendrogram ###
    rowGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[1,0],wspace=0.0,hspace=0.0,width_ratios=[1,0.25])
    row_denAX = fig.add_subplot(rowGSSS[0,0])
    row_denD = sch.dendrogram(row_clusters,color_threshold=np.inf,orientation='right')
    _clean_axis(row_denAX)

    ### row colorbar ###
    row_cbAX = fig.add_subplot(rowGSSS[0,1])
    tmp = [ [x] for x in row_labels[row_denD['leaves']] ]
    row_axi = row_cbAX.imshow(tmp,interpolation='nearest',aspect='auto',origin='lower')
    _clean_axis(row_cbAX)
    print tmp

    ### heatmap ###
    heatmapAX = fig.add_subplot(heatmapGS[1,1])
    axi = heatmapAX.imshow(testDF.ix[den['leaves'],col_denD['leaves']],interpolation='nearest',aspect='auto',origin='lower',cmap=cm.RdBu)
    _clean_axis(heatmapAX)

    ### scale colorbar ###
    scale_cbGSSS = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=heatmapGS[0,0],wspace=0.0,hspace=0.0)
    scale_cbAX = fig.add_subplot(scale_cbGSSS[0,0]) # colorbar for scale in upper left corner
    cb = fig.colorbar(axi,scale_cbAX) # note that we tell colorbar to use the scale_cbAX axis
    cb.set_label('Measurements')
    cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
    cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
    cb.outline.set_linewidth(0)

    fig.tight_layout()

    # save figure.
    plt.savefig(path)


def plot_sim(args):
    """ plot the simulation """

    # load testing data.
    data = load_pickle(args.test_file)
    Xs = data['Xs']
    Zs = data['Zs']
    ys = data['ys']
    k = args.k


    # create master array for each.
    Xfull = np.zeros((Xs[0].shape[0], Xs[0].shape[1]*len(Xs)), dtype=np.float)
    Zfull = np.zeros((Zs[0].shape[0], Zs[0].shape[1]*len(Zs)), dtype=np.float)
    yfull = np.zeros(ys[0].shape[0]*len(ys), dtype=np.int)

    # loop over each experiment.
    xj = 0
    zj = 0
    yi = 0
    for  X, Z, y, wdir, C_path, S_path, idx in _sim_gen(Xs, Zs, ys, "bla"):

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

    # call the plot command.
    pca_X_Z(Xfull, Zfull, yfull, args.fig_file)


def plot_singlecell(args):
    """ plot the simulation """

    # simplify parameters.
    base_dir = os.path.abspath(args.base_dir)
    tlist = [int(x) for x in args.tlist.split(",")]
    mlist = [x for x in args.mlist.split(",")]
    c = args.c
    e = args.e
    g = args.g

    # print mlist.
    print ',' + ','.join(mlist)

    # loop over each singlecell.
    for t in tlist:

        # set the reference files.
        dat_dir = "%s/data/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)
        ref_file= "%s/ref_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)
        test_file= "%s/test_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)

        # load them.
        test = load_pickle(test_file)
        ref = load_pickle(ref_file)

        # set the work dir.
        work_dir = "%s/work/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)

        # loop over each test case.
        lookup = dict()
        for m in mlist:

            # bootstrap.
            if m not in lookup:
                lookup[m] = list()

            # loop over instances.
            for  X_test, Z_test, y_test, wdir, C_path, S_path, idx in _sim_gen(test['Xs'], test['Zs'], test['ys'], m, work_dir):

                # simplify.
                X_ref = ref['Xs'][idx]
                Z_ref = ref['Zs'][idx]
                C_ref = ref['Cs'][idx]
                y_ref = ref['ys'][idx]

                # load the test matrix.
                if os.path.isfile(C_path):
                    C_test = np.load(C_path)
                else:
                    # silenty skip missing.
                    continue

                # round to 5 decimals.
                C_ref = np.round(C_ref, decimals=5)
                C_test = np.round(C_test, decimals=5)

                # set the scoring function.
                metric = rmse_vector

                # compute column wise average.
                vals = list()
                for j in range(C_ref.shape[1]):
                    v = metric(C_ref[:,j], C_test[:,j])
                    vals.append(v)
                total = np.average(np.array(vals))

                # put into lookup.
                lookup[m].append(total)


        # print this row.
        while lookup != {}:
            byrow = list()
            for m in mlist:

                # add if present.
                if m in lookup:

                    # add to this row.
                    if len(lookup[m]) > 0:
                        byrow.append('%.3f' % lookup[m].pop())

                    # clear if empty.
                    if len(lookup[m]) == 0:
                        del lookup[m]

                # add empty.
                else:
                    byrow.append("")

            # print the row as a function of # single-cells.
            print '%i,' % t + ','.join(byrow)



def plot_varygene(args):
    """ plot the simulation """

    # simplify parameters.
    base_dir = os.path.abspath(args.base_dir)
    glist = [int(x) for x in args.glist.split(",")]
    mlist = [x for x in args.mlist.split(",")]
    c = args.c
    e = args.e
    t = args.t

    # print mlist.
    print ',' + ','.join(mlist)

    # loop over each singlecell.
    for g in glist:

        # set the reference files.
        dat_dir = "%s/data/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)
        ref_file= "%s/ref_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)
        test_file= "%s/test_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)

        # load them.
        test = load_pickle(test_file)
        ref = load_pickle(ref_file)

        # set the work dir.
        work_dir = "%s/work/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)

        # loop over each test case.
        lookup = dict()
        for m in mlist:

            # bootstrap.
            if m not in lookup:
                lookup[m] = list()

            # loop over instances.
            for  X_test, Z_test, y_test, wdir, C_path, S_path, idx in _sim_gen(test['Xs'], test['Zs'], test['ys'], m, work_dir):

                # simplify.
                X_ref = ref['Xs'][idx]
                Z_ref = ref['Zs'][idx]
                C_ref = ref['Cs'][idx]
                y_ref = ref['ys'][idx]

                # load the test matrix.
                if os.path.isfile(C_path):
                    C_test = np.load(C_path)
                else:
                    # silenty skip missing.
                    continue

                # round to 5 decimals.
                C_ref = np.round(C_ref, decimals=5)
                C_test = np.round(C_test, decimals=5)

                # set the scoring function.
                metric = rmse_vector

                # compute column wise average.
                vals = list()
                for j in range(C_ref.shape[1]):
                    v = metric(C_ref[:,j], C_test[:,j])
                    vals.append(v)
                total = np.average(np.array(vals))

                # put into lookup.
                lookup[m].append(total)


        # print this row.
        while lookup != {}:
            byrow = list()
            for m in mlist:

                # add if present.
                if m in lookup:

                    # add to this row.
                    if len(lookup[m]) > 0:
                        byrow.append('%.4f' % lookup[m].pop())

                    # clear if empty.
                    if len(lookup[m]) == 0:
                        del lookup[m]

                # add empty.
                else:
                    byrow.append("")

            # print the row as a function of # single-cells.
            print '%i,' % g + ','.join(byrow)

def plot_truepred(args):
    """ plot the simulation """

    # simplify parameters.
    base_dir = os.path.abspath(args.base_dir)
    tlist = [int(x) for x in args.tlist.split(",")]
    mlist = [x for x in args.mlist.split(",")]
    c = args.c
    e = args.e
    g = args.g

    # print mlist.
    print ',' + ','.join(mlist)

    # loop over each singlecell.
    for t in tlist:

        # set the reference files.
        dat_dir = "%s/data/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)
        ref_file= "%s/ref_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)
        test_file= "%s/test_%i_%i_%i_%i_%i_%i.cpickle" % (dat_dir, t*5, args.q, t, c, e, g)

        # load them.
        test = load_pickle(test_file)
        ref = load_pickle(ref_file)

        # set the work dir.
        work_dir = "%s/work/%i_%i_%i_%i_%i_%i" % (base_dir, t*5, args.q, t, c, e, g)

        # loop over each test case.
        lookup = dict()
        for m in mlist:

            # bootstrap.
            if m not in lookup:
                lookup[m] = list()

            # loop over instances.
            for  X_test, Z_test, y_test, wdir, C_path, S_path, idx in _sim_gen(test['Xs'], test['Zs'], test['ys'], m, work_dir):

                # simplify.
                X_ref = ref['Xs'][idx]
                Z_ref = ref['Zs'][idx]
                C_ref = ref['Cs'][idx]
                y_ref = ref['ys'][idx]

                # load the test matrix.
                if os.path.isfile(C_path):
                    C_test = np.load(C_path)
                else:
                    # silenty skip missing.
                    continue

                # round to 5 decimals.
                C_ref = np.round(C_ref, decimals=5)
                C_test = np.round(C_test, decimals=5)

                # set the scoring function.
                metric = rmse_vector

                # compute column wise average.
                vals = list()
                for j in range(C_ref.shape[1]):
                    v = metric(C_ref[:,j], C_test[:,j])
                    vals.append(v)
                total = np.average(np.array(vals))

                # put into lookup.
                lookup[m].append(total)


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


def plot_C(args):
    """ evaluates the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir
    c_lbls = np.load(args.c_lbls)
    
    # extract method info.
    method_name = args.method_name
    
    # load the simulation data stuff.
    master = load_pickle(mas_obj)
    results = load_pickle(res_obj)

    # sort the keys.
    keys = sorted(results.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # build the list.
    true = dict()
    pred = dict()
    for l in range(args.k):
        true[l] = list()
        pred[l] = list()

    # loop over each dependent.
    r = -1
    for dkey in keys:

        # skip short keys.
        #if len(dkey) != 6: continue
        if len(dkey) != 7: continue

        # expand the key.
        n, k, e, c, r, q, m = dkey
        #n, k, e, c, q, m = dkey
        mkey = (n, k, e, c, r, q)
        #mkey = (n, k, e, c, q)
        skey = n, k, e, c, r, m        # remove reference ot repeat variable
        #skey = n, k, e, c, m        # remove reference ot repeat variable and cell types

        # skip till selected.
        if n != args.n: continue
        if k != args.k: continue
        if e != args.e: continue
        if c != args.c: continue
        if m != args.m: continue

        # load the true concentrations.
        S_true = np.load('%s.npy' % master[mkey]['H'])
        S_true = S_true[0:m,:]
        C_true = np.load('%s.npy' % master[mkey]['C'])
        
        # load the predicted.
        S_pred = np.load(results[dkey][method_name]['S'])
        C_pred = np.load(results[dkey][method_name]['C'])
               
        # remap if its not DECONF
        if method_name != "DECONF":

            # remap to known order.
            if r != -1:
                C_pred, S_pred = _remap_missing(C_pred, S_pred, r, k)
        else:
            
            # perform matching.
            C_pred, S_pred = _match_pred(C_pred, S_pred, C_true, S_true)

        # add to data.
        for j in range(n):
            #for l in range(k):
            #    if l == r: continue
            for l in [r]:
                true[l].append(C_true[l,j])
                pred[l].append(C_pred[l,j])
        
    # cast to array.
    for l in range(args.k):
        true[l] = np.array(true[l])
        pred[l] = np.array(pred[l])
    
    # create color map.
    num_colors = args.k
    cmap = plt.get_cmap('gist_rainbow')
    cnorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    # print them
    for l in range(args.k):

        # get data.
        x = true[l]
        y = pred[l]

        # plot the regression.
        fit = plb.polyfit(x, y, 1)
        fit_fn = plb.poly1d(fit)

        # compute r^2
        yhat = fit_fn(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y-ybar)**2)
        r2 = ssreg / sstot

        # compute the color.
        color = cmap(1.*l/num_colors)

        # plot the points.
        plt.plot(x, y, '.', color=color, label='%s, r^2=%.2f' % (c_lbls[l],r2))

        # plot the regression.
        plt.plot(x, fit_fn(x), '--', color=color)

        # plot middle line.
        plt.plot(np.arange(0,1.1,.1), np.arange(0,1.1,.1), '-', color='black')

    # add legend.
    plt.legend(numpoints=1)
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    
    # add labels.
    plt.xlabel("observed")
    plt.ylabel("predicted")

    # add legend.
    #plt.legend()
    plt.savefig(args.fig_file)
    

def plot_S(args):
    """ evaluates the experiment for a given method """

    # setup directory.
    sim_dir = os.path.abspath(args.sim_dir)
    mas_obj = '%s/mas.cpickle' % sim_dir
    res_obj = '%s/res.cpickle' % sim_dir
    c_lbls = np.load(args.c_lbls)
    b_lbls = np.load(args.b_lbls)
    
    # extract method info.
    method_name = args.method_name
    
    # load the simulation data stuff.
    master = load_pickle(mas_obj)
    results = load_pickle(res_obj)

    # sort the keys.
    keys = sorted(results.keys(), key=operator.itemgetter(0,1,2,3,4,5))

    # build the list.
    true = dict()
    pred = dict()
    for l in range(args.k):
        true[l] = list()
        pred[l] = list()

    # create the geene tracker.
    genet = list()
    for i in range(args.m):
        genet.append(list())

    # loop over each dependent.
    r = -1
    for dkey in keys:

        # skip short keys.
        #if len(dkey) != 6: continue
        if len(dkey) != 7: continue

        # expand the key.
        n, k, e, c, r, q, m = dkey
        #n, k, e, c, q, m = dkey
        mkey = (n, k, e, c, r, q)
        #mkey = (n, k, e, c, q)
        skey = n, k, e, c, r, m        # remove reference ot repeat variable
        #skey = n, k, e, c, m        # remove reference ot repeat variable and cell types

        # skip till selected.
        if n != args.n: continue
        if k != args.k: continue
        if e != args.e: continue
        if c != args.c: continue
        if m != args.m: continue

        # load the true concentrations.
        S_true = np.load('%s.npy' % master[mkey]['H'])
        S_true = S_true[0:m,:]
        C_true = np.load('%s.npy' % master[mkey]['C'])
        
        # load the predicted.
        S_pred = np.load(results[dkey][method_name]['S'])
        C_pred = np.load(results[dkey][method_name]['C'])
               
        # remap if its not DECONF
        if method_name != "DECONF":

            # remap to known order.
            if r != -1:
                C_pred, S_pred = _remap_missing(C_pred, S_pred, r, k)
        else:
            
            # perform matching.
            C_pred, S_pred = _match_pred(C_pred, S_pred, C_true, S_true)

        # compute absolute difference.
        s = S_true[:, r] - S_pred[:, r]
        for i in range(m):
            genet[i].append(s[i])
    
    # create stuff.
    tmp = list()
    for i in range(args.m):
        tmp.append(genet[i])
    
    # create figure.
    pos = range(args.m)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _violin_plot(ax, tmp, pos)

    # finalize.
    #plt.title("stuff")
    ax.set_xticks(pos)
    ax.set_xticklabels(b_lbls, rotation=90)
    ax.set_ylabel('absolute difference')
    #ax.set_ylim([0,300])
    plt.savefig(args.fig_file)
    
## low-level functions ##

# helper for cleaning up axes by removing ticks, tick labels, frame, etc.
def _clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def _violin_plot(ax, data, pos, bp=False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos)-min(pos)
    w = min(0.15*max(dist,1.0),0.5)
    for d,p in zip(data,pos):
        try:
            k = gaussian_kde(d) #calculates the kernel density
            m = k.dataset.min() #lower bound of violin
            M = k.dataset.max() #upper bound of violin
            x = arange(m,M,(M-m)/100.) # support for violin
            v = k.evaluate(x) #violin profile (density curve)
            v = v/v.max()*w #scaling the violin to the available space
            ax.fill_betweenx(x,p,v+p,facecolor='y',alpha=0.3)
            ax.fill_betweenx(x,p,-v+p,facecolor='y',alpha=0.3)
        except:
            continue
    if bp:
        ax.boxplot(data,notch=1,positions=pos,vert=1)

