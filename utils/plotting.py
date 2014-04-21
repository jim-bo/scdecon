""" plotting """
## imports ##
import numpy as np
#import brewer2mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch

# violin plot
from scipy.stats import gaussian_kde
from numpy.random import normal
from numpy import arange


## high-level functions ##
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
