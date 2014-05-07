# system.
import warnings
import random

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph

import numpy as np

import scipy.spatial

# app
from utils.matops import *
#from utils.rfuncs import *
from utils.misc import *

## method ##
def randclust(SC, k):
    """ cluster using random """

    # generate labels.
    labels = np.array([random.randint(0,k-1) for x in range(SC.shape[1])])

    # compute the average.
    S, cats = avg_cat(labels, SC)

    # return it.
    return S, labels, cats


def kmeans(SC, k):
    """ clusters using k-means"""

    # compute cluster.
    algo = cluster.MiniBatchKMeans(n_clusters=k)

    # fit the data.
    algo.fit(np.transpose(SC))

    # extract assignment.
    labels = algo.labels_.astype(np.int)

    # compute the average.
    S, cats = avg_cat(labels, SC)

    # return it.
    return S, labels, cats

def spectral(SC, k):
    """ clustering using spectral """

    # create the object.
    algo = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors", assign_labels='kmeans')

    # fit the data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo.fit(np.transpose(SC))

    # extract assignment.
    labels = algo.labels_.astype(np.int)

    # compute the average.
    S, cats = avg_cat(labels, SC)

    # return it.
    return S, labels, cats

def ward(SC, k):
    """ clustering using ward-tree"""

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(np.transpose(SC), n_neighbors=k)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create the object.
    algo = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors", assign_labels='kmeans')

    # fit the data.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo.fit(np.transpose(SC))

    # extract assignment.
    labels = algo.labels_.astype(np.int)

    # compute the average.
    S, cats = avg_cat(labels, SC)

    # return it.
    return S, labels, cats
