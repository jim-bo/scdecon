"""
matrix functions
"""
import sys
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt

def norm_matrix(m):
    ''' normalizes whole matrix '''
    return m / np.max(m)

def norm_vector(v):
    ''' normalizes vector '''
    return np.divide(v, np.sum(v))

def norm_cols(m):
    ''' normalizes columns of matrix '''
    for j in range(m.shape[1]):

        # compute column sum.
        colsum = np.sum(m[:,j])

        # make it add to 1.
        np.divide(m[:,j], colsum, m[:,j])
    return m

def norm_rows(m):
    ''' normalizes rows of matrix '''
    for i in range(m.shape[0]):

        # compute column sum.
        rowsum = np.sum(m[i,:])

        # make it add to 1.
        np.divide(m[i,:], rowsum, m[i,:])
    return m

    temp = data1 - data2

    temp = temp*temp

    return math.sqrt(temp.mean())

def pearson_vector(x, y):
    ''' pearson for vector '''
    return stats.pearsonr(x,y)[0]

def pearson_matrix(x, y):
    ''' perason for matrix '''
    # calculate column wise.
    vals = list()
    for j in range(x.shape[1]):
        vals.append(pearson_vector(x[:,j], y[:,j]))
    return np.average(np.array(vals))

def rmse_vector(x, y):
    """ root mean square error """

    return np.sqrt(np.mean(np.square(np.subtract(x, y))))

def nrmse_vector(x, y):
    """ normalized root mean square error """

    return rmse_vector(x,y) / (y.max() - y.min())

def relerr_vector(x, y):
    """ relative error """

    # compute error.
    z = np.absolute(x - y)

    for i in range(len(x)):
        if z[i] > 1.0:
            print x[i], y[i], z[i]
            sys.exit()

    s = z / y

    for i in range(len(x)):
        if s[i] > 1.0:
            print x[i], y[i], z[i], s[i]
            sys.exit()

    # divide by truth
    return np.average()

def mape_vector(a, f):
    """ relative error """

    # compute average true.
    av = np.average(a)

    # compute error.
    return np.sum(np.absolute(a - f) / av) / a.shape[0]

def rmse_cols(x, y):
    """ root mean square error """

    # LOOP over columns.
    rmss = list()
    for a in range(x.shape[1]):
        rms = sqrt(mean_squared_error(x[:,a], y[:,a]))
        rmss.append(rms)

    return np.average(np.array(rmss))

def avg_cat(targets, expr):
    ''' computes S based on average category '''

    # create ordered list of categories.
    cats = sorted(list(set(list(targets))))

    # calculate dimensions.
    m = expr.shape[1]
    k = len(cats)

    # create the array.
    S = np.zeros((m,k), dtype=np.float)

    # loop over each category.
    for j in range(len(cats)):

        # select just matches of this gene, category.
        idxs = np.where(targets==cats[j])[0]

        # take subset.
        sub = expr[idxs,:]

        # loop over each gene.
        for i in range(m):

            # save the average.
            S[i,j] = np.average(sub[:,i])

    # return the matrix.
    return S, cats

def save_experiment(out_file, Xs, Cs):
    """ saves the lists of numpy arrays using pickle"""
    with open(out_file, "wb") as fout:
        pickle.dump({"Xs":Xs,"Cs":Cs}, fout)

def load_experiment(in_file):
    """ saves the lists of numpy arrays using pickle"""
    with open(in_file) as fin:
        data = pickle.load(fin)
    return data['Xs'], data['Cs']





