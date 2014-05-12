"""
matrix functions
"""
import sys
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.tools.eval_measures

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
    
    # return 1 if dey equal.
    if np.sum(x == y) == len(x): return 1.0    
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

    return statsmodels.tools.eval_measures.rmse(x, y)

def rmse_matrix(x, y):
    """ root mean square error """
    vals = list()
    for j in range(x.shape[1]):
        vals.append(rmse_vector(x[:,j], y[:,j]))
    return np.average(np.array(vals))

def nrmse_vector(x, y):
    """ normalized root mean square error """

    return rmse_vector(x,y) / (y.max() - y.min())
    
def nrmse_matrix(x, y):
    """ normalized root mean square error """
    vals = list()
    for j in range(x.shape[1]):
        vals.append(nrmse_vector(x[:,j], y[:,j]))
    return np.average(np.array(vals))

def meanabs_vector(x, y):
    return statsmodels.tools.eval_measures.meanabs(x, y)

def sumabs_vector(x, y):
    return np.sum(np.abs(x-y))

def meanabs_matrix(x, y):
    vals = list()
    for j in range(x.shape[1]):
        vals.append(meanabs_vector(x[:,j], y[:,j]))
    return np.average(np.array(vals))

def meanrel_vector(x, y):
    return np.mean(np.abs(x-y)/x)
    
def maxrel_vector(x, y):
    return np.max(np.abs(x-y)/x)

def minrel_vector(x, y):
    return np.min(np.abs(x-y)/x)
    
def maxabs_vector(x, y):
    return statsmodels.tools.eval_measures.maxabs(x, y)

def maxabs_matrix(x, y):
    vals = list()
    for j in range(x.shape[1]):
        vals.append(maxabs_vector(x[:,j], y[:,j]))
    return np.average(np.array(vals))    


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





