import numpy as np
import rpy2.robjects as R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def load_R_libraries():
    #LIBRARIES = ("CellMix", "GEOquery")
    LIBRARIES = ("CellMix",)
    load_str = ";".join(map(lambda x: "library({0})".format(x), LIBRARIES))
    R.r(load_str)
    return

def r2npy(m):
    """Convert an R matrix to a 2D numpy array.

    Parameters
    ----------
    m: rpy2.robject
        an R matrix.

    Returns
    -------
     triple: (numpy.array, list, list)
        A triple consisting of a 2D numpy array, a list of row names, and a
         list of column names.
    """
    if m is None:
        raise ValueError("m must be valid R matrix!")

    matrix = np.array(m)
    rownames = list(m.rownames) if m.rownames else []
    colnames = list(m.colnames) if m.colnames else []

    return matrix, rownames, colnames
