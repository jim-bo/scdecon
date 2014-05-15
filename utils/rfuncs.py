import numpy as np
import rpy2.robjects as R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def load_R_libraries():
    LIBRARIES = ("CellMix", "GEOquery")
    #LIBRARIES = ("CellMix",)
    load_str = ";".join(map(lambda x: "suppressMessages(library({0}))".format(x), LIBRARIES))
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


def R_DECONF(X_path, Z_path, y_path, k, S_path, C_path, wdir):
    """ run DECONF using rpy2 """

    # extend paths.
    X_path = '%s.txt' % X_path
    Z_path = '%s.txt' % Z_path
    Stmp = '%s/S.txt' % wdir
    Ctmp = '%s/C.txt' % wdir

    print wdir

    # run deconvolution.
    txt = '''# load libraries.
#suppressMessages(library(CellMix));
#suppressMessages(library(GEOquery));

# load data.
exprsFile <- file.path("{X_path}");
exprs <- as.matrix(read.table(exprsFile, header=TRUE, sep="\\t", row.names=1, as.is=TRUE));
eset <- ExpressionSet(assayData=exprs);

# run deconvolution.
res <- ged(eset, {num}, method='deconf');

# write matrix.
write.table(coef(res), file="{Ctmp}", row.names=FALSE, col.names=FALSE)
write.table(basis(res), file="{Stmp}", row.names=FALSE, col.names=FALSE)
'''.format(X_path=X_path, Stmp=Stmp, Ctmp=Ctmp, num=k)

    # execute it.
    R.r(txt)


