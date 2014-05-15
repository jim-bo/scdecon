import cPickle as pickle
import numpy as np
import networkx as nx
import operator
import sys
import itertools
from scipy.stats import pearsonr

def run_it_uconn(wdir, script_txt, n, m, k, missing=None):

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

def write_r_mat(out_file, Z):
    """ writes R formated matrix"""

    # open file,
    with open(out_file, "wb") as fout:

        # write header.
        fout.write('\t'.join(['sample_%i' % i for i in range(Z.shape[1])]) + '\n')

        # write data.
        for i in range(Z.shape[0]):
            fout.write('gene_%i\t' % i + '\t'.join(['%f' % v for v in Z[i,:]]) + '\n')

def save_pickle(out_file, the_list):
    """ saves list of numpy """
    with open(out_file, "wb") as fout:
        pickle.dump(the_list, fout)

def load_pickle(in_file):
    """ saves the lists of numpy arrays using pickle"""
    with open(in_file) as fin:
        data = pickle.load(fin)
    return data

def match_signatures(S, T):
    ''' user perason match columns '''

    # sanity check columns.
    assert S.shape[1] == T.shape[1], 'columns must have same dimension'

    # labels.
    labels = range(S.shape[1])

    # build bipartite graph.
    G = nx.Graph()
    for l in labels:
        G.add_node("S_%i" % l)
        G.add_node("T_%i" % l)

    # build edges.
    for p in labels:
        for q in labels:

            # compute perasonr.
            weight = pearsonr(S[:,p], T[:,q])

            # add edge.
            a = "S_%i" % p
            b = "T_%i" % q
            G.add_edge(a, b, weight=weight[0])

    # find matching.
    matches = nx.max_weight_matching(G, maxcardinality=True)

    # record the sort order.
    order = list()
    for l in labels:
        order.append(int(matches["T_%i" % l].replace("S_","")))

    # return the final re-ordering matrix.
    return order


def match_labels(truth, test):
    ''' max-weight matching to assign labels from clustering'''

    # assert they are same length.
    assert truth.shape == test.shape, 'cant match unequal length predictions'

    # sanity check.
    a = set(np.unique(truth))
    b = set(np.unique(test))
    if a != b:
        logging.error("cant match on different label cardinatliyt")
        sys.exit(1)
    celltypes = list(a)

    # build bipartite graph.
    g = nx.Graph()
    for l in celltypes:
        g.add_node("true_%i" % l)
        g.add_node("test_%i" % l)

    # add edges based on count.
    for i in range(len(test)):
        test_n = "test_%i" % test[i]
        true_n = "true_%i" % truth[i]

        if g.has_edge(test_n, true_n) == True:
            g[test_n][true_n]['weight'] += 1
        else:
            g.add_edge(test_n, true_n, weight=1)

    # find matching.
    matches = nx.max_weight_matching(g)

    # re-assign test.
    swap = dict()
    for a,b in matches.items():
        if a.count("true") > 0:
            x = int(a.replace("true_",""))
            y = int(b.replace("test_",""))
            swap[y] = x
    for l in range(len(test)):
        test[l] = swap[test[l]]

    # get indirect sort.
    indir = [x[0] for x in sorted(swap.items(), key=operator.itemgetter(1))]

    # return updated labels.
    return test, indir


def _rand_mix(cell_count):
    ''' random mix of percentages '''

    # get the keys.
    keys = range(cell_count)

    # choose randomly and sum to 1.
    choices = np.arange(.1,1,.1)
    picks = np.random.choice(choices, 4, replace=True)
    picks = picks / np.sum(picks)
    np.random.shuffle(picks)

    # sanity
    assert np.around(np.sum(picks)) == 1, """bad probability setup"""

    # return it.
    return picks


def load_example(args):
    ''' loads example using rbridge '''
    assert False, 'not ported to new app'
    sys.exit(1)

    # load libraries.
    _load_R_libraries()

    # execute script.
    gse_name = 'GSE20300'
    query = """# load the ACR data (both exp and cbc)
data <- gedData("{gse_name}");
acr <- ExpressionMix('{gse_name}', verbose = 1);
res <- gedBlood(acr, verbose = TRUE);

# return data: X, C, labels
list(exprs(acr),coef(res),sampleNames(phenoData(acr)))
    """.format(gse_name=gse_name)
    pair = robjects.r(query)

    # save data points.
    X, C, labels = pair
    data = CellMixData(X, None, C)
    data.save(args.proj_key)


def _sample_S(targets, expr, p):
    """ perform sampling to generate S"""

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
        sub = expr[:,np.where(targets==cats[j])[0]]

        # loop over each gene.
        for i in range(m):

            # count zeros.
            genes = sub[i, :]

            # sample randomly with replacement.
            samples = np.random.choice(genes, size=p, replace=True)

            # average.
            S[i,j] = np.average(samples)

    # return S
    return S


def _measure_cluster(SC, Strue, sc_lbls, methods):
    ''' measure accuracy of clustering methods '''

    # build results dict.
    rmse_dict = dict()
    match_dict = dict()
    for name, meth in methods:
        rmse_dict[name] = list()
        match_dict[name] = list()

    # loop for 100 times:
    for t in range(100):

        # loop over each method.
        for name, meth in methods:

            # cluster.
            Spred, labels, cats = meth(SC, args.k)
            Spred = norm_cols(Spred)

            # match labels.
            try:
                labels, reorder = match_labels(sc_lbls, labels)
            except KeyError:
                continue

            # assert shape.
            assert Strue.shape == Spred.shape, 'S not same dim'

            # re-order columns in Spred.
            Spred = Spred[:,reorder]

            # calculate accuracy.
            rmse = rmse_cols(Strue, Spred)

            # score matching.
            match_score = float(np.sum(sc_lbls == labels)) / sc_lbls.shape[0]

            # sanity check error calculation.
            '''
            if match_score == 1.0 and rmse != 0.0:
                logging.error("why score bad")
                sys.exit(1)
            '''

            # save it.
            rmse_dict[name].append(rmse)
            match_dict[name].append(match_score)

    # return em.
    return rmse_dict, match_dict


def compare_cluster_avg(args):
    """ compare clustering across simulation parameters"""

    # load the labels.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)

    # compute S.
    Strue = _avg_S(sc_lbls, SC, 0)
    Strue = norm_cols(Strue)

    # define available methods.
    methods = list()
    methods.append(("random",randclust))
    methods.append(("kmeans",kmeans))
    methods.append(("spectral",spectral))
    methods.append(("ward",ward))

    # loop over each subset of single cells.
    for b in np.arange(1.0,0.0,-0.1):

        # pick subset.
        samples = np.random.choice(range(SC.shape[1]), size=int(SC.shape[1] * b))
        subSC = SC[:,samples]
        sublbl = sc_lbls[samples]

        # compute accuracy.
        rmse_dict, match_dict = _measure_cluster(subSC, Strue, sublbl, methods)

        # print results.
        for name in rmse_dict:
            print "avg", b, name, '%.8f' % np.average(np.array(rmse_dict[name])), '%.8f' % np.average(np.array(match_dict[name]))

        sys.exit()

def compare_cluster_sample(args):
    """ compare clustering across simulation parameters"""

    # load the labels.
    SC = np.load(args.SC)
    sc_lbls = np.load(args.sc_lbls)

    # loop over sample parameters.
    for p in range(1,50,5):

        # compute S.
        Strue = _sample_S(sc_lbls, SC, p)
        Strue = norm_cols(Strue)

        # define available methods.
        methods = list()
        methods.append(("random",randclust))
        methods.append(("kmeans",kmeans))
        methods.append(("spectral",spectral))
        methods.append(("ward",ward))

        # compute accuracy.
        rmse_dict, match_dict = _measure_cluster(SC, Strue, sc_lbls, methods)

        # print results.
        for name in rmse_dict:
            print "sample_%i" % p, name, '%.8f' % np.average(np.array(rmse_dict[name])), '%.8f' % np.average(np.array(match_dict[name]))

