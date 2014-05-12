
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
    tmp = ['ACTB', 'GAPDH']
    hkeepers = list()
    for h in tmp:
        hkeepers += list(np.where(b_lbls == h)[0])
    hkeepers = sorted(hkeepers)

    # normalize S/
    Snorm = S.copy()
    for a in range(k):
        Snorm[:,a] = S[:,a] / gmean(S[hkeepers,a])

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

'''
    # debug cheat.
    Sfull = np.array([
        [0.1, 0.2, 0.7],
        [0.7, 0.5, 0.2],
        [0.3, 0.4, 0.1],
        [0.9, 0.1, 0.1],
        [0.4, 0.6, 0.7],
        [0.1, 0.2, 0.8]])

    Ctrue = np.array([
        [.3, .2, .4, .8, .6, .1],
        [.5, .4, .3, .1, .1, .2],
        [.2, .4, .3, .1, .3, .7]])

    for j in range(Ctrue.shape[1]):
        assert np.sum(Ctrue[:,j]) == 1.0, 'what happened here: %i' % j

    X = np.dot(Sfull, Ctrue)

    Sbase = Sfull[:,[0,1]]
    features = range(Sfull.shape[0])

    cheat_s = Sfull[:,2]
    cheat_C = Ctrue

    Ccheat = _solve_C(X, Sfull, num_threads=1)
    X2 = np.dot(Sfull, Ccheat)

    vals = list()
    for j in range(X.shape[1]):
        v = rmse_vector(X[:,j], X2[:,j])
        vals.append(v)
    so = np.average(np.array(vals))

    print so, 0, rmse_matrix(Ctrue, Ccheat), ' '.join([str(x) for x in Ccheat[-1,:]])


    # debug.
    Sfull = np.array([
        [0.1, 0.2, 0.7],
        [0.7, 0.5, 0.2],
        [0.3, 0.4, 0.1],
        [0.9, 0.1, 0.1],
        [0.4, 0.6, 0.7],
        [0.1, 0.2, 0.8]])

    Ctrue = np.array([[0.3], [0.5], [0.2]])
    X = np.dot(Sfull, Ctrue)

    # trim it.
    Sbase = Sfull[:,[0,1]]
    features = range(Sfull.shape[0])

  return C, S

def _solve_missing(X, Sbase, features, scorefn, cheat_s=None, cheat_C=None):
    """ solves using QP"""

    # debug cheat.
    Sfull = np.array([
        [0.1, 0.2, 0.7],
        [0.7, 0.5, 0.2],
        [0.3, 0.4, 0.1],
        [0.9, 0.1, 0.1],
        [0.4, 0.6, 0.7],
        [0.1, 0.2, 0.8]])

    Ctrue = np.array([[0.3], [0.5], [0.2]])
    X = np.dot(Sfull, Ctrue)

    Sbase = Sfull[:,[0,1]]
    features = range(Sfull.shape[0])

    cheat_s = Sfull[:,2]
    cheat_C = Ctrue

    scorefn = meanrel_vector
    scorefn = meanabs_vector

    # create C.
    n = X.shape[1]
    m = X.shape[0]
    k = Sbase.shape[1] + 1
    C = np.zeros((k,n), dtype=np.float)

    # shrink it.
    Sbase = Sbase[features,:]
    X = X[features,:]

    # create array for new S.
    Snew = np.zeros((m, n), dtype=np.float)

    # solve each concentration independently.
    bysample = list()
    for j in range(n):

        if j > 3: break

        # freeze X.
        Xqp = X[:,[j]]
        Xqp.setflags(write=False)

        # try various concentrations for missing.
        byscore = list()
        for cm in np.arange(0.0, 1.0, 0.1):

            # compute s, c
            #s, c, o = _sqp(Xqp, Sbase, cm, scorefn)
            s, c, o = _missing(Xqp, Sbase, cm, scorefn)

            # debug mode.
            if cheat_s != None or cheat_C != None:

                # simplify cheat.
                cheat_c = cheat_C[:,j]

                # compute score.
                byscore.append((o, scorefn(cheat_c, c), scorefn(cheat_s, s)))
                print '%.2f' % cm, '%.3f' % o, ' '.join(['%.3f' % x for x in c]), ' '.join(['%.3f' % x for x in s])



        # sort it.
        byscore = sorted(byscore, key=operator.itemgetter(0))

        # save it.
        bysample.append(byscore)



    # print it to the screen.
    for i in range(len(np.arange(0.0, 1.0, 0.1))):

        row = list()
        for s in bysample:
            row.append('%.5f' % s[i][1])

        print ' '.join([str(i)] + row)




    print "DEBUG DONE"
    sys.exit()

    # return it.
    return C, S
'''


    '''
    print CV.value
    print SV.value
    sys.exit()
    # case to vectors.
    s = np.array([SV.value[i,0] for i in range(m)])
    c = np.array([CV.value[l,0] for l in range(k-1)] + [cm])

    # compute x2
    S = np.zeros((m, k))
    for i, l in zip(range(m), range(k-1)):
        S[i, l] = Sbase[i, l]
    for i in range(m):
        S[i,-1] = SV.value[i]

    C = np.zeros((k,1))
    for l in range(k-1):
        C[l,0] = CV.value[l,0]
    C[-1,0] = cm

    X2 = np.dot(S, C)

    x = np.array([X[i,0] for i in range(m)])
    x2 = np.array([X2[i,0] for i in range(m)])

    # compute score.
    if scorefn == None:
        o = p.objective.value
    else:
        o = scorefn(x, x2)

    # return results.
    return s, c, o
    '''
