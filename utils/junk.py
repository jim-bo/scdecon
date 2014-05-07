
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
