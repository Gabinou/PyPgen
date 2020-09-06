# Code créé par Gabriel Taillon le 6 Septembre 2020

# [1] Diggle, Peter J. Statistical analysis of spatial and spatio-temporal point patterns. CRC press, 2013.

def HPPMultidimSamples(Samples, Rate, ArrDomainSizes, silent=0):
    if not silent:
        print('Homogeneous Poisson Process in '+str(len(ArrDomainSizes)) +
              ' dimensions, using HPPSamples. Orderly? No simultaneous events.')
    Intervals, Time, Cumul = HPPSamples(Samples, Rate)
    AllUProbs = np.random.rand(len(Intervals), len(ArrDomainSizes))
    AllEventPos = AllUProbs*ArrDomainSizes
    return(Intervals, Time, Cumul, AllEventPos)


def HPPmaxT(maxT, Rate, silent=0):
    # MPPM Computes 1D homogeneous poisson process samples until maxT
    if not silent:
        print('Taking samples until T=' + str(maxT) +
              ' time of a Homogeneous Poisson Process of rate lambda='+str(('%.4e' % Rate))+' Hz.')
    Intervals = [0]
    Time = [0]
    i = 0
    while Time[-1] < maxT:
        Intervals = np.append(Intervals, random.expovariate(Rate))
        Time = np.append(Time, Time[-1] + Intervals[-1])
        i += 1
    Cumul = np.arange(0, len(Time), 1)

    return(Intervals[1:], Time[1:], Cumul)


def HPPSamples(Samples, Rate, Realizations=1, silent=0):
    #  Computes 1D homogeneous poisson process samples.
    if not silent:
        print('Taking '+str(Samples) +
              ' samples of a Homogeneous Poisson Process of rate lambda='+str(('%.4e' % Rate))+' Hz.')
    AllIntervals = []
    AllTime = []
    AllCumul = []
    j = 0
    while j < Realizations:
        i = 0
        Intervals = []
        while i < Samples:
            Intervals = np.append(Intervals,  random.expovariate(Rate))
            i += 1
        Time = np.cumsum(Intervals)
        Cumul = np.arange(0, Samples, 1)
        if AllIntervals == []:
            AllIntervals = np.reshape(Intervals, (len(Intervals), 1))
            AllTime = np.reshape(Time, (len(Time), 1))
            AllCumul = np.reshape(Cumul, (len(Cumul), 1))
        else:
            AllIntervals = np.hstack(
                (AllIntervals, np.reshape(Intervals, (len(Intervals), 1))))
            AllTime = np.hstack((AllTime, np.reshape(Time, (len(Time), 1))))
            AllCumul = np.hstack(
                (AllCumul, np.reshape(Cumul, (len(Cumul), 1))))
        j += 1

    return(AllIntervals, AllTime, AllCumul)


def MultiVarNHPPInvAlgoSamples(lambdafunc, Boundaries, silent=0):
    """ Computes sample random points in a multidimensional space, using a multidimensional rate function. Uses the inversion algorithm. 

    **Arguments:**  
        lambdafunc : function
            Rate function. For now, the multidimensional rate cannot be constant.
        Boundaries : (2,n) matrix
            Contains the spatial boundaries in which to generate the points. Data space boundaries.
        silent: bool
            Prints debug if equal to one. TO BE REMOVED. REPLACED WITH CORRECT LOGGING.
    **Returns:**
        X2D : (samples,n) matrix
            Generated samples.

        Source : Saltzman, E. A., Drew, J. H., Leemis, L. M., & Henderson, S. G. (2012). Simulating multivariate nonhomogeneous Poisson processes using projections. ACM Transactions on Modeling and Computer Simulation (TOMACS), 22(3), 15.
    """
    if not silent:
        print('In the NHPPInversion Algo')
    ArgNum = len(inspect.getfullargspec(lambdafunc)[0])
    if (ArgNum) != Boundaries.shape[0]:
        raise NameError('lambdafunc arguments not equal Interval numbers.')
    Lambda1func, Lambda1sym = Lambda(lambdafunc, Boundaries)
    FnSymDict = InvAlgoFn(lambdafunc, Boundaries[1:])
    FnSymDict['Lambda1'] = Lambda1sym
    InvFnFuncDict = InvAlgoInvert(FnSymDict)
    # print(Boundaries.shape)
    X2D = InvSamples(Boundaries[0], Lambda1func, InvFnFuncDict)
    return(X2D)



def MultiVarNHPPThinSamples(lambdaa, Boundaries, Samples=100, blocksize=1000, silent=0):
    """ Computes sample random points in a multidimensional space, using a multidimensional rate function/rate matrix. Use the thinning/acceptance-rejection algorithm.

    **Arguments:**  
        lambdaa : function or n-d matrix
            Rate function. For now, the multidimensional rate cannot be constant.
        Boundaries : (2,n) matrix
            Contains the spatial boundaries in which to generate the points. Data space boundaries.
        Samples: integer
            Number of samples to generate
        blocksize: integer
            Number of points to generate before thinning, rejection. Instead of generating points one by one, a blocksize length array is generated. I think this makes the algo faster.
        silent: bool
            Prints debug if equal to one. TO BE REMOVED. REPLACED WITH CORRECT LOGGING.
    **Returns:**
        Thinned : (samples,n) matrix
            Generated samples.
    """
    if not silent:
        print('NHPP samples in space by thinning. lambda can be a 2D matrix or function')
    # This algorithm acts as if events do not happen outside the Boundaries.
    boundstuple = []

    def index2pos(inds, Boundaries, dimm):
        return(np.flip(inds/dimm*np.sum(np.abs(Boundaries), axis=1)+Boundaries[:, 0], axis=1))
    dim = 500
    side = np.linspace(-1, 2, dim)
    sides = ()
    dims = ()
    for bound in Boundaries:
        sides += (side, )
        dims += (dim, )
    pos_tuples = np.meshgrid(*sides)
    for pos_tuple in pos_tuples:
        if 'pos_list' not in locals():
            pos_list = np.ravel(pos_tuple)
        else:
            pos_list = np.vstack((pos_list, np.ravel(pos_tuple)))
    mat_found_max = np.where(lambdaa(pos_list).reshape(
        *dims) == np.amax(lambdaa(pos_list).reshape(*dims)))
    first_x0 = index2pos(np.array(
        [mat_found_max[0][0], mat_found_max[1][0]]).reshape(1, -1), Boundaries, dims[0])
    for i in Boundaries:
        boundstuple += (tuple(i),)
    if callable(lambdaa):
        max = scipy.optimize.minimize(
            lambda x: -lambdaa(x), x0=first_x0, bounds=boundstuple, method='TNC')
        lmax = lambdaa(max.x)
    else:
        lmax = np.amax(lambdaa)
    Thinned = []
    while len(Thinned) < Samples:
        if callable(lambdaa):
            for i in Boundaries:
                if 'Unthin' not in locals():
                    Unthin = np.random.uniform(*i, size=(blocksize))
                else:
                    Unthin = np.vstack(
                        (Unthin, np.random.uniform(*i, size=(blocksize))))
        else:
            for shape in lambdaa.shape:
                if 'Unthin' not in locals():
                    Unthin = np.random.randint(0, shape, size=(blocksize))
                else:
                    Unthin = np.vstack(
                        (Unthin, np.random.randint(0, shape, size=(blocksize))))
        if len(Unthin.shape) == 1:
            Unthin = np.reshape(Unthin, (1, len(Unthin)))
        U = np.random.uniform(size=(blocksize))
        if callable(lambdaa):
            Criteria = lambdaa(Unthin)/lmax
        else:
            Criteria2D = lambdaa/lmax
            Criteria = lambdaa[tuple(Unthin)]
        if Thinned == []:
            Thinned = Unthin.T[U < Criteria, :]
        else:
            Thinned = np.vstack((Thinned, Unthin.T[U < Criteria, :]))
        del Unthin
    if callable(lambdaa):
        Thinned = Thinned[:Samples, :]
    else:
        Thinned = index2pos(Thinned[:Samples, :], Boundaries, lambdaa)
    return(Thinned)