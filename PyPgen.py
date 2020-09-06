# Code créé par Gabriel Taillon le 6 Septembre 2020
import os
import sys
import numpy as np

# Glossary
# - *A*: Data space
# - *&lambda*: Process intensity (function for non-homogeneous)
# - PP: Point Process
# - HPP: Homogeneous Poisson Process
# - NHPP: Non-Homogeneous Poisson Process
# - MPP: Mixed Poisson Process
# - CP: Cox Process/Doubly Stochastic Point Process
# - MaPP: Markov Point Process / Finite Gibbs Point Process
#     - As far as I know, those are the same. Refer to [1-3].

# References
# [1] Diggle, Peter J. Statistical analysis of spatial and spatio-temporal point patterns. CRC press, 2013.
# [2] Illian, Janine, et al. Statistical analysis and modelling of spatial point patterns. Vol. 70. John Wiley & Sons, 2008.
#     # n: current total of points
# [3] Jensen, Eva B. Vedel, and Linda Stougaard Nielsen. "Inhomogeneous Markov point processes by transformation." Bernoulli 6.5 (2000): 761-782.
# [4] Snyder, Donald L., and Michael I. Miller. Random point processes in time and space. Springer Science & Business Media, 2012.
# [5] Resnick, Sidney I. Adventures in stochastic processes. Springer Science & Business Media, 1992.
# [6] Grandell, Jan. Mixed poisson processes. Vol. 77. CRC Press, 1997.


def HPP_rate(rate, bounds, realizations=1):
    """ Random number of HPP samples with parameter rate in bounds

    :param int samples
    :param list bounds: list-like of len = dimensions number
    :param int realizations: 
    :return: numpyndarray of samples_realization1*samples_realization2*...*dimensions
    """
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    n_volume = np.prod(bounds_high-bounds_low)
    points_per_realization = np.random.poisson(
        int(rate*n_volume), realizations).astype(int)
    points = np.random.uniform(
        bounds_low, bounds_high, (*points_per_realization, dimensions))
    return(points)


def HPP_temporal(rate, bounds, blocksize=1000):
    """ Generate HPP samples for a temporal HPP using exponential distribution
    of inter-arrival times until bounds[1] is exceeded

    :param int rate parameter of the poisson process
    :param list bounds: list-like of 2 elements
    :param int realizations: 
    :param int blocksize: exponential to generate before checking bounds 
    :return: numpyndarray of samples*dimensions*realizations
    """ 
    if (len(bounds) != 2):
        raise TypeError("Input bounds must have exactly two elements.")
    points = np.array([np.amin(bounds)])
    while(points[-1] < np.amax(bounds)):
        variates = np.random.exponential(1.0/float(rate), blocksize)
        points = np.append(points, np.amin(bounds) + np.cumsum(variates))
    points_inbound = np.sort(points[(points < np.amax(bounds))
                            * (points > np.amin(bounds))])
    return(points_inbound)


def HPP_samples(samples, bounds, realizations=1):
    """ Generate fixed HPP samples in bounds.

    :param int samples
    :param list bounds: list-like of len = dimensions number
    :param int realizations: 
    :return: numpyndarray of samples*dimensions*realizations
    """
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    points_per_realization = np.tile(samples, realizations)
    points = np.random.uniform(
        bounds_low, bounds_high, (*points_per_realization, dimensions))
    return(points)


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
