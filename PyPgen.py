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
# [7] Pasupathy, Raghu. "Generating homogeneous poisson processes." Wiley encyclopedia of operations research and management science (2010).


def HPP_samples(samples, bounds, realizations=1):
    """ Generate fixed HPP samples in bounds.

    :param int samples
    :param list bounds: list-like of len = dimensions number
    :param int realizations:
    :return: list of numpyndarray of samples
    """
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    points_per_realization = np.tile(samples, realizations)
    points = []
    for i in np.arange(realizations):
        points.append(np.random.uniform(bounds_low, bounds_high, (points_per_realization[i], dimensions)))
    if realizations == 1:
        points = points[0]
    return(points)


def HPP_rate(rate, bounds, realizations=1):
    """ Random number of HPP samples with parameter rate in bounds

    :param float rate
    :param list bounds: list-like of len = dimensions number
    :param int realizations:
    :return: list of numpyndarray of samples
    """
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    n_volume = np.prod(bounds_high-bounds_low)
    points_per_realization = np.random.poisson(
        rate*n_volume, realizations).astype(int)
    points = []
    for i in np.arange(realizations):
        points.append(np.random.uniform(
        bounds_low, bounds_high, (points_per_realization[i], dimensions)))
    if realizations == 1:
        points = points[0]
    return(points)


def HPP_temporal(rate, bounds, realizations=1, blocksize=1000):
    """ Generate HPP samples for a temporal HPP using exponential distribution
    of inter-arrival times until bounds[1] is exceeded

    :param int rate parameter of the poisson process
    :param list bounds: list-like of 2 elements
    :param int realizations:
    :param int blocksize: exponential to generate before checking bounds
    :return: list of numpyndarray of samples
    """
    if (len(bounds) != 2):
        raise TypeError("Input bounds must have exactly two elements.")
    points = []
    for i in np.arange(realizations):
        points.append(np.array([np.amin(bounds)]))
        while(points[i][-1] < np.amax(bounds)):
            variates = np.random.exponential(1.0/float(rate), blocksize)
            points[i] = np.append(points[i], np.amin(bounds) + np.cumsum(variates))
        points[i] = np.sort(points[i][(points[i] < np.amax(bounds))
                                        * (points[i] > np.amin(bounds))])
    if realizations == 1:
        points = points[0]
    return(points)


def NHPP(rate, rate_max, bounds, realizations=1, algo="thinning"):
    """Random points in n-dimensions with a multidimensional rate function/rate matrix.
    Uses the thinning/acceptance-rejection algorithm.

    :param function rate intensity function of the NHPP 
    :param list bounds: list-like of len = dimensions number
    :param int realizations:
    :return: numpyndarray of samples
    """
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])

    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    n_volume = np.prod(bounds_high-bounds_low)
    points_num = np.random.poisson(rate_max*n_volume)
    if not callable(rate):
        raise TypeError("Rate should be a function")
    if algo = "thinning":
        points_unthinned = np.random.uniform(
            bounds_low, bounds_high, (points_num, dimensions))

        rate_ratio = np.ravel(rate(*np.hsplit(points_unthinned, dimensions))/rate_max)
        accept_prob = np.random.uniform(size=points_num) 
        points_thinned = points_unthinned[rate_ratio > accept_prob, :] 
    return(points_thinned)