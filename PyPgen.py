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
# - HMaPP/NHMaPP: Homogeneous/Non-Homogeneous Markov Point Process
#     - As far as I know, those are the same as Finite Gibbs Point Process. Refer to [1-3].

# References
# [1] Diggle, Peter J. Statistical analysis of spatial and spatio-temporal point patterns. CRC press, 2013.
# [2] Illian, Janine, et al. Statistical analysis and modelling of spatial point patterns. Vol. 70. John Wiley & Sons, 2008.
#     # n: current total of points
# [3] Jensen, Eva B. Vedel, and Linda Stougaard Nielsen. "Inhomogeneous Markov point processes by transformation." Bernoulli 6.5 (2000): 761-782.
# [4] Snyder, Donald L., and Michael I. Miller. Random point processes in time and space. Springer Science & Business Media, 2012.
# [5] Resnick, Sidney I. Adventures in stochastic processes. Springer Science & Business Media, 1992.
# [6] Grandell, Jan. Mixed poisson processes. Vol. 77. CRC Press, 1997.
# [7] Pasupathy, Raghu. "Generating homogeneous poisson processes." Wiley encyclopedia of operations research and management science (2010).
# [8] Geyer, Charles J., and Jesper Møller. "Simulation procedures and likelihood inference for spatial point processes." Scandinavian journal of statistics (1994): 359-373.


def HPP_samples(samples, bounds, realizations=1):
    """ Generate fixed HPP samples in bounds.

    :param int samples
    :param list bounds: list-like of len = dimensions number
    :param int realizations:
    :return: list of numpyndarray of samples
    """
    if np.any(np.ravel(samples) < 0):
        raise ValueError(
            "Input info distribution should not produce negative values (rate > 0)")
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    if np.isscalar(samples):
        points_per_realization = np.tile(samples, realizations)
    else:
        points_per_realization = np.ravel(samples)
        if(len(points_per_realization) != realizations):
            raise ValueError(
                "Input samples list-like should have the same length as realizations")
    points = []
    for i in np.arange(realizations):
        points.append(np.random.uniform(bounds_low, bounds_high,
                                        (points_per_realization[i], dimensions)))
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
            points[i] = np.append(points[i], np.amin(
                bounds) + np.cumsum(variates))
        points[i] = np.sort(points[i][(points[i] < np.amax(bounds))
                                      * (points[i] > np.amin(bounds))])
    if realizations == 1:
        points = points[0]
    return(points)


def NHPP(rate, rate_max, bounds, realizations=1):
    """Random points in n-dimensions with a multidimensional rate function
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
    n_volume = np.prod(bounds_high - bounds_low)
    points = []
    if not callable(rate):
        raise TypeError("Rate should be a function")
    for i in np.arange(realizations):
        points_num = np.random.poisson(rate_max * n_volume)
        points_unthinned = np.random.uniform(
            bounds_low, bounds_high, (points_num, dimensions))
        rate_ratio = np.ravel(
            rate(*np.hsplit(points_unthinned, dimensions))/rate_max)
        accept_prob = np.random.uniform(size=points_num)
        points.append(points_unthinned[rate_ratio > accept_prob, :])
    if realizations == 1:
        points = points[0]
    return(points)


def MPP(info, bounds, realizations=1):
    """Random points in n-dimensions with random rate according to 
    information distribution

    :param function info information distribution. input of info should be size of samples
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
    random_rates = info(realizations)
    if np.any(random_rates < 0):
        raise ValueError(
            "Info distribution should not produce negative values (rate > 0)")
    points_per_realization = np.random.poisson(
        np.multiply(random_rates, n_volume), realizations).astype(int)
    points = []
    for i in np.arange(realizations):
        points.append(np.random.uniform(
            bounds_low, bounds_high, (points_per_realization[i], dimensions)))
    if realizations == 1:
        points = points[0]
    return(points)


def MaPP(rate, bounds, pairwisef, mix_prob=1, iterations=40000, burn_in=40000):
    """ Markov point process using the Metropolis-Hastings algorithm as in [8]

    :param function or scalar rate "intensity" term of the MaPP
    :param int realizations:
    :param list bounds: list-like of len = dimensions number
    :param function pairwisef: pairwise interation function between points
    :param scalar mix_prob: mixing probability p
    :param int iterations:  
    :param int burn_in: base number of necessary iterations
    :return: numpyndarray of samples
    """
    if not is_instance(iterations, int):
        raise TypeError("Input iterations should be an int")

    if (not np.iscalar(mix_prob)) | (mix_prob < 0) | (mix_prob > 1):
        raise TypeError(
            "Input mix_prob should be a number between 1 and 0 inclusive")

    if (not iscallable(rate)) & (not np.isscalar(rate)):
        raise TypeError(
            "Input rate should be a scalar (homogeneous) or function (non-homgeneous)")
    if np.isscalar(rate):
        rate_term = rate
    npbounds = np.array(bounds)
    npbounds = np.reshape(npbounds, (int(len(np.ravel(npbounds))/2), 2))
    dimensions = int(npbounds.shape[0])
    bounds_low = np.amin(npbounds, axis=1)
    bounds_high = np.amax(npbounds, axis=1)
    n_volume = np.prod(bounds_high - bounds_low)

    replace_prob = 1 - mix_prob
    delete_prob = mix_prob/2
    add_prob = mix_prob/2
    probs = [add_prob, delete_prob, replace_prob]
    n = 0
    for i in np.arange(iterations + burn_in):
        if (n < 2):
            tentative_points = np.random.uniform(
                bounds_low, bounds_high, (2, dimensions))
            accept_prob = 1
        else:
            chosen = np.random.choice(np.arange(len(probs)), None, p=probs)
            if (chosen == 0):
                new_point = np.random.uniform(
                    bounds[0], bounds[1], (1, dimensions))
                tentative_points = np.copy(np.append(points, new_point))
                density_point = np.prod(
                    pairwisef(np.tile(new_point, n), points))
                if iscallable(rate):
                    rate_term = rate(new_point)
            elif (chosen == 1):
                todelete = np.random.randint(0, n)
                tentative_points = np.copy(np.delete(points, (todelete % n)))
                density_point = np.prod(
                    pairwisef(np.tile(points[todelete], len(tentative_points)), tentative_points))
                if iscallable(rate):
                    rate_term = rate(points[todelete])
            elif (chosen == 2):
                new_point = np.random.uniform(
                    bounds[0], bounds[1], (1, dimensions))
                tentative_points = np.copy(points)
                toreplace = np.random.randint(0, len(points))
                tentative_points[toreplace] = new_point
                density_point = np.cumprod(
                    pairwisef(np.repeat(new_point, n), points, *pairwisef_params))[-1]
                if iscallable(rate):
                    rate_term = rate(new_point)

            m = len(tentative_points)
            if (density_point <= 0):
                accept_prob = 0
            else:
                accept_prob = min(1, float(density_point)**(m-n) *
                                  float(n_volume)**(m-n) * float(rate_term)**(m-n) * float(n)**(n > m) / float(m)**(m > n))
        if (accept_prob >= np.random.uniform()):
            points = np.copy(tentative_points)
            n = len(points)

    return(points)
