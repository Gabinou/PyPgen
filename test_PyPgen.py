# Code créé par Gabriel Taillon le 6 Septembre 2020
import PyPgen
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# NHPP TESTING
phase = 0
period = 2
amplitude = 3
offset = 1
rate_max = amplitude + offset
dimensions = 1


def rate_halfsinus1D(time):
    sins = np.sin(2.*np.pi*time/period + phase)
    out = amplitude*(sins*(sins > 0)) + offset
    return(out)


bounds = [1, 10]
out = PyPgen.NHPP(rate_halfsinus1D, rate_max, bounds)
assert(out.shape[-1] == dimensions)
assert(len(out.shape) > 1)
assert(np.amin(out) > bounds[0])
assert(np.amax(out) < bounds[1])

phase_1 = 0
phase_2 = 0
period_1 = 2
period_2 = 1
amplitude = 1
offset = 1
rate_max = amplitude + amplitude + offset
dimensions = 2


def rate_halfsinus2D(x_1, x_2):
    sins = np.sin(2.*np.pi*x_1/period_1 + phase) + \
        np.sin(2.*np.pi*x_2/period_2 + phase)
    out = amplitude*(sins*(sins > 0)) + offset
    return(out)


bounds = [[1, 10], [10, 20]]
out = PyPgen.NHPP(rate_halfsinus2D, rate_max, bounds)
assert(out.shape[-1] == dimensions)
assert(len(out.shape) > 1)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(np.amin(out[:, 1]) > bounds[1][0])
assert(np.amax(out[:, 1]) < bounds[1][1])


phase_1 = 0
phase_2 = 0
phase_3 = 0
period_1 = 1
period_2 = 2
period_3 = 3
amplitude = 1
offset = 1
rate_max = amplitude + amplitude + amplitude + offset
dimensions = 3


def rate_halfsinus3D(x_1, x_2, x_3):
    sins = np.sin(2.*np.pi*x_1/period_1 + phase) + \
        np.sin(2.*np.pi*x_2/period_2 + phase) + \
        np.sin(2.*np.pi*x_3/period_3 + phase)
    out = amplitude*(sins*(sins > 0)) + offset
    return(out)


bounds = [[1, 10], [10, 20], [20, 21]]
out = PyPgen.NHPP(rate_halfsinus3D, rate_max, bounds)
assert(out.shape[-1] == dimensions)
assert(len(out.shape) > 1)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(np.amin(out[:, 1]) > bounds[1][0])
assert(np.amax(out[:, 1]) < bounds[1][1])
assert(np.amin(out[:, 2]) > bounds[2][0])
assert(np.amax(out[:, 2]) < bounds[2][1])

# HPP TESTING

scripthpath = os.path.dirname(os.path.abspath(__file__))
path2res = os.path.join(scripthpath, os.path.splitext(
    os.path.basename(__file__))[0])
if not os.path.exists(path2res):
    os.makedirs(path2res)

rate = 5
bounds = [1, 10]
out = PyPgen.HPP_temporal(rate=rate, bounds=bounds)
assert(np.amin(out) > bounds[0])
assert(np.amax(out) < bounds[1])
plt.title(str(len(out)) + " samples of a temporal HPP of rate " +
          str(rate) + " in 10s")
plt.plot(out, np.arange(len(out)), ".k")
plt.xlabel("Time [s]")
plt.ylabel("Count")
plt.savefig(os.path.join(path2res, "HPP_temporal_test.png"),
            bbox_inches='tight', dpi=100, edgecolor='w')
# plt.show()
plt.close("all")

samples = 20
realizations = 3
bounds = [[0, 10], [0, 10]]
dimensions = len(bounds)
out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                         realizations=realizations)
assert(len(out) == realizations)
for i in np.arange(realizations):
    assert(np.amin(out[i][:, 0]) > bounds[0][0])
    assert(np.amax(out[i][:, 0]) < bounds[0][1])
    assert(np.amin(out[i][:, 1]) > bounds[1][0])
    assert(np.amax(out[i][:, 1]) < bounds[1][1])
    assert(len(out[i]) == samples)

samples = 20
realizations = 1
bounds = [[0, 10], [0, 10]]
dimensions = len(bounds)
out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                         realizations=realizations)
assert(len(out) == realizations)
for i in np.arange(realizations):
    assert(np.amin(out[i][:, 0]) > bounds[0][0])
    assert(np.amax(out[i][:, 0]) < bounds[0][1])
    assert(np.amin(out[i][:, 1]) > bounds[1][0])
    assert(np.amax(out[i][:, 1]) < bounds[1][1])
    assert(len(out[i]) == samples)


plt.title(str(samples) + " samples of a HPP in 10 unit square")
plt.scatter(out[:, 0], out[:, 1], c="k")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(os.path.join(path2res, "HPP_samples_2Dtest.png"),
            bbox_inches='tight', dpi=100, edgecolor='w')
# plt.show()
plt.close("all")


rate = 2
realizations = 1
bounds = [[0, 10], [0, 10]]
dimensions = len(bounds)
out = PyPgen.HPP_rate(rate=rate, bounds=bounds,
                      realizations=realizations)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(np.amin(out[:, 1]) > bounds[1][0])
assert(np.amax(out[:, 1]) < bounds[1][1])
assert(out.shape[-1] == dimensions)
assert((len(out.shape) - 1) == realizations)


plt.title(str(len(out)) + " samples of a HPP of rate " +
          str(rate) + " in 10 unit square")
plt.scatter(out[:, 0], out[:, 1], c="k")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(os.path.join(path2res, "HPP_rate_2Dtest.png"),
            bbox_inches='tight', dpi=100, edgecolor='w')
# plt.show()
plt.close("all")


rate = 2
realizations = 3
bounds = [[0, 10], [10, 0]]
dimensions = len(bounds)
out = PyPgen.HPP_rate(rate=rate, bounds=bounds, realizations=realizations)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(np.amin(out[:, 1]) > bounds[1][1])
assert(np.amax(out[:, 1]) < bounds[1][0])
assert(out.shape[-1] == dimensions)
assert((len(out.shape) - 1) == realizations)
sys.exit()
