# Code créé par Gabriel Taillon le 6 Septembre 2020
import PyPgen
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()

samples = 20
realizations = 3
bounds = [[0, 10], [0, 10]]
dimensions = len(bounds)
out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                         realizations=realizations)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(out.shape[-1] == dimensions)
assert(out.shape[0] == samples)
assert(out.shape == (samples, samples, samples, dimensions))
assert((len(out.shape) - 1) == realizations)

samples = 20
realizations = 1
bounds = [[0, 10], [0, 10]]
dimensions = len(bounds)
out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                         realizations=realizations)
assert(np.amin(out[:, 0]) > bounds[0][0])
assert(np.amax(out[:, 0]) < bounds[0][1])
assert(out.shape[-1] == dimensions)
assert(out.shape[0] == samples)
assert((len(out.shape) - 1) == realizations)


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
assert(out.shape[-1] == dimensions)
assert((len(out.shape) - 1) == realizations)
