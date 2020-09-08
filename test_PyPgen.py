# Code créé par Gabriel Taillon le 6 Septembre 2020
import PyPgen
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest

scripthpath = os.path.dirname(os.path.abspath(__file__))
path2res = os.path.join(scripthpath, os.path.splitext(
    os.path.basename(__file__))[0])
if not os.path.exists(path2res):
    os.makedirs(path2res)


class TestMaPP(unittest.TestCase):
    def test_basic(self):
        pass


class TestMPP(unittest.TestCase):
    def test_basic(self):
        shape = 3
        scale = 1
        bounds = [1, 10]
        dimensions = 1
        realizations = 3

        def info(size):
            return(np.random.gamma(shape=shape, scale=scale, size=size))

        out = PyPgen.MPP(info=info, bounds=bounds, realizations=realizations)
        self.assertTrue(len(out) == realizations)
        for i in np.arange(realizations):
            self.assertTrue(np.amin(out[i][:, 0]) > bounds[0])
            self.assertTrue(np.amax(out[i][:, 0]) < bounds[1])
            self.assertTrue(out[i].shape[-1] == dimensions)


class TestNHPP(unittest.TestCase):
    def test_halfsinus1D(self):
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
        self.assertTrue(out.shape[-1] == dimensions)
        self.assertTrue(len(out.shape) > 1)
        self.assertTrue(np.amin(out) > bounds[0])
        self.assertTrue(np.amax(out) < bounds[1])

    def test_halfsinus2D(self):
        phase_1 = 0
        phase_2 = 0
        period_1 = 2
        period_2 = 1
        amplitude = 1
        offset = 1
        rate_max = amplitude + amplitude + offset
        dimensions = 2

        def rate_halfsinus2D(x_1, x_2):
            sins = np.sin(2.*np.pi*x_1/period_1 + phase_1) + \
                np.sin(2.*np.pi*x_2/period_2 + phase_2)
            out = amplitude*(sins*(sins > 0)) + offset
            return(out)

        bounds = [[1, 10], [10, 20]]
        out = PyPgen.NHPP(rate_halfsinus2D, rate_max, bounds)
        self.assertTrue(out.shape[-1] == dimensions)
        self.assertTrue(len(out.shape) > 1)
        self.assertTrue(np.amin(out[:, 0]) > bounds[0][0])
        self.assertTrue(np.amax(out[:, 0]) < bounds[0][1])
        self.assertTrue(np.amin(out[:, 1]) > bounds[1][0])
        self.assertTrue(np.amax(out[:, 1]) < bounds[1][1])

    def test_halfsinus3D(self):
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
            sins = np.sin(2.*np.pi*x_1/period_1 + phase_1) + \
                np.sin(2.*np.pi*x_2/period_2 + phase_2) + \
                np.sin(2.*np.pi*x_3/period_3 + phase_3)
            out = amplitude*(sins*(sins > 0)) + offset
            return(out)

        bounds = [[1, 10], [10, 20], [20, 21]]
        out = PyPgen.NHPP(rate_halfsinus3D, rate_max, bounds)
        self.assertTrue(out.shape[-1] == dimensions)
        self.assertTrue(len(out.shape) > 1)
        self.assertTrue(np.amin(out[:, 0]) > bounds[0][0])
        self.assertTrue(np.amax(out[:, 0]) < bounds[0][1])
        self.assertTrue(np.amin(out[:, 1]) > bounds[1][0])
        self.assertTrue(np.amax(out[:, 1]) < bounds[1][1])
        self.assertTrue(np.amin(out[:, 2]) > bounds[2][0])
        self.assertTrue(np.amax(out[:, 2]) < bounds[2][1])


class TestHPP_temporal(unittest.TestCase):
    def test_basic(self):
        rate = 5
        bounds = [1, 10]
        out = PyPgen.HPP_temporal(rate=rate, bounds=bounds)
        self.assertTrue(np.amin(out) > bounds[0])
        self.assertTrue(np.amax(out) < bounds[1])
        plt.title(str(len(out)) + " samples of a temporal HPP of rate " +
                  str(rate) + " in 10s")
        plt.plot(out, np.arange(len(out)), ".k")
        plt.xlabel("Time [s]")
        plt.ylabel("Count")
        plt.savefig(os.path.join(path2res, "HPP_temporal_test.png"),
                    bbox_inches='tight', dpi=100, edgecolor='w')
        # plt.show()
        plt.close("all")

        realizations = 3
        rate = 5
        bounds = [1, 10]
        out = PyPgen.HPP_temporal(
            rate=rate, bounds=bounds, realizations=realizations)
        self.assertTrue(len(out) == realizations)
        for i in np.arange(realizations):
            self.assertTrue(np.amin(out[i]) > bounds[0])
            self.assertTrue(np.amax(out[i]) < bounds[1])


class TestHPP_samples(unittest.TestCase):
    def test_2D(self):
        samples = 20
        realizations = 3
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                                 realizations=realizations)
        self.assertTrue(len(out) == realizations)
        for i in np.arange(realizations):
            self.assertTrue(np.amin(out[i][:, 0]) > bounds[0][0])
            self.assertTrue(np.amax(out[i][:, 0]) < bounds[0][1])
            self.assertTrue(np.amin(out[i][:, 1]) > bounds[1][0])
            self.assertTrue(np.amax(out[i][:, 1]) < bounds[1][1])
            self.assertTrue(len(out[i]) == samples)

    def test_3D(self):
        samples = [3, 4, 5]
        realizations = 3
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                                 realizations=realizations)
        self.assertTrue(len(out) == realizations)
        for i in np.arange(realizations):
            self.assertTrue(np.amin(out[i][:, 0]) > bounds[0][0])
            self.assertTrue(np.amax(out[i][:, 0]) < bounds[0][1])
            self.assertTrue(np.amin(out[i][:, 1]) > bounds[1][0])
            self.assertTrue(np.amax(out[i][:, 1]) < bounds[1][1])
            self.assertTrue(len(out[i]) == samples[i])

    def test_3D(self):
        samples = [20, 10]
        realizations = 1
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        try:
            out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                                     realizations=realizations)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_2Dlistsample(self):
        samples = [20, 10]
        realizations = 3
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        try:
            out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                                     realizations=realizations)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_2Dscalarsampleplot(self):
        samples = 20
        realizations = 1
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        out = PyPgen.HPP_samples(samples=samples, bounds=bounds,
                                 realizations=realizations)
        self.assertTrue(np.amin(out[:, 0]) > bounds[0][0])
        self.assertTrue(np.amax(out[:, 0]) < bounds[0][1])
        self.assertTrue(np.amin(out[:, 1]) > bounds[1][0])
        self.assertTrue(np.amax(out[:, 1]) < bounds[1][1])
        self.assertTrue(len(out) == samples)

        plt.title(str(samples) + " samples of a HPP in 10 unit square")
        plt.scatter(out[:, 0], out[:, 1], c="k")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path2res, "HPP_samples_2Dtest.png"),
                    bbox_inches='tight', dpi=100, edgecolor='w')
        # plt.show()
        plt.close("all")

    def test_2Dscalarsampleplot(self):

        rate = 2
        realizations = 1
        bounds = [[0, 10], [0, 10]]
        dimensions = len(bounds)
        out = PyPgen.HPP_rate(rate=rate, bounds=bounds,
                              realizations=realizations)
        self.assertTrue(np.amin(out[:, 0]) > bounds[0][0])
        self.assertTrue(np.amax(out[:, 0]) < bounds[0][1])
        self.assertTrue(np.amin(out[:, 1]) > bounds[1][0])
        self.assertTrue(np.amax(out[:, 1]) < bounds[1][1])

        plt.title(str(len(out)) + " samples of a HPP of rate " +
                  str(rate) + " in 10 unit square")
        plt.scatter(out[:, 0], out[:, 1], c="k")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path2res, "HPP_rate_2Dtest.png"),
                    bbox_inches='tight', dpi=100, edgecolor='w')
        # plt.show()
        plt.close("all")

    def test_2Dscalarrealizations(self):
        rate = 2
        realizations = 3
        bounds = [[0, 10], [10, 0]]
        dimensions = len(bounds)
        out = PyPgen.HPP_rate(rate=rate, bounds=bounds,
                              realizations=realizations)
        self.assertTrue(len(out) == realizations)
        for i in np.arange(realizations):
            self.assertTrue(np.amin(out[i][:, 0]) > bounds[0][0])
            self.assertTrue(np.amax(out[i][:, 0]) < bounds[0][1])
            self.assertTrue(np.amin(out[i][:, 1]) > bounds[1][1])
            self.assertTrue(np.amax(out[i][:, 1]) < bounds[1][0])


if __name__ == '__main__':
    unittest.main()
