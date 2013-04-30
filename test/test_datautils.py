import numpy as np

from numpy.testing import assert_array_equal
from pyoperators.utils.testing import assert_eq
from pysimulators.datautils import (
    airy_disk,
    distance,
    gaussian,
    profile,
    integrated_profile,
    _distance_slow,
)


def test_distance1():
    origin = (0.0,)
    d0 = np.arange(5.0) * 0.5
    d1 = distance(5, origin=origin, resolution=0.5)
    d2 = _distance_slow((5,), origin, [0.5], None)
    assert_array_equal(d0, d1)
    assert_array_equal(d0, d2)


def test_distance2():
    origin = (0.0, 1.0)
    d0 = np.array([[2.0, np.sqrt(5), np.sqrt(8)], [0, 1, 2]])
    d1 = distance((2, 3), origin=origin, resolution=(1.0, 2.0))
    d2 = _distance_slow((2, 3), origin, [1.0, 2.0], None)
    assert_array_equal(d0, d1)
    assert_array_equal(d0, d2)


def test_fwhm():
    def func(f, fwhm, resolution, n):
        m = f((1000, 1000), fwhm=fwhm, resolution=resolution)
        assert np.sum(m[500, :] > np.max(m) / 2) == n

    for f in [gaussian, airy_disk]:
        for fwhm, resolution, n in zip([10, 10, 100], [0.1, 1, 10], [100, 10, 10]):
            yield func, f, fwhm, resolution, n


def test_profile():
    def profile_slow(input, origin=None, bin=1.0):
        d = distance(input.shape, origin=origin)
        d /= bin
        d = d.astype(int)
        m = np.max(d)
        p = np.ndarray(int(m + 1))
        n = np.zeros(m + 1, int)
        for i in range(m + 1):
            p[i] = np.mean(input[d == i])
            n[i] = np.sum(d == i)
        return p, n

    d = distance((10, 20))
    x, y, n = profile(d, origin=(4, 5), bin=2.0, histogram=True)
    y2, n2 = profile_slow(d, origin=(4, 5), bin=2.0)
    assert_eq(y, y2[0 : y.size])
    assert_eq(n, n2[0 : n.size])


def test_integrated_profile():
    def integrated_profile_slow(input, origin=None, bin=1.0):
        d = distance(input.shape, origin=origin)
        m = np.max(d)
        x = np.ndarray(int(m + 1))
        y = np.ndarray(int(m + 1))
        for i in range(m + 1):
            x[i] = (i + 1) * bin
            y[i] = np.sum(input[d < x[i]])
        return x, y

    d = distance((10, 20))
    x, y = integrated_profile(d, origin=(4, 5), bin=2.0)
    x2, y2 = integrated_profile_slow(d, origin=(4, 5), bin=2.0)
    assert_eq(x, x2[0 : y.size])
    assert_eq(y, y2[0 : y.size])
