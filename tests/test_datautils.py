import numpy as np
import pytest

from pyoperators.utils.testing import assert_same
from pysimulators.datautils import (
    _distance2_slow,
    _distance_slow,
    airy_disk,
    distance,
    distance2,
    gaussian,
    integrated_profile,
    profile,
    psd2,
)

from .common import FLOAT_TYPES


@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_distance_1d(dtype):
    shape = (5,)
    center = 1
    scale = 0.5
    ref = abs((np.arange(shape[0]) - center) * scale)

    d1 = distance(shape, center=center, scale=scale, dtype=dtype)
    d2 = _distance_slow(shape, (center,), (scale,), dtype)
    assert_same(d1, ref.astype(dtype))
    assert_same(d2, ref.astype(dtype))
    distance(shape, center=center, scale=scale, dtype=dtype, out=d1)
    _distance_slow(shape, (center,), (scale,), dtype, d2)
    assert_same(d1, ref.astype(dtype))
    assert_same(d2, ref.astype(dtype))


@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_distance_2d(dtype):
    shape = (2, 3)
    center = (0.0, 1.0)
    scale = (1.0, 2.0)
    ref = np.array([[2.0, np.sqrt(5), np.sqrt(8)], [0, 1, 2]])

    d1 = distance(shape, center=center, scale=scale, dtype=dtype)
    d2 = _distance_slow(shape, center, scale, dtype)
    assert_same(d1, ref)
    assert_same(d2, ref)
    distance(shape, center=center, scale=scale, dtype=dtype, out=d1)
    _distance_slow(shape, center, scale, dtype, d2)
    assert_same(d1, ref)
    assert_same(d2, ref)


@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_distance2_1d(dtype):
    shape = (5,)
    center = 1
    scale = 0.5
    ref = ((np.arange(shape[0]) - center) * scale) ** 2

    d1 = distance2(shape, center=center, scale=scale, dtype=dtype)
    d2 = _distance2_slow(shape, (center,), (scale,), dtype)
    assert_same(d1, ref.astype(dtype))
    assert_same(d2, ref.astype(dtype))
    distance2(shape, center=center, scale=scale, dtype=dtype, out=d1)
    _distance2_slow(shape, (center,), (scale,), dtype, d2)
    assert_same(d1, ref.astype(dtype))
    assert_same(d2, ref.astype(dtype))


@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_distance2_2d(dtype):
    shape = (2, 3)
    center = (0.0, 1.0)
    scale = (1.0, 2.0)
    ref = np.array([[2.0, np.sqrt(5), np.sqrt(8)], [0, 1, 2]])

    d1 = distance(shape, center=center, scale=scale, dtype=dtype)
    d2 = _distance_slow(shape, center, scale, dtype)
    assert_same(d1, ref)
    assert_same(d2, ref)
    distance(shape, center=center, scale=scale, dtype=dtype, out=d1)
    _distance_slow(shape, center, scale, dtype, d2)
    assert_same(d1, ref)
    assert_same(d2, ref)


@pytest.mark.parametrize('func', [airy_disk])
@pytest.mark.parametrize(
    'fwhm, scale, n',
    [
        (10, 0.1, 100),
        (10, 1, 10),
        (100, 10, 10),
    ],
)
def test_fwhm(func, fwhm, scale, n):
    m = func((1000, 1000), fwhm=fwhm, scale=scale)
    assert np.sum(m[500, :] > np.max(m) / 2) == n


def test_gaussian():
    shape = (100, 90)
    sigma = 2.1, 2.4
    center = 14.3, 52.2
    dtype = float
    y, x = np.ogrid[0 : shape[0], 0 : shape[1]]
    ref = np.exp(
        -((x - center[0]) ** 2) / (2 * sigma[0] ** 2)
        + -((y - center[1]) ** 2) / (2 * sigma[1] ** 2)
    ) / (2 * np.pi * sigma[0] * sigma[1])
    actual = gaussian(shape, sigma=sigma, center=center, dtype=dtype)
    assert_same(actual, ref, atol=2)


def test_profile():
    def profile_slow(input, center=None, scale=1.0, bin=1.0):
        d = distance(input.shape, center=center, scale=scale)
        d /= bin
        d = d.astype(int)
        m = int(np.max(d))
        p = np.ndarray(m + 1)
        n = np.zeros(m + 1, int)
        for i in range(m + 1):
            mask = d == i
            p[i] = np.mean(input[mask])
            n[i] = np.sum(mask)
        return p, n

    d = distance((10, 20))
    center = (4, 5)
    scale = 0.4
    bin = 2
    x, y, n = profile(d, center=center, scale=scale, bin=bin, histogram=True)
    y_ref, n_ref = profile_slow(d, center=center, scale=scale, bin=bin)
    assert_same(y, y_ref[0 : y.size])
    assert_same(n, n_ref[0 : n.size])


@pytest.mark.xfail(reason='reason: correct values but wrong abscissae.')
def test_integrated_profile():
    def integrated_profile_slow(input, center=None, scale=1, bin=1):
        d = distance(input.shape, center=center, scale=scale)
        m = int(np.max(d))
        x = np.ndarray(m + 1)
        y = np.ndarray(m + 1)
        for i in range(m + 1):
            x[i] = (i + 1) * bin
            y[i] = np.sum(input[d < x[i]])
        return x * scale, y * scale**2

    d = distance((10, 20))
    bin = 2
    center = (4, 5)
    scale = 0.4
    x, y = integrated_profile(d, bin=bin, center=center, scale=scale)
    x_ref, y_ref = integrated_profile_slow(d, bin=bin, center=center, scale=scale)
    assert_same(x, x_ref[0 : y.size])
    assert_same(y, y_ref[0 : y.size])


def test_psd2():
    image = np.random.standard_normal((122, 122))
    fs = 1 / 0.11
    psd = psd2(image, sampling_frequency=fs)
    assert_same(np.sum(psd * (fs**2 / image.size)), np.mean(image**2))
