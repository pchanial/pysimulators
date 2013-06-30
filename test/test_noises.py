from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from pyoperators.utils.testing import assert_same
from pysimulators.noises import (_fold_psd, _unfold_psd, _interp,
                                 _gaussian_psd_1f, _gaussian_sample)


def test_fold():
    p = [1, 1, 1, 1]
    p_folded = _fold_psd(p)
    assert_same(p_folded, [1, 2, 1])


def test_unfold():
    p = [1, 2, 1]
    p_unfolded = _unfold_psd(p)
    assert_same(p_unfolded, [1, 1, 1, 1])


def test_interp():
    x = [1, 2, 3, 5, 7]
    y = [[1, 2, 3, 5, 7], [0, 1, 2, 4, 6]]
    zs = ([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 7.5, 8],
          [-1, -0.5], [2.25, 2.5], [8, 10])

    def func(z):
        out1 = _interp(z, x, y[0])
        assert_same(z, out1)
        out2 = _interp(z, x, y[1])
        out = _interp(z, x, y)
        assert_same(out, np.vstack([out1, out2]))
    for z in zs:
        yield func, z


def test_gaussian():
    fsamp = 5
    nsamples = 2**10
    white = [0.3, 0.5]
    fknee = 0.
    alpha = 1

    def func(twosided):
        np.random.seed(0)
        p = _gaussian_psd_1f(nsamples, fsamp, white, fknee, alpha,
                             twosided=twosided)
        t = _gaussian_sample(nsamples, fsamp, p, twosided=twosided)
        return p, t

    p1, t1 = func(False)
    p2, t2 = func(True)

    bandwidth = fsamp / nsamples
    assert_same(np.sum(p1, axis=-1) * bandwidth,
                np.sum(p2, axis=-1) * bandwidth, rtol=1000)

    assert_allclose(np.sqrt(np.sum(p1, axis=-1) * bandwidth), white, rtol=1e-3)
    assert_same(t1, t2)
