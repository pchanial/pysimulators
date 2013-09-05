from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from pyoperators.utils.testing import assert_eq, assert_raises
from pysimulators import Acquisition, Instrument, Layout, MaskPolicy, Pointing

flags = ['bad', 'u1', 'u2']


def test_mask_policy1():
    good_policy = ['kEep', 'removE', 'MASK']
    mask_policy = MaskPolicy(flags, good_policy)
    assert_eq(np.array(mask_policy), (0, 2, 1))
    assert mask_policy.bad == 'keep' and mask_policy.u1 == 'remove' and \
        mask_policy.u2 == 'mask'


def test_mask_policy2():
    bad_policy = ['remove', 'KKeep']
    assert_raises(ValueError, MaskPolicy, flags, bad_policy)
    assert_raises(KeyError, MaskPolicy, flags[0:2], bad_policy)


def test_get_noise():
    fsamp = 5
    sigma = 0.3
    p = Pointing.zeros(2e4)
    p.removed[5000:10000] = True
    pointings = (p, (p, Pointing.zeros(3e4)))
    shapes = ((1,), (2, 3))
    np.random.seed(0)

    def func1(shape, p, fknee):
        instrument = Instrument('', Layout(shape))
        conf = Acquisition(instrument, p)
        noise = conf.get_noise(sampling_frequency=fsamp, sigma=sigma,
                               fknee=fknee)
        assert noise.shape == (conf.get_ndetectors(),
                               sum(conf.get_nsamples()))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)
    for shape in shapes:
        for p in pointings:
            for fknee in (0, 1e-7):
                yield func1, shape, p, fknee

    freq = np.arange(6) / 6 * fsamp
    psd = np.array([0, 1, 1, 1, 1, 1], float) * sigma**2 / fsamp

    def func2(shape, p):
        instrument = Instrument('', Layout(shape))
        conf = Acquisition(instrument, p)
        noise = conf.get_noise(psd=psd, bandwidth=freq[1], twosided=True,
                               sampling_frequency=fsamp)
        assert noise.shape == (conf.get_ndetectors(),
                               sum(conf.get_nsamples()))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)

    for shape in shapes:
        for p in pointings:
            yield func2, shape, p

    freq = np.arange(4) / 6 * fsamp
    psd = np.array([0, 2, 2, 1], float) * sigma**2 / fsamp

    def func3(shape, p):
        instrument = Instrument('', Layout(shape))
        conf = Acquisition(instrument, p)
        noise = conf.get_noise(bandwidth=freq[1], psd=psd, twosided=False,
                               sampling_frequency=fsamp)
        assert noise.shape == (conf.get_ndetectors(),
                               sum(conf.get_nsamples()))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)

    for shape in shapes:
        for p in pointings:
            yield func3, shape, p
