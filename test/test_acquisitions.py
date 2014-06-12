from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from pyoperators.utils.testing import assert_eq, assert_raises
from pysimulators import (
    Acquisition, Instrument, MaskPolicy, PackedTable, Sampling, Scene)

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
    scene = Scene(10)
    sampling = Sampling(2e4, period=1 / fsamp)
    shapes = ((1,), (2, 3))
    np.random.seed(0)

    class MyAcquisition1(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(self.sampling, sigma=sigma,
                                             fknee=fknee)

    def func1(shape, fknee):
        instrument = Instrument('', PackedTable(shape))
        acq = MyAcquisition1(instrument, sampling, scene)
        noise = acq.get_noise()
        assert noise.shape == (len(acq.instrument), len(acq.sampling))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)
    for shape in shapes:
        for fknee in (0, 1e-7):
            yield func1, shape, fknee

    freq = np.arange(6) / 6 * fsamp
    psd = np.array([0, 1, 1, 1, 1, 1], float) * sigma**2 / fsamp

    class MyAcquisition2(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(self.sampling, psd=psd,
                                             bandwidth=freq[1], twosided=True)

    def func2(shape):
        instrument = Instrument('', PackedTable(shape))
        acq = MyAcquisition2(instrument, sampling, scene)
        noise = acq.get_noise()
        assert noise.shape == (len(acq.instrument), len(acq.sampling))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)

    for shape in shapes:
        yield func2, shape

    freq = np.arange(4) / 6 * fsamp
    psd = np.array([0, 2, 2, 1], float) * sigma**2 / fsamp

    class MyAcquisition3(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(self.sampling, psd=psd,
                                             bandwidth=freq[1], twosided=False)

    def func3(shape):
        instrument = Instrument('', PackedTable(shape))
        acq = MyAcquisition3(instrument, sampling, scene)
        noise = acq.get_noise()
        assert noise.shape == (len(acq.instrument), len(acq.sampling))
        assert_allclose(np.std(noise), sigma, rtol=1e-2)

    for shape in shapes:
        yield func3, shape
