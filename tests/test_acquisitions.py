import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from pysimulators import (
    Acquisition,
    Instrument,
    MaskPolicy,
    PackedTable,
    Sampling,
    Scene,
)

flags = ['bad', 'u1', 'u2']


def test_mask_policy1():
    good_policy = ['kEep', 'removE', 'MASK']
    mask_policy = MaskPolicy(flags, good_policy)
    assert_equal(np.array(mask_policy), (0, 2, 1))
    assert (
        mask_policy.bad == 'keep'
        and mask_policy.u1 == 'remove'
        and mask_policy.u2 == 'mask'
    )


def test_mask_policy2():
    bad_policy = ['remove', 'KKeep']
    with pytest.raises(ValueError):
        MaskPolicy(flags, bad_policy)
    with pytest.raises(KeyError):
        MaskPolicy(flags[0:2], bad_policy)


@pytest.mark.parametrize('shape', [(1,), (2, 3)])
@pytest.mark.parametrize('fknee', [0, 1e-7])
def test_get_noise1(shape, fknee):
    fsamp = 5
    sigma = 0.3
    scene = Scene(10)
    sampling = Sampling(2e4, period=1 / fsamp)
    np.random.seed(0)

    class MyAcquisition1(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(self.sampling, sigma=sigma, fknee=fknee)

    instrument = Instrument('', PackedTable(shape))
    acq = MyAcquisition1(instrument, sampling, scene)
    noise = acq.get_noise()
    assert noise.shape == (len(acq.instrument), len(acq.sampling))
    assert_allclose(np.std(noise), sigma, rtol=1e-2)


@pytest.mark.parametrize('shape', [(1,), (2, 3)])
def test_get_noise2(shape):
    fsamp = 5
    sigma = 0.3
    scene = Scene(10)
    sampling = Sampling(2e4, period=1 / fsamp)
    freq = np.arange(6) / 6 * fsamp
    psd = np.array([0, 1, 1, 1, 1, 1], float) * sigma**2 / fsamp

    class MyAcquisition2(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(
                self.sampling, psd=psd, bandwidth=freq[1], twosided=True
            )

    instrument = Instrument('', PackedTable(shape))
    acq = MyAcquisition2(instrument, sampling, scene)
    noise = acq.get_noise()
    assert noise.shape == (len(acq.instrument), len(acq.sampling))
    assert_allclose(np.std(noise), sigma, rtol=1e-2)


@pytest.mark.parametrize('shape', [(1,), (2, 3)])
def test_get_noise3(shape):
    fsamp = 5
    sigma = 0.3
    scene = Scene(10)
    sampling = Sampling(2e4, period=1 / fsamp)
    freq = np.arange(4) / 6 * fsamp
    psd = np.array([0, 2, 2, 1], float) * sigma**2 / fsamp

    class MyAcquisition3(Acquisition):
        def get_noise(self):
            return self.instrument.get_noise(
                self.sampling, psd=psd, bandwidth=freq[1], twosided=False
            )

    instrument = Instrument('', PackedTable(shape))
    acq = MyAcquisition3(instrument, sampling, scene)
    noise = acq.get_noise()
    assert noise.shape == (len(acq.instrument), len(acq.sampling))
    assert_allclose(np.std(noise), sigma, rtol=1e-2)
