from __future__ import division

import astropy.io.fits as pyfits
import numpy as np
from numpy.testing import assert_allclose
from pyoperators import DiagonalOperator, PackOperator, pcg
from pyoperators.utils.testing import assert_same, assert_eq
from pysimulators import Map
from pysimulators.interfaces.madmap1 import MadMap1Observation, _read_filters


def mapper_naive(tod, H):
    m, c = H.T(tod), H.T(np.ones_like(tod))
    m /= c
    m.coverage = c
    return m

profile = None #'test_madcap.png'
path = 'test/data/madmap1/'
map_ref = pyfits.open(path + 'naivemapSpirePSW.fits')['image'].data
name = 'SPIRE/PSW'
obs = MadMap1Observation(name, 135, path + 'todSpirePSW_be',
                         path + 'invnttSpirePSW_be',
                         path + 'madmapSpirePSW.fits[coverage]',
                         bigendian=True, missing_value=np.nan)

tod = obs.get_tod(unit='Jy/beam')
projection = obs.get_operator()
packing = PackOperator(np.isnan(map_ref))

model = projection * packing
map_naive_1d = mapper_naive(tod, projection)
map_naive_2d = mapper_naive(tod, model)


def test_naive1():
    assert_same(map_naive_1d, map_ref[np.isfinite(map_ref)])


def test_naive2():
    assert_same(map_naive_2d, map_ref)


M = DiagonalOperator(packing(1/map_naive_2d.coverage))
assert np.all(np.isfinite(M.data))

invntt = obs.get_invntt_operator(fftw_flag='FFTW_PATIENT')


class Callback:
    def __init__(self):
        self.niterations = 0

    def __call__(self, x):
        self.niterations += 1

callback = Callback()
#callback = None


def test_ls():
    H = projection
    m = pcg(H.T * invntt * H, (H.T*invntt)(tod), M=M, tol=1e-7)
    map_ls_packed = Map(m['x'])
    map_ls_packed.header['TIME'] = m['time']

    #print 'Elapsed time:', map_ls_packed.header['TIME']
    #from tamasis import mapper_ls
    #map_ls_packed = mapper_ls(tod, projection, invntt=invntt, tol=1e-7, M=M,
    #                          callback=callback, criterion=False,
    #                          profile=profile)
    if profile:
        return
    print 'Elapsed time:', map_ls_packed.header['TIME']
    assert m['nit'] < 50
    ref = packing(Map(path + 'madmapSpirePSW.fits'))
    assert_allclose(map_ls_packed, ref, atol=1e-5)


def test_endian():
    bendian = _read_filters(path + 'invntt_be', bigendian=True)
    lendian = _read_filters(path + 'invntt_le')
    assert len(bendian) == 6
    assert bendian[0]['data'].size == 101
    assert_same(bendian[0]['data'][0], 5597147.4155586753)
    assert_eq(lendian, bendian)


if __name__ == '__main__':
    test_ls()
