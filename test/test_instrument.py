import numpy as np

from pyoperators.utils import product
from pyoperators.utils.testing import assert_eq
from pysimulators import Instrument

class MyInstrument(Instrument):
    def get_valid_detectors(self, masked=False):
        coords = np.mgrid[tuple(slice(0,s) for s in self.detector.shape)]
        coords[0] = (coords[0] +  1) % 2
        coords = tuple(c.ravel() for c in coords)
        mask = ~self.detector.removed
        if masked:
            mask &= ~self.detector.masked
        return tuple(c[mask[coords]] for c in coords)

def assert_pack_unpack(instrument):
    shape = instrument.detector.shape
    u = np.arange(product(shape)*2.).reshape(shape + (2,))
    u.T[...] *= ~instrument.detector.removed.T
    p = instrument.pack(u)
    u2 = instrument.unpack(p)
    assert_eq(u, u2)

def test_pack_unpack1():
    for ndim in range(1,4):
        shape = tuple(np.arange(2, 2 + ndim))
        mask = np.random.random_integers(0, 1, size=shape).astype(bool)
        instrument = MyInstrument(ndim+1, shape, removed=mask)
        yield assert_pack_unpack, instrument

def test_pack_unpack2():
    for ndim in range(1,4):
        shape = tuple(np.arange(2, 2 + ndim))
        mask = np.zeros(shape, bool)
        instrument = MyInstrument(ndim+1, shape, removed=mask)
        yield assert_pack_unpack, instrument

def test_pack_unpack3():
    for ndim in range(1,4):
        shape = tuple(np.arange(2, 2 + ndim))
        mask = np.ones(shape, bool)
        instrument = MyInstrument(ndim+1, shape, removed=mask)
        yield assert_pack_unpack, instrument
