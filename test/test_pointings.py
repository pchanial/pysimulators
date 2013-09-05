import numpy as np
from pysimulators import Pointing
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils.testing import assert_is

POINTING_DTYPE = Pointing.default_dtype
SUPER_POINTING_DTYPE = POINTING_DTYPE + [('junk', 'S1')]


def test_coords():
    ras = 1., (1.,), [1.], np.ones(10), np.ones((2, 3))
    decs = 2., (2.,), [2.], 2*np.ones(10), 2*np.ones((2, 3))
    pas = 3., (3.,), [3.], 3*np.ones(10), 3*np.ones((2, 3))

    for ra in ras:
        for fmt in (
                lambda ra: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra]).T,
                lambda ra: (ra,),
                lambda ra: [ra],
                lambda ra: {'ra': ra}):
            yield assert_raises, ValueError, Pointing, fmt(ra)

    def func2(p, ra, dec):
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
    for ra, dec in zip(ras, decs):
        for fmt in (
                lambda ra, dec: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec]).T,
                lambda ra, dec: (ra, dec),
                lambda ra, dec: [ra, dec],
                lambda ra, dec: {'ra': ra, 'dec': dec}):
            p = Pointing(fmt(ra, dec))
            yield func2, p, ra, dec

    def func3(p, ra, dec, pa):
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, pa)
    for ra, dec, pa in zip(ras, decs, pas):
        for fmt in (
                lambda ra, dec, pa: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa]).T,
                lambda ra, dec, pa: (ra, dec, pa),
                lambda ra, dec, pa: [ra, dec, pa],
                lambda ra, dec, pa: {'ra': ra, 'dec': dec, 'pa': pa}):
            p = Pointing(fmt(ra, dec, pa))
            yield func3, p, ra, dec, pa

    junks = 4., (4.,), [4.], 4*np.ones(10), 4*np.ones((2, 3))
    for ra, dec, pa, junk in zip(ras, decs, pas, junks):
        for fmt in (
                lambda ra, dec, pa, j: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa,
                    j.T if isinstance(j, np.ndarray) else j]).T,
                lambda ra, dec, pa, junk: (ra, dec, pa, junk),
                lambda ra, dec, pa, junk: [ra, dec, pa, junk]):
            yield assert_raises, ValueError, Pointing, fmt(ra, dec, pa, junk)


def test_dtype_subclass():
    class SubPointing(Pointing):
        pass

    def func(coords, dtype, subok):
        p = Pointing(coords, dtype=dtype, subok=subok)
        assert_is(type(p), SubPointing if type(coords) is SubPointing and subok
                  else Pointing)
        assert_equal(dtype, p.dtype)

    for coords in ([1, 2, 3], Pointing([1, 2, 3]), SubPointing([1, 2, 3])):
        for dtype in (POINTING_DTYPE, SUPER_POINTING_DTYPE):
            for subok in (False, True):
                yield func, coords, dtype, subok
