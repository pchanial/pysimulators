import numpy as np
from pysimulators import Pointing, PointingEquatorial
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils.testing import assert_is, assert_same

POINTING_DTYPE = Pointing.DEFAULT_DTYPE
SUPER_POINTING_DTYPE = POINTING_DTYPE + [('junk', 'S1')]


def test_coords():
    ras = 1., (1.,), [1.], np.ones(10), np.ones((2, 3))
    decs = 2., (2.,), [2.], 2 * np.ones(10), 2 * np.ones((2, 3))
    pas = 3., (3.,), [3.], 3 * np.ones(10), 3 * np.ones((2, 3))
    times = 1., (1.,), [1.], np.r_[1:11], np.r_[1:4]
    junks = 4., (4.,), [4.], 4 * np.ones(10), 4 * np.ones((2, 3))

    def func1(fmt, ra):
        assert_raises(ValueError, PointingEquatorial, fmt(ra))
    for ra in ras:
        for fmt in (lambda ra: {'ra': ra},):
            yield func1, fmt, ra

    def func2(fmt, ra, dec):
        p = PointingEquatorial(fmt(ra, dec))
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, 0)
    for ra, dec in zip(ras, decs):
        for fmt in (
                lambda ra, dec: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec]).T,
                lambda ra, dec: (ra, dec),
                lambda ra, dec: [ra, dec],
                lambda ra, dec: {'ra': ra, 'dec': dec}):
            yield func2, fmt, ra, dec

    def func3(fmt, ra, dec, pa):
        p = PointingEquatorial(fmt(ra, dec, pa))
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
            yield func3, fmt, ra, dec, pa

    def func4(fmt, ra, dec, pa, time):
        p = PointingEquatorial(fmt(ra, dec, pa, time))
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, pa)
        assert_same(p.time, time, broadcasting=True)
    for ra, dec, pa, time in zip(ras, decs, pas, times):
        for fmt in (
                lambda ra, dec, pa, t: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa,
                    (t if ra.ndim < 2 else np.tile(t, ra.shape[:-1] + (1,))).T
                    if isinstance(t, np.ndarray) else t]).T,
                lambda ra, dec, pa, t: (ra, dec, pa, t),
                lambda ra, dec, pa, t: [ra, dec, pa, t]):
            yield func4, fmt, ra, dec, pa, time

    def func5(fmt, ra, dec, pa, time, junk):
        assert_raises(ValueError, PointingEquatorial,
                      fmt(ra, dec, pa, time, junk))
    for ra, dec, pa, time, junk in zip(ras, decs, pas, times, junks):
        for fmt in (
                lambda ra, dec, pa, t, j: np.array([
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa,
                    t.T if isinstance(t, np.ndarray) else t,
                    j.T if isinstance(j, np.ndarray) else j]).T,
                lambda ra, dec, pa, t, junk: (ra, dec, pa, t, junk),
                lambda ra, dec, pa, t, junk: [ra, dec, pa, t, junk]):
            yield func5, fmt, ra, dec, pa, time, junk


def test_dtype_subclass():
    class SubPointing(Pointing):
        pass

    def func(coords, subok):
        p = Pointing(coords, subok=subok)
        assert_is(type(p), SubPointing if type(coords) is SubPointing and subok
                  else Pointing)
        assert_equal(p.dtype, p.DEFAULT_DTYPE)

    for coords in ([1, 2], Pointing([1, 2]), SubPointing([1, 2])):
        for subok in (False, True):
            yield func, coords, subok
