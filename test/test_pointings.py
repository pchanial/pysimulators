import numpy as np
from pysimulators import Pointing, PointingEquatorial
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils.testing import assert_same


def test_coords():
    ras = 1.0, (1.0,), [1.0], np.ones(10), np.ones((2, 3))
    decs = 2.0, (2.0,), [2.0], 2 * np.ones(10), 2 * np.ones((2, 3))
    pas = 3.0, (3.0,), [3.0], 3 * np.ones(10), 3 * np.ones((2, 3))
    times = 1.0, (1.0,), [1.0], np.r_[1:11], np.r_[1:4]
    junks = 4.0, (4.0,), [4.0], 4 * np.ones(10), 4 * np.ones((2, 3))

    def func1(fmt, ra):
        args, keywords = fmt(ra)
        assert_raises(ValueError, PointingEquatorial, *args, **keywords)

    for ra in ras:
        for fmt in (lambda ra: ((ra,), {}), lambda ra: ((), {'ra': ra})):
            yield func1, fmt, ra

    def func2(fmt, ra, dec):
        args, keywords = fmt(ra, dec)
        if isinstance(ra, np.ndarray) and ra.ndim == 2:
            assert_raises(ValueError, PointingEquatorial, *args, **keywords)
            return
        p = PointingEquatorial(*args, **keywords)
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, 0)

    for ra, dec in zip(ras, decs):
        for fmt in (
            lambda ra, dec: ((ra, dec), {}),
            lambda ra, dec: ((), {'ra': ra, 'dec': dec}),
        ):
            yield func2, fmt, ra, dec

    def func3(fmt, ra, dec, pa):
        args, keywords = fmt(ra, dec, pa)
        if isinstance(ra, np.ndarray) and ra.ndim == 2:
            assert_raises(ValueError, PointingEquatorial, *args, **keywords)
            return
        p = PointingEquatorial(*args, **keywords)
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, pa)

    for ra, dec, pa in zip(ras, decs, pas):
        for fmt in (
            lambda ra, dec, pa: ((ra, dec, pa), {}),
            lambda ra, dec, pa: ((), {'ra': ra, 'dec': dec, 'pa': pa}),
        ):
            yield func3, fmt, ra, dec, pa

    def func4(fmt, ra, dec, pa, time):
        args, keywords = fmt(ra, dec, pa, time)
        if isinstance(ra, np.ndarray) and ra.ndim == 2:
            assert_raises(ValueError, PointingEquatorial, *args, **keywords)
            return
        p = PointingEquatorial(*args, **keywords)
        assert_same(p.ra, ra)
        assert_same(p.dec, dec)
        assert_same(p.pa, pa)
        assert_same(p.time, time)

    for ra, dec, pa, time in zip(ras, decs, pas, times):
        for fmt in (
            lambda ra, dec, pa, t: ((ra, dec, pa), {'time': t}),
            lambda ra, dec, pa, t: ((), {'ra': ra, 'dec': dec, 'pa': pa, 'time': t}),
        ):
            yield func4, fmt, ra, dec, pa, time

    def func5(fmt, ra, dec, pa, time, junk):
        args, keywords = fmt(ra, dec, pa, time, junk)
        assert_raises(ValueError, PointingEquatorial, *args, **keywords)

    for ra, dec, pa, time, junk in zip(ras, decs, pas, times, junks):
        fmt = lambda ra, dec, pa, t, junk: ((ra, dec, pa, junk), {'time': t})
        yield func5, fmt, ra, dec, pa, time, junk
