import numpy as np
from pysimulators import Pointing
from numpy.testing import assert_equal, assert_raises


def test_coords():
    ras = 1.0, (1.0,), [1.0], np.ones(10), np.ones((2, 3))
    decs = 2.0, (2.0,), [2.0], 2 * np.ones(10), 2 * np.ones((2, 3))
    pas = 3.0, (3.0,), [3.0], 3 * np.ones(10), 3 * np.ones((2, 3))

    for ra in ras:
        for format in (
            lambda ra: np.array([ra.T if isinstance(ra, np.ndarray) else ra]).T,
            lambda ra: (ra,),
            lambda ra: [
                ra,
            ],
            lambda ra: {'ra': ra},
        ):
            yield assert_raises, ValueError, Pointing, format(ra)

    def func2(p, ra, dec):
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)

    for ra, dec in zip(ras, decs):
        for format in (
            lambda ra, dec: np.array(
                [
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                ]
            ).T,
            lambda ra, dec: (ra, dec),
            lambda ra, dec: [ra, dec],
            lambda ra, dec: {'ra': ra, 'dec': dec},
        ):
            p = Pointing(format(ra, dec))
            yield func2, p, ra, dec

    def func3(p, ra, dec, pa):
        assert_equal(p.ra, ra)
        assert_equal(p.dec, dec)
        assert_equal(p.pa, pa)

    for ra, dec, pa in zip(ras, decs, pas):
        for format in (
            lambda ra, dec, pa: np.array(
                [
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa,
                ]
            ).T,
            lambda ra, dec, pa: (ra, dec, pa),
            lambda ra, dec, pa: [ra, dec, pa],
            lambda ra, dec, pa: {'ra': ra, 'dec': dec, 'pa': pa},
        ):
            p = Pointing(format(ra, dec, pa))
            yield func3, p, ra, dec, pa

    junks = 4.0, (4.0,), [4.0], 4 * np.ones(10), 4 * np.ones((2, 3))
    for ra, dec, pa, junk in zip(ras, decs, pas, junks):
        for format in (
            lambda ra, dec, pa, junk: np.array(
                [
                    ra.T if isinstance(ra, np.ndarray) else ra,
                    dec.T if isinstance(dec, np.ndarray) else dec,
                    pa.T if isinstance(pa, np.ndarray) else pa,
                    junk.T if isinstance(junk, np.ndarray) else junk,
                ]
            ).T,
            lambda ra, dec, pa, junk: (ra, dec, pa, junk),
            lambda ra, dec, pa, junk: [ra, dec, pa, junk],
        ):
            yield assert_raises, ValueError, Pointing, format(ra, dec, pa, junk)
