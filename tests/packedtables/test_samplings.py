import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators.utils import isscalarlike
from pyoperators.utils.testing import assert_same
from pysimulators import SamplingEquatorial


@pytest.mark.parametrize('fmt', [lambda ra: ((ra,), {}), lambda ra: ((), {'ra': ra})])
@pytest.mark.parametrize('ra', [1.0, (1.0,), [1.0], np.ones(10), np.ones((2, 3))])
def test_coords_ra(fmt, ra):

    args, keywords = fmt(ra)
    if len(args) > 0 and isscalarlike(args[0]):
        p = SamplingEquatorial(*args, **keywords)
        assert p.ra is None
        assert p.dec is None
        assert p.pa == 0
        return
    with pytest.raises(ValueError):
        SamplingEquatorial(*args, **keywords)


@pytest.mark.parametrize(
    'fmt',
    [
        lambda ra, dec: ((ra, dec), {}),
        lambda ra, dec: ((), {'ra': ra, 'dec': dec}),
    ],
)
@pytest.mark.parametrize(
    'ra, dec',
    [
        (1.0, 2.0),
        ((1.0,), (2.0,)),
        ([1.0], [2.0]),
        (np.ones(10), 2 * np.ones(10)),
        (np.ones((2, 3)), 2 * np.ones((2, 3))),
    ],
)
def test_coords_radec(fmt, ra, dec):
    args, keywords = fmt(ra, dec)
    if isinstance(ra, np.ndarray) and ra.ndim == 2:
        with pytest.raises(ValueError):
            SamplingEquatorial(*args, **keywords)
        return
    p = SamplingEquatorial(*args, **keywords)
    assert_equal(p.ra, ra)
    assert_equal(p.dec, dec)
    assert p.pa == 0


@pytest.mark.parametrize(
    'fmt',
    [
        lambda ra, dec, pa: ((ra, dec, pa), {}),
        lambda ra, dec, pa: ((), {'ra': ra, 'dec': dec, 'pa': pa}),
    ],
)
@pytest.mark.parametrize(
    'ra, dec, pa',
    [
        (1.0, 2.0, 3.0),
        ((1.0,), (2.0,), (3.0,)),
        ([1.0], [2.0], [3.0]),
        (np.ones(10), 2 * np.ones(10), 3 * np.ones(10)),
        (np.ones((2, 3)), 2 * np.ones((2, 3)), 3 * np.ones((2, 3))),
    ],
)
def test_coords_radecpa(fmt, ra, dec, pa):
    args, keywords = fmt(ra, dec, pa)
    if isinstance(ra, np.ndarray) and ra.ndim == 2:
        with pytest.raises(ValueError):
            SamplingEquatorial(*args, **keywords)
        return
    p = SamplingEquatorial(*args, **keywords)
    assert_equal(p.ra, ra)
    assert_equal(p.dec, dec)
    assert_equal(p.pa, pa)


@pytest.mark.parametrize(
    'fmt',
    [
        lambda ra, dec, pa, t: ((ra, dec, pa), {'time': t}),
        lambda ra, dec, pa, t: ((), {'ra': ra, 'dec': dec, 'pa': pa, 'time': t}),
    ],
)
@pytest.mark.parametrize(
    'ra, dec, pa, time',
    [
        (1.0, 2.0, 3.0, 1.0),
        ((1.0,), (2.0,), (3.0,), (1.0,)),
        ([1.0], [2.0], [3.0], [1.0]),
        (np.ones(10), 2 * np.ones(10), 3 * np.ones(10), np.r_[1:11]),
        (np.ones((2, 3)), 2 * np.ones((2, 3)), 3 * np.ones((2, 3)), np.r_[1:4]),
    ],
)
def test_coords_radecpatime(fmt, ra, dec, pa, time):
    args, keywords = fmt(ra, dec, pa, time)
    if isinstance(ra, np.ndarray) and ra.ndim == 2:
        with pytest.raises(ValueError):
            SamplingEquatorial(*args, **keywords)
        return
    p = SamplingEquatorial(*args, **keywords)
    assert_same(p.ra, ra)
    assert_same(p.dec, dec)
    assert_same(p.pa, pa)
    assert_same(p.time, time)


@pytest.mark.parametrize(
    'fmt',
    [lambda ra, dec, pa, t, other: ((ra, dec, pa, other), {'time': t})],
)
@pytest.mark.parametrize(
    'ra, dec, pa, time, junk',
    [
        (1.0, 2.0, 3.0, 1.0, 4.0),
        ((1.0,), (2.0,), (3.0,), (1.0,), (4.0,)),
        ([1.0], [2.0], [3.0], [1.0], [4.0]),
        (np.ones(10), 2 * np.ones(10), 3 * np.ones(10), np.r_[1:11], 4 * np.ones(10)),
        (
            np.ones((2, 3)),
            2 * np.ones((2, 3)),
            3 * np.ones((2, 3)),
            np.r_[1:4],
            4 * np.ones((2, 3)),
        ),
    ],
)
def test_coords_radecpatimejunk(fmt, ra, dec, pa, time, junk):
    args, keywords = fmt(ra, dec, pa, time, junk)
    with pytest.raises(ValueError):
        SamplingEquatorial(*args, **keywords)
