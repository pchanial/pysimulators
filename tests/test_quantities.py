import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from pyoperators.utils.testing import assert_eq
from pysimulators.quantities import Quantity, UnitError

from .common import assert_equal_subclass


def assert_quantity(q, m, u):
    assert isinstance(q, Quantity)
    assert_array_equal(q.magnitude, m)
    if isinstance(u, dict):
        assert q._unit == u
    else:
        assert q.unit == u


def test_si():
    q = Quantity(1, 'km')
    assert_equal_subclass(q.SI, Quantity(1000, 'm'))


def test_tounit1():
    q = Quantity(1, 'm')
    assert_equal_subclass(q, q.tounit('m'))


def test_tounit2():
    q = Quantity(1, 'km')
    assert_equal_subclass(Quantity(1000, 'm'), q.tounit('m'))


def test_copy():
    q = Quantity(1, 'm')
    q2 = q.copy()
    q.inunit('m')
    assert_equal_subclass(q, q2)


def test_inunit():
    q = Quantity(1, 'km')
    q.inunit('m')
    assert_equal_subclass(Quantity(1000, 'm'), q)


@pytest.mark.parametrize('other', [Quantity(1.0), np.array(1.0), 1])
def test_add1(other):
    q = Quantity(1.0)
    assert_equal_subclass(q + other, Quantity(2))
    assert_equal_subclass(other + q, Quantity(2))


@pytest.mark.parametrize('other', [Quantity(1.0), np.array(1.0), 1])
def test_add2(other):
    q = Quantity(1.0, 'm')
    assert_equal_subclass(q + other, Quantity(2, 'm'))
    assert_equal_subclass(other + q, Quantity(2, 'm'))


def test_add3():
    q1 = Quantity(1.0)
    q2 = Quantity(1.0, 'km')
    assert_equal_subclass(q1 + q2, Quantity(2, 'km'))


def test_add4():
    q1 = Quantity(1.0, 'm')
    q2 = Quantity(1.0, 'km')
    assert_equal_subclass(q1 + q2, Quantity(1001, 'm'))


def test_add5():
    q1 = Quantity(1.0, 'km')
    q2 = Quantity(1.0, 'm')
    assert_equal_subclass(q1 + q2, Quantity(1.001, 'km'))


def test_add6():
    with pytest.raises(UnitError):
        Quantity(1, 'kloug') + Quantity(3.0, 'babar')


def check_unit_sub(x, y, v, u):
    c = x - y
    assert_quantity(c, v, u)


@pytest.mark.parametrize('other', [Quantity(1.0), np.array(1.0), 1])
def test_sub1(other):
    q = Quantity(1.0)
    assert_equal_subclass(q - other, Quantity(0.0))
    assert_equal_subclass(other - q, Quantity(0.0))


@pytest.mark.parametrize('other', [Quantity(1.0), np.array(1.0), 1])
def test_sub2(other):
    q = Quantity(1.0, 'm')
    assert_equal_subclass(q - other, Quantity(0.0, 'm'))
    assert_equal_subclass(other - q, Quantity(0.0, 'm'))


def test_sub3():
    q1 = Quantity(1.0)
    q2 = Quantity(1.0, 'km')
    assert_equal_subclass(q1 - q2, Quantity(0.0, 'km'))


def test_sub4():
    q1 = Quantity(1.0, 'm')
    q2 = Quantity(1.0, 'km')
    assert_equal_subclass(q1 - q2, Quantity(-999.0, 'm'))


def test_sub5():
    q1 = Quantity(1.0, 'km')
    q2 = Quantity(1.0, 'm')
    assert_equal_subclass(q1 - q2, Quantity(0.999, 'km'))


def test_sub6():
    with pytest.raises(UnitError):
        Quantity(1, 'kloug') - Quantity(3.0, 'babar')


def test_conversion_error():
    a = Quantity(1.0, 'kloug')
    with pytest.raises(UnitError):
        a.inunit('notakloug')
    with pytest.raises(UnitError):
        a.inunit('kloug^2')


def test_conversion_sr1():
    a = Quantity(1, 'MJy/sr').tounit('uJy/arcsec^2')
    assert_almost_equal(a, 23.5044305391)
    assert a.unit == 'uJy / arcsec^2'


def test_conversion_sr2():
    a = (Quantity(1, 'MJy/sr') / Quantity(1, 'uJy/arcsec^2')).SI
    assert_almost_equal(a, 23.5044305391)
    assert a.unit == ''


def test_array_prepare():
    assert Quantity(10, 'm') <= Quantity(1, 'km')
    assert Quantity(10, 'm') < Quantity(1, 'km')
    assert Quantity(1, 'km') >= Quantity(10, 'm')
    assert Quantity(1, 'km') > Quantity(10, 'm')
    assert Quantity(1, 'km') != Quantity(1, 'm')
    assert Quantity(1, 'km') == Quantity(1000, 'm')
    assert np.maximum(Quantity(10, 'm'), Quantity(1, 'km')) == 1000
    assert np.minimum(Quantity(10, 'm'), Quantity(1, 'km')) == 10


@pytest.mark.parametrize('array', [[1.3, 2, 3], [[1.0, 2, 3], [4, 5, 6]]])
@pytest.mark.parametrize(
    'func', [np.min, np.max, np.mean, np.ptp, np.round, np.sum, np.std, np.var]
)
def test_function(array, func):
    a = Quantity(array, unit='Jy')
    b = func(a)
    if func not in (np.round,):
        assert np.isscalar(b)
    assert_array_equal(b, func(a.view(np.ndarray)))
    if func in (np.round,) or a.ndim == 1:
        return
    b = func(a, axis=0)
    assert_array_equal(b, func(a.view(np.ndarray), axis=0))
    assert b.unit == 'Jy' if func is not np.var else 'Jy^2'


@pytest.mark.parametrize(
    'value, expected_type',
    [
        (Quantity(1), float),
        (Quantity(1, dtype='float32'), np.float32),
        (Quantity(1.0), float),
        (Quantity(complex(1, 0)), np.complex128),
        (Quantity(1.0, dtype=np.complex64), np.complex64),
        (Quantity(1.0, dtype=np.complex128), np.complex128),
        (Quantity(np.array(complex(1, 0))), complex),
        (Quantity(np.array(np.complex64(1.0))), np.complex64),
        (Quantity(np.array(np.complex128(1.0))), complex),
    ]
    + [
        (Quantity(1.0, dtype=np.complex256), np.complex256),
    ]
    if hasattr(np, 'complex256')
    else [],
)
def test_dtype(value, expected_type):
    assert value.dtype == expected_type


def test_derived_units1():
    du = {
        'detector[rightward]': Quantity([1, 1 / 10.0], 'm^2'),
        'time[leftward]': Quantity([1, 1 / 2, 1 / 3, 1 / 4], 's'),
    }
    a = Quantity([[1, 2, 3, 4], [10, 20, 30, 40]], 'detector', derived_units=du)
    assert_equal_subclass(a.SI, Quantity([[1, 2, 3, 4], [1, 2, 3, 4]], 'm^2', du))

    a = Quantity([[1, 2, 3, 4], [10, 20, 30, 40]], 'time', derived_units=du)
    assert_equal_subclass(a.SI, Quantity([[1, 1, 1, 1], [10, 10, 10, 10]], 's', du))

    a = Quantity(1, 'detector C', du)
    assert a.SI.shape == (2,)

    a = Quantity(np.ones((1, 10)), 'detector', derived_units=du)
    assert a.SI.shape == (2, 10)


def test_derived_units2():
    a = Quantity(4.0, 'Jy/detector', {'detector': Quantity(2, 'arcsec^2')})
    a.inunit(a.unit + ' / arcsec^2 * detector')
    assert_quantity(a, 2, 'Jy / arcsec^2')


def test_derived_units3():
    a = Quantity(
        2.0,
        'brou',
        {
            'brou': Quantity(2.0, 'bra'),
            'bra': Quantity(2.0, 'bri'),
            'bri': Quantity(2.0, 'bro'),
            'bro': Quantity(2, 'bru'),
            'bru': Quantity(2.0, 'stop'),
        },
    )
    b = a.tounit('bra')
    assert b.magnitude == 4
    b = a.tounit('bri')
    assert b.magnitude == 8
    b = a.tounit('bro')
    assert b.magnitude == 16
    b = a.tounit('bru')
    assert b.magnitude == 32
    b = a.tounit('stop')
    assert b.magnitude == 64
    b = a.SI
    assert b.magnitude == 64
    assert b.unit == 'stop'


_ = slice(None)
DERIVED_UNITS_KEYS = [
    1,
    (1,),
    (1, 2),
    (1, 2, 3),
    (1, 2, 3, 4),
    _,
    (_,),
    (_, 1),
    (_, 1, 2),
    (_, 1, 2, 3),
    (1, _),
    (1, _, 2),
    (1, _, 2, 3),
    (1, 2, _),
    (1, 2, _, 3),
    (1, 2, 3, _),
    (_, _),
    (_, _, 1),
    (_, 1, _),
    (1, _, _),
    (_, _, 1, 2),
    (_, 1, _, 2),
    (_, 1, 2, _),
    (1, _, _, 2),
    (1, _, 2, _),
    (1, 2, _, _),
    (_, _, _),
    (_, _, _, 1),
    (_, _, 1, _),
    (_, 1, _, _),
    (1, _, _, _),
    (_, _, _, _),
    Ellipsis,
]


@pytest.mark.parametrize('key', DERIVED_UNITS_KEYS)
def test_derived_units_leftward(key):
    l = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    a_leftward = Quantity.ones((2, 3, 4, 5), 'r', {'r[leftward]': Quantity(l)})

    b = a_leftward[key]
    if np.isscalar(b):
        assert b.dtype == np.float64
        return
    if key is Ellipsis:
        assert_eq(
            b.derived_units['r[leftward]'], a_leftward.derived_units['r[leftward]']
        )
    if key in ((1, 2, 3, 4), (_, 1, 2, 3)):
        assert 'r' in b.derived_units
        if key == (1, 2, 3, 4):
            assert b.derived_units['r'] == Quantity(59)
        else:
            assert b.derived_units['r'] == Quantity(33)
    else:
        assert 'r' not in b.derived_units
        if not isinstance(key, tuple):
            key = (key,)
        key += (4 - len(key)) * (slice(None),)
        assert_eq(
            b.derived_units['r[leftward]'],
            a_leftward.derived_units['r[leftward]'][key[-3:]],
        )


@pytest.mark.parametrize('key', DERIVED_UNITS_KEYS)
def test_derived_units_rightward(key):
    r = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    a_rightward = Quantity.ones((2, 3, 4, 5), 'r', {'r[rightward]': Quantity(r)})

    b = a_rightward[key]
    if np.isscalar(b):
        assert b.dtype == np.float64
        return
    if key is Ellipsis:
        assert_eq(
            b.derived_units['r[rightward]'], a_rightward.derived_units['r[rightward]']
        )
    if key in ((1, 2, 3), (1, 2, 3, 4), (1, 2, 3, _)):
        assert 'r' in b.derived_units
        assert b.derived_units['r'] == Quantity(23)
    else:
        assert 'r' not in b.derived_units
        if not isinstance(key, tuple):
            key = (key,)
        assert_eq(
            b.derived_units['r[rightward]'],
            a_rightward.derived_units['r[rightward]'][key[:3]],
        )


def test_pixels():
    actual = Quantity(1, 'pixel/sr/pixel_reference').SI
    expected = Quantity(1, 'sr^-1')
    assert_equal_subclass(actual, expected)
