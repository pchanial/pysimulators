from __future__ import division

import numpy as np

from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_almost_equal, assert_array_equal
from pyoperators.utils.testing import assert_eq, assert_is, assert_is_instance
from pysimulators.quantities import Quantity, UnitError


def assert_quantity(q, m, u):
    assert_is_instance(q, Quantity)
    assert_array_equal(q.magnitude, m)
    if isinstance(u, dict):
        assert_equal(q._unit, u)
    else:
        assert_equal(q.unit, u)


def test1():
    q = Quantity(1, 'km')
    assert_eq(q.SI, Quantity(1000, 'm'))

    q = Quantity(1, 'm')
    assert_eq(q, q.tounit('m'))
    q2 = q.copy()
    q.inunit('m')
    assert_eq(q, q2)


def test2():
    q = Quantity(1, 'km').tounit('m')
    assert_quantity(q, 1000, 'm')


def test3():
    q = Quantity(1, 'km')
    q.inunit('m')
    assert_quantity(q, 1000, 'm')


def check_unit_add(x, y, v, u):
    c = x + y
    assert_quantity(c, v, u)


def test_add1():
    q = Quantity(1.)
    for other in Quantity(1.), np.array(1.), 1:
        yield check_unit_add, q, other, 2, {}
        yield check_unit_add, other, q, 2, {}


def test_add2():
    q = Quantity(1., 'm')
    for other in Quantity(1.), np.array(1.), 1:
        yield check_unit_add, q, other, 2, {'m': 1.0}
        yield check_unit_add, other, q, 2, {'m': 1.0}


def test_add3():
    q = Quantity(1.)
    q2 = Quantity(1., 'km')
    yield check_unit_add, q, q2, 2, {'km': 1.0}


def test_add4():
    q = Quantity(1., 'm')
    q2 = Quantity(1., 'km')
    yield check_unit_add, q, q2, 1001, {'m': 1.0}


def test_add5():
    q = Quantity(1., 'km')
    q2 = Quantity(1., 'm')
    yield check_unit_add, q, q2, 1.001, {'km': 1.0}


def test_add6():
    assert_raises(UnitError, lambda: Quantity(1, 'kloug') +
                  Quantity(3., 'babar'))


def check_unit_sub(x, y, v, u):
    c = x - y
    assert_quantity(c, v, u)


def test_sub1():
    q = Quantity(1.)
    for other in Quantity(1.), np.array(1.), 1:
        yield check_unit_sub, q, other, 0, {}
        yield check_unit_sub, other, q, 0, {}


def test_sub2():
    q = Quantity(1., 'm')
    for other in Quantity(1.), np.array(1.), 1:
        yield check_unit_sub, q, other, 0, {'m': 1.0}
        yield check_unit_sub, other, q, 0, {'m': 1.0}


def test_sub3():
    q = Quantity(1.)
    q2 = Quantity(1., 'km')
    yield check_unit_sub, q, q2, 0, {'km': 1.0}


def test_sub4():
    q = Quantity(1., 'm')
    q2 = Quantity(1., 'km')
    yield check_unit_sub, q, q2, -999, {'m': 1.0}


def test_sub5():
    q = Quantity(1., 'km')
    q2 = Quantity(1., 'm')
    yield check_unit_sub, q, q2, 0.999, {'km': 1.0}


def test_sub6():
    assert_raises(UnitError, lambda: Quantity(1, 'kloug') -
                  Quantity(3., 'babar'))


def test_conversion_error():
    a = Quantity(1., 'kloug')
    assert_raises(UnitError, lambda: a.inunit('notakloug'))
    assert_raises(UnitError, lambda: a.inunit('kloug^2'))


def test_conversion_sr1():
    a = Quantity(1, 'MJy/sr').tounit('uJy/arcsec^2')
    assert_almost_equal(a, 23.5044305391)
    assert_equal(a.unit, 'uJy / arcsec^2')


def test_conversion_sr2():
    a = (Quantity(1, 'MJy/sr')/Quantity(1, 'uJy/arcsec^2')).SI
    assert_almost_equal(a, 23.5044305391)
    assert_equal(a.unit, '')


def test_array_prepare():
    assert Quantity(10, 'm') <= Quantity(1, 'km')
    assert Quantity(10, 'm') < Quantity(1, 'km')
    assert Quantity(1, 'km') >= Quantity(10, 'm')
    assert Quantity(1, 'km') > Quantity(10, 'm')
    assert Quantity(1, 'km') != Quantity(1, 'm')
    assert Quantity(1, 'km') == Quantity(1000, 'm')
    assert np.maximum(Quantity(10, 'm'), Quantity(1, 'km')) == 1000
    assert np.minimum(Quantity(10, 'm'), Quantity(1, 'km')) == 10


def test_function():
    arrays = ([1.3, 2, 3], [[1., 2, 3], [4, 5, 6]])

    def func(array, f):
        a = Quantity(array, unit='Jy')
        b = f(a)
        if f not in (np.round,):
            assert np.isscalar(b)
        assert_array_equal(b, f(a.view(np.ndarray)))
        if f in (np.round,) or a.ndim == 1:
            return
        b = f(a, axis=0)
        assert_array_equal(b, f(a.view(np.ndarray), axis=0))
        assert_equal(b.unit, 'Jy' if f is not np.var else 'Jy^2')
    for array in arrays:
        for f in (np.min, np.max, np.mean, np.ptp, np.round, np.sum, np.std,
                  np.var):
            yield func, array, f


def test_dtype():
    assert_is(Quantity(1).dtype, np.dtype(float))
    assert_is(Quantity(1, dtype='float32').dtype, np.dtype(np.float32))
    assert_is(Quantity(1.).dtype, np.dtype(float))
    assert_is(Quantity(complex(1, 0)).dtype, np.dtype(np.complex128))
    assert_is(Quantity(1., dtype=np.complex64).dtype, np.dtype(np.complex64))
    assert_is(Quantity(1., dtype=np.complex128).dtype, np.dtype(np.complex128))
    assert_is(Quantity(1., dtype=np.complex256).dtype, np.dtype(np.complex256))
    assert_is(Quantity(np.array(complex(1, 0))).dtype, np.dtype(complex))
    assert_is(Quantity(np.array(np.complex64(1.))).dtype,
              np.dtype(np.complex64))
    assert_is(Quantity(np.array(np.complex128(1.))).dtype, np.dtype(complex))


def test_derived_units1():
    du = {'detector[rightward]': Quantity([1, 1/10.], 'm^2'),
          'time[leftward]': Quantity([1, 1/2, 1/3, 1/4], 's')}
    a = Quantity([[1, 2, 3, 4],
                  [10, 20, 30, 40]], 'detector', derived_units=du)
    assert_almost_equal(a.SI, [[1, 2, 3, 4], [1, 2, 3, 4]])
    a = Quantity([[1, 2, 3, 4], [10, 20, 30, 40]], 'time', derived_units=du)
    assert_almost_equal(a.SI, [[1, 1, 1, 1], [10, 10, 10, 10]])
    a = Quantity(1, 'detector C', du)
    assert a.SI.shape == (2,)
    a = Quantity(np.ones((1, 10)), 'detector', derived_units=du)
    assert a.SI.shape == (2, 10)


def test_derived_units2():
    a = Quantity(4., 'Jy/detector', {'detector': Quantity(2,'arcsec^2')})
    a.inunit(a.unit + ' / arcsec^2 * detector')
    assert_quantity(a, 2, 'Jy / arcsec^2')


def test_derived_units3():
    a = Quantity(2., 'brou', {'brou': Quantity(2., 'bra'),
                              'bra': Quantity(2., 'bri'),
                              'bri': Quantity(2., 'bro'),
                              'bro': Quantity(2, 'bru'),
                              'bru': Quantity(2., 'stop')})
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


def test_derived_units4():
    l = np.arange(3*4*5).reshape((3, 4, 5))
    a_leftward = Quantity.ones((2, 3, 4, 5), 'r', {'r[leftward]': Quantity(l)})
    r = np.arange(2*3*4).reshape((2, 3, 4))
    a_rightward = Quantity.ones((2, 3, 4, 5), 'r',
                                {'r[rightward]': Quantity(r)})
    _ = slice(None)

    def func_leftward(a, key):
        b = a[key]
        if np.isscalar(b):
            assert b.dtype == np.float64
            return
        if key is Ellipsis:
            assert_eq(b.derived_units['r[leftward]'],
                      a.derived_units['r[leftward]'])
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
            key += (4-len(key)) * (slice(None),)
            assert_eq(b.derived_units['r[leftward]'],
                      a.derived_units['r[leftward]'][key[-3:]])

    def func_rightward(a, key):
        b = a[key]
        if np.isscalar(b):
            assert b.dtype == np.float64
            return
        if key is Ellipsis:
            assert_eq(b.derived_units['r[rightward]'],
                      a.derived_units['r[rightward]'])
        if key in ((1, 2, 3), (1, 2, 3, 4), (1, 2, 3, _)):
            assert 'r' in b.derived_units
            assert b.derived_units['r'] == Quantity(23)
        else:
            assert 'r' not in b.derived_units
            if not isinstance(key, tuple):
                key = (key,)
            assert_eq(b.derived_units['r[rightward]'],
                      a.derived_units['r[rightward]'][key[:3]])
    for key in (1, (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4),
                _, (_,), (_, 1), (_, 1, 2), (_, 1, 2, 3),
                (1, _), (1, _, 2), (1, _, 2, 3),
                (1, 2, _), (1, 2, _, 3),
                (1, 2, 3, _),
                (_, _),  (_, _, 1), (_, 1, _), (1, _, _),
                (_, _, 1, 2), (_, 1, _, 2), (_, 1, 2, _), (1, _, _, 2),
                (1, _, 2, _), (1, 2, _, _),
                (_, _, _), (_, _, _, 1), (_, _, 1, _), (_, 1, _, _),
                (1, _, _, _),
                (_, _, _, _), Ellipsis):
        yield func_leftward, a_leftward, key
        yield func_rightward, a_rightward, key


def test_pixels():
    assert_quantity(Quantity(1, 'pixel/sr/pixel_reference').SI, 1, 'sr^-1')
