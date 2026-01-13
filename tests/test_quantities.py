from typing import Any

import numpy as np
import pytest
from numpy.lib import NumpyVersion
from numpy.testing import assert_almost_equal, assert_array_equal

from pyoperators.utils.testing import assert_eq
from pysimulators.quantities import Quantity, UnitError

from .common import assert_equal_subclass


def assert_quantity_equal(q1: Any, q2: Any):
    assert isinstance(q1, Quantity)
    assert type(q1) is type(q2)
    assert_array_equal(q1.magnitude, q2.magnitude)
    assert q1._unit == q2._unit


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


def test_array_ufunc():
    assert Quantity(10, 'm') <= Quantity(1, 'km')
    assert Quantity(10, 'm') < Quantity(1, 'km')
    assert Quantity(1, 'km') >= Quantity(10, 'm')
    assert Quantity(1, 'km') > Quantity(10, 'm')
    assert Quantity(1, 'km') != Quantity(1, 'm')
    assert Quantity(1, 'km') == Quantity(1000, 'm')
    assert np.maximum(Quantity(10, 'm'), Quantity(1, 'km')) == 1000
    assert np.minimum(Quantity(10, 'm'), Quantity(1, 'km')) == 10


@pytest.mark.parametrize(
    'func, args, keywords',
    [
        (np.amax, (), {}),
        (np.amin, (), {}),
        (np.around, (), {}),
        (np.atleast_1d, (), {}),
        (np.atleast_2d, (), {}),
        (np.atleast_3d, (), {}),
        (np.average, (), {}),
        (np.broadcast_to, ((3, 2, 2),), {}),
        (np.clip, (1, 2), {}),
        (np.copy, (), {}),
        (np.cumsum, (), {}),
        (getattr(np, 'cumulative_sum', 'cumulative_sum'), (), {'axis': 1}),
        (np.delete, ([0],), {}),
        (np.diag, (), {}),
        (np.diagflat, (), {}),
        (np.diagonal, (), {}),
        (np.diff, (), {}),
        (np.expand_dims, (), {'axis': 0}),
        (np.fix, (), {}),
        (np.flip, (), {}),
        (np.fliplr, (), {}),
        (np.flipud, (), {}),
        (np.imag, (), {}),
        (np.interp, ([1, 2], [1, 1.5]), {}),
        (np.max, (), {}),
        (np.mean, (), {}),
        (np.median, (), {}),
        (np.min, (), {}),
        (np.moveaxis, (0, 1), {}),
        (np.nan_to_num, (), {}),
        (np.nancumsum, (), {}),
        (np.nanmax, (), {}),
        (np.nanmean, (), {}),
        (np.nanmedian, (), {}),
        (np.nanmin, (), {}),
        (np.nanstd, (), {}),
        (np.nansum, (), {}),
        (np.pad, ([2, 3],), {}),
        (np.partition, (1,), {}),
        (np.permute_dims, (), {}),
        (np.ptp, (), {}),
        (np.ravel, (), {}),
        (np.real, (), {}),
        (np.real_if_close, (), {}),
        (np.repeat, (2,), {}),
        (np.reshape, ((4, 1),), {}),
        (np.resize, ((3, 3),), {}),
        (np.roll, (1,), {}),
        (np.rollaxis, (), {'axis': 1}),
        (np.rot90, (), {}),
        (np.round, (), {}),
        (np.sort, (), {}),
        (np.sort_complex, (), {}),
        (np.squeeze, (), {}),
        (np.std, (), {}),
        (np.sum, (), {}),
        (np.swapaxes, (0, 1), {}),
        (np.take, ([0, 0]), {}),
        (np.take_along_axis, (), {'axis': 0, 'indices': np.array([[0]])}),
        (np.tile, (2,), {}),
        (np.trace, (), {}),
        (np.transpose, (), {}),
        (np.trim_zeros, (), {}),
        (np.unique_values, (), {}),
        (np.unwrap, (), {}),
    ],
)
def test_function_default_conversion(func, args, keywords) -> None:
    if isinstance(func, str):
        pytest.skip(f'Function {func!r} unavailable for Numpy {np.__version__}')
    input = Quantity([[1, 2], [3, 4]], 'm')
    if NumpyVersion(np.__version__) < '2.2.0' and func is np.trim_zeros:
        input = input.ravel()

    actual_output = func(input, *args, **keywords)
    assert type(input) is Quantity
    assert_array_equal(
        actual_output.magnitude, func(input.view(np.ndarray), *args, **keywords)
    )
    assert actual_output._unit == {'m': 1.0}


@pytest.mark.parametrize(
    'func, inputs, expected_output',
    [
        (
            np.append,
            [Quantity([1, 2], 'm'), Quantity(1, 'km')],
            Quantity([1, 2, 1000], 'm'),
        ),
        (
            np.intersect1d,
            [Quantity([1, 2, 1000], 'm'), Quantity([1, 2], 'km')],
            Quantity([1000], 'm'),
        ),
        (
            np.setdiff1d,
            [Quantity([1, 2, 1000], 'm'), Quantity([1, 2], 'km')],
            Quantity([1, 2], 'm'),
        ),
        (
            np.setxor1d,
            [Quantity([1, 2, 1000], 'm'), Quantity([1, 2], 'km')],
            Quantity([1, 2, 2000], 'm'),
        ),
        (
            np.union1d,
            [Quantity([1, 2, 1000], 'm'), Quantity([1, 2], 'km')],
            Quantity([1, 2, 1000, 2000], 'm'),
        ),
    ],
)
def test_function_homogenize1(func, inputs, expected_output):
    actual_output = func(*inputs)
    assert_quantity_equal(actual_output, expected_output)


def test_function_fill_diagonal():
    array = Quantity.zeros((2, 2), 'm')
    diagonal = Quantity.ones(2, 'km')
    np.fill_diagonal(array, diagonal)
    assert_quantity_equal(array, Quantity([[1000, 0], [0, 1000]], 'm'))

    array = np.zeros((2, 2))
    diagonal = Quantity.ones(2, 'km')
    np.fill_diagonal(array, diagonal)
    assert_array_equal(array, np.array([[1, 0], [0, 1]]))


@pytest.mark.filterwarnings('ignore:`row_stack` alias is deprecated')
@pytest.mark.parametrize(
    'func, q1, q2, expected_output',
    [
        (
            np.column_stack,
            Quantity([1, 2], 'm'),
            Quantity([2, 3], 'km'),
            Quantity([[1, 2000], [2, 3000]], 'm'),
        ),
        (
            np.concat,
            Quantity([1, 2], 'm'),
            Quantity([3, 4], 'km'),
            Quantity([1, 2, 3000, 4000], 'm'),
        ),
        (
            np.concatenate,
            Quantity([1, 2], 'm'),
            Quantity([3, 4], 'km'),
            Quantity([1, 2, 3000, 4000], 'm'),
        ),
        (
            np.dstack,
            Quantity([[1], [2], [3]], 'm'),
            Quantity([[1], [2], [3]], 'km'),
            Quantity([[[1, 1000]], [[2, 2000]], [[3, 3000]]], 'm'),
        ),
        (
            np.hstack,
            Quantity([[1], [2]], 'm'),
            Quantity([[2], [3]], 'km'),
            Quantity([[1, 2000], [2, 3000]], 'm'),
        ),
        (
            np.row_stack,
            Quantity([1, 2], 'm'),
            Quantity([2, 3], 'km'),
            Quantity([[1, 2], [2000, 3000]], 'm'),
        ),
        (
            np.stack,
            Quantity([1], 'm'),
            Quantity([1], 'km'),
            Quantity([[1], [1000]], 'm'),
        ),
        (
            np.vstack,
            Quantity([1, 2], 'm'),
            Quantity([2, 3], 'km'),
            Quantity([[1, 2], [2000, 3000]], 'm'),
        ),
    ],
)
def test_function_homogenize2(func, q1, q2, expected_output):
    actual_output = func([q1, q2])
    assert_quantity_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    'func, array, args',
    [
        (np.array_split, Quantity([[1, 2], [3, 4]], 'm'), (2,)),
        (np.dsplit, Quantity([[[1, 2], [3, 4]]], 'm'), (2,)),
        (np.hsplit, Quantity([[1, 2], [3, 4]], 'm'), (2,)),
        (np.split, Quantity([[1, 2], [3, 4]], 'm'), (2,)),
        (np.vsplit, Quantity([[1, 2], [3, 4]], 'm'), (2,)),
        (getattr(np, 'unstack', 'unstack'), Quantity([[1, 2], [3, 4]], 'm'), ()),
    ],
)
def test_function_output_sequence(func, array, args):
    if isinstance(func, str):
        pytest.skip(f'Function {func!r} unavailable for Numpy {np.__version__}')
    actual_outputs = func(array, *args)
    expected_outputs = func(array.view(np.ndarray), *args)
    assert isinstance(actual_outputs, (list, tuple))
    assert len(actual_outputs) == len(expected_outputs)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


@pytest.mark.parametrize(
    'func, keywords',
    [
        (np.convolve, {}),
        (np.correlate, {}),
        (np.cov, {}),
        (np.cross, {}),
        (np.dot, {}),
        (np.inner, {}),
        (np.kron, {}),
        (np.outer, {}),
        (np.tensordot, {'axes': 0}),
        (np.vdot, {}),
    ],
)
def test_function_action_prod(func, keywords):
    array = Quantity([1, 2, 3], 'm')
    actual_output = func(array, array, **keywords)
    expected_output = func(array.view(np.ndarray), array.view(np.ndarray), **keywords)
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm^2'))


@pytest.mark.parametrize(
    'func', [np.min, np.max, np.mean, np.ptp, np.sum, np.std, np.var]
)
def test_function_with_axis(func):
    a = Quantity([[1.0, 2, 3], [4, 5, 6]], unit='Jy')
    actual_output = func(a, axis=0)
    expected_unit = 'Jy^2' if func is np.var else 'Jy'
    expected_output = Quantity(func(a.view(np.ndarray), axis=0), expected_unit)
    assert_quantity_equal(actual_output, expected_output)


def test_function_broadcast_arrays():
    q1 = Quantity([1, 2], 'm')
    q2 = Quantity([[1], [2], [3]], 's')
    actual_outputs = np.broadcast_arrays(q1, q2)
    expected_outputs = np.broadcast_arrays(q1.view(np.ndarray), q2.view(np.ndarray))
    assert isinstance(actual_outputs, (list, tuple))
    assert len(actual_outputs) == 2
    assert_quantity_equal(actual_outputs[0], Quantity(expected_outputs[0], 'm'))
    assert_quantity_equal(actual_outputs[1], Quantity(expected_outputs[1], 's'))


def test_function_choose():
    choices = [
        Quantity([1, 2, 3], 'm'),
        Quantity([4, 5, 6], 'km'),
        Quantity([7, 8, 9], 'm'),
    ]
    indices = np.array([0, 1, 2])
    actual_output = np.choose(indices, choices)
    # Should homogenize to 'm' (first non-empty unit) and convert km to m
    homogenized_choices = [
        choices[0].view(np.ndarray),
        choices[1].tounit('m').view(np.ndarray),
        choices[2].view(np.ndarray),
    ]
    expected_output = np.choose(indices, homogenized_choices)
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


@pytest.mark.parametrize('func', [np.compress, np.extract])
def test_function_condition_second(func):
    condition = [True, False, True]
    array = Quantity([1, 2, 3], 'm')
    actual_output = func(condition, array)
    expected_output = func(condition, array.view(np.ndarray))
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


def test_function_einsum():
    a = Quantity([1, 2], 'm')
    b = Quantity([3, 4], 's')
    actual_output = np.einsum('i,i->', a, b)
    expected_output = np.einsum('i,i->', a.view(np.ndarray), b.view(np.ndarray))
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm s'))


def test_function_insert():
    array = Quantity([1, 2, 3], 'm')
    values = Quantity([100, 200], 'km')
    actual_output = np.insert(array, 1, values)
    expected_output = np.insert(
        array.view(np.ndarray), 1, values.tounit('m').view(np.ndarray)
    )
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


@pytest.mark.parametrize('func', [np.var, np.nanvar])
def test_function_action_square(func):
    array = Quantity([1, 2, 3], 'm')
    actual_output = func(array)
    expected_output = func(array.view(np.ndarray))
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm^2'))


def test_function_select():
    condlist = [np.array([True, False, False]), np.array([False, True, False])]
    choicelist = [Quantity([1, 2, 3], 'm'), Quantity([4, 5, 6], 'km')]
    actual_output = np.select(condlist, choicelist, default=0)
    expected_output = np.select(
        condlist,
        [choicelist[0].view(np.ndarray), choicelist[1].tounit('m').view(np.ndarray)],
        default=0,
    )
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


def test_function_unique():
    array = Quantity([1, 2, 1, 3, 2], 'm')
    # Test with return_index=False (returns single array)
    actual_output = np.unique(array)
    expected_output = np.unique(array.view(np.ndarray))
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))

    # Test with return_index=True (returns tuple)
    actual_output, actual_indices = np.unique(array, return_index=True)
    expected_output, expected_indices = np.unique(
        array.view(np.ndarray), return_index=True
    )
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))
    assert_array_equal(actual_indices, expected_indices)


def test_function_unique_all():
    array = Quantity([1, 2, 1, 3, 2], 'm')
    result = np.unique_all(array)

    assert isinstance(result, np.lib._arraysetops_impl.UniqueAllResult)
    assert type(result.values) is Quantity
    assert type(result.indices) is np.ndarray
    assert type(result.inverse_indices) is np.ndarray
    assert type(result.counts) is np.ndarray

    expected = np.unique_all(array.view(np.ndarray))
    assert_quantity_equal(result.values, Quantity(expected.values, 'm'))
    assert_array_equal(result.indices, expected.indices)
    assert_array_equal(result.inverse_indices, expected.inverse_indices)
    assert_array_equal(result.counts, expected.counts)


def test_function_where():
    condition = np.array([True, False, True])
    x = Quantity([1, 2, 3], 'm')
    y = Quantity([4, 5, 6], 'km')
    actual_output = np.where(condition, x, y)
    expected_output = np.where(
        condition, x.view(np.ndarray), y.tounit('m').view(np.ndarray)
    )
    assert_quantity_equal(actual_output, Quantity(expected_output, 'm'))


@pytest.mark.parametrize(
    'value, expected_type',
    (
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
        else []
    ),
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
    assert_quantity_equal(a, Quantity(2, 'Jy / arcsec^2'))


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
