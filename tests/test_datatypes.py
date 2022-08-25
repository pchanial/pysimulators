import pickle

import astropy.io.fits as fits
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from pysimulators import FitsArray, Map, Quantity, Tod, create_fitsheader

TYPES = [Quantity, FitsArray, Map, Tod]
KEYWORDS_Q = {'unit': 'km', 'derived_units': {'km': Quantity(1000, 'm')}}
KEYWORDS_F = KEYWORDS_Q.copy()
KEYWORDS_F.update({'header': fits.Header({'NAXIS': 0})})
KEYWORDS_M = KEYWORDS_F.copy()
KEYWORDS_M.update({'coverage': 8, 'error': 1.0, 'origin': 'upper'})
KEYWORDS_T = KEYWORDS_F.copy()
KEYWORDS_T.update({'mask': True})
KEYWORDS = [KEYWORDS_Q, KEYWORDS_F, KEYWORDS_M, KEYWORDS_T]
DTYPES = [bool, int, np.float32, np.float64, np.complex64, np.complex128]


def get_datatypes_objects():
    a = np.ones((4, 3))
    a[1, 2] = 4

    q = Quantity(a, unit='myunit', derived_units={'myunit': Quantity(2.0, 'Jy')})

    header = create_fitsheader(fromdata=q, cdelt=0.5, crval=(4.0, 8.0))
    header['BUNIT'] = 'myunit'
    f = FitsArray(q, header=header)

    m = Map(f, origin='upper', error=a * 2, coverage=a * 3)

    mask = np.zeros((4, 3), np.bool8)
    mask[0, 2] = True
    t = Tod(f, mask=mask)
    return q, f, m, t


@pytest.mark.parametrize('obj', get_datatypes_objects())
@pytest.mark.parametrize('type_', TYPES)
def test_copy_false_subok_true(obj, type_):
    other = type_(obj, copy=False, subok=True)
    if isinstance(obj, type_):
        assert obj is other
    else:
        assert obj is not other
        assert_equal(obj, other)


@pytest.mark.parametrize('obj', get_datatypes_objects())
@pytest.mark.parametrize('type_', TYPES)
def test_copy_false_subok_false(obj, type_):
    other = type_(obj, copy=False, subok=False)
    if type(obj) is type_:
        assert obj is other
    else:
        assert obj is not other
        assert_equal(obj, other)


@pytest.mark.parametrize('obj', get_datatypes_objects())
@pytest.mark.parametrize('type_', TYPES)
def test_copy_true_subok_true(obj, type_):
    other = type_(obj, copy=True, subok=True)
    assert obj is not other
    assert_equal(obj, other)
    if isinstance(obj, type_):
        assert type(other) is type(obj)
    else:
        assert type(other) is type_


@pytest.mark.parametrize('obj', get_datatypes_objects())
@pytest.mark.parametrize('type_', TYPES)
def test_copy_true_subok_false(obj, type_):
    other = type_(obj, copy=True, subok=False)
    assert obj is not other
    assert_equal(obj, other)
    assert type(other) is type_


@pytest.mark.parametrize('type_', TYPES)
@pytest.mark.parametrize('input', [(2,), [2], np.array([2])])
def test_input1(type_, input):
    d = type_(input)
    assert d.shape == (1,)
    assert d[0] == 2


@pytest.mark.parametrize('type_', TYPES)
@pytest.mark.parametrize('input', [2, np.array(2)])
def test_input2(type_, input):
    d = type_(input)
    assert d.shape == ()
    assert d[...] == 2


@pytest.mark.parametrize('type_', TYPES)
def test_input3(type_):
    d = type_([])
    assert d.shape == (0,)


@pytest.mark.parametrize(
    'type_, attrs',
    [
        (Quantity, ['unit', 'derived_units']),
        (FitsArray, ['unit', 'derived_units', 'header']),
        (Map, ['unit', 'derived_units', 'header', 'coverage', 'error', 'origin']),
        (Tod, ['unit', 'derived_units', 'header', 'mask']),
    ],
)
def test_view(type_, attrs):

    data = np.ones((2, 4)).view(type_)
    for attr in attrs:
        assert hasattr(data, attr)


@pytest.mark.parametrize('type_', TYPES)
def test_operation(type_):
    a = type_.ones(10)
    assert type(a + 1) is type_


@pytest.mark.parametrize('type_', TYPES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_dtype1(type_, dtype):
    a = type_(np.array([10, 20], dtype=dtype))
    assert a.dtype == (dtype if dtype != int else float)


@pytest.mark.parametrize('type_', TYPES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_dtype2(type_, dtype):
    a = type_(np.array([10, 20], dtype=dtype), dtype=dtype)
    assert a.dtype == dtype


@pytest.mark.parametrize('type_', TYPES)
@pytest.mark.parametrize('dtype', DTYPES)
def test_dtype3(type_, dtype):
    a = type_(np.ones((4, 3)), dtype=dtype, copy=False)
    assert a.dtype == dtype


@pytest.mark.parametrize('type_, keywords', zip(TYPES, KEYWORDS))
@pytest.mark.parametrize('method', ['empty', 'ones', 'zeros'])
@pytest.mark.parametrize('dtype', [None, *DTYPES])
def test_empty_ones_zeros(type_, keywords, method, dtype):
    method = getattr(type_, method)
    a = method((2, 3), dtype=dtype, **keywords)
    assert a.dtype == type_.default_dtype if dtype is None else dtype
    for k_, v in keywords.items():
        assert getattr(a, k_) == v


def test_map():
    class MAP(Map):
        def __init__(self, data):
            self.info = 'info'

    m = MAP(np.ones(3))
    assert sorted(m.__dict__) == [
        '_derived_units',
        '_header',
        '_unit',
        'coverage',
        'error',
        'info',
        'origin',
    ]


@pytest.mark.parametrize('obj', get_datatypes_objects())
@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL))
def test_pickling(obj, protocol):
    actual = pickle.loads(pickle.dumps(obj, protocol))
    assert_equal(obj, actual)


@pytest.mark.parametrize('obj', get_datatypes_objects())
def test_save(tmp_path, obj):
    if not isinstance(obj, FitsArray):
        assert not hasattr(obj, 'save')
        return

    filename = str(tmp_path / 'obj.fits')
    obj.save(filename)
    actual = type(obj)(filename)
    assert_equal(actual, obj)


@pytest.mark.parametrize('cls', [Quantity, FitsArray, Map, Tod])
@pytest.mark.parametrize(
    'func, expected_unit_power',
    [
        (np.min, 1),
        (np.max, 1),
        (np.sum, 1),
        (np.mean, 1),
        (np.ptp, 1),
        (np.std, 1),
        (np.var, 2),
    ],
)
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_ndarray_reduction(cls, func, expected_unit_power, axis):
    data = [[1.0, 2, 3, 4], [5, 6, 7, 8]]

    array = cls(data, unit='u')
    result = func(array, axis=axis)

    if cls in {Quantity, FitsArray} and axis is None:
        assert type(result) is not cls
        pytest.xfail(f'{func.__name__}({cls.__name__}(...)) is not a {cls.__name__}.')

    assert type(result) is cls
    assert 'u' in result._unit
    assert result._unit['u'] == expected_unit_power

    ref = func(data, axis=axis)
    assert_allclose(result.view(np.ndarray), ref)

    if cls is Map:
        assert result.coverage is None
        assert result.error is None

    elif cls is Tod:
        if axis in {0, 1} and func not in {np.var}:
            assert result.mask is not None
            pytest.xfail(
                f'{func.__name__}({cls.__name__}(..., axis={axis}, mask=None)).mask is not None'
            )
        assert result.mask is None


@pytest.mark.parametrize(
    'func, expected_unit_power',
    [
        (np.min, 1),
        (np.max, 1),
        (np.sum, 1),
        (np.mean, 1),
        (np.ptp, 1),
        (np.std, 1),
        (np.var, 2),
    ],
)
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_ndarray_reduction_tod(func, expected_unit_power, axis):
    data = [[1.0, 2, 3, 4], [5, 6, 7, 8]]
    mask = [[True, False, False, False], [False, True, True, False]]

    array = Tod(data, mask=mask, unit='u')
    result = func(array, axis=axis)
    ref = func(np.ma.MaskedArray(data, mask=mask), axis=axis)
    if not isinstance(ref, np.ndarray):
        ref = np.ma.MaskedArray(ref)
    assert_allclose(result.view(np.ndarray), ref.view(np.ndarray))

    assert 'u' in result._unit
    assert result._unit['u'] == expected_unit_power

    if result.mask is None and ref.mask is not None:
        pytest.xfail('Tod should emulate MaskedArrays.')
    assert result.mask is not None
    assert result.mask.shape == ref.mask.shape
    assert_equal(result.mask, ref.mask)


@pytest.mark.parametrize('cls', [Quantity, FitsArray, Map, Tod])
@pytest.mark.parametrize(
    'func, expected_unit',
    [
        (np.round, {'u': 1}),
        (np.exp, {}),
        (np.square, {'u': 2}),
        (np.reciprocal, {'u': -1}),
    ],
)
def test_ndarray_elementwise(cls, func, expected_unit):
    data = [[1.0, 2, 3, 4], [5, 6, 7, 8]]

    array = cls(data, unit='u')
    result = func(array)
    assert type(result) is cls

    ref = func(data)
    assert_allclose(result.view(np.ndarray), ref.view(np.ndarray))

    assert result._unit == expected_unit

    if cls is Map:
        assert result.coverage is None
        assert result.error is None

    if cls is Tod:
        if func is np.round:
            assert result.mask is not None
            pytest.xfail('tod.mask is not None')
        assert result.mask is None


@pytest.mark.parametrize(
    'func, expected_unit',
    [
        (np.round, {'u': 1}),
        (np.exp, {}),
        (np.square, {'u': 2}),
        (np.reciprocal, {'u': -1}),
    ],
)
def test_ndarray_elementwise_tod(func, expected_unit):
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mask = [[True, False, False, False], [False, True, True, False]]

    array = Tod(data, mask=mask, unit='u')
    result = func(array)
    assert type(result) is Tod

    ref = func(array.view(np.ndarray))
    assert_allclose(result.view(np.ndarray), ref)

    assert result._unit == expected_unit

    assert result.mask is not None
    assert result.mask.shape == (2, 4)
    assert_equal(result.mask, mask)


@pytest.mark.parametrize(
    'dtype1', [np.int8, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize(
    'dtype2', [np.int8, np.int32, np.int64, np.float32, np.float64]
)
def test_astype(dtype1, dtype2):
    m = Map(
        np.array([1, 2, 3], dtype=dtype1),
        coverage=np.array([0, 1, 0], dtype=dtype1),
        error=np.array([2, 2, 2], dtype=dtype1),
    )
    m2 = m.astype(dtype2)
    assert m2.dtype == dtype2
    assert m2.coverage.dtype == dtype2
    assert m2.error.dtype == dtype2
