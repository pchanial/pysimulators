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


def assert_datatype_allclose(d1, d2) -> None:
    assert isinstance(d2, Quantity)
    cls = type(d2)
    assert type(d1) is cls
    assert_allclose(d1.magnitude, d2.magnitude)

    assert d1._unit == d2._unit
    assert d1._derived_units == d2._derived_units

    if cls is FitsArray:
        assert d1._header == d2._header

    elif cls is Map:
        if d2.coverage is None:
            assert d1.coverage is None
        else:
            assert d1.coverage.shape == d2.coverage.shape
            assert_equal(d1.coverage, d2.coverage)
        if d2.error is None:
            assert d1.error is None
        else:
            assert d1.error.shape == d2.error.shape
            assert_equal(d1.error, d2.error)

    elif cls is Tod:
        if d2.mask is None:
            assert d1.mask is None
        else:
            assert d1.mask.shape == d2.mask.shape
            assert_equal(d1.mask, d2.mask)


def get_datatypes_objects():
    a = np.ones((4, 3))
    a[1, 2] = 4

    q = Quantity(a, unit='myunit', derived_units={'myunit': Quantity(2.0, 'Jy')})

    header = create_fitsheader(fromdata=q, cdelt=0.5, crval=(4.0, 8.0))
    header['BUNIT'] = 'myunit'
    f = FitsArray(q, header=header)

    m = Map(f, origin='upper', error=a * 2, coverage=a * 3)

    mask = np.zeros((4, 3), np.bool_)
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
    a = type_(np.ones((4, 3)), dtype=dtype, copy=None)
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
    'func, expected_unit',
    [
        (np.min, {'u': 1}),
        (np.max, {'u': 1}),
        (np.sum, {'u': 1}),
        (np.mean, {'u': 1}),
        (np.ptp, {'u': 1}),
        (np.std, {'u': 1}),
        (np.var, {'u': 2}),
    ],
)
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_function_reduction(cls, func, expected_unit, axis):
    data = [[1.0, 2, 3, 4], [5, 6, 7, 8]]
    array = cls(data, unit='u')

    actual = func(array, axis=axis)

    np_output = func(data, axis=axis)
    expected = cls(np_output, unit=expected_unit)
    assert_datatype_allclose(actual, expected)


@pytest.mark.parametrize(
    'func, expected_unit',
    [
        (np.min, {'u': 1}),
        (np.max, {'u': 1}),
        (np.sum, {'u': 1}),
        (np.mean, {'u': 1}),
        (np.ptp, {'u': 1}),
        (np.std, {'u': 1}),
        (np.var, {'u': 2}),
    ],
)
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_function_reduction_tod(func, expected_unit, axis):
    data = [[1.0, 2, 3, 4], [5, 6, 7, 8]]
    mask = [[True, False, False, False], [False, True, True, False]]
    array = Tod(data, mask=mask, unit='u')

    actual = func(array, axis=axis)

    ma_output = func(np.ma.MaskedArray(data, mask=mask), axis=axis)
    if np.isscalar(ma_output):
        expected = Tod(ma_output, mask=None, unit=expected_unit)
    else:
        expected = Tod(ma_output.data, mask=ma_output.mask, unit=expected_unit)
    assert_datatype_allclose(actual, expected)


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
def test_ufunc(cls, func, expected_unit):
    array = cls([[1.0, 2, 3, 4], [5, 6, 7, 8]], unit='u')

    actual = func(array)

    output = func(array.view(np.ndarray))
    expected = cls(output, unit=expected_unit)
    assert_datatype_allclose(actual, expected)


@pytest.mark.parametrize(
    'func, expected_unit',
    [
        (np.round, {'u': 1}),
        (np.exp, {}),
        (np.square, {'u': 2}),
        (np.reciprocal, {'u': -1}),
    ],
)
def test_ufunc_tod(func, expected_unit):
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    mask = [[True, False, False, False], [False, True, True, False]]
    array = Tod(data, mask=mask, unit='u')

    actual = func(array)

    output = func(array.view(np.ndarray))
    expected = Tod(output, mask=mask, unit=expected_unit)
    assert_datatype_allclose(actual, expected)


@pytest.mark.parametrize(
    'array',
    [
        Quantity([[1.0, 2]], unit='u'),
        FitsArray([[1.0, 2]], unit='u'),
        Map([[1.0, 2]], unit='u'),
        Map([[1.0, 2]], coverage=[[0.0, 1]], error=[[0.5, 1]], unit='u'),
        Tod([[1.0, 2]], unit='u'),
        Tod([[1.0, 2]], mask=[[True, False]], unit='u'),
    ],
)
@pytest.mark.parametrize(
    'func, new_shape',
    [
        (lambda a: np.ravel(a), (2,)),
        (lambda a: a.ravel(), (2,)),
        (lambda a: np.reshape(a, (2, 1)), (2, 1)),
        (lambda a: a.reshape(2, 1), (2, 1)),
        (lambda a: a.reshape((2, 1)), (2, 1)),
    ],
)
def test_ravel_reshape(array, func, new_shape):
    reshaped_array = func(array)

    assert type(reshaped_array) is type(array)
    assert reshaped_array.shape == new_shape
    if isinstance(array, Map):
        if array.coverage is None:
            assert reshaped_array.coverage is None
        else:
            assert reshaped_array.coverage.shape == new_shape
        if array.error is None:
            assert reshaped_array.error is None
        else:
            assert reshaped_array.error.shape == new_shape
    elif isinstance(array, Tod):
        if array.mask is None:
            assert reshaped_array.mask is None
        else:
            assert reshaped_array.mask.shape == new_shape


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
