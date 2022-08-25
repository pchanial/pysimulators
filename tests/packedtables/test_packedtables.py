import string

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators.utils import isalias, isscalarlike, product
from pyoperators.utils.testing import assert_same
from pysimulators import PackedTable
from pysimulators.geometry import create_grid_squares

if np.__version__ < '1.8':

    def full(shape, fill_value, dtype=None, order='C'):
        out = np.empty(shape, dtype=dtype, order=order)
        out[...] = fill_value
        return out

    np.full = full


def setattr_unpacked(x, a, v):
    setattr(x.all, a, v)


@pytest.mark.parametrize('shape', [(), (1,), (1, 2), (1, 2, 3)])
def test_ndim1(shape):
    layout = PackedTable(shape)
    assert layout.ndim == len(shape)
    assert layout.shape == shape


@pytest.mark.parametrize(
    'shape, ndim',
    [
        ((1,), 1),
        ((1, 2), 1),
        ((1, 2), 2),
        ((1, 2, 3), 1),
        ((1, 2, 3), 2),
        ((1, 2, 3), 3),
    ],
)
def test_ndim2(shape, ndim):

    layout = PackedTable(shape, ndim=ndim)
    assert_equal(layout.ndim, ndim)
    assert_equal(layout.shape, shape)
    assert_equal(layout._shape_actual, shape[:ndim])


def test_ndim3():
    shape = (13, 3)
    l = PackedTable(shape, ndim=1)
    l1, l2 = l.split(2)
    assert len(l1) == 7
    assert l1.shape == shape
    assert len(l2) == 6
    assert l2.shape == shape


@pytest.mark.parametrize('method', [setattr, setattr_unpacked])
@pytest.mark.parametrize('attr', PackedTable._reserved_attributes)
def test_reserved(method, attr):
    layout = PackedTable((6, 6))

    with pytest.raises(AttributeError):
        method(layout, attr, 1)


SA_SHAPE = (6, 6)
SA_PACKED = np.arange(36.0)
SA_UNPACKED = SA_PACKED.reshape(SA_SHAPE)


@pytest.mark.parametrize(
    'method, value', [(setattr, SA_PACKED), (setattr_unpacked, SA_UNPACKED)]
)
def test_special_attribute_array(method, value):
    layout = PackedTable(SA_SHAPE, key=np.ones(SA_SHAPE, int))
    method(layout, 'key', value)
    assert_same(layout.key, SA_PACKED)
    assert layout.key.dtype == float
    assert_same(layout.all.key, SA_UNPACKED)
    assert layout.all.key.dtype == float
    if method is setattr:
        assert isalias(layout.key, SA_UNPACKED)
        assert isalias(layout.all.key, SA_UNPACKED)
    else:
        assert not isalias(layout.key, SA_UNPACKED)
        assert not isalias(layout.all.key, SA_UNPACKED)


@pytest.mark.parametrize('method', [setattr, setattr_unpacked])
@pytest.mark.parametrize('value', [lambda: SA_PACKED, lambda self: SA_PACKED])
def test_special_attribute_func1(method, value):
    layout = PackedTable(SA_SHAPE, key=None)

    if method is setattr_unpacked:
        with pytest.raises(TypeError):
            method(layout, 'key', value)
        return

    method(layout, 'key', value)
    assert isalias(layout.key, SA_PACKED)
    assert isalias(layout.all.key, SA_PACKED)
    assert_same(layout.key, SA_PACKED)
    assert_same(layout.all.key, SA_PACKED.reshape(SA_SHAPE))


@pytest.fixture
def setup_special_attribute_func():
    shape = (6, 6)
    ordering = np.arange(product(shape))[::-1].reshape(shape)
    val = np.arange(product(shape)).reshape(shape)
    layout = PackedTable(shape, ordering=ordering, val=val, key=None)
    layout.myscalar1 = 2
    layout._myscalar2 = 5
    return layout, val


def test_special_attribute_func2(setup_special_attribute_func):
    layout, val = setup_special_attribute_func
    layout.key = lambda: val.ravel()[::-1] * 6
    assert_equal(layout.all.key, val * 6)
    assert_equal(layout.key, val.ravel()[::-1] * 6)
    assert layout.all.key.shape == layout.shape
    assert layout.key.shape == (len(layout),)


def test_special_attribute_func3(setup_special_attribute_func):
    layout, val = setup_special_attribute_func
    layout.key = lambda self: self.val * self.myscalar1 * self._myscalar2
    assert_equal(layout.all.key, val * 10)
    assert_equal(layout.key, val.ravel()[::-1] * 10)
    assert layout.all.key.shape == layout.shape
    assert layout.key.shape == (len(layout),)


def test_special_attribute_func4(setup_special_attribute_func):
    layout, val = setup_special_attribute_func
    layout.key = lambda s, x: x * s.val * s.myscalar1 * s._myscalar2
    assert_same(layout.key(2), val.ravel()[::-1] * 20)
    assert_same(layout.all.key(2), val * 20)


def test_special_attribute_func5():
    class P(PackedTable):
        def __init__(self, shape, ordering, val):
            PackedTable.__init__(
                self, shape, ordering=ordering, val=val, key1=None, key2=None, key3=None
            )

        myscalar1 = 2
        _myscalar2 = 3

        def key1(self):
            return self.val * self.myscalar1 * self._myscalar2

        @property
        def key2(self):
            return self.val * self.myscalar1 * self._myscalar2

        def key3(self, x):
            return x * self.val * self.myscalar1 * self._myscalar2

    shape = (6, 6)
    ordering = np.arange(product(shape))[::-1].reshape(shape)
    val = np.arange(product(shape)).reshape(shape)
    layout = P(shape, ordering, val)
    assert_same(layout.key1, val.ravel()[::-1] * 6)
    assert_same(layout.all.key1, val * 6)
    assert_same(layout.key2, val.ravel()[::-1] * 6)
    assert_same(layout.all.key2, val * 6)
    assert_same(layout.key3(2), val.ravel()[::-1] * 12)
    assert_same(layout.all.key3(2), val * 12)


@pytest.mark.parametrize('method', [setattr, setattr_unpacked])
@pytest.mark.parametrize('value', [np.int32(10), 10.0, np.array(10.0)])
def test_special_attribute_scalar(method, value):
    layout = PackedTable((6, 6), key=value)
    assert 'key' in layout._special_attributes
    assert isscalarlike(layout.key)
    assert_equal(layout.key, value)
    assert layout.key.shape == ()
    assert_same(layout.all.key, value, broadcasting=True)
    assert_same(layout.all.key.shape, layout.shape)
    method(layout, 'key', value)
    assert isscalarlike(layout.key)
    assert_equal(layout.key, value)
    assert layout.key.shape == ()
    assert_same(layout.all.key, value, broadcasting=True)
    assert_same(layout.all.key.shape, layout.shape)


@pytest.mark.parametrize('method', [setattr, setattr_unpacked])
@pytest.mark.parametrize('value', [None, 1])
def test_special_attribute_none(method, value):
    layout = PackedTable((6, 6), key=value)
    assert 'key' in layout._special_attributes
    assert_equal(layout.key, value)
    assert_equal(layout.all.key, value)
    method(layout, 'key', None)
    assert layout.key is None
    assert layout.all.key is None


def test_layout_errors():
    shape = (4,)
    with pytest.raises(ValueError):
        PackedTable(shape, selection=[True, False])

    with pytest.raises(ValueError):
        PackedTable(shape, ordering=[1, 3])

    with pytest.raises(ValueError):
        l = PackedTable(shape, selection=[True, False, False, True])
        l.all.removed[1] = False

    layout = PackedTable(shape, special=None)
    with pytest.raises(AttributeError):
        setattr(layout, 'removed', True)

    with pytest.raises(AttributeError):
        setattr(layout.all, 'notspecial', 'value')

    with pytest.raises(ValueError):
        setattr(layout, 'special', [1, 2, 3])

    with pytest.raises(ValueError):
        setattr(layout.all, 'special', [1, 2, 3])

    with pytest.raises(ValueError):
        layout.pack([1, 2, 3])

    with pytest.raises(ValueError):
        layout.unpack([1, 2, 3])

    with pytest.raises(TypeError):
        layout.pack([1, 2, 3, 4], out=[])

    with pytest.raises(TypeError):
        layout.unpack([1, 2, 3, 4], out=[])

    with pytest.raises(ValueError):
        layout.pack([1, 2, 3, 4], out=np.array([1]))

    with pytest.raises(ValueError):
        layout.unpack([1, 2, 3, 4], out=np.array([1]))


@pytest.mark.parametrize(
    'selection, ordering',
    [
        ([], None),
        ((False, False, True, True), (1, 4, -1, -1)),
        (Ellipsis, (-1, -1, -1, -1)),
    ],
)
def test_no_component(selection, ordering):
    layout = PackedTable((4,), selection=selection, ordering=ordering, key=[1, 2, 3, 4])
    assert len(layout) == 0
    assert len(layout.all) == 4
    assert_equal(layout.key.size, 0)
    assert_equal(layout.all.key, [-1, -1, -1, -1])


def test_pack_none():
    layout = PackedTable((3, 3))
    assert layout.pack(None) is None
    assert layout.unpack(None) is None


PU_SHAPE = (3, 3)
PU_SIZE = product(PU_SHAPE)
PU_ORDERING1 = np.arange(PU_SIZE)[::-1].reshape(PU_SHAPE)
PU_ORDERING2 = PU_ORDERING1.copy()
PU_ORDERING2[0, 2] = -1
PU_SELECTION = [[False, True, True], [True, True, True], [False, True, False]]


@pytest.mark.parametrize(
    'selection, ordering',
    [
        (None, None),
        (None, PU_ORDERING1),
        (None, PU_ORDERING2),
        (PU_SELECTION, None),
        (PU_SELECTION, PU_ORDERING1),
        (PU_SELECTION, PU_ORDERING2),
    ],
)
def test_pack_unpack(selection, ordering):
    def missing(x):
        kind = x.dtype.kind
        if kind == 'b':
            return ~x
        if kind == 'i':
            return x == -1
        if kind == 'u':
            return x == 0
        if kind == 'f':
            return ~np.isfinite(x)
        if kind == 'V':
            if x.dtype.names == ('x', 'y'):
                return ~np.isfinite(x.x) & ~np.isfinite(x.y)
        assert False

    vertex = create_grid_squares(PU_SHAPE, 0.1)
    center = np.mean(vertex, axis=-2)
    reccenter = np.recarray(center.shape[:-1], [('x', float), ('y', float)])
    reccenter.x, reccenter.y = center[..., 0], center[..., 1]
    recvertex = np.recarray(vertex.shape[:-1], [('x', float), ('y', float)])
    recvertex.x, recvertex.y = vertex[..., 0], vertex[..., 1]

    for data in [
        (np.arange(PU_SIZE, dtype=int) % 2).astype(bool).reshape(PU_SHAPE),
        np.arange(PU_SIZE, dtype=int).reshape(PU_SHAPE),
        np.arange(PU_SIZE, dtype=np.uint64).reshape(PU_SHAPE),
        np.arange(PU_SIZE, dtype=float).reshape(PU_SHAPE),
        (np.arange(PU_SIZE * 2, dtype=int) % 2).astype(bool).reshape(PU_SHAPE + (2,)),
        np.arange(PU_SIZE * 2, dtype=int).reshape(PU_SHAPE + (2,)),
        np.arange(PU_SIZE * 2, dtype=np.uint64).reshape(PU_SHAPE + (2,)),
        np.arange(PU_SIZE * 2, dtype=float).reshape(PU_SHAPE + (2,)),
        (np.arange(PU_SIZE * 2 * 3, dtype=int) % 2)
        .astype(bool)
        .reshape(PU_SHAPE + (2, 3)),
        np.arange(PU_SIZE * 2 * 3, dtype=int).reshape(PU_SHAPE + (2, 3)),
        np.arange(PU_SIZE * 2 * 3, dtype=np.uint64).reshape(PU_SHAPE + (2, 3)),
        np.arange(PU_SIZE * 2 * 3, dtype=float).reshape(PU_SHAPE + (2, 3)),
        center,
        reccenter,
        vertex,
        recvertex,
    ]:
        layout = PackedTable(
            PU_SHAPE,
            vertex=vertex,
            center=center,
            selection=selection,
            ordering=ordering,
        )
        packed = layout.pack(data)
        assert_equal(packed.shape, (len(layout),) + data.shape[2:])
        d_ = layout.unpack(packed)
        assert_equal(d_.shape, data.shape)
        assert np.all((data == d_) | missing(d_))
        out_packed = np.empty((len(layout),) + data.shape[2:], data.dtype).view(
            type(data)
        )
        out_unpacked = np.empty(data.shape, data.dtype).view(type(data))
        layout.pack(data, out=out_packed)
        layout.unpack(out_packed, out=out_unpacked)
        assert np.all((data == out_unpacked) | missing(out_unpacked))
        out_unpacked = np.empty((6,) + data.shape[1:], data.dtype)[::2].view(type(data))
        out_packed = np.empty(
            (out_packed.shape[0] * 2,) + out_packed.shape[1:], data.dtype
        )[::2].view(type(data))
        layout.pack(data, out=out_packed)
        layout.unpack(out_packed, out=out_unpacked)
        assert np.all((data == out_unpacked) | missing(out_unpacked))


def test_selection1():
    selection = [[False, True, False], [True, True, True], [False, True, False]]
    val = np.arange(9, dtype=int).reshape((3, 3))
    layout = PackedTable((3, 3), selection=selection, key=val)
    pexpected = [1, 3, 4, 5, 7]
    uexpected = val.copy()
    uexpected[layout.all.removed] = -1
    assert_equal(layout.key, pexpected)
    assert_equal(layout.all.key, uexpected)


def test_selection2():
    selection = [[False, True, False], [True, True, True], [False, True, False]]
    val = np.arange(9, dtype=float).reshape((3, 3))
    layout = PackedTable((3, 3), selection=selection, key=val)
    pexpected = [1, 3, 4, 5, 7]
    uexpected = val.copy()
    uexpected[layout.all.removed] = np.nan
    assert_equal(layout.key, pexpected)
    assert_equal(layout.all.key, uexpected)


def test_ordering1():
    ordering = [
        [11, 14, 17, 0, 1, 2],
        [10, 13, 16, 3, 4, 5],
        [9, 12, 15, 6, 7, 8],
        [26, 25, 24, 33, 30, 27],
        [23, 22, 21, 34, 31, 28],
        [20, 19, 18, 35, 32, 29],
    ]
    layout = PackedTable((6, 6), ordering=ordering)

    # fmt: off
    expected = (
        [3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 6, 0, 13, 7, 1, 14, 8, 2, 32, 31, 30, 26] +
        [25, 24, 20, 19, 18, 23, 29, 35, 22, 28, 34, 21, 27, 33]
    )
    # fmt: on
    assert_equal(layout._index, expected)


def test_ordering2():
    ordering = [
        [11, 14, 17, 0, 1, 2],
        [10, 13, 16, 3, 4, 5],
        [9, 12, 15, 6, 7, 8],
        [26, 25, 24, 33, 30, 27],
        [23, 22, 21, 34, 31, 28],
        [20, 19, 18, 35, 32, 29],
    ]
    selection = [
        [False, True, True, True, True, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [False, True, True, True, True, False],
    ]
    layout = PackedTable((6, 6), selection=selection, ordering=ordering)
    # fmt: off
    expected = (
        [3, 4, 9, 10, 11, 15, 16, 17, 12, 6, 13, 7, 1, 14, 8, 2, 32, 31, 26, 25, 24] +
        [20, 19, 18, 23, 29, 22, 28, 34, 21, 27, 33]
    )
    # fmt: on
    assert_equal(layout._index, expected)


def test_ordering3():
    ordering = [
        [-1, 12, 15, 0, 1, -1],
        [9, 11, 14, 2, 3, 4],
        [8, 10, 13, 5, 6, 7],
        [23, 22, 21, 29, 26, 24],
        [20, 19, 18, 30, 27, 25],
        [-1, 17, 16, 31, 28, -1],
    ]
    layout = PackedTable((6, 6), ordering=ordering)
    # fmt: off
    expected = (
        [3, 4, 9, 10, 11, 15, 16, 17, 12, 6, 13, 7, 1, 14, 8, 2, 32, 31, 26, 25, 24] +
        [20, 19, 18, 23, 29, 22, 28, 34, 21, 27, 33]
    )
    # fmt: on
    assert_equal(layout._index, expected)


@pytest.mark.parametrize(
    'ordering, expected',
    [
        ([[0, 1, 2], [3, 4, 5]], Ellipsis),
        ([[-1, 1, 2], [3, 4, -1]], slice(1, 5, 1)),
        ([[-1, 1, -1], [3, -1, -1]], slice(1, 5, 2)),
        ([[5, 4, 3], [2, 1, 0]], slice(5, -1, -1)),
        ([[5, -1, 3], [-1, 1, -1]], slice(4, -2, -2)),
        ([[-1, 4, -1], [2, -1, 0]], slice(5, -1, -2)),
        ([[-1, -1, 3], [-1, -1, -1]], [2]),
    ],
)
def test_ordering4(ordering, expected):

    layout = PackedTable((2, 3), ordering=ordering)
    assert_equal(layout._index, expected)


SLICE_SIZE_N = 4


@pytest.mark.parametrize('start', [None, *range(-SLICE_SIZE_N - 1, SLICE_SIZE_N + 1)])
@pytest.mark.parametrize('stop', [None, *range(-SLICE_SIZE_N - 1, SLICE_SIZE_N + 1)])
@pytest.mark.parametrize('step', range(-SLICE_SIZE_N - 1, SLICE_SIZE_N + 1))
def test_slice_size(start, stop, step):
    if step == 0:
        return
    array = tuple(string.ascii_lowercase[:SLICE_SIZE_N])
    slice_ = slice(start, stop, step)

    sn = PackedTable._normalize_slice(slice_, SLICE_SIZE_N)
    if sn is Ellipsis:
        assert_equal(array, array[slice_])
        return

    assert (sn.stop - sn.start) // sn.step == len(array[slice_])
    if sn.stop < 0:
        sn = slice(sn.start, None, sn.step)
    assert_equal(array[sn], array[slice_])


SELECTION_UNPACKED_INT = (0, 2, 5, 3, 1, 4), (0, 3, 0, 2, 1, 5)
SELECTION_UNPACKED_BOOL = np.zeros((6, 6), bool)
SELECTION_UNPACKED_BOOL[SELECTION_UNPACKED_INT] = True


@pytest.mark.xfail(reason='reason: Layout.all.__getitem__ not implemented')
@pytest.mark.parametrize(
    'selection, expected',
    [
        ((), None),
        ([], []),
        (Ellipsis, None),
        (2, [12, 13, 14, 15, 16, 17]),
        ((2,), [12, 13, 14, 15, 16, 17]),
        ([3], [18, 19, 20, 21, 22, 23]),
        ((0, 0), [0]),
        ((1, 1), [7]),
        ([1, 1], [6, 7, 8, 9, 10, 11]),
        ((slice(0, 2), slice(0, 3)), [0, 1, 2, 6, 7, 8]),
        (SELECTION_UNPACKED_BOOL, [0, 7, 15, 20, 29, 30]),
        (SELECTION_UNPACKED_INT, [0, 7, 15, 20, 29, 30]),
    ],
)
def test_selection_unpacked1(selection, expected):
    val = np.array(
        [
            [12, 2, 18, 15, 22, 19],
            [33, 21, 16, 26, 31, 1],
            [30, 28, 7, 14, 0, 6],
            [27, 4, 11, 24, 3, 5],
            [8, 20, 32, 9, 29, 35],
            [25, 17, 13, 23, 10, 34],
        ]
    )
    layout = PackedTable((6, 6), val=val)

    selected = layout.all[selection]
    assert_equal(selected._index, expected)
    assert_equal(selected.all.val[selection], val[selection])


SELECTION_UNPACKED2_ORDERING = [
    [-1, 12, 15, 0, 1, -1],
    [9, 11, 14, 2, 3, 4],
    [8, 10, 13, 5, 6, 7],
    [23, 22, 21, 29, 26, 24],
    [20, 19, 18, 30, 27, 25],
    [-1, 17, 16, 31, 28, -1],
]
SELECTION_UNPACKED2_VAL = np.array(
    [
        [-1, 2, 18, 15, 22, -1],
        [33, 21, 16, 26, 31, 1],
        [30, 28, 7, 14, 0, 6],
        [27, 4, 11, 24, 3, 5],
        [8, 20, 32, 9, 29, 35],
        [-1, 17, 13, 23, 10, -1],
    ]
)
SELECTION_UNPACKED2_LAYOUT = PackedTable(
    (6, 6), ordering=SELECTION_UNPACKED2_ORDERING, all_val=SELECTION_UNPACKED2_VAL
)


@pytest.mark.xfail(reason='reason: Layout.all.__getitem__ not implemented')
@pytest.mark.parametrize(
    'selection, expected',
    [
        ((), SELECTION_UNPACKED2_LAYOUT._index),
        ([], []),
        (Ellipsis, SELECTION_UNPACKED2_LAYOUT._index),
        (2, [15, 16, 17, 12, 13, 14]),
        ((2,), [15, 16, 17, 12, 13, 14]),
        ([3], [20, 19, 18, 23, 22, 21]),
        ((0, 0), []),
        ((1, 1), [7]),
        ([1, 1], [9, 10, 11, 6, 7, 8]),
        ((slice(0, 2), slice(0, 3)), [6, 7, 1, 8, 2]),
        (SELECTION_UNPACKED_BOOL, [15, 7, 20, 29]),
        (SELECTION_UNPACKED_INT, [15, 7, 20, 29]),
    ],
)
def test_selection_unpacked2(selection, expected):
    selected = SELECTION_UNPACKED2_LAYOUT.all[selection]
    assert selected._index.dtype == np.int32
    assert_equal(selected._index, expected)
    assert_equal(selected.all.val[selection], SELECTION_UNPACKED2_VAL[selection])


SELECTION_PACKED1_INT = [13, 14, 6, 19, 0, 35, 3]
SELECTION_PACKED1_BOOL = np.zeros(36, bool)
SELECTION_PACKED1_BOOL[SELECTION_PACKED1_INT] = True


@pytest.mark.parametrize(
    'selection, expected',
    [
        ((), Ellipsis),
        ([], []),
        (Ellipsis, Ellipsis),
        (2, [2]),
        ((0,), [0]),
        ([3], [3]),
        (slice(4, 9), slice(4, 9, 1)),
        (slice(4, 16, 3), slice(4, 16, 3)),
        ([4, 7, 10, 13], slice(4, 16, 3)),
        (SELECTION_PACKED1_INT, SELECTION_PACKED1_INT),
        (SELECTION_PACKED1_BOOL, sorted(SELECTION_PACKED1_INT)),
    ],
)
def test_selection_packed1(selection, expected):
    # layout index is Ellipsis
    val = np.array(
        [
            [12, 2, 18, 15, 22, 19],
            [33, 21, 16, 26, 31, 1],
            [30, 28, 7, 14, 0, 6],
            [27, 4, 11, 24, 3, 5],
            [8, 20, 32, 9, 29, 35],
            [25, 17, 13, 23, 10, 34],
        ]
    )
    layout = PackedTable((6, 6), val=val)
    assert layout._index is Ellipsis

    selected = layout[selection]
    assert_equal(selected._index, expected)
    assert_equal(selected.val, layout.val[selection])


SELECTION_PACKED2_INT = [13, 14, 6, 19, 0, 31, 3]
SELECTION_PACKED2_BOOL = np.zeros(32, bool)
SELECTION_PACKED2_BOOL[SELECTION_PACKED2_INT] = True
# fmt: off
SELECTION_PACKED2_INDEX = np.array(
    [ 3,  4,  9, 10, 11, 15, 16, 17, 12,  6, 13,  7,  1, 14,  8,  2, 32, 31, 26, 25, 24,
    20, 19, 18, 23, 29, 22, 28, 34, 21, 27, 33]
)
# fmt: on


@pytest.mark.parametrize(
    'selection, expected',
    [
        ((), SELECTION_PACKED2_INDEX),
        ([], []),
        (Ellipsis, SELECTION_PACKED2_INDEX),
        (2, [9]),
        ((0,), [3]),
        ([3], [10]),
        (slice(4, 9), [11, 15, 16, 17, 12]),
        (slice(4, 16, 3), [11, 17, 13, 14]),
        (SELECTION_PACKED2_INT, SELECTION_PACKED2_INDEX[SELECTION_PACKED2_INT]),
        (SELECTION_PACKED2_BOOL, SELECTION_PACKED2_INDEX[SELECTION_PACKED2_BOOL]),
    ],
)
def test_selection_packed2(selection, expected):
    # layout index is an array
    ordering = [
        [-1, 12, 15, 0, 1, -1],
        [9, 11, 14, 2, 3, 4],
        [8, 10, 13, 5, 6, 7],
        [23, 22, 21, 29, 26, 24],
        [20, 19, 18, 30, 27, 25],
        [-1, 17, 16, 31, 28, -1],
    ]
    val = np.array(
        [
            [-1, 2, 18, 15, 22, -1],
            [33, 21, 16, 26, 31, 1],
            [30, 28, 7, 14, 0, 6],
            [27, 4, 11, 24, 3, 5],
            [8, 20, 32, 9, 29, 35],
            [-1, 17, 13, 23, 10, -1],
        ]
    )
    layout = PackedTable((6, 6), ordering=ordering, val=val)
    assert type(layout._index) is np.ndarray
    assert_equal(layout._index, SELECTION_PACKED2_INDEX)

    selected = layout[selection]
    assert selected._index.dtype == layout._dtype_index
    assert_equal(selected._index, expected)
    assert_equal(selected.val, layout.val[selection])


SELECTION_PACKED3_INT = [4, 2, 8, 0, 7, 1]
SELECTION_PACKED3_BOOL = np.zeros(9, bool)
SELECTION_PACKED3_BOOL[SELECTION_PACKED3_INT] = True
SELECTION_PACKED3_INDEX = slice(4, 31, 3)  # [4,  7, 10, 13, 16, 19, 22, 25, 28]


@pytest.mark.parametrize(
    'selection, expected',
    [
        ((), SELECTION_PACKED3_INDEX),
        ([], []),
        (Ellipsis, SELECTION_PACKED3_INDEX),
        (2, [10]),
        ((0,), [4]),
        ([3], [13]),
        (slice(2, 5), slice(10, 19, 3)),
        (slice(2, 16, 2), slice(10, 34, 6)),
        (SELECTION_PACKED3_INT, [16, 10, 28, 4, 25, 7]),
        (SELECTION_PACKED3_BOOL, [4, 7, 10, 16, 25, 28]),
    ],
)
def test_selection_packed3(selection, expected):
    # layout index is a slice
    val = np.array(
        [
            [12, 2, 18, 15, 22, 19],
            [33, 21, 16, 26, 31, 1],
            [30, 28, 7, 14, 0, 6],
            [27, 4, 11, 24, 3, 5],
            [8, 20, 32, 9, 29, 35],
            [25, 17, 13, 23, 10, 34],
        ]
    )
    layout_selection = np.zeros((6, 6), bool)
    layout_selection.ravel()[4:30:3] = True
    layout = PackedTable((6, 6), selection=layout_selection, val=val)
    index = layout._index
    assert type(index) is slice
    assert layout._index == SELECTION_PACKED3_INDEX

    selected = layout[selection]
    assert_equal(selected._index, expected)
    assert_equal(selected.val, layout.val[selection])


@pytest.mark.parametrize(
    'size, selection',
    [
        (1, Ellipsis),
        (1, 0),
        (2, Ellipsis),
        (2, 0),
        (2, 1),
        (3, Ellipsis),
        (3, 0),
        (3, 1),
        (3, 2),
    ],
)
@pytest.mark.parametrize('nsplit', range(1, 7))
def test_split(size, selection, nsplit):
    if selection is not Ellipsis:
        tmp = np.ones(size, bool)
        tmp[selection] = False
        selection = tmp
    layout = PackedTable(size, selection=selection, val=np.arange(size) * 2.0)
    slices = layout.split(nsplit)
    assert len(slices) == nsplit
    o = np.zeros(layout.shape, int)
    v = np.full(layout.shape, np.nan)
    for selection in slices:
        o[selection._index] += 1
        v[selection._index] = selection.val
    o[o == 0] = -1
    assert_same(o, layout.unpack(1))
    assert_same(v, layout.all.val)
    assert_same(np.concatenate([_.val for _ in slices]), layout.val)
