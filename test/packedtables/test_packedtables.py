from __future__ import division
import numpy as np
import string
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils import isalias, isscalarlike, product
from pyoperators.utils.testing import (
    assert_eq, assert_is_none, assert_same, skiptest)
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


def test_ndim1():
    shapes = (), (1,), (1, 2), (1, 2, 3)

    def func(s):
        layout = PackedTable(s)
        assert_equal(layout.ndim, len(s))
        assert_equal(layout.shape, s)
    for s in shapes:
        yield func, s


def test_ndim2():
    shapes = (), (1,), (1, 2), (1, 2, 3)

    def func(s, n):
        layout = PackedTable(s, ndim=n)
        assert_equal(layout.ndim, n)
        assert_equal(layout.shape, s)
        assert_equal(layout._shape_actual, s[:n])
    for s in shapes:
        for n in range(1, len(s) + 1):
            yield func, s, n


def test_ndim3():
    shape = (13, 3)
    l = PackedTable(shape, ndim=1)
    l1, l2 = l.split(2)
    assert_equal(len(l1), 7)
    assert_equal(l1.shape, shape)
    assert_equal(len(l2), 6)
    assert_equal(l2.shape, shape)


def test_reserved():
    layout = PackedTable((6, 6))

    def func(s, key):
        assert_raises(AttributeError, s, layout, key, 1)
    for s in (setattr, setattr_unpacked):
        for key in layout._reserved_attributes:
            yield func, s, key


def test_special_attribute_array():
    shape = (6, 6)
    val = np.arange(36.).reshape(shape)

    def func(s, v):
        layout = PackedTable(shape, key=np.ones(shape, int))
        s(layout, 'key', v)
        assert_same(layout.key, val.ravel())
        assert_equal(layout.key.dtype, float)
        assert_same(layout.all.key, val)
        assert_equal(layout.all.key.dtype, float)
        if s is setattr:
            assert isalias(layout.key, val)
            assert isalias(layout.all.key, val)
        else:
            assert not isalias(layout.key, val)
            assert not isalias(layout.all.key, val)
    for s, v in zip((setattr, setattr_unpacked), (val.ravel(), val)):
        yield func, s, v


def test_special_attribute_func1():
    shape = (6, 6)
    val = np.arange(36)
    layout = PackedTable(shape, key=None)

    def func(s, v):
        if s is setattr_unpacked:
            assert_raises(TypeError, s, layout, 'key', v)
            return
        s(layout, 'key', v)
        assert isalias(layout.key, val)
        assert isalias(layout.all.key, val)
        assert_same(layout.key, val)
        assert_same(layout.all.key, val.reshape(shape))
    for s in (setattr, setattr_unpacked):
        for v in (lambda: val, lambda s: val):
            yield func, s, v


def test_special_attribute_func2():
    shape = (6, 6)
    ordering = np.arange(product(shape))[::-1].reshape(shape)
    val = np.arange(product(shape)).reshape(shape)
    layout = PackedTable(shape, ordering=ordering, val=val, key=None)
    layout.myscalar1 = 2
    layout._myscalar2 = 3
    layout_funcs = (lambda: val.ravel()[::-1] * 6,
                    lambda s: s.val * s.myscalar1 * s._myscalar2)

    def func(v):
        setattr(layout, 'key', v)
        assert_equal(layout.all.key, val * 6)
        assert_equal(layout.key, val.ravel()[::-1] * 6)
        assert_same(layout.all.key.shape, layout.shape)
        assert_same(layout.key.shape, (len(layout),))
    for v in layout_funcs:
        yield func, v


def test_special_attribute_func3():
    shape = (6, 6)
    ordering = np.arange(product(shape))[::-1].reshape(shape)
    val = np.arange(product(shape)).reshape(shape)
    func = lambda s, x: x * s.val * s.myscalar1 * s._myscalar2
    layout = PackedTable(shape, ordering=ordering, val=val, key=func)
    layout.myscalar1 = 2
    layout._myscalar2 = 3
    assert_same(layout.key(2), val.ravel()[::-1] * 12)
    assert_same(layout.all.key(2), val * 12)


def test_special_attribute_func4():
    class P(PackedTable):
        def __init__(self, shape, ordering, val):
            PackedTable.__init__(self, shape, ordering=ordering, val=val,
                                 key1=None, key2=None, key3=None)

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


def test_special_attribute_scalar():
    def func(s, v):
        layout = PackedTable((6, 6), key=v)
        assert 'key' in layout._special_attributes
        assert isscalarlike(layout.key)
        assert_eq(layout.key, v)
        assert_eq(layout.key.shape, ())
        assert_same(layout.all.key, v, broadcasting=True)
        assert_same(layout.all.key.shape, layout.shape)
        s(layout, 'key', v)
        assert isscalarlike(layout.key)
        assert_eq(layout.key, v)
        assert_eq(layout.key.shape, ())
        assert_same(layout.all.key, v, broadcasting=True)
        assert_same(layout.all.key.shape, layout.shape)
    for s in (setattr, setattr_unpacked):
        for v in (np.int32(10), 10., np.array(10.)):
            yield func, s, v


def test_special_attribute_none():
    def func(s, v):
        layout = PackedTable((6, 6), key=v)
        assert 'key' in layout._special_attributes
        assert_equal(layout.key, v)
        assert_equal(layout.all.key, v)
        s(layout, 'key', None)
        assert_is_none(layout.key)
        assert_is_none(layout.all.key)
    for s in setattr, setattr_unpacked:
        for v in (None, 1):
            yield func, s, v


def test_layout_errors():
    shape = (4,)
    assert_raises(ValueError, PackedTable, shape, selection=[True, False])
    assert_raises(ValueError, PackedTable, shape, ordering=[1, 3])

    def func():
        l = PackedTable(shape, selection=[True, False, False, True])
        l.all.removed[1] = False
    assert_raises(ValueError, func)

    layout = PackedTable(shape, special=None)
    assert_raises(AttributeError, setattr, layout, 'removed', True)
    assert_raises(AttributeError, setattr, layout.all, 'notspecial', 'value')
    assert_raises(ValueError, setattr, layout, 'special', [1, 2, 3])
    assert_raises(ValueError, setattr, layout.all, 'special', [1, 2, 3])
    assert_raises(ValueError, layout.pack, [1, 2, 3])
    assert_raises(ValueError, layout.unpack, [1, 2, 3])
    assert_raises(TypeError, layout.pack, [1, 2, 3, 4], out=[])
    assert_raises(TypeError, layout.unpack, [1, 2, 3, 4], out=[])
    assert_raises(ValueError, layout.pack, [1, 2, 3, 4], out=np.array([1]))
    assert_raises(ValueError, layout.unpack, [1, 2, 3, 4], out=np.array([1]))


def test_no_component():
    selections = ([], (False, False, True, True), Ellipsis)
    orderings = (None, (1, 4, -1, -1), (-1, -1, -1, -1))

    def func(selection, ordering):
        layout = PackedTable((4,), selection=selection, ordering=ordering,
                             key=[1, 2, 3, 4])
        assert_equal(len(layout), 0)
        assert_equal(len(layout.all), 4)
        assert_equal(layout.key.size, 0)
        assert_equal(layout.all.key, [-1, -1, -1, -1])
    for selection, ordering in zip(selections, orderings):
        yield func, selection, ordering


def test_pack_none():
    layout = PackedTable((3, 3))
    assert_is_none(layout.pack(None))
    assert_is_none(layout.unpack(None))


def test_packunpack():
    shape = (3, 3)
    size = 9
    ordering1 = np.arange(size)[::-1].reshape(shape)
    ordering2 = ordering1.copy()
    ordering2[0, 2] = -1
    selection = [[False, True, True],
                 [True,  True, True],
                 [False, True, False]]
    vertex = create_grid_squares(shape, 0.1)
    center = np.mean(vertex, axis=-2)
    reccenter = np.recarray(center.shape[:-1], [('x', float), ('y', float)])
    reccenter.x, reccenter.y = center[..., 0], center[..., 1]
    recvertex = np.recarray(vertex.shape[:-1], [('x', float), ('y', float)])
    recvertex.x, recvertex.y = vertex[..., 0], vertex[..., 1]
    valb = (np.arange(size, dtype=int) % 2).astype(bool).reshape(shape)
    vali = np.arange(size, dtype=int).reshape(shape)
    valu = np.arange(size, dtype=np.uint64).reshape(shape)
    valf = np.arange(size, dtype=float).reshape(shape)
    valb2 = (np.arange(size * 2, dtype=int) % 2).astype(bool).reshape(
        shape + (2,))
    vali2 = np.arange(size * 2, dtype=int).reshape(shape + (2,))
    valu2 = np.arange(size * 2, dtype=np.uint64).reshape(shape + (2,))
    valf2 = np.arange(size * 2, dtype=float).reshape(shape + (2,))
    valb3 = (np.arange(size * 2 * 3, dtype=int) % 2).astype(bool).reshape(
        shape + (2, 3))
    vali3 = np.arange(size * 2 * 3, dtype=int).reshape(shape + (2, 3))
    valu3 = np.arange(size * 2 * 3, dtype=np.uint64).reshape(shape + (2, 3))
    valf3 = np.arange(size * 2 * 3, dtype=float).reshape(shape + (2, 3))
    data = (valb, vali, valu, valf, valb2, vali2, valu2, valf2, valb3, vali3,
            valu3, valf3, center, reccenter, vertex, recvertex)

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

    def func(s, i, d):
        layout = PackedTable(shape, vertex=vertex, center=center,
                             selection=s, ordering=i)
        packed = layout.pack(d)
        assert_equal(packed.shape, (len(layout),) + d.shape[2:])
        d_ = layout.unpack(packed)
        assert_equal(d_.shape, d.shape)
        assert np.all((d == d_) | missing(d_))
        out_packed = np.empty((len(layout),) + d.shape[2:],
                              d.dtype).view(type(d))
        out_unpacked = np.empty(d.shape, d.dtype).view(type(d))
        layout.pack(d, out=out_packed)
        layout.unpack(out_packed, out=out_unpacked)
        assert np.all((d == out_unpacked) | missing(out_unpacked))
        out_unpacked = np.empty((6,)+d.shape[1:], d.dtype)[::2].view(type(d))
        out_packed = np.empty((out_packed.shape[0]*2,)+out_packed.shape[1:],
                              d.dtype)[::2].view(type(d))
        layout.pack(d, out=out_packed)
        layout.unpack(out_packed, out=out_unpacked)
        assert np.all((d == out_unpacked) | missing(out_unpacked))

    for s, i in ((None, None), (None, ordering1), (None, ordering2),
                 (selection, None), (selection, ordering1),
                 (selection, ordering2)):
        for d in data:
            yield func, s, i, d


def test_selection1():
    selection = [[False, True, False],
                 [True,  True, True],
                 [False, True, False]]
    val = np.arange(9, dtype=int).reshape((3, 3))
    layout = PackedTable((3, 3), selection=selection, key=val)
    pexpected = [1, 3, 4, 5, 7]
    uexpected = val.copy()
    uexpected[layout.all.removed] = -1
    assert_equal(layout.key, pexpected)
    assert_equal(layout.all.key, uexpected)


def test_selection2():
    selection = [[False, True, False],
                 [True,  True, True],
                 [False, True, False]]
    val = np.arange(9, dtype=float).reshape((3, 3))
    layout = PackedTable((3, 3), selection=selection, key=val)
    pexpected = [1, 3, 4, 5, 7]
    uexpected = val.copy()
    uexpected[layout.all.removed] = np.nan
    assert_equal(layout.key, pexpected)
    assert_equal(layout.all.key, uexpected)


def test_ordering1():
    ordering = [[11, 14, 17,  0,  1,  2],
                [10, 13, 16,  3,  4,  5],
                [ 9, 12, 15,  6,  7,  8],
                [26, 25, 24, 33, 30, 27],
                [23, 22, 21, 34, 31, 28],
                [20, 19, 18, 35, 32, 29]]
    layout = PackedTable((6, 6), ordering=ordering)
    expected = [ 3,  4,  5,  9, 10, 11, 15, 16, 17,
                12,  6,  0, 13,  7,  1, 14,  8,  2,
                32, 31, 30, 26, 25, 24, 20, 19, 18,
                23, 29, 35, 22, 28, 34, 21, 27, 33]
    assert_equal(layout._index, expected)


def test_ordering2():
    ordering = [[11, 14, 17,  0,  1,  2],
                [10, 13, 16,  3,  4,  5],
                [ 9, 12, 15,  6,  7,  8],
                [26, 25, 24, 33, 30, 27],
                [23, 22, 21, 34, 31, 28],
                [20, 19, 18, 35, 32, 29]]
    selection = [[False, True, True, True, True, False],
                 [True,  True, True, True, True, True ],
                 [True,  True, True, True, True, True ],
                 [True,  True, True, True, True, True ],
                 [True,  True, True, True, True, True ],
                 [False, True, True, True, True, False]]
    layout = PackedTable((6, 6), selection=selection, ordering=ordering)
    expected = [ 3,  4,  9, 10, 11, 15, 16, 17,
                12,  6, 13,  7,  1, 14,  8,  2,
                32, 31, 26, 25, 24, 20, 19, 18,
                23, 29, 22, 28, 34, 21, 27, 33]
    assert_equal(layout._index, expected)


def test_ordering3():
    ordering = [[-1, 12, 15,  0,  1, -1],
                [ 9, 11, 14,  2,  3,  4],
                [ 8, 10, 13,  5,  6,  7],
                [23, 22, 21, 29, 26, 24],
                [20, 19, 18, 30, 27, 25],
                [-1, 17, 16, 31, 28, -1]]
    layout = PackedTable((6, 6), ordering=ordering)
    expected = [ 3,  4,  9, 10, 11, 15, 16, 17,
                12,  6, 13,  7,  1, 14,  8,  2,
                32, 31, 26, 25, 24, 20, 19, 18,
                23, 29, 22, 28, 34, 21, 27, 33]
    assert_equal(layout._index, expected)


def test_ordering4():
    orderings = ([[0, 1, 2],
                  [3, 4, 5]],
                 [[-1, 1, 2],
                  [3, 4, -1]],
                 [[-1, 1, -1],
                  [3, -1, -1]],
                 [[5, 4, 3],
                  [2, 1, 0]],
                 [[5, -1,  3],
                  [-1, 1, -1]],
                 [[-1, 4, -1],
                  [2, -1,  0]],
                 [[-1, -1,  3],
                  [-1, -1, -1]])
    expecteds = (Ellipsis, slice(1, 5, 1), slice(1, 5, 2), slice(5, -1, -1),
                 slice(4, -2, -2), slice(5, -1, -2), [2])

    def func(o, e):
        layout = PackedTable((2, 3), ordering=o)
        assert_eq(layout._index, e)
    for o, e in zip(orderings, expecteds):
        yield func, o, e


def test_slice_size():
    n = 4
    array = tuple(string.ascii_lowercase[:n])

    def func(s):
        sn = PackedTable._normalize_slice(s, n)
        if sn is Ellipsis:
            assert_equal(array, array[s])
            return
        assert_equal((sn.stop - sn.start) // sn.step, len(array[s]))
        if sn.stop < 0:
            sn = slice(sn.start, None, sn.step)
        assert_equal(array[sn], array[s])
    for start in [None] + range(-n-1, n+1):
        for stop in [None] + range(-n-1, n+1):
            for step in range(-n - 1, n + 1):
                if step == 0:
                    continue
                yield func, slice(start, stop, step)


@skiptest
def test_selection_unpacked1():
    val = np.array([[12,  2, 18, 15, 22, 19],
                    [33, 21, 16, 26, 31,  1],
                    [30, 28,  7, 14,  0,  6],
                    [27,  4, 11, 24,  3,  5],
                    [ 8, 20, 32,  9, 29, 35],
                    [25, 17, 13, 23, 10, 34]])
    int_selection = (0, 2, 5, 3, 1, 4), (0, 3, 0, 2, 1, 5)
    bool_selection = np.zeros((6, 6), bool)
    bool_selection[int_selection] = True
    layout = PackedTable((6, 6), val=val)
    selections = ((), [], Ellipsis, 2, (2,), [3], (0, 0), (1, 1), [1, 1],
                  (slice(0, 2), slice(0, 3)), bool_selection, int_selection)
    expecteds = (None, [], None, [12, 13, 14, 15, 16, 17],
                 [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [0], [7],
                 [6, 7, 8, 9, 10, 11], [0, 1, 2, 6, 7, 8],
                 [0, 7, 15, 20, 29, 30], [0, 7, 15, 20, 29, 30])

    def func(selection, expected):
        selected = layout.all[selection]
        assert_equal(selected._index, expected)
        assert_equal(selected.all.val[selection], val[selection])
    for selection, expected in zip(selections, expecteds):
        yield func, selection, expected


@skiptest
def test_selection_unpacked2():
    ordering = [[-1, 12, 15,  0,  1, -1],
                [ 9, 11, 14,  2,  3,  4],
                [ 8, 10, 13,  5,  6,  7],
                [23, 22, 21, 29, 26, 24],
                [20, 19, 18, 30, 27, 25],
                [-1, 17, 16, 31, 28, -1]]
    val = np.array([[-1,  2, 18, 15, 22, -1],
                    [33, 21, 16, 26, 31,  1],
                    [30, 28,  7, 14,  0,  6],
                    [27,  4, 11, 24,  3,  5],
                    [ 8, 20, 32,  9, 29, 35],
                    [-1, 17, 13, 23, 10, -1]])
    int_selection = (0, 2, 5, 3, 1, 4), (0, 3, 0, 2, 1, 5)
    bool_selection = np.zeros((6, 6), bool)
    bool_selection[int_selection] = True
    layout = PackedTable((6, 6), ordering=ordering, all_val=val)
    selections = ((), [], Ellipsis, 2, (2,), [3], (0, 0), (1, 1), [1, 1],
                  (slice(0, 2), slice(0, 3)), bool_selection, int_selection)
    expecteds = (layout._index, [], layout._index,
                 [15, 16, 17, 12, 13, 14], [15, 16, 17, 12, 13, 14],
                 [20, 19, 18, 23, 22, 21], [], [7], [9, 10, 11, 6, 7, 8],
                 [6, 7, 1, 8, 2], [15, 7, 20, 29], [15, 7, 20, 29])

    def func(selection, expected):
        selected = layout.all[selection]
        assert selected._index.dtype == np.int32
        assert_equal(selected._index, expected)
        assert_equal(selected.all.val[selection], val[selection])
    for selection, expected in zip(selections, expecteds):
        yield func, selection, expected


def test_selection_packed1():
    # layout index is Ellipsis
    val = np.array([[12,  2, 18, 15, 22, 19],
                    [33, 21, 16, 26, 31,  1],
                    [30, 28,  7, 14,  0,  6],
                    [27,  4, 11, 24,  3,  5],
                    [ 8, 20, 32,  9, 29, 35],
                    [25, 17, 13, 23, 10, 34]])
    int_selection = [13, 14, 6, 19, 0, 35, 3]
    bool_selection = np.zeros(36, bool)
    bool_selection[int_selection] = True
    layout = PackedTable((6, 6), val=val)
    selections = ((), [], Ellipsis, 2, (0,), [3], slice(4, 9), slice(4, 16, 3),
                  [4, 7, 10, 13], int_selection, bool_selection)
    expecteds = (Ellipsis, [], Ellipsis, [2], [0], [3], slice(4, 9, 1),
                 slice(4, 16, 3), slice(4, 16, 3), int_selection,
                 sorted(int_selection))

    def func(selection, expected):
        selected = layout[selection]
        assert_equal(selected._index, expected)
        assert_equal(selected.val, layout.val[selection])
    for selection, expected in zip(selections, expecteds):
        yield func, selection, expected


def test_selection_packed2():
    # layout index is an array
    ordering = [[-1, 12, 15,  0,  1, -1],
                [ 9, 11, 14,  2,  3,  4],
                [ 8, 10, 13,  5,  6,  7],
                [23, 22, 21, 29, 26, 24],
                [20, 19, 18, 30, 27, 25],
                [-1, 17, 16, 31, 28, -1]]
    val = np.array([[-1,  2, 18, 15, 22, -1],
                    [33, 21, 16, 26, 31,  1],
                    [30, 28,  7, 14,  0,  6],
                    [27,  4, 11, 24,  3,  5],
                    [ 8, 20, 32,  9, 29, 35],
                    [-1, 17, 13, 23, 10, -1]])
    int_selection = [13, 14, 6, 19, 0, 31, 3]
    bool_selection = np.zeros(32, bool)
    bool_selection[int_selection] = True
    layout = PackedTable((6, 6), ordering=ordering, val=val)
    i = layout._index
    selections = ((), [], Ellipsis, 2, (0,), [3], slice(4, 9), slice(4, 16, 3),
                  int_selection, bool_selection)
    expecteds = (i, [], i, [9], [3], [10], [11, 15, 16, 17, 12],
                 [11, 17, 13, 14], i[int_selection], i[bool_selection])

    def func(selection, expected):
        selected = layout[selection]
        assert selected._index.dtype == layout._dtype_index
        assert_equal(selected._index, expected)
        assert_equal(selected.val, layout.val[selection])
    for selection, expected in zip(selections, expecteds):
        yield func, selection, expected


def test_selection_packed3():
    # layout index is a slice
    val = np.array([[12,  2, 18, 15, 22, 19],
                    [33, 21, 16, 26, 31,  1],
                    [30, 28,  7, 14,  0,  6],
                    [27,  4, 11, 24,  3,  5],
                    [ 8, 20, 32,  9, 29, 35],
                    [25, 17, 13, 23, 10, 34]])
    selection = np.zeros((6, 6), bool)
    selection.ravel()[4:30:3] = True
    layout = PackedTable((6, 6), selection=selection, val=val)
    i = layout._index  # slice(4, 31, 3): [ 4,  7, 10, 13, 16, 19, 22, 25, 28]
    int_selection = [4, 2, 8, 0, 7, 1]
    bool_selection = np.zeros(9, bool)
    bool_selection[int_selection] = True
    assert_equal(i, slice(4, 31, 3))
    selections = ((), [], Ellipsis, 2, (0,), [3], slice(2, 5), slice(2, 16, 2),
                  int_selection, bool_selection)
    expecteds = (i, [], i, [10], [4], [13], slice(10, 19, 3), slice(10, 34, 6),
                 [16, 10, 28,  4, 25,  7], [4, 7, 10, 16, 25, 28])

    def func(selection, expected):
        selected = layout[selection]
        assert_equal(selected._index, expected)
        assert_equal(selected.val, layout.val[selection])
    for selection, expected in zip(selections, expecteds):
        yield func, selection, expected


def test_split():
    def func(n, s, m):
        if s is not Ellipsis:
            tmp = np.ones(n, bool)
            tmp[s] = False
            s = tmp
        layout = PackedTable(n, selection=s, val=np.arange(n)*2.)
        slices = layout.split(m)
        assert_eq(len(slices), m)
        o = np.zeros(layout.shape, int)
        v = np.full(layout.shape, np.nan)
        for s in slices:
            o[s._index] += 1
            v[s._index] = s.val
        o[o == 0] = -1
        assert_same(o, layout.unpack(1))
        assert_same(v, layout.all.val)
        assert_same(np.concatenate([_.val for _ in slices]), layout.val)
    for n in range(1, 4):
        for s in (Ellipsis,) + tuple(range(n)):
            for m in range(1, 7):
                yield func, n, s, m
