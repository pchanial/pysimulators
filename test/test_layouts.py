from __future__ import division

import numpy as np

from numpy.testing import assert_equal, assert_raises
from pyoperators.utils import isalias
from pyoperators.utils.testing import (assert_same, assert_is_instance,
                                       assert_is_none, assert_is_type)
from pysimulators import Quantity
from pysimulators.geometry import create_grid, create_grid_squares
from pysimulators.layouts import (Layout, LayoutGrid, LayoutGridCircles,
                                  LayoutGridSquares)


def test_dimensions():
    shapes = ((), (1,), (1, 2), (1, 2, 3))

    def func(s):
        layout = Layout(s)
        assert_equal(layout.ndim, len(s))
        assert_equal(layout.shape, s)
    for s in shapes:
        yield func, s


def test_reserved():
    layout = Layout((6, 6))

    def func(key):
        assert_raises(KeyError, layout.setattr_packed, key, 1)
        assert_raises(KeyError, layout.setattr_unpacked, key, 1)

    for key in layout._reserved_attributes:
        yield func, key


def test_special_attribute_array():
    layout = Layout((6, 6))
    val = np.arange(len(layout))

    def func(setattr_, v):
        setattr_('key', v)
        assert isalias(layout.key, val)
        assert isalias(layout.packed.key, val)
        assert_same(layout.key.shape, layout.shape)
        assert_same(layout.packed.key.shape, (len(layout),))
    for setattr_, v in zip((layout.setattr_unpacked, layout.setattr_packed),
                           (val.reshape(layout.shape), val)):
        yield func, setattr_, v


def test_special_attribute_func():
    layout = Layout((6, 6))
    val = np.arange(len(layout))

    def func(setattr_, v):
        setattr_('key', v)
        assert isalias(layout.key, val)
        assert isalias(layout.packed.key, val)
        assert_same(layout.key.shape, layout.shape)
        assert_same(layout.packed.key.shape, (len(layout),))
    for setattr_, v in zip((layout.setattr_unpacked, layout.setattr_packed),
                           (lambda: val.reshape(layout.shape), lambda: val)):
        yield func, setattr_, v


def test_special_attribute_scalar():
    layout = Layout((6, 6))
    val = np.array(10.)

    def func(setattr_, v):
        setattr_('key', v)
        assert isalias(layout.key, val)
        assert isalias(layout.packed.key, val)
        assert_same(layout.key.shape, ())
        assert_same(layout.packed.key.shape, ())
    for setattr_ in (layout.setattr_unpacked, layout.setattr_packed):
        yield func, setattr_, val


def test_special_attribute_none():
    layout = Layout((6, 6))

    def func(setattr_):
        setattr_('key', None)
        assert_is_none(layout.key)
        assert_is_none(layout.packed.key)
    for setattr_ in (layout.setattr_unpacked, layout.setattr_packed):
        yield func, setattr_


def test_error():
    shape = (4,)
    layout = Layout(shape)
    assert_raises(KeyError, setattr, layout.packed, 'mykey', 'value')
    assert_raises(ValueError, Layout, shape, removed=[True, False])
    assert_raises(ValueError, Layout, shape, index=[1, 3])
    assert_raises(ValueError, layout.setattr_packed, 'mykey', [1, 2, 3])
    assert_raises(ValueError, layout.setattr_unpacked, 'mykey', [1, 2, 3])
    assert_raises(ValueError, layout.pack, [1, 2, 3])
    assert_raises(ValueError, layout.unpack, [1, 2, 3])
    assert_raises(TypeError, layout.pack, [1, 2, 3, 4], out=[])
    assert_raises(TypeError, layout.unpack, [1, 2, 3, 4], out=[])
    assert_raises(ValueError, layout.pack, [1, 2, 3, 4], out=np.array([1]))
    assert_raises(ValueError, layout.unpack, [1, 2, 3, 4], out=np.array([1]))
    assert_raises(ValueError, LayoutGrid, shape, 0.1, origin=[1, 2])

    def func1():
        layout.removed = True
    assert_raises(RuntimeError, func1)

    def func2():
        l = Layout(shape, removed=[True, False, False, True])
        l.removed[1] = True
    assert_raises(RuntimeError, func2)


def test_zero_component():
    removeds = (True, (True, True, False, False), False)
    indexs = (None, (1, 4, -1, -1), (-1, -1, -1, -1))

    def func(removed, index):
        layout = Layout((4,), removed=removed, index=index, key=[1, 2, 3, 4])
        assert_equal(len(layout), 4)
        assert_equal(len(layout.packed), 0)
        assert_equal(layout.key, [-1, -1, -1, -1])
        assert_equal(layout.packed.key.size, 0)
    for removed, index in zip(removeds, indexs):
        yield func, removed, index


def test_center():
    shape = (4, 4)
    vertex = create_grid_squares(shape, 0.1, filling_factor=0.8)
    center = np.mean(vertex, axis=-2)

    def func(setattr_, v):
        layout = Layout(shape)
        getattr(layout, setattr_)('vertex', v)
        assert_same(layout.center, center)
        assert_same(layout.packed.center, center.reshape((-1, 2)))
    for setattr_, v in zip(('setattr_unpacked', 'setattr_packed'),
                           (vertex, vertex.reshape((-1, 4, 2)))):
        yield func, setattr_, v


def test_pack_none():
    layout = Layout((3, 3))
    assert_is_none(layout.pack(None))
    assert_is_none(layout.unpack(None))


def test_packunpack():
    shape = (3, 3)
    size = 9
    index1 = np.arange(size)[::-1].reshape(shape)
    index2 = index1.copy()
    index2[0, 2] = -1
    removed = [[True,  False, False],
               [False, False, False],
               [True,  False, True ]]
    vertex = create_grid_squares(shape, 0.1)
    center = np.mean(vertex, axis=-1)
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

    def func(r, i, d):
        layout = Layout(shape, vertex=vertex, center=center, removed=r,
                        index=i)
        packed = layout.pack(d)
        assert_equal(packed.shape, (len(layout.packed),) + d.shape[2:])
        d_ = layout.unpack(packed)
        assert_equal(d_.shape, d.shape)
        assert np.all((d == d_) | missing(d_))
        out_packed = np.empty((len(layout.packed),) + d.shape[2:],
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

    for r, i in ((False, None), (False, index1), (False, index2),
                 (removed, None), (removed, index1), (removed, index2)):
        for d in data:
            yield func, r, i, d


def test_removed1():
    removed = np.array([[True,  False, True ],
                        [False, False, False],
                        [True,  False, True ]])
    val = np.arange(9, dtype=int).reshape((3, 3))
    layout = Layout((3, 3), removed=removed, key=val)
    uexpected = val.copy()
    uexpected[removed] = -1
    pexpected = [1, 3, 4, 5, 7]
    assert_equal(layout.key, uexpected)
    assert_equal(layout.packed.key, pexpected)


def test_removed2():
    removed = np.array([[True,  False, True ],
                        [False, False, False],
                        [True,  False, True ]])
    val = np.arange(9, dtype=float).reshape((3, 3))
    layout = Layout((3, 3), removed=removed, key=val)
    uexpected = val.copy()
    uexpected[removed] = np.nan
    pexpected = [1, 3, 4, 5, 7]
    assert_equal(layout.key, uexpected)
    assert_equal(layout.packed.key, pexpected)


def test_index1():
    index = [[11, 14, 17,  0,  1,  2],
             [10, 13, 16,  3,  4,  5],
             [ 9, 12, 15,  6,  7,  8],
             [26, 25, 24, 33, 30, 27],
             [23, 22, 21, 34, 31, 28],
             [20, 19, 18, 35, 32, 29]]
    layout = Layout((6, 6), index=index)
    expected = [ 3,  4,  5,  9, 10, 11, 15, 16, 17,
                12,  6,  0, 13,  7,  1, 14,  8,  2,
                32, 31, 30, 26, 25, 24, 20, 19, 18,
                23, 29, 35, 22, 28, 34, 21, 27, 33]
    assert_equal(layout.packed.index, expected)


def test_index2():
    index = [[11, 14, 17,  0,  1,  2],
             [10, 13, 16,  3,  4,  5],
             [ 9, 12, 15,  6,  7,  8],
             [26, 25, 24, 33, 30, 27],
             [23, 22, 21, 34, 31, 28],
             [20, 19, 18, 35, 32, 29]]
    removed = [[True,  False, False, False, False, True ],
               [False, False, False, False, False, False],
               [False, False, False, False, False, False],
               [False, False, False, False, False, False],
               [False, False, False, False, False, False],
               [True,  False, False, False, False, True ]]
    layout = Layout((6, 6), index=index, removed=removed)
    expected = [ 3,  4,  9, 10, 11, 15, 16, 17,
                12,  6, 13,  7,  1, 14,  8,  2,
                32, 31, 26, 25, 24, 20, 19, 18,
                23, 29, 22, 28, 34, 21, 27, 33]
    assert_equal(layout.packed.index, expected)


def test_index3():
    index = [[-1, 12, 15,  0,  1, -1],
             [ 9, 11, 14,  2,  3,  4],
             [ 8, 10, 13,  5,  6,  7],
             [23, 22, 21, 29, 26, 24],
             [20, 19, 18, 30, 27, 25],
             [-1, 17, 16, 31, 28, -1]]
    layout = Layout((6, 6), index=index)
    expected = [ 3,  4,  9, 10, 11, 15, 16, 17,
                12,  6, 13,  7,  1, 14,  8,  2,
                32, 31, 26, 25, 24, 20, 19, 18,
                23, 29, 22, 28, 34, 21, 27, 33]
    assert_equal(layout.packed.index, expected)


def test_origin():
    shape = (3, 1)
    center = create_grid(shape, 0.1)
    layout = Layout(shape, center=center)
    assert layout.nvertices == 0
    assert_same(layout.center, center)
    assert_same(layout.masked, False)
    assert_same(layout.removed, np.zeros(shape, bool))


def test_layout_vertex():
    shape = (3, 1)
    center = create_grid(shape, 0.1)
    vertex = create_grid_squares(shape, 0.1, filling_factor=0.9)
    layout = Layout(shape, vertex=vertex)
    assert layout.nvertices == 4
    assert_same(layout.center, center)
    assert_same(layout.vertex, vertex)
    assert_same(layout.masked, False)
    assert_same(layout.removed, np.zeros(shape, bool))


def test_layout_grid():
    shape = (3, 2)
    spacing = Quantity(0.1, 'mm')
    xreflection = True
    yreflection = True
    origin = (1, 1)
    angle = 10
    center = create_grid(shape, spacing, center=origin,
                         xreflection=xreflection, yreflection=yreflection,
                         angle=angle)
    layout = LayoutGrid(shape, spacing, origin=origin,
                        xreflection=xreflection, yreflection=yreflection,
                        angle=angle)
    assert layout.nvertices == 0
    assert not isinstance(layout.__dict__['center'], np.ndarray)
    assert_same(layout.center, center)
    assert_same(layout.masked, False)
    assert_same(layout.removed, np.zeros(shape, bool))


def test_layout_grid_circles():
    shape = (3, 2)
    spacing = 0.1
    xreflection = True
    yreflection = True
    origin = (1, 1)
    angle = 10
    center = create_grid(
        shape, spacing, center=origin, xreflection=xreflection,
        yreflection=yreflection, angle=angle)
    layout = LayoutGridCircles(
        shape, spacing, origin=origin, xreflection=xreflection,
        yreflection=yreflection, angle=angle)
    assert layout.nvertices == 0
    assert layout.radius == spacing / 2
    assert not isinstance(layout.__dict__['center'], np.ndarray)
    assert_same(layout.center, center)
    assert_same(layout.masked, False)
    assert_same(layout.removed, np.zeros(shape, bool))


def test_layout_grid_circles_unit():
    shape = (3, 2)
    removed = [[False, True ],
               [False, False],
               [False, False]]
    spacing = (1., Quantity(1.), Quantity(1., 'mm'))
    r = np.zeros(shape) + 0.4
    radius = (0.4, Quantity(0.4), Quantity(0.4, 'mm'), Quantity(0.4e-3, 'm'),
              r, Quantity(r), Quantity(r, 'mm'), Quantity(r*1e-3, 'm'))

    def func(s, r):
        layout = LayoutGridCircles(shape, s, radius=r, removed=removed)
        if (not isinstance(s, Quantity) or s.unit == '') and \
           isinstance(r, Quantity) and r.unit == 'm':
            expected = 0.4e-3
        else:
            expected = 0.4
        if np.size(r) > 1:
            expected, val = np.empty(shape), expected
            expected[...] = val
            expected[0, 1] = np.nan
        assert_same(layout.radius, expected, broadcasting=True)
        vals = (layout.radius, layout.packed.radius, layout.center,
                layout.packed.center)
        unit = getattr(s, 'unit', '') or getattr(r, 'unit', '')
        if unit:
            for val in vals:
                assert_is_instance(val, Quantity)
                assert_equal(val.unit, unit)
        else:
            r = np.asanyarray(r)
            expecteds = (type(r), type(r), np.ndarray, np.ndarray)
            for val, e in zip(vals, expecteds):
                assert_is_type(val, e)
    for s in spacing:
        for r in radius:
            yield func, s, r


def test_layout_grid_squares():
    shape = (3, 2)
    spacing = 0.1
    filling_factor = 0.9
    xreflection = True
    yreflection = True
    origin = (1, 1)
    angle = 10
    vertex = create_grid_squares(
        shape, spacing, filling_factor=filling_factor, center=origin,
        xreflection=xreflection, yreflection=yreflection, angle=angle)
    layout = LayoutGridSquares(
        shape, spacing, filling_factor=filling_factor,
        origin=origin, xreflection=xreflection,
        yreflection=yreflection, angle=angle)
    assert layout.nvertices == 4
    assert not isinstance(layout.__dict__['center'], np.ndarray)
    assert not isinstance(layout.__dict__['vertex'], np.ndarray)
    assert_same(layout.center, np.mean(vertex, axis=-2))
    assert_same(layout.vertex, vertex)
    assert_same(layout.masked, False)
    assert_same(layout.removed, np.zeros(shape, bool))


def test_layout_grid_squares_unit():
    shape = (3, 2)
    removed = [[False, False],
               [False, False],
               [False, False]]
    spacing = (1., Quantity(1.), Quantity(1., 'mm'))
    lcenter = ((1., 1.), Quantity((1., 1.)), Quantity((1., 1.), 'mm'),
               Quantity((1e-3, 1e-3), 'm'))

    def func(s, l):
        layout = LayoutGridSquares(shape, s, removed=removed, origin=l)
        if (not isinstance(s, Quantity) or s.unit == '') and \
           isinstance(l, Quantity) and l.unit == 'm':
            expected = np.array((1e-3, 1e-3))
        else:
            expected = np.array((1, 1))
        actual = np.mean(layout.packed.center, axis=0).view(np.ndarray)
        assert_same(actual, expected, rtol=1000)
        vals = (layout.center, layout.packed.center, layout.vertex,
                layout.packed.vertex)
        unit = getattr(s, 'unit', '') or getattr(l, 'unit', '')
        if unit:
            for val in vals:
                assert_is_instance(val, Quantity)
                assert_equal(val.unit, unit)
        else:
            for val in vals:
                assert_is_type(val, np.ndarray)
    for s in spacing:
        for l in lcenter:
            yield func, s, l
