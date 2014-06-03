from __future__ import division

import numpy as np
from numpy.testing import assert_equal, assert_raises
from pyoperators.utils import isalias, isscalarlike, product
from pyoperators.utils.testing import (
    assert_eq, assert_is, assert_is_instance, assert_is_none, assert_is_type,
    assert_same, skiptest)
from pysimulators import Quantity
from pysimulators.geometry import create_grid, create_grid_squares
from pysimulators import (
    Layout, LayoutGrid, LayoutGridCircles, LayoutVertex, LayoutGridSquares)


def test_layout_grid_errors():
    assert_raises(TypeError, LayoutGrid, 3, 1.2)
    assert_raises(ValueError, LayoutGrid, (1,), 1.2)
    assert_raises(ValueError, LayoutGrid, (1, 2, 3), 1.2)
    assert_raises(ValueError, LayoutGrid, (4, 2), 0.1, origin=[1])


def test_layout():
    shape = (4, 4)
    center = create_grid(shape, 0.1)
    get_center = lambda: center.reshape(-1, 2)

    class Spatial1(Layout):
        @property
        def center(self):
            return get_center()

    class Spatial2(Layout):
        def center(self):
            return get_center()

    layouts = (Layout(shape, center=center),
               Layout(shape, center=get_center),
               Spatial1(shape), Spatial2(shape))

    def func(layout):
        assert_same(layout.center, center.reshape(-1, 2))
        assert_same(layout.all.center, center)
    for layout in layouts:
        yield func, layout


def test_layout_vertex():
    shape = (4, 4)
    vertex = create_grid_squares(shape, 0.1, filling_factor=0.8)
    center = np.mean(vertex, axis=-2)
    get_vertex = lambda: vertex.reshape(-1, 4, 2)

    class Vertex1(LayoutVertex):
        @property
        def vertex(self):
            return get_vertex()

    class Vertex2(LayoutVertex):
        def vertex(self):
            return get_vertex()

    layouts = (LayoutVertex(shape, 4, vertex=vertex),
               LayoutVertex(shape, 4, vertex=get_vertex),
               Vertex1(shape, 4), Vertex2(shape, 4))

    def func(layout):
        assert_equal(layout.nvertices, 4)
        assert_same(layout.center, center.reshape((-1, 2)))
        assert_same(layout.all.center, center)
        assert_same(layout.vertex, vertex.reshape((-1, 4, 2)))
        assert_same(layout.all.vertex, vertex)
    for layout in layouts:
        yield func, layout


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
    layout = LayoutGrid(
        shape, spacing, origin=origin, xreflection=xreflection,
        yreflection=yreflection, angle=angle)
    assert not hasattr(layout, 'nvertices')
    assert_same(layout.all.center, center)
    assert_same(layout.all.removed, np.zeros(shape, bool))


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
    assert not hasattr(layout, 'nvertices')
    assert_same(layout.radius, spacing / 2)
    assert_same(layout.all.center, center)
    assert_same(layout.all.removed, np.zeros(shape, bool))


def test_layout_grid_circles_unit():
    shape = (3, 2)
    selection = [[True, False],
                 [True, True ],
                 [True, True ]]
    spacing = (1., Quantity(1.), Quantity(1., 'mm'))
    r = np.zeros(shape) + 0.4
    radius = (0.4, Quantity(0.4), Quantity(0.4, 'mm'), Quantity(0.4e-3, 'm'),
              r, Quantity(r), Quantity(r, 'mm'), Quantity(r*1e-3, 'm'))

    def func(s, r):
        layout = LayoutGridCircles(shape, s, radius=r,
                                   selection=selection)
        if (not isinstance(s, Quantity) or s.unit == '') and \
           isinstance(r, Quantity) and r.unit == 'm':
            expectedval = 0.4e-3
        else:
            expectedval = 0.4
        expected = np.empty(shape)
        expected[...] = expectedval
        expected[0, 1] = np.nan
        assert_same(layout.radius, expectedval, broadcasting=True)
        assert_same(layout.all.radius, expected)
        vals = (layout.radius, layout.all.radius, layout.center,
                layout.all.center)
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
    assert_same(layout.all.center, np.mean(vertex, axis=-2))
    assert_same(layout.all.vertex, vertex)
    assert_same(layout.all.removed, np.zeros(shape, bool))


def test_layout_grid_squares_unit():
    shape = (3, 2)
    selection = [[True, True],
                 [True, True],
                 [True, True]]
    spacing = (1., Quantity(1.), Quantity(1., 'mm'))
    lcenter = ((1., 1.), Quantity((1., 1.)), Quantity((1., 1.), 'mm'),
               Quantity((1e-3, 1e-3), 'm'))

    def func(s, l):
        layout = LayoutGridSquares(shape, s, selection=selection,
                                   origin=l)
        if (not isinstance(s, Quantity) or s.unit == '') and \
           isinstance(l, Quantity) and l.unit == 'm':
            expected = np.array((1e-3, 1e-3))
        else:
            expected = np.array((1, 1))
        actual = np.mean(layout.center, axis=0).view(np.ndarray)
        assert_same(actual, expected, rtol=1000)
        vals = (layout.center, layout.all.center, layout.vertex,
                layout.all.vertex)
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


def test_layout_grid_colrow():
    ordering = [[-1, 3, 0],
                [1, -1, -1],
                [-1, 2, -1]]
    expected_row = [[-1, 0, 0],
                    [1, -1, -1],
                    [-1, 2, -1]]
    expected_col = [[-1, 1, 2],
                    [0, -1, -1],
                    [-1, 1, -1]]
    grid = LayoutGrid((3, 3), 1.2, ordering=ordering)
    assert_equal(grid.all.column, expected_col)
    assert_equal(grid.all.row, expected_row)
