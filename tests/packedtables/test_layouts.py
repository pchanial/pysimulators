import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators.utils.testing import assert_same
from pysimulators import Layout, LayoutGrid, LayoutGridSquares, Quantity
from pysimulators.geometry import create_grid, create_grid_squares


def test_layout_grid_errors():
    with pytest.raises(TypeError):
        LayoutGrid(3, 1.2)
    with pytest.raises(ValueError):
        LayoutGrid((1,), 1.2)
    with pytest.raises(ValueError):
        LayoutGrid((1, 2, 3), 1.2)
    with pytest.raises(ValueError):
        LayoutGrid((4, 2), 0.1, origin=[1])


LAYOUT_SHAPE = (4, 4)
LAYOUT_CENTERS = create_grid(LAYOUT_SHAPE, 0.1)


class LayoutProperty(Layout):
    @property
    def center(self):
        return LAYOUT_CENTERS.reshape(-1, 2)


class LayoutMethod(Layout):
    def center(self):
        return LAYOUT_CENTERS.reshape(-1, 2)


@pytest.mark.parametrize(
    'cls, center',
    [
        (Layout, LAYOUT_CENTERS),
        (Layout, lambda: LAYOUT_CENTERS.reshape(-1, 2)),
        (LayoutProperty, None),
        (LayoutMethod, None),
    ],
)
def test_layout(cls, center):
    layout = cls(LAYOUT_SHAPE, center=center)
    assert_same(layout.center, LAYOUT_CENTERS.reshape(-1, 2))
    assert_same(layout.all.center, LAYOUT_CENTERS)


LAYOUT_VERTICES = create_grid_squares(LAYOUT_SHAPE, 0.1, filling_factor=0.8)


class LayoutVertexProperty(Layout):
    @property
    def vertex(self):
        return LAYOUT_VERTICES.reshape(-1, 4, 2)


class LayoutVertexMethod(Layout):
    def vertex(self):
        return LAYOUT_VERTICES.reshape(-1, 4, 2)


@pytest.mark.parametrize(
    'cls, vertex',
    [
        (Layout, LAYOUT_VERTICES),
        (Layout, lambda: LAYOUT_VERTICES.reshape(-1, 4, 2)),
        (LayoutVertexProperty, None),
        (LayoutVertexMethod, None),
    ],
)
def test_layout_vertex(cls, vertex):
    layout = cls(LAYOUT_SHAPE, vertex=vertex)
    center = np.mean(LAYOUT_VERTICES, axis=-2)

    assert layout.nvertices == 4
    assert_same(layout.center, center.reshape((-1, 2)))
    assert_same(layout.all.center, center)
    assert_same(layout.vertex, LAYOUT_VERTICES.reshape((-1, 4, 2)))
    assert_same(layout.all.vertex, LAYOUT_VERTICES)


def test_layout_grid():
    shape = (3, 2)
    spacing = Quantity(0.1, 'mm')
    xreflection = True
    yreflection = True
    origin = (1, 1)
    angle = 10
    center = create_grid(
        shape,
        spacing,
        center=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
    )
    layout = LayoutGrid(
        shape,
        spacing,
        origin=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
    )
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
        shape,
        spacing,
        center=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
    )
    layout = LayoutGrid(
        shape,
        spacing,
        origin=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
        radius=spacing / 2,
    )
    assert not hasattr(layout, 'nvertices')
    assert_same(layout.radius, spacing / 2)
    assert_same(layout.all.center, center)
    assert_same(layout.all.removed, np.zeros(shape, bool))


LAYOUT_GRID_CIRCLES_UNIT_SHAPE = (3, 2)


@pytest.mark.parametrize('spacing', [1.0, Quantity(1.0), Quantity(1.0, 'mm')])
@pytest.mark.parametrize(
    'radius',
    [
        0.4,
        Quantity(0.4),
        Quantity(0.4, 'mm'),
        Quantity(0.4e-3, 'm'),
        np.zeros(LAYOUT_GRID_CIRCLES_UNIT_SHAPE) + 0.4,
        Quantity(np.zeros(LAYOUT_GRID_CIRCLES_UNIT_SHAPE) + 0.4),
        Quantity(np.zeros(LAYOUT_GRID_CIRCLES_UNIT_SHAPE) + 0.4, 'mm'),
        Quantity((np.zeros(LAYOUT_GRID_CIRCLES_UNIT_SHAPE) + 0.4) * 1e-3, 'm'),
    ],
)
def test_layout_grid_circles_unit(spacing, radius):
    selection = [[True, False], [True, True], [True, True]]

    layout = LayoutGrid(
        LAYOUT_GRID_CIRCLES_UNIT_SHAPE, spacing, radius=radius, selection=selection
    )
    if (
        (not isinstance(spacing, Quantity) or spacing.unit == '')
        and isinstance(radius, Quantity)
        and radius.unit == 'm'
    ):
        expectedval = 0.4e-3
    else:
        expectedval = 0.4
    expected = np.empty(LAYOUT_GRID_CIRCLES_UNIT_SHAPE)
    expected[...] = expectedval
    expected[0, 1] = np.nan
    assert_same(layout.radius, expectedval, broadcasting=True)
    assert_same(layout.all.radius, expected)
    vals = (layout.radius, layout.all.radius, layout.center, layout.all.center)
    unit = getattr(spacing, 'unit', '') or getattr(radius, 'unit', '')
    if unit:
        for val in vals:
            assert isinstance(val, Quantity)
            assert val.unit == unit
    else:
        radius = np.asanyarray(radius)
        expecteds = (type(radius), type(radius), np.ndarray, np.ndarray)
        for val, e in zip(vals, expecteds):
            assert type(val) is e


def test_layout_grid_squares():
    shape = (3, 2)
    spacing = 0.1
    filling_factor = 0.9
    xreflection = True
    yreflection = True
    origin = (1, 1)
    angle = 10
    vertex = create_grid_squares(
        shape,
        spacing,
        filling_factor=filling_factor,
        center=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
    )
    layout = LayoutGridSquares(
        shape,
        spacing,
        filling_factor=filling_factor,
        origin=origin,
        xreflection=xreflection,
        yreflection=yreflection,
        angle=angle,
    )
    assert layout.nvertices == 4
    assert_same(layout.all.center, np.mean(vertex, axis=-2))
    assert_same(layout.all.vertex, vertex)
    assert_same(layout.all.removed, np.zeros(shape, bool))


@pytest.mark.parametrize('spacing', [1.0, Quantity(1.0), Quantity(1.0, 'mm')])
@pytest.mark.parametrize(
    'lcenter',
    [
        (1.0, 1.0),
        Quantity((1.0, 1.0)),
        Quantity((1.0, 1.0), 'mm'),
        Quantity((1e-3, 1e-3), 'm'),
    ],
)
def test_layout_grid_squares_unit(spacing, lcenter):
    shape = (3, 2)
    selection = [[True, True], [True, True], [True, True]]

    layout = LayoutGridSquares(shape, spacing, selection=selection, origin=lcenter)
    if (
        (not isinstance(spacing, Quantity) or spacing.unit == '')
        and isinstance(lcenter, Quantity)
        and lcenter.unit == 'm'
    ):
        expected = np.array((1e-3, 1e-3))
    else:
        expected = np.array((1, 1))
    actual = np.mean(layout.center, axis=0).view(np.ndarray)
    assert_same(actual, expected, rtol=1000)
    vals = (layout.center, layout.all.center, layout.vertex, layout.all.vertex)
    unit = getattr(spacing, 'unit', '') or getattr(lcenter, 'unit', '')
    if unit:
        for val in vals:
            assert isinstance(val, Quantity)
            assert val.unit == unit
    else:
        for val in vals:
            assert type(val) is np.ndarray


def test_layout_grid_colrow():
    ordering = [[-1, 3, 0], [1, -1, -1], [-1, 2, -1]]
    expected_row = [[-1, 0, 0], [1, -1, -1], [-1, 2, -1]]
    expected_col = [[-1, 1, 2], [0, -1, -1], [-1, 1, -1]]
    grid = LayoutGrid((3, 3), 1.2, ordering=ordering)
    assert_equal(grid.all.column, expected_col)
    assert_equal(grid.all.row, expected_row)
