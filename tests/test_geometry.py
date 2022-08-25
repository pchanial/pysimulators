import numpy as np
import pytest

from pyoperators.utils import product
from pyoperators.utils.testing import assert_same
from pysimulators.geometry import (
    convex_hull,
    create_circle,
    create_grid,
    create_grid_squares,
    create_rectangle,
    create_regular_polygon,
    create_square,
    rotate,
    surface_simple_polygon,
)

from .common import BIGGEST_FLOAT_TYPE, FLOAT_TYPES

SHAPES = [(), (1,), (1, 1), (1, 2), (2, 1), (2, 2)]

CONVEX_HULL_SQUARE = np.array([[1.0, 1], [1, -1], [-1, -1], [-1, 1]])


@pytest.mark.parametrize(
    'point',
    [
        CONVEX_HULL_SQUARE,
        np.vstack((CONVEX_HULL_SQUARE, [[0, 0]])),
        np.vstack((CONVEX_HULL_SQUARE, [[-1, 1]])),
    ],
)
def test_convex_hull(point):
    expected = CONVEX_HULL_SQUARE[[0, 3, 2, 1]]
    h = convex_hull(point)
    assert_same(h, expected)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_rotate(shape, dtype):
    expected = [
        np.array(1, BIGGEST_FLOAT_TYPE) / 2,
        np.sqrt(np.array(3, BIGGEST_FLOAT_TYPE)) / 2,
    ]

    sqrt3 = np.sqrt(np.array(3, dtype))
    c = [sqrt3 / 2, np.array(1, dtype) / 2]
    coords = np.empty(shape + (2,), dtype)
    coords[...] = c
    assert_same(rotate(coords, 30), expected, broadcasting=True)
    assert_same(rotate(coords, 30, out=coords), expected, broadcasting=True)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_circle(shape, dtype):
    origin = [1, 1]

    n = max(product(shape), 1)
    radius = np.arange(1, n + 1, dtype=dtype).reshape(shape)
    circle = create_circle(radius, center=origin, dtype=dtype)
    actual = np.sqrt(np.sum((circle - origin) ** 2, axis=-1))
    assert_same(actual, radius[..., None], broadcasting=True)


@pytest.mark.parametrize(
    'shape', [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
)
@pytest.mark.parametrize('xreflection', [False, True])
@pytest.mark.parametrize('yreflection', [False, True])
def test_grid(shape, xreflection, yreflection):
    def create_grid_slow(
        shape, spacing, xreflection=False, yreflection=True, center=(0, 0)
    ):
        x = np.arange(shape[1], dtype=float) * spacing
        x -= x.mean()
        if xreflection:
            x = x[::-1]
        y = (np.arange(shape[0], dtype=float) * spacing)[::-1]
        y -= y.mean()
        if yreflection:
            y = y[::-1]
        grid_x, grid_y = np.meshgrid(x, y)
        return np.asarray([grid_x.T, grid_y.T]).T + center

    spacing = 0.4
    origin = [1, 1]

    actual = create_grid(
        shape,
        spacing,
        xreflection=xreflection,
        yreflection=yreflection,
        center=origin,
    )
    expected = create_grid_slow(
        shape,
        spacing,
        xreflection=xreflection,
        yreflection=yreflection,
        center=origin,
    )
    assert_same(actual, expected)


@pytest.mark.parametrize(
    'shape', [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
)
@pytest.mark.parametrize('xreflection', [False, True])
@pytest.mark.parametrize('yreflection', [False, True])
def test_grid_squares(shape, xreflection, yreflection):
    def rotate_slow(x, angle, xreflection=False, yreflection=False):
        angle = np.radians(angle)
        r11 = np.cos(angle)
        r21 = np.sin(angle)
        r12 = -r21
        r22 = r11
        if xreflection:
            r11 = -r11
            r21 = -r21
        if yreflection:
            r12 = -r12
            r22 = -r22
        m = np.array([[r11, r12], [r21, r22]])
        return np.dot(x, m.T)

    def create_grid_squares_slow(
        shape,
        spacing,
        filling_factor=1,
        xreflection=False,
        yreflection=False,
        center=(0, 0),
        angle=0,
    ):
        x = np.arange(shape[1], dtype=float) * spacing
        x -= x.mean()
        y = (np.arange(shape[0], dtype=float) * spacing)[::-1]
        y -= y.mean()
        grid_x, grid_y = np.meshgrid(x, y)
        nodes = np.asarray([grid_x.T, grid_y.T]).T
        d = np.sqrt(filling_factor * spacing**2) / 2
        offset = np.array([[d, d], [-d, d], [-d, -d], [d, -d]])
        corner = nodes[..., None, :] + offset
        corner = rotate_slow(corner, angle, xreflection, yreflection)
        return corner + center

    spacing = 0.4
    filling_factor = 0.5
    origin = [1, 1]

    actual = create_grid_squares(
        shape,
        spacing,
        filling_factor=filling_factor,
        xreflection=xreflection,
        yreflection=yreflection,
        center=origin,
    )
    expected = create_grid_squares_slow(
        shape,
        spacing,
        filling_factor=filling_factor,
        xreflection=xreflection,
        yreflection=yreflection,
        center=origin,
    )
    assert_same(actual, expected)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_square(dtype, shape):
    origin = np.asarray([1, 1])

    n = max(product(shape), 1)
    size = np.arange(1, n + 1, dtype=dtype).reshape(shape)
    actual = create_square(size, center=origin, dtype=dtype)
    expected = np.empty(shape + (4, 2))
    expected[..., 0, 0] = origin[0] + size / 2
    expected[..., 0, 1] = origin[1] + size / 2
    expected[..., 1, 0] = origin[0] - size / 2
    expected[..., 1, 1] = origin[1] + size / 2
    expected[..., 2, 0] = origin[0] - size / 2
    expected[..., 2, 1] = origin[1] - size / 2
    expected[..., 3, 0] = origin[0] + size / 2
    expected[..., 3, 1] = origin[1] - size / 2
    assert_same(actual, expected)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_rectangle(shape, dtype):
    origin = np.asarray([1, 1])

    n = max(product(shape), 1)
    size_x = np.arange(1, n + 1, dtype=dtype).reshape(shape)
    size_y = size_x / 2
    actual = create_rectangle(
        np.asarray([size_x.T, size_y.T]).T, center=origin, dtype=dtype
    )
    expected = np.empty(shape + (4, 2))
    expected[..., 0, 0] = origin[0] + size_x / 2
    expected[..., 0, 1] = origin[1] + size_y / 2
    expected[..., 1, 0] = origin[0] - size_x / 2
    expected[..., 1, 1] = origin[1] + size_y / 2
    expected[..., 2, 0] = origin[0] - size_x / 2
    expected[..., 2, 1] = origin[1] - size_y / 2
    expected[..., 3, 0] = origin[0] + size_x / 2
    expected[..., 3, 1] = origin[1] - size_y / 2
    assert_same(actual, expected)


@pytest.mark.parametrize('radius', [np.int64(1), np.array([1, 2])])
@pytest.mark.parametrize(
    'nvertex, expected', ((n, n * np.sin(2 * np.pi / n) / 2) for n in [3, 4, 5, 6, 7])
)
def test_surface_simple_polygon(radius, nvertex, expected):
    origin = np.asarray([1, 1])

    polygon = create_regular_polygon(nvertex, radius, center=origin)
    assert_same(surface_simple_polygon(polygon), expected * radius**2)
    out = np.empty(np.shape(radius))
    surface_simple_polygon(polygon, out=out)
    assert_same(out, expected * radius**2)
