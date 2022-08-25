import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators.utils.testing import assert_same
from pysimulators import LayoutGridSquares, Quantity, SceneGrid, create_fitsheader
from pysimulators.geometry import create_grid_squares

ITYPES = np.int32, np.int64
FTYPES = np.float32, np.float64


@pytest.mark.parametrize(
    'center, expected',
    [
        ((0, 0), [[9, 10], [5, 6]]),
        (
            (-0.5, 0.5),
            [
                [0.25 * (8 + 9 + 12 + 13), 0.25 * (9 + 10 + 13 + 14)],
                [0.25 * (4 + 5 + 8 + 9), 0.25 * (5 + 6 + 9 + 10)],
            ],
        ),
    ],
)
@pytest.mark.parametrize('ncolmax', [0, 1, 4, 5])
@pytest.mark.parametrize('itype', ITYPES)
@pytest.mark.parametrize('ftype', FTYPES)
def test_spatial_integration(center, ncolmax, itype, ftype, expected):
    #  -------------
    #  |12|13|14|15|
    #  |--+--+--+--+
    #  | 8| 9|10|11|
    #  |--+--+--+--+
    #  | 4| 5| 6| 7|
    #  |--+--+--+--+
    #  | 0| 1| 2| 3|
    #  -------------
    #  The plane center has world coordinates (0, 0)

    plane_shape = (4, 4)
    pixsize = 1  # um
    header = create_fitsheader(
        plane_shape,
        ctype=['X---CAR', 'Y---CAR'],
        cdelt=[pixsize, pixsize],
        crval=(0, 0),
        cunit=['um', 'um'],
    )
    plane = SceneGrid.fromfits(header)

    detectors = create_grid_squares((2, 2), Quantity(pixsize, 'um'), center=center)
    proj = plane.get_integration_operator(
        plane.topixel(detectors), ncolmax=ncolmax, dtype=ftype, dtype_index=itype
    )
    x = np.arange(len(plane)).reshape(plane_shape)
    y = proj(x)

    assert_equal(proj.matrix.dtype, ftype)
    assert_equal(proj.matrix.data.index.dtype, itype)
    assert_equal(proj.shapein, plane.shape)
    assert_equal(proj.shapeout, detectors.shape[:-2])
    assert_equal(proj.matrix.shape, (4, len(plane)))
    expected_min_ncolmax = 1 if center == (0, 0) else 4
    assert_equal(proj.matrix.data.shape, (4, max(expected_min_ncolmax, ncolmax)))
    assert_equal(proj.min_ncolmax, expected_min_ncolmax)
    assert_same(y, expected)


@pytest.mark.parametrize('spacing', [0.5, 1, 2])
@pytest.mark.parametrize('angle', [0, 30])
def test_spatial_integration2(spacing, angle):
    shape_grid = (16, 16)
    filling_factor = 0.8

    origin = shape_grid * np.array(spacing) * np.sqrt(2) / 2
    grid = LayoutGridSquares(
        shape_grid,
        spacing,
        origin=origin,
        angle=angle,
        filling_factor=filling_factor,
    )
    shape_plane = np.ceil(shape_grid * np.array(spacing) * np.sqrt(2))
    scene = SceneGrid(shape_plane)
    proj = scene.get_integration_operator(grid.vertex)
    assert not proj.outside
    assert_same(
        np.sum(proj.matrix.data.value),
        len(grid) * spacing**2 * filling_factor,
        rtol=100,
    )
