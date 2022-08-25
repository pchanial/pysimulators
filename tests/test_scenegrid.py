from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from pyoperators.utils.testing import assert_same
from pysimulators import create_fitsheader, LayoutGridSquares, Quantity, SceneGrid
from pysimulators.geometry import create_grid_squares

itypes = np.int32, np.int64
ftypes = np.float32, np.float64


def test_spatial_integration():
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

    def func(center, ncolmax, itype, ftype, expected):
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

    centers = [(0, 0), (-0.5, 0.5)]
    ncolmaxs = (0, 1, 4, 5)
    expecteds = [
        [[9, 10], [5, 6]],
        [
            [0.25 * (8 + 9 + 12 + 13), 0.25 * (9 + 10 + 13 + 14)],
            [0.25 * (4 + 5 + 8 + 9), 0.25 * (5 + 6 + 9 + 10)],
        ],
    ]
    for center, expected in zip(centers, expecteds):
        for ncolmax in ncolmaxs:
            for itype in itypes:
                for ftype in ftypes:
                    yield func, center, ncolmax, itype, ftype, expected


def test_spatial_integration2():
    shape_grid = (16, 16)
    spacings = 0.5, 1, 2
    angles = 0, 30
    filling_factor = 0.8

    def func(spacing, angle):
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

    for spacing in spacings:
        for angle in angles:
            yield func, spacing, angle
