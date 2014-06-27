from __future__ import division

import numpy as np
from numpy.testing import assert_equal
from pyoperators.utils import product
from pyoperators.utils.testing import assert_same
from pysimulators import Quantity, create_fitsheader, SceneGrid
from pysimulators.geometry import create_grid_squares


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
    header = create_fitsheader(plane_shape, ctype=['X---CAR', 'Y---CAR'],
                               cdelt=[pixsize, pixsize],
                               crval=(0, 0), cunit=['um', 'um'])
    plane = SceneGrid.fromfits(header)

    def func(center, ncolmax, expected):
        detectors = create_grid_squares((2, 2), Quantity(pixsize, 'um'),
                                        center=center)
        proj = plane.get_integration_operator(plane.topixel(detectors),
                                              ncolmax=ncolmax)
        x = np.arange(len(plane)).reshape(plane_shape)
        y = proj(x)

        assert_equal(proj.shapein, plane.shape)
        assert_equal(proj.shapeout, detectors.shape[:-2])
        assert_equal(proj.matrix.shape, (4, len(plane)))
        expected_min_ncolmax = 1 if center == (0, 0) else 4
        assert_equal(proj.matrix.data.shape, (4, max(expected_min_ncolmax,
                                                     ncolmax)))
        assert_equal(proj.min_ncolmax, expected_min_ncolmax)
        assert_same(y, expected)

    centers = [(0, 0), (-0.5, 0.5)]
    ncolmaxs = (0, 1, 4, 5)
    expecteds = [[[9, 10],
                  [5, 6]],
                 [[0.25*(8+9+12+13), 0.25*(9+10+13+14)],
                  [0.25*(4+5+8+9),   0.25*(5+6+9+10)]]]
    for center, expected in zip(centers, expecteds):
        for ncolmax in ncolmaxs:
            yield func, center, ncolmax, expected
