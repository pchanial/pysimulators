from __future__ import division

import numpy as np
from pyoperators.utils import product
from pyoperators.utils.testing import assert_same
from pysimulators import Quantity, create_fitsheader, DiscreteSurface
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
    plane = DiscreteSurface.fromfits(header)

    def func(center, npps, expected):
        detectors = create_grid_squares((2, 2), Quantity(pixsize, 'um'),
                                        center=center)
        proj = plane.get_integration_operator(detectors)
        x = np.arange(product(plane_shape), dtype=float).reshape(plane_shape)
        y = proj(x)

        assert proj.matrix.shape == (2, 2, npps)
        assert_same(y, expected)

    centers = [(0, 0), (-0.5, 0.5)]
    nppss = (1, 4)
    expecteds = [[[9, 10],
                  [5, 6]],
                 [[0.25*(8+9+12+13), 0.25*(9+10+13+14)],
                  [0.25*(4+5+8+9),   0.25*(5+6+9+10)]]]
    for center, npps, expected in zip(centers, nppss, expecteds):
        yield func, center, npps, expected
