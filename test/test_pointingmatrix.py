from __future__ import division

import numpy as np
from numpy.testing import assert_equal

from pysimulators import PointingMatrix


def test_constructor():
    shape = (2,)
    shape_input = (3, 4)
    array = np.zeros(shape, dtype=np.int64)

    def func(matrix):
        assert matrix.shape_input == shape_input
        assert matrix.value.shape == shape
        assert matrix.index.shape == shape

    for matrix in [PointingMatrix(array, shape_input, copy=False),
                   PointingMatrix.empty(shape, shape_input),
                   PointingMatrix.zeros(shape, shape_input)]:
        yield func, matrix


# we assume that the input map is 2x5
#  -----------
#  |0|1|2|3|4|
#  |-+-+-+-+-|
#  |5|6|7|8|9|
#  -----------
# The first sample intersects a quarter of 0, 1, 5 and 6
# and the second sample intersects half of 6 and 7.
matrix = PointingMatrix.empty((2, 4), (2, 5))
matrix[0] = [(0, .25), (1, .25), (5, .25), (6, .25)]
matrix[1] = [(7, .50), (8, .50), (-1, 0),  (-1, 0)]


def test_isvalid():
    assert matrix.isvalid()

    invalid1 = matrix.copy()
    invalid1[0, 2] = (-2, 0)
    assert not invalid1.isvalid()

    invalid2 = matrix.copy()
    invalid2[0, 2] = (10, 0)
    assert not invalid2.isvalid()


def test_mask():
    expected = [[False, False, True, True, True],
                [False, False, False, False, True]]
    assert_equal(matrix.get_mask(), expected)


def test_pack():
    expected = PointingMatrix.empty((2, 4), (6,))
    expected[0] = [(0, .25), (1, .25), (2, .25), (3, .25)]
    expected[1] = [(4, .50), (5, .50), (-1, 0),  (-1, 0)]
    packed = matrix.copy()
    packed.pack(matrix.get_mask())
    assert_equal(packed, expected)
    assert_equal(packed.shape_input, expected.shape_input)
