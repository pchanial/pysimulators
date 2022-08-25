import numpy as np
import pytest
from numpy.testing import assert_equal

from pysimulators import PointingMatrix

CONSTRUCTOR_SHAPE = (2,)
CONSTRUCTOR_SHAPE_INPUT = (3, 4)


@pytest.mark.parametrize(
    'matrix',
    [
        PointingMatrix(
            np.zeros(CONSTRUCTOR_SHAPE, dtype=np.int64),
            CONSTRUCTOR_SHAPE_INPUT,
            copy=False,
        ),
        PointingMatrix.empty(CONSTRUCTOR_SHAPE, CONSTRUCTOR_SHAPE_INPUT),
        PointingMatrix.zeros(CONSTRUCTOR_SHAPE, CONSTRUCTOR_SHAPE_INPUT),
    ],
)
def test_constructor(matrix):
    assert matrix.shape_input == CONSTRUCTOR_SHAPE_INPUT
    assert matrix.value.shape == CONSTRUCTOR_SHAPE
    assert matrix.index.shape == CONSTRUCTOR_SHAPE


# we assume that the input map is 2x5
#  -----------
#  |0|1|2|3|4|
#  |-+-+-+-+-|
#  |5|6|7|8|9|
#  -----------
# The first sample intersects a quarter of 0, 1, 5 and 6
# and the second sample intersects half of 6 and 7.
matrix = PointingMatrix.empty((2, 4), (2, 5))
matrix[0] = [(0, 0.25), (1, 0.25), (5, 0.25), (6, 0.25)]
matrix[1] = [(7, 0.50), (8, 0.50), (-1, 0), (-1, 0)]


def test_isvalid():
    assert matrix.isvalid()

    invalid1 = matrix.copy()
    invalid1[0, 2] = (-2, 0)
    assert not invalid1.isvalid()

    invalid2 = matrix.copy()
    invalid2[0, 2] = (10, 0)
    assert not invalid2.isvalid()


def test_mask():
    expected = [[False, False, True, True, True], [False, False, False, False, True]]
    assert_equal(matrix.get_mask(), expected)


def test_pack():
    expected = PointingMatrix.empty((2, 4), (6,))
    expected[0] = [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]
    expected[1] = [(4, 0.50), (5, 0.50), (-1, 0), (-1, 0)]
    packed = matrix.copy()
    packed.pack(matrix.get_mask())
    assert_equal(packed, expected)
    assert_equal(packed.shape_input, expected.shape_input)
