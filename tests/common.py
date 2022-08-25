import numpy as np
from numpy.testing import assert_equal

from pysimulators import FitsArray, Map, Quantity, Tod

SINT_TYPES = [np.int8, np.int16, np.int32, np.int64]
UINT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
INT_TYPES = SINT_TYPES + UINT_TYPES
FLOAT_TYPES = [np.float16, np.float32, np.float64]
if hasattr(np, 'float128'):
    FLOAT_TYPES.append(np.float128)
    BIGGEST_FLOAT_TYPE = np.float128
else:
    BIGGEST_FLOAT_TYPE = np.float64


def assert_equal_subclass(x, y):
    assert type(x) is type(y)
    assert x.dtype == y.dtype
    assert_equal(x.view(np.ndarray), y.view(np.ndarray))

    if isinstance(x, Quantity):
        assert x._unit == y._unit
        assert x._derived_units == y._derived_units

    if isinstance(x, FitsArray):
        assert x.header == y.header

    if isinstance(x, Map):
        if x.coverage is None:
            assert y.coverage is None
        else:
            assert y.coverage is not None
            assert x.coverage.shape == y.coverage.shape
            assert_equal(x.coverage, y.coverage)
        if x.error is None:
            assert y.error is None
        else:
            assert y.error is not None
            assert x.error.shape == y.error.shape
            assert_equal(x.error, y.error)

    if isinstance(x, Tod):
        if x.mask is None:
            assert y is None
        else:
            assert y.mask is not None
            assert x.mask.shape == y.mask.shape
            assert_equal(x.mask, y.mask)
