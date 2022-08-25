import numpy as np

SINT_TYPES = [np.int8, np.int16, np.int32, np.int64]
UINT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
INT_TYPES = SINT_TYPES + UINT_TYPES
FLOAT_TYPES = [np.float16, np.float32, np.float64]
if hasattr(np, 'float128'):
    FLOAT_TYPES.append(np.float128)
    BIGGEST_FLOAT_TYPE = np.float128
else:
    BIGGEST_FLOAT_TYPE = np.float64
