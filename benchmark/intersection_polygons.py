import numpy as np
from pybenchmarks import benchmark

from pysimulators import LayoutGridSquares

grid = LayoutGridSquares(
    (256, 256), 3, filling_factor=0.8, angle=10.0, origin=(128 * 3, 128 * 3)
)
ncolmax = 16
nx, ny = grid.shape * np.array(3.0)

setup = """
from __main__ import grid, ncolmax, nx, ny, FSRMatrix, np, flib
matrix = FSRMatrix((len(grid), nx * ny), dtype=dtype,
                   dtype_index=itype, ncolmax=ncolmax)
data = matrix.data.ravel().view(np.int8)
func = f'matrix_polygon_integration_i{itype.itemsize}_r{dtype.itemsize}'
f = getattr(flib.projection, func)
"""

b = benchmark(
    'f(grid.vertex.T, nx, ny, data, ncolmax)',
    dtype=(np.dtype(np.float32), np.dtype(np.float64)),
    itype=(np.dtype(np.int32), np.dtype(np.int64)),
    setup=setup,
    repeat=10,
)
