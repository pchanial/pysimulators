import numpy as np
from pybenchmarks import benchmark

import pysimulators


def benchmark_rotate_2d():
    rotate_2d_fortran = pysimulators._flib.geometry.rotate_2d
    rotate_2d_fortran_inplace = pysimulators._flib.geometry.rotate_2d_inplace

    def rotate1(n):
        coords = x[:n]
        coords = np.asarray(coords)
        out = np.empty_like(coords)
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        m = np.asarray([[cosangle, -sinangle], [sinangle, cosangle]])
        return np.dot(coords, m.T, out)

    def rotate2(n):
        coords = x[:n]
        out = np.empty_like(coords)
        rotate_2d_fortran(coords.T, out.T, angle_deg)
        return out

    def rotate2_inplace(n):
        coords = x[:n]
        rotate_2d_fortran_inplace(coords.T, angle_deg)
        return x

    ns = (10, 100, 1000, 10000, 100000, 1000000, 10000000)
    x = np.random.random_sample((ns[-1], 2))
    angle = np.radians(30)
    angle_deg = 30
    b1 = benchmark(rotate1, ns, verbose=False)
    b2 = benchmark(rotate2, ns, verbose=False)
    b3 = benchmark(rotate2_inplace, ns, verbose=False)

    print('{:^9}{:^12}{:^12}{:15}'.format('N', 'Dot', 'Fortran', 'Fortran Inplace'))
    print(48 * '-')
    for n, t1, t2, t3 in zip(ns, b1['time'], b2['time'], b3['time']):
        print(f'{n:<9}{t1:12.9f}{t2:12.9f}{t3:12.9f}')


def benchmark_create_grid():
    from pysimulators.geometry import create_grid

    def create_grid_slow(shape, spacing, center=(0, 0), angle=0):
        x = np.arange(shape[1], dtype=float) * spacing
        x -= x.mean()
        y = (np.arange(shape[0], dtype=float) * spacing)[::-1]
        y -= y.mean()
        grid_x, grid_y = np.meshgrid(x, y)
        centers = np.asarray([grid_x.T, grid_y.T]).T.copy()
        pysimulators._flib.geometry.rotate_2d_inplace(centers.T, angle)
        centers += center
        return centers

    ns = list(range(11))
    ids = ['{0}x{0}'.format(2**n) for n in ns]
    shapes = [2 * (2**n,) for n in ns]
    spacing = 0.4
    b = benchmark([create_grid_slow, create_grid], shapes, spacing, verbose=False)
    header = '{:^9}{:^12}{:^12}'.format('shape', 'Python', 'Fortran')
    print(header)
    print(len(header) * '-')
    for n, t1, t2 in zip(ids, b['time'][0], b['time'][1]):
        print(f'{n:<9}{t1:12.9f}{t2:12.9f}')


def benchmark_create_grid_squares():
    from pysimulators.geometry import create_grid_squares

    def create_grid_squares_slow(
        shape, spacing, filling_factor=1, center=(0, 0), angle=0
    ):
        x = np.arange(shape[1], dtype=float) * spacing
        x -= x.mean()
        y = (np.arange(shape[0], dtype=float) * spacing)[::-1]
        y -= y.mean()
        grid_x, grid_y = np.meshgrid(x, y)
        nodes = np.asarray([grid_x.T, grid_y.T]).T.copy()
        d = np.sqrt(filling_factor * spacing**2) / 2
        offset = np.array([[d, d], [-d, d], [-d, -d], [d, -d]])
        corners = nodes[..., None, :] + offset
        pysimulators._flib.geometry.rotate_2d_inplace(corners.T, angle)
        corners += center
        return corners

    ns = list(range(11))
    ids = ['{0}x{0}'.format(2**n) for n in ns]
    shapes = [2 * (2**n,) for n in ns]
    spacing = 0.4
    b = benchmark(
        [create_grid_squares_slow, create_grid_squares], shapes, spacing, verbose=False
    )
    header = '{:^9}{:^12}{:^12}'.format('shape', 'Python', 'Fortran')
    print(header)
    print(len(header) * '-')
    for n, t1, t2 in zip(ids, b['time'][0], b['time'][1]):
        print(f'{n:<9}{t1:12.9f}{t2:12.9f}')


if __name__ == '__main__':
    print()
    print('BENCHMARK ROTATE_2D')
    print('===================')
    benchmark_rotate_2d()

    print()
    print('BENCHMARK CREATE_GRID')
    print('=====================')
    benchmark_create_grid()

    print()
    print('BENCHMARK CREATE_GRID_SQUARES')
    print('=============================')
    benchmark_create_grid_squares()

"""
Dell XPS 14, i7, 2 cores, 4 threads.

BENCHMARK ROTATE_2D
===================
    N        DOT       Fortran   Fortran Inplace
------------------------------------------------
10        0.000027011 0.000005626 0.000003858
100       0.000028501 0.000005788 0.000004093
1000      0.000038466 0.000006799 0.000004699
10000     0.000108178 0.000015927 0.000012606
100000    0.000974177 0.000121181 0.000094844
1000000   0.011121390 0.005500898 0.003829451
10000000  0.155856705 0.051112604 0.037185216

BENCHMARK CREATE_GRID
=====================
  shape     Python     Fortran
---------------------------------
1x1       0.000095569 0.000006566
2x2       0.000099937 0.000006630
4x4       0.000105157 0.000006692
8x8       0.000122708 0.000007098
16x16     0.000187389 0.000007307
32x32     0.000436772 0.000008488
64x64     0.001379221 0.000012488
128x128   0.003800218 0.000028666
256x256   0.012710600 0.000091763
512x512   0.059188199 0.000388953
1024x1024 0.327460051 0.003921621

BENCHMARK CREATE_GRID_SQUARES
=============================
  shape     Python     Fortran
---------------------------------
1x1       0.000144939 0.000006842
2x2       0.000149147 0.000007037
4x4       0.000159001 0.000007273
8x8       0.000183863 0.000007592
16x16     0.000273348 0.000008598
32x32     0.000616008 0.000012221
64x64     0.001936396 0.000026899
128x128   0.005158420 0.000086011
256x256   0.018835349 0.000419362
512x512   0.086557007 0.003928289
1024x1024 0.507800102 0.014882929

"""
