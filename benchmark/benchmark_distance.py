from __future__ import division, print_function
from pybenchmarks import benchmark
from pyoperators.utils.testing import assert_same
from pysimulators._flib import datautils as du
import numpy as np


def benchmark_distance_1d():
    def distance_1d(n):
        return abs(scale * (np.arange(n) - origin))

    def distance_1d_fortran(n):
        out = np.empty(n)
        du.distance_1d_r8(out, origin, scale)
        return out

    def distance2_1d(n):
        return (scale * (np.arange(n) - origin)) ** 2

    def distance2_1d_fortran(n):
        out = np.empty(n)
        du.distance2_1d_r8(out, origin, scale)
        return out

    origin = 0.2
    scale = 1.3
    ns = [2**_ for _ in range(1, 27)]
    assert_same(distance_1d(4), distance_1d_fortran(4))
    assert_same(distance2_1d(4), distance2_1d_fortran(4))

    print()
    print('BENCHMARK DISTANCE_1D')
    print('=====================')
    b = benchmark([distance_1d, distance_1d_fortran], ns, verbose=False)
    title = '{:^9}{:^12}{:^12}'.format('N', 'Python', 'Fortran')
    print(title)
    print(len(title) * '-')
    for n, t1, t2 in zip(ns, b['time'][0], b['time'][1]):
        print('{:<9}{:12.9f}{:12.9f}'.format(n, t1, t2))

    print()
    print('BENCHMARK DISTANCE2_1D')
    print('======================')
    b = benchmark([distance2_1d, distance2_1d_fortran], ns, verbose=False)
    title = '{:^9}{:^12}{:^12}'.format('N', 'Python', 'Fortran')
    print(title)
    print(len(title) * '-')
    for n, t1, t2 in zip(ns, b['time'][0], b['time'][1]):
        print('{:<9}{:12.9f}{:12.9f}'.format(n, t1, t2))


def benchmark_distance_2d():
    def distance_2d(n):
        x, y = np.meshgrid(
            scale[0] * (np.arange(n) - origin[0]),
            scale[1] * (np.arange(n) - origin[1]),
            sparse=True,
        )
        return np.sqrt(x**2 + y**2)

    def distance_2d_fortran(n):
        out = np.empty((n, n))
        du.distance_2d_r8(out.T, origin, scale)
        return out

    def distance2_2d(n):
        x, y = np.meshgrid(
            scale[0] * (np.arange(n) - origin[0]),
            scale[1] * (np.arange(n) - origin[1]),
            sparse=True,
        )
        return x**2 + y**2

    def distance2_2d_fortran(n):
        out = np.empty((n, n))
        du.distance2_2d_r8(out.T, origin, scale)
        return out

    origin = [0.2, 0.1]
    scale = [1.3, 1.7]
    ns = [2**_ for _ in range(1, 14)]
    assert_same(distance_2d(4), distance_2d_fortran(4))
    assert_same(distance2_2d(4), distance2_2d_fortran(4))

    print()
    print('BENCHMARK DISTANCE_2D')
    print('=====================')
    b = benchmark([distance_2d, distance_2d_fortran], ns, verbose=False)
    title = '{:^9}{:^12}{:^12}'.format('N', 'Python', 'Fortran')
    print(title)
    print(len(title) * '-')
    for n, t1, t2 in zip(ns, b['time'][0], b['time'][1]):
        print('{:<9}{:12.9f}{:12.9f}'.format(n, t1, t2))

    print()
    print('BENCHMARK DISTANCE2_2D')
    print('======================')
    b = benchmark([distance2_2d, distance2_2d_fortran], ns, verbose=False)
    title = '{:^9}{:^12}{:^12}'.format('N', 'Python', 'Fortran')
    print(title)
    print(len(title) * '-')
    for n, t1, t2 in zip(ns, b['time'][0], b['time'][1]):
        print('{:<9}{:12.9f}{:12.9f}'.format(n, t1, t2))


if __name__ == '__main__':
    benchmark_distance_1d()
    benchmark_distance_2d()

"""
Dell XPS 14, i7, 2 cores, 4 threads.

BENCHMARK DISTANCE_1D: Without / With OpenMP
============================================
    N       Python     Fortran      Python     Fortran
--------------------------------- -----------------------
1         0.000009100 0.000002391 0.000009010 0.000005209
2         0.000009050 0.000002182 0.000008850 0.000005200
4         0.000009110 0.000002210 0.000008781 0.000005190
8         0.000009110 0.000002239 0.000009301 0.000005250
16        0.000009370 0.000002258 0.000009041 0.000005360
32        0.000009489 0.000002279 0.000009208 0.000005360
64        0.000010488 0.000002320 0.000009730 0.000005550
128       0.000010209 0.000003300 0.000010250 0.000005581
256       0.000010920 0.000002561 0.000010321 0.000005531
512       0.000012009 0.000002770 0.000011551 0.000005701
1024      0.000013790 0.000003259 0.000013220 0.000005920
2048      0.000017171 0.000004292 0.000016670 0.000006361
4096      0.000024681 0.000006039 0.000023689 0.000007300
8192      0.000038280 0.000009460 0.000037949 0.000009201
16384     0.000068128 0.000016429 0.000067759 0.000012560
32768     0.000122368 0.000030601 0.000126019 0.000019739
65536     0.000228209 0.000058210 0.000229030 0.000034451
131072    0.000442739 0.000113080 0.000958319 0.000062201
262144    0.001211410 0.000223241 0.002959731 0.000121298
524288    0.005219369 0.000480220 0.006446519 0.000254200
1048576   0.011413503 0.001644402 0.011335993 0.001533241
2097152   0.021918893 0.003661730 0.025349092 0.003831649
4194304   0.040002418 0.008069351 0.037138915 0.011675310
8388608   0.072293806 0.015509486 0.073173690 0.023699284
16777216  0.139316082 0.031145000 0.139379025 0.047821593
33554432  0.280709982 0.062763214 0.321547985 0.095936584
67108864  0.571758986 0.122627020 0.580856085 0.186897993

BENCHMARK DISTANCE2_1D
======================
    N       Python     Fortran      Python     Fortran
--------------------------------- -----------------------
1         0.000009861 0.000002151 0.000009389 0.000005331
2         0.000009849 0.000002439 0.000009601 0.000005331
4         0.000009909 0.000002432 0.000009630 0.000005250
8         0.000009849 0.000002229 0.000009689 0.000005240
16        0.000010202 0.000002260 0.000009789 0.000005369
32        0.000010281 0.000002251 0.000010030 0.000005369
64        0.000010579 0.000002329 0.000010090 0.000005288
128       0.000011179 0.000002518 0.000011001 0.000005531
256       0.000011621 0.000002871 0.000011292 0.000005560
512       0.000012851 0.000002809 0.000012438 0.000005620
1024      0.000014400 0.000003290 0.000013938 0.000005801
2048      0.000018039 0.000004351 0.000017519 0.000006330
4096      0.000025599 0.000006170 0.000024302 0.000007210
8192      0.000041230 0.000010231 0.000038161 0.000009191
16384     0.000073040 0.000016780 0.000067799 0.000012860
32768     0.000131450 0.000033460 0.000122101 0.000020671
65536     0.000244770 0.000064681 0.000226848 0.000035989
131072    0.000500481 0.000119319 0.000574291 0.000065670
262144    0.001504960 0.000225790 0.001660640 0.000124931
524288    0.005657821 0.000467889 0.005320239 0.000795999
1048576   0.011146712 0.001697240 0.011627316 0.001522610
2097152   0.026704001 0.003602762 0.023074102 0.003891790
4194304   0.036989713 0.008211350 0.037293196 0.013640499
8388608   0.072314692 0.015716505 0.072789001 0.024048281
16777216  0.151414871 0.031411004 0.139575958 0.048907208
33554432  0.278008938 0.063442612 0.282323122 0.095510387
67108864  0.596297979 0.124128103 0.575090885 0.187508821

BENCHMARK DISTANCE_2D
=====================
    N       Python     Fortran      Python     Fortran
--------------------------------- -----------------------
1         0.000038750 0.000004041 0.000037870 0.000010111
2         0.000043080 0.000004051 0.000042701 0.000010052
4         0.000043430 0.000004220 0.000043130 0.000010440
8         0.000044248 0.000004382 0.000043700 0.000009971
16        0.000045989 0.000005240 0.000046101 0.000010850
32        0.000049400 0.000007010 0.000049410 0.000011702
64        0.000064080 0.000014460 0.000063322 0.000015500
128       0.000111701 0.000043700 0.000113189 0.000030861
256       0.000285668 0.000157139 0.000295730 0.000091588
512       0.001436720 0.000606620 0.001858909 0.000328491
1024      0.008369911 0.002468810 0.008406839 0.001787989
2048      0.022696304 0.013827896 0.023167610 0.011939502
4096      0.089199781 0.057017493 0.107831001 0.050725007
8192      0.300878048 0.217860937 0.295413017 0.194954872

BENCHMARK DISTANCE2_2D
======================
    N       Python     Fortran      Python     Fortran
--------------------------------- -----------------------
1         0.000038052 0.000004032 0.000036201 0.000010650
2         0.000041571 0.000004082 0.000040848 0.000009911
4         0.000041950 0.000004551 0.000041292 0.000010130
8         0.000042369 0.000004220 0.000043061 0.000009968
16        0.000044169 0.000004561 0.000044880 0.000010328
32        0.000045860 0.000005000 0.000045280 0.000010641
64        0.000052559 0.000005860 0.000052831 0.000011189
128       0.000072169 0.000009639 0.000070989 0.000013299
256       0.000137920 0.000029049 0.000136189 0.000022330
512       0.000394671 0.000098791 0.000390120 0.000060642
1024      0.002374721 0.001626749 0.002386160 0.001570811
2048      0.012163997 0.006354010 0.010233283 0.011715698
4096      0.042893004 0.026740098 0.043405104 0.050754213
8192      0.111250877 0.108854055 0.111384153 0.194573164

"""
