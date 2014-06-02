# file test_scatter.py
from __future__ import division, print_function
import numpy as np
from mpi4py import MPI
from pysimulators import LayoutTemporal
rank = MPI.COMM_WORLD.rank
samples = LayoutTemporal(10, angle1=np.arange(10) * 360 / 10, angle2=None)
samples = samples.scatter()
samples.angle2 = rank * 90 + np.arange(len(samples))
print(rank, len(samples), len(samples.angle1), len(samples.angle2))
angle1 = samples.all.angle1
angle2 = samples.all.angle2
if rank == 0:
    print(angle1)
    print(angle2)
