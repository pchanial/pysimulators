from __future__ import division
import numpy as np
from pyoperators import MPI
from pyoperators.utils import split
from pyoperators.utils.testing import assert_same
from pysimulators import PackedTable

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def test():
    np.random.seed(0)
    n = 4
    x = np.random.random(n)
    layout = PackedTable(n, x=x)
    s = split(n, size, rank)
    scattered = layout.scatter()
    assert_same(scattered.x, x[s])
    assert_same(scattered.all.x, x)
