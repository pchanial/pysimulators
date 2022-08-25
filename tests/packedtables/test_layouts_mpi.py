import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import MPI
from pyoperators.utils import product, split
from pyoperators.utils.testing import assert_same
from pysimulators import PackedTable

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def test_scatter():
    np.random.seed(0)
    n = 4
    x = np.random.random(n)
    layout = PackedTable(n, x=x)
    s = split(n, size, rank)
    scattered = layout.scatter()
    assert_same(scattered.x, x[s])
    assert_same(scattered.all.x, x)


@pytest.mark.parametrize(
    'shape, ndim',
    [
        ((4,), 1),
        ((4, 4), 2),
        ((4, 4, 2), 3),
    ],
)
def test_gather(shape, ndim):
    bshape = shape[:ndim]
    ntot = product(bshape)
    o = np.arange(ntot).reshape(bshape)
    os = o, o.copy(), o.copy(), o.copy(), o.copy()
    os[1].ravel()[...] = os[1].ravel()[::-1]
    os[2].ravel()[-2:] = -1
    os[3].ravel()[::2] = -1
    os[4].ravel()[[0, 2, 3]] = -1
    value1 = None
    value2 = 2.3
    value3 = np.array(2.4)
    value4 = np.arange(product(bshape), dtype=np.int8).reshape(bshape)
    value5 = np.arange(product(bshape), dtype=complex).reshape(bshape)
    value6 = (
        np.arange(product(bshape), dtype=complex)
        .reshape(bshape)
        .view([('re', float), ('im', float)])
    )
    value7 = np.arange(product(bshape) * 3, dtype=np.int8).reshape(bshape + (3,))
    value8 = np.arange(product(bshape) * 3, dtype=complex).reshape(bshape + (3,))
    value9 = (
        np.arange(product(bshape) * 3, dtype=complex)
        .reshape(bshape + (3,))
        .view([('re', float), ('im', float)])
    )
    for ordering in os:
        for value in (
            value1,
            value2,
            value3,
            value4,
            value5,
            value6,
            value7,
            value8,
            value9,
        ):
            table = PackedTable(shape, ndim=ndim, ordering=ordering, value=value)
            table_local = table.scatter()
            table_global = table_local.gather()
            assert table_global.shape == table.shape
            assert table_global.ndim == table.ndim
            assert_equal(table_global._index, table._index)
            assert_equal(table_global.removed, table.removed)
            assert_equal(table_global.value, table.value)
