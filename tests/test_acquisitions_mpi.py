from itertools import chain, product

import numpy as np
import pytest

from pyoperators import MPI, MPIDistributionIdentityOperator
from pyoperators.utils.testing import assert_same
from pysimulators import Acquisition, Instrument, PackedTable, Sampling, Scene
from pysimulators.operators import ProjectionOperator
from pysimulators.sparse import FSRMatrix

pytestmark = pytest.mark.mpi

RANK = MPI.COMM_WORLD.rank
SIZE = MPI.COMM_WORLD.size
NPROCS_INSTRUMENT = sorted(
    {int(n) for n in [1, SIZE / 3, SIZE / 2, SIZE] if int(n) == n}
)
NSCENE = 10
NSAMPLING_GLOBAL = 100
NDETECTOR_GLOBAL = 16
SCENE = Scene(NSCENE)
SAMPLING = Sampling(NSAMPLING_GLOBAL, period=1.0)
INSTRUMENT = Instrument('', PackedTable(NDETECTOR_GLOBAL))


class MyAcquisition(Acquisition):
    def get_projection_operator(self):
        dtype = [('index', int), ('value', float)]
        data = np.recarray((len(self.instrument), len(self.sampling), 1), dtype=dtype)
        for ilocal, iglobal in enumerate(self.instrument.detector.index):
            data[ilocal].value = iglobal
            data[ilocal, :, 0].index = [
                (iglobal + int(t)) % NSCENE for t in self.sampling.time
            ]

        matrix = FSRMatrix(
            (len(self.instrument) * len(self.sampling), NSCENE),
            data=data.reshape((-1, 1)),
        )

        return ProjectionOperator(
            matrix, dtype=float, shapeout=(len(self.instrument), len(self.sampling))
        )


def get_acquisition(comm, nprocs_instrument):
    return MyAcquisition(
        INSTRUMENT, SAMPLING, SCENE, comm=comm, nprocs_instrument=nprocs_instrument
    )


@pytest.mark.parametrize('nprocs_instrument', NPROCS_INSTRUMENT)
def test_communicators(nprocs_instrument):
    sky = SCENE.ones()
    nprocs_sampling = SIZE // nprocs_instrument
    serial_acq = get_acquisition(MPI.COMM_SELF, 1)
    assert serial_acq.comm.size == 1
    assert serial_acq.instrument.comm.size == 1
    assert serial_acq.sampling.comm.size == 1
    assert len(serial_acq.instrument) == NDETECTOR_GLOBAL
    assert len(serial_acq.sampling) == NSAMPLING_GLOBAL

    parallel_acq = get_acquisition(MPI.COMM_WORLD, nprocs_instrument)
    assert parallel_acq.comm.size == SIZE
    assert parallel_acq.instrument.comm.size == nprocs_instrument
    assert parallel_acq.sampling.comm.size == nprocs_sampling
    assert (
        parallel_acq.instrument.comm.allreduce(len(parallel_acq.instrument))
        == NDETECTOR_GLOBAL
    )
    assert (
        parallel_acq.sampling.comm.allreduce(len(parallel_acq.sampling))
        == NSAMPLING_GLOBAL
    )

    serial_H = serial_acq.get_projection_operator()
    ref_tod = serial_H(sky)

    parallel_H = (
        parallel_acq.get_projection_operator()
        * MPIDistributionIdentityOperator(parallel_acq.comm)
    )
    local_tod = parallel_H(sky)
    actual_tod = np.vstack(
        parallel_acq.instrument.comm.allgather(
            np.hstack(parallel_acq.sampling.comm.allgather(local_tod))
        )
    )
    assert_same(actual_tod, ref_tod, atol=20)

    ref_backproj = serial_H.T(ref_tod)
    actual_backproj = parallel_H.T(local_tod)
    assert_same(actual_backproj, ref_backproj, atol=20)


@pytest.mark.parametrize('nprocs_instrument', NPROCS_INSTRUMENT)
@pytest.mark.parametrize(
    'selection',
    [
        Ellipsis,
        slice(None),
    ]
    + list(chain(*(product([slice(None), Ellipsis], repeat=n) for n in [1, 2, 3]))),
)
def test_communicators_getitem_all(nprocs_instrument, selection):
    acq = get_acquisition(MPI.COMM_WORLD, nprocs_instrument)
    assert acq.instrument.comm.size == nprocs_instrument
    assert acq.sampling.comm.size == MPI.COMM_WORLD.size / nprocs_instrument
    assert acq.comm.size == MPI.COMM_WORLD.size
    restricted_acq = acq[selection]
    assert restricted_acq.instrument.comm.size == nprocs_instrument
    assert restricted_acq.sampling.comm.size == MPI.COMM_WORLD.size / nprocs_instrument
    assert restricted_acq.comm.size == MPI.COMM_WORLD.size


@pytest.mark.parametrize('nprocs_instrument', NPROCS_INSTRUMENT)
@pytest.mark.parametrize('selection', [0, slice(None, 1), np.array])
def test_communicators_getitem_instrument(nprocs_instrument, selection):
    acq = get_acquisition(MPI.COMM_WORLD, nprocs_instrument)
    if selection is np.array:
        selection = np.zeros(len(acq.instrument), bool)
        selection[0] = True
    restricted_acq = acq[selection]
    assert restricted_acq.instrument.comm.size == 1
    assert restricted_acq.sampling.comm.size == acq.sampling.comm.size
    assert restricted_acq.comm.size == acq.sampling.comm.size


SELECTION_GETITEM_SAMPLING = np.zeros(NSAMPLING_GLOBAL, bool)
SELECTION_GETITEM_SAMPLING[0] = True


@pytest.mark.parametrize('nprocs_instrument', NPROCS_INSTRUMENT)
@pytest.mark.parametrize('selection', [0, slice(None, 1), np.array])
def test_communicators_getitem_sampling(nprocs_instrument, selection):
    acq = get_acquisition(MPI.COMM_WORLD, nprocs_instrument)
    if selection is np.array:
        selection = np.zeros(len(acq.sampling), bool)
        selection[0] = True
    restricted_acq = acq[:, selection]
    assert restricted_acq.instrument.comm.size == acq.instrument.comm.size
    assert restricted_acq.sampling.comm.size == 1
    assert restricted_acq.comm.size == acq.instrument.comm.size
