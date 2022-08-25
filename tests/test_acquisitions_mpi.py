import pytest

from pyoperators import MPI
from pysimulators import Acquisition, Instrument, PackedTable, Sampling, Scene

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def test():
    scene = Scene(1024)
    instrument = Instrument('instrument', PackedTable((32, 32)))
    sampling = Sampling(1000)
    acq = Acquisition(instrument, sampling, scene, nprocs_sampling=max(size // 2, 1))
    print(
        acq.comm.rank,
        acq.instrument.detector.comm.rank,
        '/',
        acq.instrument.detector.comm.size,
        acq.sampling.comm.rank,
        '/',
        acq.sampling.comm.size,
    )
    pytest.xfail('the test is not finished.')
