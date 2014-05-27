from __future__ import division

from pyoperators import MPI
from pysimulators import Acquisition, Instrument, Layout, LayoutTemporal

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def test():
    instrument = Instrument('instrument', Layout((32, 32)))
    sampling = LayoutTemporal(1000)
    acq = Acquisition(instrument, sampling, nprocs_sampling=max(size // 2, 1))
    print(
        acq.comm.rank,
        acq.instrument.detector.comm.rank,
        '/',
        acq.instrument.detector.comm.size,
        acq.sampling.comm.rank,
        '/',
        acq.sampling.comm.size,
    )
