from __future__ import division

import numpy as np
from pyoperators.utils.testing import assert_same
from pysimulators.beams import BeamGaussian, BeamUniformHalfSpace

angles = np.radians([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_gaussian():
    primary = BeamGaussian(np.radians(14))
    secondary = BeamGaussian(np.radians(14), backward=True)
    assert_same(primary(np.pi - angles, 0), secondary(angles, 0))


def test_uniform():
    beam = BeamUniformHalfSpace()
    assert_same(beam(10, 0), 1)
