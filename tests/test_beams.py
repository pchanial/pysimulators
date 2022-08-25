import numpy as np
from numpy.testing import assert_allclose

from pysimulators.beams import BeamGaussian, BeamUniformHalfSpace

angles = np.radians([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_gaussian():
    primary = BeamGaussian(np.radians(14))
    secondary = BeamGaussian(np.radians(14), backward=True)
    assert_allclose(primary(np.pi - angles, 0), secondary(angles, 0))


def test_uniform():
    beam = BeamUniformHalfSpace()
    assert_allclose(beam(10, 0), 1)
