import numpy as np

from numpy.testing import assert_array_equal
from pyoperators.utils.testing import skiptest
from pysimulators.datautils import distance, gaussian, airy_disk, _distance_slow

@skiptest
def test_distance1():
    origin = (1.,)
    d0 = np.arange(5.)*0.5
    d1 = distance(5, origin=origin, resolution=0.5)
    d2 = _distance_slow((5,), origin, [0.5], None)
    assert_array_equal(d0, d1)
    assert_array_equal(d0, d2)

@skiptest
def test_distance2():
    origin = (1.,2.)
    d0 = np.array([[2., np.sqrt(5), np.sqrt(8)], [0,1,2]])
    d1 = distance((2,3), origin=origin, resolution=(1.,2.))
    d2 = _distance_slow((2,3), origin, [1.,2.], None)
    assert_array_equal(d0, d1)
    assert_array_equal(d0, d2)

@skiptest
def test_fwhm():
    def func(f, fwhm, resolution, n):
        m = f((1000,1000), fwhm=fwhm, resolution=resolution)
        assert np.sum(m[500,:] > np.max(m)/2) == n
    for f in [gaussian, airy_disk]:
        for fwhm, resolution, n in zip([10,10,100], [0.1,1,10], [100,10,10]):
            yield func, f, fwhm, resolution, n

