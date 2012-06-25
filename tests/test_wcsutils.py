import numpy as np

from kapteyn import wcs
from numpy.testing import assert_almost_equal
from tamasis.utils import all_eq
from pysimulators.wcsutils import angle_lonlat, barycenter_lonlat, combine_fitsheader, create_fitsheader, fitsheader2shape, get_cdelt_pa, has_wcs, mean_degrees


def test_mean_degrees():
    assert mean_degrees([1,2]) == 1.5
    assert_almost_equal(mean_degrees([1,359.1]), 0.05, 12)
    assert_almost_equal(mean_degrees([0.1,359.1]), 359.6, 12)

def test_angle_lonlat1():
    assert all_eq(angle_lonlat(30, 0, 40, 0), 10)

def test_angle_lonlat2():
    input = (((30,0), (40,0), 10),
             ((39,0), (92, 90), 90),
             ((39,0), (37,-90), 90),
             ((37,-90), (39,0), 90),
             ((100,90),(134,-90), 180),
             ((24,30),(24,32), 2))
    def func(c1, c2, angle):
        assert_almost_equal(angle_lonlat(c1,c2), angle, 10)
    for c1, c2, angle in input:
        yield func, c1, c2, angle

def test_barycenter_lonlat():
    assert all_eq(barycenter_lonlat([30,40], [0, 0]), [35,0])
    assert all_eq(barycenter_lonlat([20,20,20], [-90,0,90]), [20,0])
    assert all_eq(barycenter_lonlat([20,20,20], [0,45,90]), [20,45])

def test_get_cdelt_pa1():
    header = create_fitsheader((10,10), cdelt=(-1.2,3))
    cdelt, pa = get_cdelt_pa(header)
    assert all_eq(cdelt, (-1.2,3))
    assert all_eq(pa, 0)

def test_get_cdelt_pa2():
    cdelt  = (-1.5, 3)
    pa = -25.
    header = create_fitsheader((10,10), cdelt=cdelt, pa=pa)
    cdelt_, pa_ = get_cdelt_pa(header)
    assert all_eq(cdelt, cdelt_)
    assert all_eq(pa, pa_)

def test_combine_fitsheader():
    headers = [
        create_fitsheader((1,1), cdelt=3., crval=(0,0)),
        create_fitsheader((3,3), cdelt=1., crval=(1,1)),
        create_fitsheader((5,5), cdelt=1., crval=(3,3)),
        create_fitsheader((2,2), cdelt=1., crval=(5,5)),
    ]
    header0 = combine_fitsheader(headers)
    proj0 = wcs.Projection(header0)
    epsilon = 1.e-10
    def func(iheader, header):
        nx = header['NAXIS1']
        ny = header['NAXIS2']
        x = header['CRPIX1']
        y = header['CRPIX2']
        edges = (np.array(3*(0.5,x,nx+0.5)),
                 np.array((0.5,0.5,0.5,y,y,y,ny+0.5,ny+0.5,ny+0.5)))
        a,d = wcs.Projection(header).toworld(edges)
        x0,y0 = proj0.topixel((a,d))
        assert np.all(x0 > 0.5-epsilon)
        assert np.all(y0 > 0.5-epsilon)
        assert np.all(x0 < header0['NAXIS1'] + 0.5 + epsilon)
        assert np.all(y0 < header0['NAXIS2'] + 0.5 + epsilon)
        if iheader == 0:
            assert y0[0] <= 1.5 and x0[-1] >= header0['NAXIS1']-0.5
        if iheader == len(headers)-1:
            assert x0[0] <= 1.5 and y0[-1] >= header0['NAXIS2']-0.5
    for iheader, header in enumerate(headers):
        yield func, iheader, header

def test_has_wcs():
    header = create_fitsheader([2,4])
    assert not has_wcs(header)
    header = create_fitsheader([2,4], cdelt=1)
    assert has_wcs(header)

def test_create_fitsheader():
    shapes = [ (), (1,), (1,2), (0,1), (1,0), (2,3)]
    arrays = [np.ones(()) for s in shapes ]
    def func(a):
        h1 = create_fitsheader(a.shape, dtype=float)
        h2 = create_fitsheader(fromdata=a)
        assert h1 == h2
        assert h1['NAXIS'] == max(a.ndim, 1)
        assert h1['NAXIS1'] == a.shape[-1] if a.ndim > 0 else 1
    for a in arrays:
        yield func, a

def test_fitsheader2shape():
    def func(naxes):
        header = create_fitsheader(naxes)
        if len(naxes) == 0:
            naxes = (1,)
        assert all_eq(fitsheader2shape(header), naxes[::-1])
    for naxes in [(), (1,), (1,2), (2,0,3)]:
        yield func, naxes
