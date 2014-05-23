import numpy as np
import os

from astropy.io import fits as pyfits
from astropy.wcs import WCS
from pyoperators.utils.testing import assert_eq, assert_same, skiptest_unless_module
from pysimulators.wcsutils import (
    angle_lonlat, barycenter_lonlat, combine_fitsheader, create_fitsheader,
    create_fitsheader_for, fitsheader2shape, get_cdelt_pa, has_wcs, mean_degrees,
    WCSToWorldOperator)


def test_mean_degrees():
    assert mean_degrees([1, 2]) == 1.5
    assert_same(mean_degrees([1, 359.1]), 0.05, atol=100)
    assert_same(mean_degrees([0.1, 359.1]), 359.6)


def test_angle_lonlat1():
    assert_same(angle_lonlat(30, 0, 40, 0), 10, atol=100)


def test_angle_lonlat2():
    input = (((30, 0), (40, 0), 10),
             ((39, 0), (92, 90), 90),
             ((39, 0), (37, -90), 90),
             ((37, -90), (39, 0), 90),
             ((100, 90), (134, -90), 180),
             ((24, 30), (24, 32), 2))

    def func(c1, c2, angle):
        assert_same(angle_lonlat(c1, c2), angle, rtol=1000)
    for c1, c2, angle in input:
        yield func, c1, c2, angle


def test_barycenter_lonlat():
    assert_eq(barycenter_lonlat([30, 40], [0, 0]), (35, 0))
    assert_eq(barycenter_lonlat([20, 20, 20], [-90, 0, 90]), (20, 0))
    assert_eq(barycenter_lonlat([20, 20, 20], [0, 45, 90]), (20, 45))


def test_get_cdelt_pa1():
    header = create_fitsheader((10, 10), cdelt=(-1.2, 3))
    cdelt, pa = get_cdelt_pa(header)
    assert_eq(cdelt, (-1.2, 3))
    assert_eq(pa, 0)


def test_get_cdelt_pa2():
    cdelt = (-1.5, 3)
    pa = -25.
    header = create_fitsheader((10, 10), cdelt=cdelt, pa=pa)
    cdelt_, pa_ = get_cdelt_pa(header)
    assert_eq(cdelt, cdelt_)
    assert_eq(pa, pa_)


def test_combine_fitsheader():
    headers = [
        create_fitsheader((1, 1), cdelt=3., crval=(0, 0)),
        create_fitsheader((3, 3), cdelt=1., crval=(1, 1)),
        create_fitsheader((5, 5), cdelt=1., crval=(3, 3)),
        create_fitsheader((2, 2), cdelt=1., crval=(5, 5)),
    ]
    header0 = combine_fitsheader(headers)
    proj0 = WCS(header0)
    epsilon = 1.e-10

    def func(iheader, header):
        nx = header['NAXIS1']
        ny = header['NAXIS2']
        x = header['CRPIX1']
        y = header['CRPIX2']
        edges = (np.array(3*(0.5, x, nx+0.5)),
                 np.array((0.5, 0.5, 0.5, y, y, y, ny+0.5, ny+0.5, ny+0.5)))
        a, d = WCS(header).wcs_pix2world(edges[0], edges[1], 1)
        x0, y0 = proj0.wcs_world2pix(a, d, 1)
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
    header = create_fitsheader([2, 4])
    assert not has_wcs(header)
    header = create_fitsheader([2, 4], cdelt=1)
    assert has_wcs(header)


def test_create_fitsheader1():
    shapes = [(), (1,), (1, 2), (0, 1), (1, 0), (2, 3)]
    arrays = [np.ones(s) for s in shapes]

    def func(a):
        h1 = create_fitsheader(a.shape[::-1],
                               dtype=float if a.ndim > 0 else np.int32)
        h2 = create_fitsheader_for(a if a.ndim > 0 else None)
        assert h1 == h2
        assert h1['NAXIS'] == a.ndim
        for i, s in enumerate(a.shape[::-1]):
            assert h1['NAXIS'+str(i+1)] == s
    for a in arrays:
        yield func, a


def test_create_fitsheader2():
    header = create_fitsheader_for(None)
    assert header['NAXIS'] == 0


def test_create_fitsheader3():
    header = create_fitsheader((10, 3), cdelt=1)
    assert_eq(header['RADESYS'], 'ICRS')
    assert 'EQUINOX' not in header


def test_fitsheader2shape():
    def func(naxes):
        header = create_fitsheader(naxes)
        assert_eq(fitsheader2shape(header), naxes[::-1])
    for naxes in [(), (1,), (1, 2), (2, 0, 3)]:
        yield func, naxes


@skiptest_unless_module('kapteyn')
def test_wcsoperator_kapteyn():
    from pysimulators.wcsutils import WCSKapteynToWorldOperator
    path = os.path.join(os.path.dirname(__file__), 'data/header_gnomonic.fits')
    header = pyfits.open(path)[0].header

    toworld_kapteyn = WCSKapteynToWorldOperator(header)
    crpix = (header['CRPIX1'], header['CRPIX2'])
    crval = (header['CRVAL1'], header['CRVAL2'])
    assert_eq(toworld_kapteyn(crpix), crval)
    assert_eq(toworld_kapteyn.I(crval), crpix)


def test_wcsoperator():
    path = os.path.join(os.path.dirname(__file__), 'data/header_gnomonic.fits')
    header = pyfits.open(path)[0].header

    def func(origin):
        crval = (header['CRVAL1'], header['CRVAL2'])
        crpix = np.array((header['CRPIX1'], header['CRPIX2'])) - 1 + origin
        toworld = WCSToWorldOperator(header, origin)
        assert_same(toworld(crpix), crval, atol=10)
        assert_same(toworld.I(crval), crpix, atol=100000)

    for origin in (0, 1):
        yield func, origin
