from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as pyfits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS

from pyoperators.utils.testing import assert_eq, assert_same
from pysimulators.wcsutils import (
    WCSToWorldOperator,
    angle_lonlat,
    barycenter_lonlat,
    combine_fitsheader,
    create_fitsheader,
    create_fitsheader_for,
    fitsheader2shape,
    get_cdelt_pa,
    has_wcs,
    mean_degrees,
)

DATAPATH = Path(__file__).parent / 'data'


def test_mean_degrees():
    assert mean_degrees([1, 2]) == 1.5
    assert_same(mean_degrees([1, 359.1]), 0.05, atol=100)
    assert_same(mean_degrees([0.1, 359.1]), 359.6)


def test_angle_lonlat1():
    assert_same(angle_lonlat(30, 0, 40, 0), 10, atol=100)


@pytest.mark.parametrize(
    'c1, c2, angle',
    [
        ((30, 0), (40, 0), 10),
        ((39, 0), (92, 90), 90),
        ((39, 0), (37, -90), 90),
        ((37, -90), (39, 0), 90),
        ((100, 90), (134, -90), 180),
        ((24, 30), (24, 32), 2),
    ],
)
def test_angle_lonlat2(c1, c2, angle):
    assert_same(angle_lonlat(c1, c2), angle, rtol=1000)


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
    pa = -25.0
    header = create_fitsheader((10, 10), cdelt=cdelt, pa=pa)
    cdelt_, pa_ = get_cdelt_pa(header)
    assert_eq(cdelt, cdelt_)
    assert_eq(pa, pa_)


COMBINE_HEADERS = [
    create_fitsheader((1, 1), cdelt=3.0, crval=(0, 0)),
    create_fitsheader((3, 3), cdelt=1.0, crval=(1, 1)),
    create_fitsheader((5, 5), cdelt=1.0, crval=(3, 3)),
    create_fitsheader((2, 2), cdelt=1.0, crval=(5, 5)),
]
COMBINED_HEADER = combine_fitsheader(COMBINE_HEADERS)


@pytest.mark.parametrize('iheader, header', enumerate(COMBINE_HEADERS))
def test_combine_fitsheader(iheader, header):
    proj0 = WCS(COMBINED_HEADER)
    epsilon = 1.0e-10

    nx = header['NAXIS1']
    ny = header['NAXIS2']
    x = header['CRPIX1']
    y = header['CRPIX2']
    edges = (
        np.array(3 * (0.5, x, nx + 0.5)),
        np.array((0.5, 0.5, 0.5, y, y, y, ny + 0.5, ny + 0.5, ny + 0.5)),
    )
    a, d = WCS(header).wcs_pix2world(edges[0], edges[1], 1)
    x0, y0 = proj0.wcs_world2pix(a, d, 1)
    assert np.all(x0 > 0.5 - epsilon)
    assert np.all(y0 > 0.5 - epsilon)
    assert np.all(x0 < COMBINED_HEADER['NAXIS1'] + 0.5 + epsilon)
    assert np.all(y0 < COMBINED_HEADER['NAXIS2'] + 0.5 + epsilon)
    if iheader == 0:
        assert y0[0] <= 1.5 and x0[-1] >= COMBINED_HEADER['NAXIS1'] - 0.5
    if iheader == 3:
        assert x0[0] <= 1.5 and y0[-1] >= COMBINED_HEADER['NAXIS2'] - 0.5


def test_has_wcs():
    header = create_fitsheader([2, 4])
    assert not has_wcs(header)
    header = create_fitsheader([2, 4], cdelt=1)
    assert has_wcs(header)


@pytest.mark.parametrize('shape', [(), (1,), (1, 2), (0, 1), (1, 0), (2, 3)])
def test_create_fitsheader1(shape):
    array = np.ones(shape)

    h1 = create_fitsheader(
        array.shape[::-1], dtype=float if array.ndim > 0 else np.int32
    )
    h2 = create_fitsheader_for(array if array.ndim > 0 else None)
    assert h1 == h2
    assert h1['NAXIS'] == array.ndim
    for i, s in enumerate(array.shape[::-1]):
        assert h1['NAXIS' + str(i + 1)] == s


def test_create_fitsheader2():
    header = create_fitsheader_for(None)
    assert header['NAXIS'] == 0


def test_create_fitsheader3():
    header = create_fitsheader((10, 3), cdelt=1)
    assert_eq(header['RADESYS'], 'ICRS')
    assert 'EQUINOX' not in header


@pytest.mark.parametrize('naxes', [(), (1,), (1, 2), (2, 0, 3)])
def test_fitsheader2shape(naxes):
    header = create_fitsheader(naxes)
    assert_eq(fitsheader2shape(header), naxes[::-1])


@pytest.mark.parametrize('origin', [0, 1])
def test_wcsoperator(origin):
    with pytest.warns(AstropyUserWarning, match='File may have been truncated'):
        header = pyfits.open(DATAPATH / 'header_gnomonic.fits')[0].header

    crval = (header['CRVAL1'], header['CRVAL2'])
    crpix = np.array((header['CRPIX1'], header['CRPIX2'])) - 1 + origin
    toworld = WCSToWorldOperator(header, origin)
    assert_same(toworld(crpix), crval, atol=10)
    assert_same(toworld.I(crval), crpix, atol=100000)
