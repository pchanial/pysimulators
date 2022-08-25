from pathlib import Path

import pytest
from astropy.io import fits
from numpy.testing import assert_allclose

from pysimulators.wcsutils import WCSKapteynToWorldOperator

pytest.importorskip('kapteyn')

DATAPATH = Path(__file__).parent / 'data'


def test_wcsoperator_kapteyn():
    path = DATAPATH / 'header_gnomonic.fits'
    header = fits.open(path)[0].header

    toworld_kapteyn = WCSKapteynToWorldOperator(header)
    crpix = (header['CRPIX1'], header['CRPIX2'])
    crval = (header['CRVAL1'], header['CRVAL2'])
    assert_allclose(toworld_kapteyn(crpix), crval)
    assert_allclose(toworld_kapteyn.I(crval), crpix)
