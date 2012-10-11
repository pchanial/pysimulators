# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#

from __future__ import division

import numpy as np
import pyfits
import scipy.interpolate as interp

from kapteyn import wcs as kwcs
from pyoperators import Operator
from pyoperators.decorators import real, square, inplace
from pyoperators.utils import isscalar, operation_assignment
from . import _flib as flib

__all__ = [ 
    'angle_lonlat',
    'barycenter_lonlat',
    'create_fitsheader',
    'fitsheader2shape',
    'str2fitsheader',
    'DistortionOperator',
    'WCSToPixelOperator',
    'WCSToWorldOperator',
]

def angle_lonlat(lon1, lat1, lon2=None, lat2=None):
    """
    Returns the angle between vectors on the celestial sphere in degrees.

    Parameters
    ----------
    lon1, lon2 : array of numbers
        longitude in degrees
    lat1, lat2 : array of numbers
        latitude in degrees

    Example
    -------
    >>> angle_lonlat((ra1,dec1), (ra2,dec2))
    >>> angle_lonlat(ra1, dec1, ra2, dec2)
    """

    if lon2 is None and lat2 is None:
        lon2, lat2 = lat1
        lon1, lat1 = lon1
    lon1 = np.array(lon1, float, ndmin=1, copy=False).ravel()
    lat1 = np.array(lat1, float, ndmin=1, copy=False).ravel()
    lon2 = np.array(lon2, float, ndmin=1, copy=False).ravel()
    lat2 = np.array(lat2, float, ndmin=1, copy=False).ravel()
    angle = flib.wcsutils.angle_lonlat(lon1, lat1, lon2, lat2)
    if angle.size == 1:
        angle = float(angle)
    return angle


#-------------------------------------------------------------------------------


def barycenter_lonlat(lon, lat):
    """
    Returns the barycenter of vectors on the celestial sphere.

    Parameters
    ----------
    lon : array of numbers
        Longitude in degrees.
    lat : array of numbers
        Latitude in degrees.
    """
    lon = np.array(lon, float, ndmin=1, copy=False).ravel()
    lat = np.array(lat, float, ndmin=1, copy=False).ravel()
    return flib.wcsutils.barycenter_lonlat(lon, lat)


#-------------------------------------------------------------------------------


def combine_fitsheader(headers, cdelt=None, pa=None):
    """
    Returns a FITS header which encompasses all input headers.

    Parameters
    ----------
    headers : list of pyfits.Header
        The headers of the maps that must be contained in the map of the
        returned header.
    cdelt : float or 2 element array
        Physical increment at the reference pixel. Default is the lowest
        header cdelt.
    pa : float
        Position angle of the Y=AXIS2 axis (=-CROTA2). Default is the mean 
        header pa.
    """

    if not isinstance(headers, (list, tuple)):
        headers = (headers,)

    if len(headers) == 1 and cdelt is None and pa is None:
        return headers[0]

    # compute the world coordinates of the reference pixel of the combined map
    crval = barycenter_lonlat([h['CRVAL1'] for h in headers],
                              [h['CRVAL2'] for h in headers])

    # compute the combined map cdelt & pa
    if cdelt is None or pa is None:
        cdeltpa = [get_cdelt_pa(h) for h in headers]

    if cdelt is None:
        cdelt = min([np.min(abs(cdelt)) for cdelt, pa in cdeltpa])

    if pa is None:
        pa = flib.wcsutils.mean_degrees(np.array([pa for cdelt, pa in cdeltpa]))

    # first, create the combined header with a single pixel centred on the
    # reference coordinates, with correct orientation and cdelt
    header0 = create_fitsheader((1,1), cdelt=cdelt, pa=pa, crpix=(1,1),
                                crval=crval)

    # then, 'enlarge' it to make it fit the edges of each header
    proj0 = kwcs.Projection(header0)
    xy0 = []
    for h in headers:
        proj = kwcs.Projection(h)
        nx = h['NAXIS1']
        ny = h['NAXIS2']
        x  = h['CRPIX1']
        y  = h['CRPIX2']
        edges = proj.toworld(([0.5, x, nx+0.5, 0.5, x, nx+0.5, 0.5, x, nx+0.5],
                              [0.5, 0.5, 0.5, y, y, y, ny+0.5, ny+0.5, ny+0.5]))
        xy0.append(proj0.topixel(edges))

    xmin0 = np.round(min([min(c[0]) for c in xy0]))
    xmax0 = np.round(max([max(c[0]) for c in xy0]))
    ymin0 = np.round(min([min(c[1]) for c in xy0]))
    ymax0 = np.round(max([max(c[1]) for c in xy0]))

    header0['NAXIS1'] = int(xmax0 - xmin0 + 1)
    header0['NAXIS2'] = int(ymax0 - ymin0 + 1)
    header0['CRPIX1'] = 2 - xmin0
    header0['CRPIX2'] = 2 - ymin0

    return header0


#-------------------------------------------------------------------------------


def create_fitsheader(naxes=None, dtype=None, fromdata=None, extname=None,
                      crval=(0.,0.), crpix=None, ctype=('RA---TAN','DEC--TAN'),
                      cunit='deg', cd=None, cdelt=None, pa=None, equinox=2000.):
    """
    Helper to create a FITS header.

    Parameters
    ----------
    naxes : array, optional
        (NAXIS1,NAXIS2,...) tuple, which specifies the data dimensions.
        Note that the dimensions in FITS and ndarrays are reversed.
    dtype : data-type, optional
        Desired data type for the data stored in the FITS file
    fromdata : array_like, optional
        An array from which the dimensions and typewill be extracted. Note
        that following the FITS convention, the dimension along X is the
        second value of the array shape and that the dimension along the
        Y axis is the first one.
    extname : None or string
        if a string is specified ('' can be used), the returned header
        type will be an Image HDU (otherwise a Primary HDU).
    crval : 2 element array, optional
        Reference pixel values (FITS convention).
    crpix : 2 element array, optional
        Reference pixel (FITS convention).
    ctype : 2 element string array, optional
        Projection types.
    cunit : string or 2 element string array
        Units of the CD matrix (default is degrees/pixel).
    cd : 2 x 2 array
        FITS parameters
            CD1_1 CD1_2
            CD2_1 CD2_2
    cdelt : float or 2 element array
        Physical increment at the reference pixel.
    pa : float
        Position angle of the Y=AXIS2 axis (=-CROTA2).
    equinox : float
        Reference equinox

    Examples
    --------
    >>> map = Map.ones((10,100), unit='Jy/pixel')
    >>> map.header = create_fitsheader(map.shape[::-1], cd=[[-1,0],[0,1]])
    >>> map.header = create_fitsheader(fromdata=map)

    """
    if naxes is not None and fromdata is not None or naxes is None and \
       fromdata is None:
        raise ValueError("Either keyword 'naxes' or 'fromdata' must be specifie"
                         "d.")

    if fromdata is None:
        naxes = np.array(naxes, dtype=int, ndmin=1)
        naxes = tuple(naxes)
        if len(naxes) > 8:
            raise ValueError('First argument is naxes=(NAXIS1,NAXIS2,...)')
        if dtype is not None:
            dtype = np.dtype(dtype)
            typename = dtype.name
        else:
            typename = 'float64'
    else:
        array = np.array(fromdata, copy=False)
        naxes = tuple(reversed(array.shape))
        if dtype is not None:
            array = array.astype(dtype)
        if array.dtype.itemsize == 1:
            typename = 'uint8'
        elif array.dtype.names is not None:
            typename = None
        else:
            typename = array.dtype.name

    # FITS format does not handle scalar values
    if len(naxes) == 0:
        naxes = (1,)

    numaxis = len(naxes)

    if extname is None:
        card = pyfits.create_card('simple', True)
    else:
        card = pyfits.create_card('xtension', 'IMAGE', 'Image extension')
    header = pyfits.Header([card])
    if typename is not None:
        header.update('bitpix', pyfits.PrimaryHDU.ImgCode[typename],
                      'array data type')
    header.update('naxis', numaxis, 'number of array dimensions')
    for dim in range(numaxis):
        header.update('naxis'+str(dim+1), naxes[dim])
    if extname is None:
        header.update('extend', True)
    else:
        header.update('pcount', 0, 'number of parameters')
        header.update('gcount', 1, 'number of groups')
        header.update('extname', extname)

    if cd is not None:
        cd = np.asarray(cd, dtype=float)
        if cd.shape != (2,2):
            raise ValueError('The CD matrix is not a 2x2 matrix.')
    else:
        if cdelt is None:
            return header
        cdelt = np.array(cdelt, float)
        if isscalar(cdelt):
            cdelt = np.array([-cdelt, cdelt])
        if pa is None:
            pa = 0.
        theta=np.deg2rad(-pa)
        cd = np.diag(cdelt).dot(np.array([[ np.cos(theta), np.sin(theta)],
                                          [-np.sin(theta), np.cos(theta)]]))

    crval = np.asarray(crval, float)
    if crval.size != 2:
        raise ValueError('CRVAL does not have two elements.')

    if crpix is None:
        crpix = (np.array(naxes) + 1) / 2
    else:
        crpix = np.asarray(crpix, float)
    if crpix.size != 2:
        raise ValueError('CRPIX does not have two elements.')

    ctype = np.asarray(ctype, dtype=np.string_)
    if ctype.size != 2:
        raise ValueError('CTYPE does not have two elements.')

    if isscalar(cunit):
        cunit = (cunit, cunit)
    cunit = np.asarray(cunit, dtype=np.string_)
    if cunit.size != 2:
        raise ValueError('CUNIT does not have two elements.')

    header.update('crval1', crval[0])
    header.update('crval2', crval[1])
    header.update('crpix1', crpix[0])
    header.update('crpix2', crpix[1])
    header.update('cd1_1' , cd[0,0])
    header.update('cd2_1' , cd[1,0])
    header.update('cd1_2' , cd[0,1])
    header.update('cd2_2' , cd[1,1])
    header.update('ctype1', ctype[0])
    header.update('ctype2', ctype[1])
    header.update('cunit1', cunit[0])
    header.update('cunit2', cunit[1])
    header.update('equinox', 2000.)

    return header


#-------------------------------------------------------------------------------


def fitsheader2shape(header):
    ndim = header['NAXIS']
    return tuple(header['NAXIS{0}'.format(i)] for i in range(ndim,0,-1))


#-------------------------------------------------------------------------------


def get_cdelt_pa(header):
    """
    Extract scale and rotation from a FITS header.

    Parameter
    ---------
    header : pyfits Header
        The FITS header

    Returns
    -------
    cdelt : 2 element array
        Physical increment at the reference pixel.
    pa : float
        Position angle of the Y=AXIS2 axis (=-CROTA2).
    """
    try:
        cd = np.array([[header['CD1_1'], header['CD1_2']],
                       [header['CD2_1'], header['CD2_2']]], float)
    except KeyError:
        if any([not k in header for k in ('CDELT1', 'CDELT2', 'CROTA2')]):
            raise KeyError('Header has no astrometry.')
        return np.array([header['CDELT1'], header['CDELT2']]), -header['CROTA2']

    det = np.linalg.det(cd)
    sgn = det / np.abs(det)
    cdelt = np.array([sgn * np.sqrt(cd[0,0]**2 + cd[0,1]**2),
                      np.sqrt(cd[1,1]**2 + cd[1,0]**2)])
    pa = -np.arctan2(-cd[1,0], cd[1,1])
    return cdelt, np.rad2deg(pa)
    

#-------------------------------------------------------------------------------


def has_wcs(header):
    """
    Returns True is the input FITS header has a defined World Coordinate System.
    """

    required = 'CRPIX,CRVAL,CTYPE'.split(',')
    keywords = np.concatenate([(lambda i: [r+str(i+1) for r in required])(i) 
                               for i in range(header['NAXIS'])])
    return all([k in header for k in keywords])


#-------------------------------------------------------------------------------


def mean_degrees(array):
    """
    Returns the mean value of an array of values in degrees, by taking into 
    account the discrepancy at 0 degree
    """
    return flib.wcsutils.mean_degrees(np.asarray(array, dtype=float).ravel())


#-------------------------------------------------------------------------------


def minmax_degrees(array):
    """
    Returns the minimum and maximum value of an array of values in degrees, 
    by taking into account the discrepancy at 0 degree.
    """
    return flib.wcsutils.minmax_degrees(np.asarray(array, dtype=float).ravel())


#-------------------------------------------------------------------------------


def str2fitsheader(string):
    """
    Convert a string into a pyfits.Header object

    All cards are extracted from the input string until the END keyword is
    reached.
    """
    header = pyfits.Header()
    cards = header.ascard()
    iline = 0
    while (iline*80 < len(string)):
        line = string[iline*80:(iline+1)*80]
        if line[0:3] == 'END': break
        cards.append(pyfits.Card().fromstring(line))
        iline += 1
    return header


#-------------------------------------------------------------------------------


@real
@square
class DistortionOperator(Operator):
    """
    Distortion operator.

    The interpolation is performed using the Clough-Tocher method. 
    Interpolation of points outside the object plane coordinates convex hull
    will return NaN.

    Parameters
    ----------
    xin : array of shape (npoints, ndims)
       Coordinates in the object plane.
    xout : array of shape (npoints, ndims)
       Coordinates in the image plane.

    """
    def __init__(self, xin, xout, **keywords):
        xin = np.asarray(xin)
        xout = np.asarray(xout)
        if xin.shape[-1] != 2:
            raise ValueError('The shape of the object plane coordinates should '
                             'be (npoints,ndims), where ndims is only implement'
                             'ed for 2.')
        if xin.shape != xout.shape:
            raise ValueError('The object and image coordinates do not have the '
                             'same shape.')

        keywords['dtype'] = float
        Operator.__init__(self, **keywords)
        self.interp0 = interp.CloughTocher2DInterpolator(xin, xout[...,0])
        self.interp1 = interp.CloughTocher2DInterpolator(xin, xout[...,1])
        self.set_rule('.I', lambda s:DistortionOperator(xout, xin))

    def direct(self, input, output):
        output[...,0] = self.interp0(input)
        output[...,1] = self.interp1(input)


@real
@square
@inplace
class _WCSKapteynOperator(Operator):
    def __init__(self, wcs, **keywords):
        """
        wcs : FITS header or Kapteyn wcs.Projection instance
            Representation of the world coordinate system
        """
        if isinstance(wcs, pyfits.Header):
            wcs = kwcs.Projection(wcs)
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        wcs.allow_invalid = True
        Operator.__init__(self, **keywords)
        self.wcs = wcs

    def validatein(self, shapein):
        if len(shapein) == 0:
            raise ValueError('Invalid scalar input.')
        if shapein[-1] != 2:
            raise ValueError("Invalid last dimension: '{0}'.".format(
                             shapein[-1]))


class WCSToPixelOperator(_WCSKapteynOperator):
    """
    Operator for WCS world-to-pixel transforms.
    
    Example
    -------
    >>> header = pyfits.open('myfile.fits')
    >>> w = WCSToPixelOperator(header)
    >>> pix = w((ra0,dec0))

    """
    def __init__(self, wcs, **keywords):
        _WCSKapteynOperator.__init__(self, wcs, **keywords)
        self.set_rule('.I', lambda s: WCSToWorldOperator(s.wcs))
    __init__.__doc__ = _WCSKapteynOperator.__init__.__doc__

    def direct(self, input, output, operation=operation_assignment):
        operation(output, self.wcs.topixel(input))


class WCSToWorldOperator(_WCSKapteynOperator):
    """
    Operator for WCS pixel-to-world transforms.
    
    Example
    -------
    >>> header = pyfits.open('myfile.fits')
    >>> w = WCSToWorldOperator(header)
    >>> radec = w((x,y))

    """
    def __init__(self, wcs, **keywords):
        _WCSKapteynOperator.__init__(self, wcs, **keywords)
        self.set_rule('.I', lambda s: WCSToPixelOperator(s.wcs))
    __init__.__doc__ = _WCSKapteynOperator.__init__.__doc__

    def direct(self, input, output, operation=operation_assignment):
        operation(output, self.wcs.toworld(input))

