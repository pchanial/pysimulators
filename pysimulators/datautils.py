from __future__ import division

import numpy as  np
import scipy.signal
import scipy.special

from pyoperators.utils import isscalar
#from . import tamasisfortran as tmf
from .datatypes import Map

__all__ = [ 
    'airy_disk',
    'aperture_circular',
    'distance',
    'ds9',
    'gaussian',
    'phasemask_fourquadrant',
    'psd2',
]


def airy_disk(shape, fwhm=None, r0=None, origin=None, resolution=1.):
    """
    Return a two-dimensional map with an Airy pattern.

    Parameters
    ----------
    shape : tuple
        The map shape
    fwhw : float
        The Full Width Half Maximum of the primary lobe.
    f0 : float
        The radius of the first dark ring, where the intensity is zero.
    origin : tuple of float
        The pattern center (FITS convention)
    resolution : float
        The pixel scale
    """
    if fwhm is None and r0 is None:
        raise ValueError('No scale parameter.')
    d  = distance(shape, origin=origin, resolution=resolution)
    index = np.where(d == 0)
    d[index] = 1.e-30
    if fwhm is not None:
        d *= 1.61633 / (fwhm / 2)
    else:
        d *= 3.8317 / r0
    d  = (2 * scipy.special.jn(1,d) / d)**2
    d /= np.sum(d)
    return d


#-------------------------------------------------------------------------------


def aperture_circular(shape, diameter, origin=None, resolution=1.):
    array = distance(shape, origin=origin, resolution=resolution)
    m = array > diameter / 2
    array[ m] = 0
    array[~m] = 1
    return array


#-------------------------------------------------------------------------------


def distance(shape, origin=None, resolution=1.):
    """
    Returns an array whose values are the distances to a given origin.
    
    Parameters
    ----------
    shape : tuple of integer
        dimensions of the output array. For a 2d array, the first integer
        is for the Y-axis and the second one for the X-axis.
    origin : array-like in the form of (x0, y0, ...), optional
        The coordinates, according to the FITS convention (i.e. first
        column and row is one), from which the distance is calculated.
        Default value is the array center.
    resolution : array-like in the form of (dx, dy, ...), optional
        Inter-pixel distance. Default is one. If resolution is a Quantity, its
        unit will be carried over to the returned distance array

    Example
    -------
    nx, ny = 3, 3
    print(distance((ny,nx)))
    [[ 1.41421356  1.          1.41421356]
    [ 1.          0.          1.        ]
    [ 1.41421356  1.          1.41421356]]
    """
    if isscalar(shape):
        shape = (shape,)
    else:
        shape = tuple(shape)
    shape = tuple(int(np.round(s)) for s in shape)
    rank = len(shape)

    if origin is None:
        origin = (np.array(shape[::-1], dtype=float) + 1) / 2
    else:
        origin = np.ascontiguousarray(origin, dtype=float)

    unit = getattr(resolution, '_unit', None)

    if isscalar(resolution):
        resolution = np.resize(resolution, rank)
    resolution = np.asanyarray(resolution, dtype=float)

    if rank == 1:
        d = tmf.distance_1d(shape[0], origin[0], resolution[0])
    elif rank == 2:
        d = tmf.distance_2d(shape[1], shape[0], origin, resolution).T
    elif rank == 3:
        d = tmf.distance_3d(shape[2], shape[1], shape[0], origin, resolution).T
    else:
        d = _distance_slow(shape, origin, resolution, float)

    return Map(d, copy=False, unit=unit)


#-------------------------------------------------------------------------------


def _distance_slow(shape, origin, resolution, dtype):
    """
    Returns an array whose values are the distances to a given origin.

    This routine is written using np.meshgrid routine. It is slower
    than the Fortran-based `distance` routine, but can handle any number
    of dimensions.

    Refer to `tamasis.distance` for full documentation.
    """

    if dtype is None:
        dtype = float
    index = []
    for n, o, r in zip(reversed(shape), origin, resolution):
        index.append(slice(0,n))
    d = np.asarray(np.mgrid[index], dtype).T
    d -= np.asanyarray(origin) - 1.
    d *= resolution
    np.square(d, d)
    d = Map(np.sqrt(np.sum(d, axis=d.shape[-1])), dtype=dtype, copy=False)
    return d
    

#-------------------------------------------------------------------------------


class Ds9(object):
    """
    Helper around the ds9 package.
    
    Examples
    --------
    >>> ds9.open_in_new_window = False
    >>> d = ds9.current
    >>> d.set('scale linear')
    >>> ds9(mynewmap, 'zoom to fit')
    """
    def __call__(self, array, xpamsg=None, origin=None, **keywords):
        if not isinstance(array, str):
            array = np.asanyarray(array)
            dtype = array.dtype
        else:
            dtype = None
        array = Map(array, dtype=dtype, copy=False, origin=origin)
        array.ds9(xpamsg=xpamsg, new=self.open_in_new_window, **keywords)

    @property
    def targets(self):
        """
        Returns a list of the ids of the running ds9 instances.
        """
        import ds9
        return ds9.ds9_targets()

    @property
    def current(self):
        """
        Return current ds9 instance.
        """
        targets = self.targets
        if targets is None:
            return None
        return self.get(targets[-1])

    def get(self, id):
        """
        Return ds9 instance matching a specified id.
        """
        import ds9
        return ds9.ds9(id)

    open_in_new_window = True

ds9 = Ds9()


#-------------------------------------------------------------------------------


def gaussian(shape, fwhm, origin=None, resolution=1., unit=None):
    if len(shape) == 2:
        sigma = fwhm / np.sqrt(8*np.log(2))
    else:
        raise NotImplementedError()
    d = distance(shape, origin=origin, resolution=resolution)
    d.unit = ''
    d = np.exp(-d**2 / (2*sigma**2))
    d /= np.sum(d)
    if unit:
        d.unit = unit
    return d


#-------------------------------------------------------------------------------


def phasemask_fourquadrant(shape, phase=-1):
    array = Map.ones(shape, complex)
    array[0:shape[0]//2,shape[1]//2:] = phase
    array[shape[0]//2:,0:shape[1]//2] = phase
    return array


#-------------------------------------------------------------------------------


def psd2(input):
    input = np.asanyarray(input)
    s = np.abs(scipy.signal.fft2(input))
    for axis, n in zip((-2,-1), s.shape[-2:]):
        s = np.roll(s, n // 2, axis=axis)
    return Map(s)

    
