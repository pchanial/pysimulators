from __future__ import division

import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.special

from matplotlib import pyplot as mp
from pyoperators import FFTOperator
from pyoperators.utils import isscalar, product
from . import _flib as flib
from .datatypes import Map

__all__ = [
    'airy_disk',
    'aperture_circular',
    'distance',
    'ds9',
    'gaussian',
    'integrated_profile',
    'phasemask_fourquadrant',
    'profile',
    'profile_psd2',
    'psd2',
]


def airy_disk(shape, fwhm=None, r0=None, origin=None, resolution=1):
    """
    Return a two-dimensional map with an Airy pattern.

    Parameters
    ----------
    shape : tuple
        The map shape (ny, nx).
    fwhw : float
        The Full Width Half Maximum of the primary lobe.
    f0 : float
        The radius of the first dark ring, where the intensity is zero.
    origin : tuple of float
        The pattern center of the form (x0,y0)
    resolution : float
        The pixel scale.

    """
    if fwhm is None and r0 is None:
        raise ValueError('No scale parameter.')
    d = distance(shape, origin=origin, resolution=resolution)
    index = np.where(d == 0)
    d[index] = 1.e-30
    if fwhm is not None:
        d *= 1.61633 / (fwhm / 2)
    else:
        d *= 3.8317 / r0
    d = (2 * scipy.special.jn(1, d) / d)**2
    d /= np.sum(d)
    return d


def aperture_circular(shape, diameter, origin=None, resolution=1):
    """
    Return a two-dimensional map with circular mask.

    Parameters
    ----------
    shape : tuple
        The map shape (ny, nx).
    fwhw : float
        The Full Width Half Maximum of the primary lobe.
    origin : tuple of float
        The pattern center of the form (x0,y0)
    resolution : float
        The pixel scale.

    """
    array = distance(shape, origin=origin, resolution=resolution)
    m = array > diameter / 2
    array[m] = 0
    array[~m] = 1
    return array


def distance(shape, origin=None, resolution=1.):
    """
    Returns an array whose values are the distances to a given origin.

    Parameters
    ----------
    shape : tuple of integer
        dimensions of the output array. For a 2d array, the first integer
        is for the Y-axis and the second one for the X-axis.
    origin : array-like in the form of (x0, y0, ...), optional
        The coordinates of the point from which the distance is calculated,
        assuming a zero-based coordinate indexing. Default value is the array
        center.
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
        origin = (np.array(shape[::-1], dtype=float) - 1) / 2
    else:
        origin = np.ascontiguousarray(origin, dtype=float)

    unit = getattr(resolution, '_unit', None)

    if isscalar(resolution):
        resolution = np.resize(resolution, rank)
    resolution = np.asanyarray(resolution, dtype=float)

    du = flib.datautils
    if rank == 1:
        d = du.distance_1d(shape[0], origin[0], resolution[0])
    elif rank == 2:
        d = du.distance_2d(shape[1], shape[0], origin, resolution).T
    elif rank == 3:
        d = du.distance_3d(shape[2], shape[1], shape[0], origin, resolution).T
    else:
        d = _distance_slow(shape, origin, resolution, float)

    return Map(d, copy=False, unit=unit)


def _distance_slow(shape, origin, resolution, dtype):
    """
    Returns an array whose values are the distances to a given origin.

    This routine is written using np.meshgrid routine. It is slower
    than the Fortran-based `distance` routine, but can handle any number
    of dimensions.

    Refer to `pysimulators.distance` for full documentation.
    """

    if dtype is None:
        dtype = float
    index = []
    for n, o, r in zip(reversed(shape), origin, resolution):
        index.append(slice(0, n))
    d = np.asarray(np.mgrid[index], dtype).T
    d -= np.asanyarray(origin)
    d *= resolution
    np.square(d, d)
    d = Map(np.sqrt(np.sum(d, axis=d.shape[-1])), dtype=dtype, copy=False)
    return d


class Ds9(object):
    """
    Helper around the ds9 package.

    Examples
    --------
    >>> ds9.open_in_new_window = False
    >>> d = ds9.current
    >>> d.set('scale linear')
    >>> mynewmap = gaussian((128, 128), sigma=32)
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


def gaussian(shape, sigma=None, fwhm=None, origin=None, resolution=1,
             unit=None):
    """
    Returns an array whose values are the distances to a given origin.

    Parameters
    ----------
    shape : tuple of integer
        dimensions of the output array. For a 2d array, the first integer
        is for the Y-axis and the second one for the X-axis.
    fwhm : float or tuple of float
        The Full Width Half Maximum of the gaussian (fwhm_x, fwhm_y, ...).
    sigma : float or tuple of float
        The sigma parameter (sigma_x, sigma_y, ...).
    origin : array-like in the form of (x0, y0, ...), optional
        The coordinates of the point from which the distance is calculated,
        assuming a zero-based coordinate indexing. Default value is the array
        center.
    resolution : array-like in the form of (dx, dy, ...), optional
        Inter-pixel distance. Default is one.
    unit : string or dict
        The unit of the returned Map.

    """
    if sigma is None and fwhm is None:
        raise ValueError('The shape of the gaussian is not specified.')
    if sigma is None:
        sigma = fwhm / np.sqrt(8 * np.log(2))
    d = distance(shape, origin=origin, resolution=resolution / (np.sqrt(2) *
                 sigma))
    d.unit = ''
    d = np.exp(-d**2)
    d /= np.sum(d)
    if unit:
        d.unit = unit
    return d


def integrated_profile(input, origin=None, bin=1., nbins=None):
    """
    Returns axisymmetric integrated profile of a 2d image.
    x, y = integrated_profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: array
        2d input array.
    origin: (x0,y0)
        center of the profile. (Fits convention). Default is the image center.
    bin: number
        width of the profile bins (in unit of pixels).
    nbins: integer
        number of profile bins.

    Returns
    -------
    x : array
        The strict upper boundary within which each integration is performed.
    y : array
        The integrated profile.

    """
    x, y, n = profile(input, origin=origin, bin=bin, nbins=nbins,
                      histogram=True)
    x = np.arange(1, y.size + 1) * bin
    y[~np.isfinite(y)] = 0
    y *= n
    return x, np.add.accumulate(y)


def phasemask_fourquadrant(shape, phase=-1):
    array = Map.ones(shape, dtype=complex)
    array[0:shape[0]//2, shape[1]//2:] = phase
    array[shape[0]//2:, 0:shape[1]//2] = phase
    return array


def plot_tod(tod, mask=None, **kw):
    """Plot the signal timelines in a Tod and show masked samples.

    Plotting every detector timelines may be time consuming, so it is
    recommended to use this method on one or few detectors like this:
    >>> plot_tod(tod[idetector])

    """
    if mask is None:
        mask = getattr(tod, 'mask', None)

    ndetectors = product(tod.shape[0:-1])
    tod = tod.view().reshape((ndetectors, -1))
    if mask is not None:
        mask = mask.view().reshape((ndetectors, -1))
        if np.all(mask):
            print('There is no valid sample.')
            return

    for idetector in range(ndetectors):
        mp.plot(tod[idetector], **kw)
        if mask is not None:
            index = np.where(mask[idetector])
            mp.plot(index, tod[idetector, index], 'ro')

    unit = getattr(tod, 'unit', '')
    if unit:
        mp.ylabel('Signal [' + unit + ']')
    else:
        mp.ylabel('Signal')
    mp.xlabel('Time sample')


def profile(input, origin=None, bin=1., nbins=None, histogram=False):
    """
    Returns axisymmetric profile of a 2d image.
    x, y[, n] = profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: array
        2d input array
    origin: (x0,y0)
        Center of the profile. (Fits convention). Default is the image center.
    bin: number
        Width of the profile bins (in unit of pixels).
    nbins: integer
        Number of profile bins.
    histogram: boolean
        If set to True, return the histogram.

    """
    input = np.ascontiguousarray(input, float)
    if origin is None:
        origin = (np.array(input.shape[::-1], float) + 1) / 2
    else:
        origin = np.ascontiguousarray(origin, float)

    if nbins is None:
        nbins = int(max(input.shape[0]-origin[1], origin[1],
                        input.shape[1]-origin[0], origin[0]) / bin)

    x, y, n = flib.datautils.profile_axisymmetric_2d(input.T, origin, bin,
                                                     nbins)
    if histogram:
        return x, y, n
    else:
        return x, y


def psd2(array, sampling_frequency=1, fftw_flag='FFTW_MEASURE'):
    """
    Return two-dimensional PSD.

    Parameters
    ----------
    array : array-like
       Two-dimensional array.
    sampling_frequency : float
       The sampling frequency.
    fftw_flag : string
       The FFTW planner flag.

    Returns
    -------
    psd : ndarray
       The Power Spectrum Density, such as
           sum psd * bandwidth = mean(array**2)
       with bandwidth = sampling_frequency ** 2 / array.size is the PSD bin
       area.

    """
    array = np.asarray(array)
    s = np.abs(FFTOperator(array.shape, fftw_flag=fftw_flag)(array))**2
    s /= array.size**2
    bandwidth = sampling_frequency**2 / array.size
    s /= bandwidth
    return Map(scipy.fftpack.fftshift(s))


def profile_psd2(array, sampling_frequency=1, fftw_flag='FFTW_MEASURE'):
    """
    Return the axisymmetric profile of the PSD of a 2-dimensional image.

    Parameters
    ----------
    array : array-like
       Two-dimensional input array.
    sampling_frequency : float
       The sampling frequency.
    fftw_flag : string
       The FFTW planner flag.

    Returns
    -------
    x : ndarray
       The frequency bin centers, spaced by sampling_frequency / array.shape[1]
    y : ndarray
       The PSD profile. For white noise of standard deviation sigma, the
       following relationship stands:
           np.mean(y) * sampling_frequency**2 ~= sigma**2

    """
    array = np.asanyarray(array)
    if array.ndim != 2:
        raise ValueError('The input array is not two-dimensional.')
    if array.shape[0] != array.shape[1]:
        raise ValueError('The input array is not square.')
    psd = psd2(array, sampling_frequency=sampling_frequency,
               fftw_flag=fftw_flag)
    x, y = profile(psd, origin=array.shape // np.array(2) + 1)
    x *= sampling_frequency / array.shape[1]
    return x, y
