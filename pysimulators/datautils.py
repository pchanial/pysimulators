from __future__ import absolute_import, division, print_function
from pyoperators import FFTOperator
from pyoperators.utils import (
    float_intrinsic_dtype,
    isalias,
    isscalarlike,
    product,
    tointtuple,
)
from pyoperators.utils.ufuncs import abs2
from . import _flib as flib
from .datatypes import Map
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.special

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


def airy_disk(shape, fwhm=None, r0=None, origin=None, scale=1, dtype=float):
    """
    Return a two-dimensional map with an Airy pattern.

    Parameters
    ----------
    shape : tuple
        The map shape (ny, nx).
    fwhw : float
        The Full Width Half Maximum of the primary lobe.
    f0 : float, optional
        The radius of the first dark ring, where the intensity is zero.
    origin : array-like, optional
        Center (x0, y0) of the Airy disk, in pixel coordinates. By convention,
        the coordinates of the center of the pixel [0, 0] are (0, 0).
        Default is the image center.
    scale : float or array-like of two elements, optional
        The inter-pixel distance (dx, dy).
    dtype : np.dtype, optional
        The output data type.

    """
    if fwhm is None and r0 is None:
        raise ValueError('No scale parameter.')
    d = distance(shape, origin=origin, scale=scale, dtype=dtype)
    index = np.where(d == 0)
    d[index] = 1.0e-30
    if fwhm is not None:
        d *= 1.61633 / (fwhm / 2)
    else:
        d *= 3.8317 / r0
    d = (2 * scipy.special.jn(1, d) / d) ** 2
    d /= np.sum(d)
    return d


def aperture_circular(shape, diameter, origin=None, scale=1, dtype=float):
    """
    Return a two-dimensional map with circular mask.

    Parameters
    ----------
    shape : tuple
        The map shape (ny, nx).
    diameter : float
        The aperture diameter.
    origin : array-like, optional
        Center (x0, y0) of the aperture, in pixel coordinates. By convention,
        the coordinates of the center of the pixel [0, 0] are (0, 0).
        Default is the image center.
    scale : float or array-like of two elements, optional
        The inter-pixel distance (dx, dy).
    dtype : np.dtype, optional
        The output data type.

    """
    array = distance2(shape, origin=origin, scale=scale, dtype=dtype)
    array[...] = array <= (diameter / 2) ** 2
    return array


def distance(shape, origin=None, scale=1, dtype=float, out=None):
    """
    Returns an array whose values are the distances to a given origin.

    Parameters
    ----------
    shape : tuple of integer
        dimensions of the output array. For a 2d array, the first integer
        is for the Y-axis and the second one for the X-axis.
    origin : array-like, optional
        The coordinates (x0, y0, ...) of the point from which the distance is
        calculated, assuming a zero-based coordinate indexing. Default value
        is the array center.
    scale : float or array-like, optional
        Inter-pixel distance (dx, dy, ...). If scale is a Quantity, its
        unit will be carried over to the returned distance array
    dtype : np.dtype, optional
        The output data type.

    Example
    -------
    nx, ny = 3, 3
    print(distance((ny,nx)))
    [[ 1.41421356  1.          1.41421356]
     [ 1.          0.          1.        ]
     [ 1.41421356  1.          1.41421356]]

    """
    shape = tointtuple(shape)
    dtype = np.dtype(dtype)
    unit = getattr(scale, '_unit', None)
    if out is None:
        out = np.empty(shape, dtype)
    ndim = out.ndim
    dtype = float_intrinsic_dtype(out.dtype)

    if ndim in (1, 2) and dtype != out.dtype:
        out_ = np.empty(shape, dtype)
    else:
        out_ = out
    if origin is None:
        origin = (np.array(shape[::-1]) - 1) / 2
    else:
        origin = np.ascontiguousarray(origin, dtype)
    if isscalarlike(scale):
        scale = np.resize(scale, out.ndim)
    scale = np.ascontiguousarray(scale, dtype)

    if ndim in (1, 2):
        fname = 'distance_{0}d_r{1}'.format(ndim, dtype.itemsize)
        func = getattr(flib.datautils, fname)
        if ndim == 1:
            func(out_, origin[0], scale[0])
        else:
            func(out_.T, origin, scale)
        if not isalias(out, out_):
            out[...] = out_
    else:
        _distance_slow(shape, origin, scale, dtype, out)
    return Map(out, copy=False, unit=unit)


def _distance_slow(shape, origin, scale, dtype, out=None):
    """
    Returns an array whose values are the distances to a given origin.

    This routine is written using np.meshgrid routine. It is slower
    than the Fortran-based `distance` routine, but can handle any number
    of dimensions.

    Refer to `pysimulators.distance` for full documentation.
    """
    index = []
    for n in shape:
        index.append(slice(0, n))
    d = np.ogrid[index]
    d = [
        (s * (a.astype(dtype) - o)) ** 2
        for a, o, s in zip(d, origin[::-1], scale[::-1])
    ]
    return np.sqrt(sum(d), out)


def distance2(shape, origin=None, scale=1, dtype=float, out=None):
    """
    Returns an array whose values are the squared distances to a given origin.

    Parameters
    ----------
    shape : tuple of integer
        dimensions of the output array. For a 2d array, the first integer
        is for the Y-axis and the second one for the X-axis.
    origin : array-like, optional
        The coordinates (x0, y0, ...) of the point from which the distance is
        calculated, assuming a zero-based coordinate indexing. Default value
        is the array center.
    scale : float or array-like, optional
        Inter-pixel distance (dx, dy, ...). If scale is a Quantity, its
        unit will be carried over to the returned distance array
    dtype : np.dtype, optional
        The output data type.

    Example
    -------
    nx, ny = 3, 3
    print(distance2((ny,nx)))
    [[ 2. 1.  2.]
     [ 1. 0.  1.]
     [ 2. 1.  2.]]

    """
    shape = tointtuple(shape)
    dtype = np.dtype(dtype)
    unit = getattr(scale, '_unit', None)
    if out is None:
        out = np.empty(shape, dtype)
    ndim = out.ndim
    dtype = float_intrinsic_dtype(out.dtype)

    if ndim in (1, 2) and dtype != out.dtype:
        out_ = np.empty(shape, dtype)
    else:
        out_ = out
    if origin is None:
        origin = (np.array(shape[::-1]) - 1) / 2
    else:
        origin = np.ascontiguousarray(origin, dtype)
    if isscalarlike(scale):
        scale = np.resize(scale, out.ndim)
    scale = np.ascontiguousarray(scale, dtype)

    if ndim in (1, 2):
        fname = 'distance2_{0}d_r{1}'.format(ndim, dtype.itemsize)
        func = getattr(flib.datautils, fname)
        if ndim == 1:
            func(out_, origin[0], scale[0])
        else:
            func(out_.T, origin, scale)
        if not isalias(out, out_):
            out[...] = out_
    else:
        _distance2_slow(shape, origin, scale, dtype, out)
    return Map(out, copy=False, unit=unit)


def _distance2_slow(shape, origin, scale, dtype, out=None):
    """
    Returns an array whose values are the distances to a given origin.

    This routine is written using np.meshgrid routine. It is slower
    than the Fortran-based `distance` routine, but can handle any number
    of dimensions.

    Refer to `pysimulators.distance` for full documentation.
    """
    if out is None:
        out = np.empty(shape, dtype)
    index = []
    for n in shape:
        index.append(slice(0, n))
    d = np.ogrid[index]
    d = [
        (s * (a.astype(dtype) - o)) ** 2
        for a, o, s in zip(d, origin[::-1], scale[::-1])
    ]
    if len(d) == 1:
        out[...] = d[0]
        return out
    np.sum(sum(d[:-1]), d[-1], out)
    return out


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


def gaussian(shape, sigma=None, fwhm=None, origin=None, scale=1, dtype=float):
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
    origin : array-like, optional
        Center (x0, y0, ...) of the gaussian, in pixel coordinates. By
        convention, the coordinates of the center of the pixel [0, 0]
        are (0, 0). Default is the image center.
    scale : float or array-like, optional
        Inter-pixel distance (dx, dy, ...).
    dtype : np.dtype, optional
        The output data type.

    """
    if sigma is None and fwhm is None:
        raise ValueError('The shape of the gaussian is not specified.')
    if sigma is None:
        sigma = fwhm / np.sqrt(8 * np.log(2))
    d = distance2(shape, origin=origin, scale=scale / (np.sqrt(2) * sigma))
    d = np.exp(-d)
    d /= np.sum(d)
    return d


def integrated_profile(input, bin=1, nbins=None, origin=None, scale=1):
    """
    Returns axisymmetric integrated profile of a 2d image.
    x, y = integrated_profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: array
        2d input array.
    bin: float, optional
        width of the profile bins.
    nbins: integer, optional
        number of profile bins.
    origin : array-like, optional
        Center (x0, y0) of the profile, in pixel coordinates. By convention,
        the coordinates of the center of the pixel [0, 0] are (0, 0).
        Default is the image center.
    scale : float, optional
        The inter-pixel distance.

    Returns
    -------
    x : array
        The strict upper boundary within which each integration is performed.
    y : array
        The integrated profile.

    """
    x, y, n = profile(
        input, bin=bin, nbins=nbins, histogram=True, origin=origin, scale=scale
    )
    y[~np.isfinite(y)] = 0
    y *= n * scale**2
    return x, np.cumsum(y)


def phasemask_fourquadrant(shape, phase=-1):
    array = Map.ones(shape, dtype=complex)
    array[0 : shape[0] // 2, shape[1] // 2 :] = phase
    array[shape[0] // 2 :, 0 : shape[1] // 2] = phase
    return array


def plot_tod(tod, mask=None, **kw):
    """Plot the signal timelines in a Tod and show masked samples.

    Plotting every detector timelines may be time consuming, so it is
    recommended to use this method on one or few detectors like this:
    >>> plot_tod(tod[idetector])

    """
    import matplotlib.pyplot as mp

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


def profile(input, bin=1, nbins=None, histogram=False, origin=None, scale=1):
    """
    Returns axisymmetric profile of a 2d image.
    x, y[, n] = profile(image, [origin, bin, nbins, histogram=True])

    Parameters
    ----------
    input: array
        2d input array
    bin: number, optional
        Width of the profile bins.
    nbins: integer, optional
        Number of profile bins.
    histogram: boolean, optional
        If set to True, return the histogram.
    origin : array-like, optional
        Center (x0, y0) of the profile, in pixel coordinates. By convention,
        the coordinates of the center of the pixel [0, 0] are (0, 0).
        Default is the image center.
    scale : float, optional
        The inter-pixel distance.

    """
    input = np.array(input, order='c', copy=False)
    dtype = float_intrinsic_dtype(input.dtype)
    input = np.array(input, dtype=dtype, copy=False)
    if origin is None:
        origin = (np.array(input.shape[::-1], dtype) - 1) / 2
    else:
        origin = np.ascontiguousarray(origin, dtype)
    bin_scaled = bin / scale

    if nbins is None:
        nbins = int(
            max(
                input.shape[0] - origin[1],
                origin[1],
                input.shape[1] - origin[0],
                origin[0],
            )
            / bin_scaled
        )

    fname = 'profile_axisymmetric_2d_r{}'.format(dtype.itemsize)
    func = getattr(flib.datautils, fname)
    x, y, n = func(input.T, origin, bin_scaled, nbins)
    x *= scale
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
    if array.ndim != 2:
        raise ValueError('The input array is not two-dimensional.')
    s = abs2(FFTOperator(array.shape, fftw_flag=fftw_flag)(array))
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
    array = np.asarray(array)
    if array.shape[0] != array.shape[1]:
        raise ValueError('The input array is not square.')
    psd = psd2(array, sampling_frequency=sampling_frequency, fftw_flag=fftw_flag)
    x, y = profile(psd, origin=array.shape // np.array(2) + 1)
    x *= sampling_frequency / array.shape[1]
    return x, y
