# Copyrights 2010-2013 Pierre Chanial
# All rights reserved

from __future__ import division
import copy
import numpy as np
import time
from pyoperators import DiagonalOperator, SymmetricBandToeplitzOperator, MPI
from pyoperators.memory import empty
from pyoperators.utils import (
    ifirst,
    isscalarlike,
    product,
    strelapsed,
    strenum,
    strnbytes,
)

from . import _flib as flib
from .datatypes import Map, Tod
from .geometry import convex_hull
from .instruments import Instrument, Imager
from .layouts import Layout
from .mpiutils import gather_fitsheader_if_needed
from .noises import (
    _fold_psd,
    _gaussian_psd_1f,
    _gaussian_sample,
    _logloginterp_psd,
    _psd2invntt,
    _unfold_psd,
)
from .operators import PointingMatrix, ProjectionOperator
from .pointings import Pointing
from .wcsutils import (
    RotationBoresightEquatorialOperator,
    barycenter_lonlat,
    combine_fitsheader,
    create_fitsheader,
    fitsheader2shape,
)

__all__ = ['Acquisition', 'AcquisitionImager', 'MaskPolicy']


class Acquisition(object):
    """
    The Acquisition class, which represents the instrument and
    pointing configurations.

    """

    def __init__(
        self,
        instrument,
        sampling,
        nprocs_instrument=None,
        nprocs_sampling=None,
        comm=None,
    ):
        """
        Parameters
        ----------
        instrument : Instrument
            The Instrument instance.
        sampling : LayoutTemporal
            The sampling information (pointing, etc.)

        """
        if not isinstance(instrument, Instrument):
            raise TypeError(
                "The instrument input has an invalid type '{}'.".format(
                    type(instrument).__name__
                )
            )
        #        if not isinstance(sampling, LayoutTemporal):
        #            raise TypeError(
        #                "The sampling input has an invalid type '{}'.".format(
        #                    type(instrument).__name__))

        if comm is None:
            comm = MPI.COMM_WORLD
        if nprocs_instrument is None and nprocs_sampling is None:
            nprocs_sampling = comm.size
        if nprocs_instrument is None:
            if nprocs_sampling < 1 or nprocs_sampling > comm.size:
                raise ValueError(
                    "Invalid value for nprocs_sampling '{0}'.".format(nprocs_sampling)
                )
            nprocs_instrument = comm.size // nprocs_sampling
        elif nprocs_sampling is None:
            if nprocs_instrument < 1 or nprocs_sampling > comm.size:
                raise ValueError(
                    "Invalid value for nprocs_instrument '{0}'.".format(
                        nprocs_instrument
                    )
                )
            nprocs_sampling = comm.size // nprocs_instrument
        if nprocs_instrument * nprocs_sampling != comm.size:
            raise ValueError('Invalid distribution of the acquisition.')

        commgrid = comm.Create_cart([nprocs_sampling, nprocs_instrument], reorder=True)

        comm_instrument = commgrid.Sub([False, True])
        comm_sampling = commgrid.Sub([True, False])
        self.instrument = instrument.scatter(comm_instrument)
        self.sampling = sampling.scatter(comm_sampling)
        self.comm = commgrid

    def __str__(self):
        return '{}\nSamplings: {}'.format(self.instrument, len(self.sampling))

    __repr__ = __str__

    def pack(self, x):
        return self.instrument.detector.pack(x)

    pack.__doc__ = Layout.pack.__doc__

    def unpack(self, x):
        return self.instrument.detector.unpack(x)

    unpack.__doc__ = Layout.unpack.__doc__

    def get_invntt_operator(
        self,
        psd=None,
        bandwidth=None,
        twosided=False,
        sigma=None,
        fknee=0,
        fslope=1,
        sampling_frequency=None,
        ncorr=None,
        fftw_flag='FFTW_MEASURE',
        nthreads=None,
    ):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        The input Power Spectrum Density can either be fully specified by using
        the 'bandwidth' and 'psd' keywords, or by providing the parameters of
        the gaussian distribution:
            psd = sigma**2 * (1 + (fknee/f)**fslope) / B
        where B is the sampling bandwidth equal to sampling_frequency / N.

        Parameters
        ----------
        psd : array-like
            The one-sided or two-sided power spectrum density
            [signal unit/sqrt Hz].
        bandwidth : float, optional
            The PSD frequency increment [Hz].
        twosided : boolean, optional
            Whether or not the input psd is one-sided (only positive
            frequencies) or two-sided (positive and negative frequencies).
        sigma : float
            Standard deviation of the white noise component.
        fknee : float
            The 1/f noise knee frequency [Hz].
        fslope : float
            The 1/f noise slope.
        sampling_frequency : float
            The sampling frequency [Hz].
        ncorr : int
            The correlation length of the time-time noise correlation matrix.
        fftw_flag : string, optional
            The flags FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT and
            FFTW_EXHAUSTIVE can be used to describe the increasing amount of
            effort spent during the planning stage to create the fastest
            possible transform. Usually, FFTW_MEASURE is a good compromise
            and is the default.
        nthreads : int, optional
            Tells how many threads to use when invoking FFTW or MKL. Default is
            the number of cores.

        """
        if (
            bandwidth is None
            and psd is not None
            or bandwidth is not None
            and psd is None
        ):
            raise ValueError('The bandwidth or the PSD is not specified.')
        if bandwidth is None and psd is None and sigma is None:
            raise ValueError('The noise model is not specified.')

        # handle the non-correlated case first
        if bandwidth is None and fknee == 0:
            return DiagonalOperator(1 / sigma**2, broadcast='rightward')

        if sampling_frequency is None:
            if not hasattr(self.sampling, 'sampling_frequency'):
                raise ValueError('The sampling frequency is not specified.')
            sampling_frequency = self.sampling.sampling_frequency

        nsamples_max = len(self.sampling)
        fftsize = 2
        while fftsize < nsamples_max:
            fftsize *= 2

        new_bandwidth = sampling_frequency / fftsize
        if bandwidth is not None and psd is not None:
            if twosided:
                psd = _fold_psd(psd)
            f = np.arange(fftsize // 2 + 1, dtype=float) * new_bandwidth
            p = _unfold_psd(_logloginterp_psd(f, bandwidth, psd))
        else:
            p = _gaussian_psd_1f(
                fftsize, sampling_frequency, sigma, fknee, fslope, twosided=True
            )
        p[..., 0] = p[..., 1]
        invntt = _psd2invntt(p, new_bandwidth, ncorr, fftw_flag=fftw_flag)

        shapein = (len(self.instrument), len(self.sampling))
        return SymmetricBandToeplitzOperator(
            shapein, invntt, fftw_flag=fftw_flag, nthreads=nthreads
        )

    def get_projection_operator(
        self, header, npixels_per_sample=0, method=None, units=None, derived_units=None
    ):
        if method == 'nearest':
            npixels_per_sample = 1
        time0 = time.time()
        info = []
        matrix = self.get_projection_matrix(
            header, npixels_per_sample=npixels_per_sample, method=method
        )
        npps_min = matrix.header['min_npixels_per_sample']

        if npps_min == 0:
            info += ['Warning, all detectors fall outside the map.']

        elif npixels_per_sample < npps_min:
            if npixels_per_sample > 0:
                info += [
                    "Warning, the value 'npixels_per_sample' is not large"
                    " enough. Set it to '{0}' instead of '{1}' to avoid r"
                    "ecomputation of the pointing matrix.".format(
                        npps_min, npixels_per_sample
                    )
                ]
            else:
                info += [
                    "Set the keyword 'npixels_per_sample' to '{0}' to avo"
                    "id recomputation of the pointing matrix.".format(npps_min)
                ]
            npixels_per_sample = npps_min
            matrix = self.get_projection_matrix(
                header, npixels_per_sample=npixels_per_sample, method=method
            )

        elif npixels_per_sample > npps_min:
            info += [
                "Warning, the value 'npixels_per_sample' is too large. Se"
                "t it to '{0}' instead of '{1}' for better memory perform"
                "ance.".format(npps_min, npixels_per_sample)
            ]

        if matrix.header['outside']:
            info += ['Warning, some detectors fall outside the map.']

        info.insert(0, strnbytes(matrix.nbytes))
        print(
            strelapsed(time0, 'Computing the projector')
            + ' ({0})'.format(' '.join(info))
        )

        return ProjectionOperator(
            matrix,
            attrin={'_header': header},
            classin=Map,
            classout=Tod,
            units=units,
            derived_units=derived_units,
        )

    def get_noise(
        self,
        psd=None,
        bandwidth=None,
        twosided=False,
        sigma=None,
        fknee=0,
        fslope=1,
        sampling_frequency=None,
        out=None,
    ):
        """
        Return the noise realization following a given PSD.

        The input Power Spectrum Density can either be fully specified by using
        the 'bandwidth' and 'psd' keywords, or by providing the parameters of
        the gaussian distribution:
            psd = sigma**2 * (1 + (fknee/f)**fslope) / B
        where B is equal to sampling_frequency / N.

        Parameters
        ----------
        psd : array-like, optional
            The one-sided or two-sided Power Spectrum Density,
            [signal unit**2/Hz].
        bandwidth : float, optional
            The PSD frequency increment [Hz].
        twosided : boolean, optional
            Whether or not the output psd is one-sided (only positive
            frequencies) or two-sided (positive and negative frequencies).
        sigma : float, optional
            Standard deviation of the white noise component.
        fknee : float, optional
            The 1/f noise knee frequency [Hz].
        fslope : float, optional
            The 1/f noise slope.
        sampling_frequency : float, optional
            The sampling frequency of the output timeline [Hz]. By default,
            it is taken from the acquisition's sampling attribute.
        out : ndarray, optional
            Placeholder for the output noise.

        """
        if (
            bandwidth is None
            and psd is not None
            or bandwidth is not None
            and psd is None
        ):
            raise ValueError('The bandwidth or the PSD is not specified.')
        if bandwidth is None and psd is None and sigma is None:
            raise ValueError('The noise model is not specified.')

        shape = (len(self.instrument), len(self.sampling))

        # handle non-correlated case first
        if bandwidth is None and fknee == 0:
            noise = np.random.randn(*shape)
            if out is None:
                out = noise
            else:
                out[...] = noise
            np.multiply(out.T, sigma, out.T)
            return out

        if sampling_frequency is None:
            sampling_frequency = 1 / self.sampling.sampling_period

        if out is None:
            out = empty(shape)

        # fold two-sided input PSD
        if bandwidth is not None and psd is not None:
            if twosided:
                psd = _fold_psd(psd)
                twosided = False
        else:
            twosided = True

        n = len(self.sampling)
        if bandwidth is None and psd is None:
            p = _gaussian_psd_1f(
                n, sampling_frequency, sigma, fknee, fslope, twosided=twosided
            )
        else:
            # log-log interpolation of one-sided PSD
            f = np.arange(n // 2 + 1, dtype=float) * (sampling_frequency / n)
            p = _logloginterp_psd(f, bandwidth, psd)

        # looping over the detectors
        for out_ in out:
            out_[...] = _gaussian_sample(n, sampling_frequency, p, twosided=twosided)

        return out

    def get_projection_matrix(
        self, header, start, stop, npixels_per_sample=0, method=None
    ):
        raise NotImplementedError()

    @classmethod
    def create_scan(
        cls,
        center,
        length,
        step,
        sampling_period,
        speed,
        acceleration,
        nlegs=3,
        angle=0,
        instrument_angle=45,
        cross_scan=True,
        dtype=None,
    ):
        """
        Return a sky scan as a Pointing array.

        Parameters
        ----------
        center : tuple
            (Right Ascension, declination) of the scan center.
        length : float
            Length of the scan lines, in arcseconds.
        step : float
            Separation between scan legs, in arcseconds.
        sampling_period : float
            Duration between two pointings.
        speed : float
            Scan speed, in arcsec/s.
        acceleration : float
            Acceleration, in arcsec/s^2.
        nlegs : integer
            Number of scan legs.
        angle : float
            Angle between the scan line direction and the North minus
            90 degrees, in degrees.
        instrument_angle : float
            Angle between the scan line direction and the instrument second
            axis, in degrees.
        cross_scan : boolean
            If true, a cross-scan is appended to the pointings.
        dtype : structured dtype
            Pointing data type.

        """
        scan = _create_scan(
            center,
            length,
            step,
            sampling_period,
            speed,
            acceleration,
            nlegs,
            angle,
            dtype,
        )
        if cross_scan:
            cross = _create_scan(
                center,
                length,
                step,
                sampling_period,
                speed,
                acceleration,
                nlegs,
                angle + 90,
                dtype,
            )
            cross.time += scan.time[-1] + sampling_period
            scan, scan.header = (np.hstack([scan, cross]).view(Pointing), scan.header)

        scan.pa = angle + instrument_angle
        scan.header.update('HIERARCH instrument_angle', instrument_angle)
        return scan

    def plot(self, map=None, header=None, new_figure=True, percentile=0.01, **keywords):
        """
        map : ndarray of dim 2
            The optional map to be displayed as background.
        header : pyfits.Header
            The optional map's FITS header.
        new_figure : boolean
            If true, plot the scan in a new window.
        percentile : float, tuple of two floats
            As a float, percentile of values to be discarded, otherwise,
            percentile of the minimum and maximum values to be displayed.

        """
        if header is None:
            header = getattr(map, 'header', None)
            if header is None:
                header = self.sampling.get_map_header(naxis=1)
        annim = self.sampling.plot(
            map=map,
            header=header,
            new_figure=new_figure,
            percentile=percentile,
            **keywords,
        )
        return annim


class AcquisitionImager(Acquisition):
    """
    The AcquisitionImager class, which represents the setups of an imager
    instrument and the pointing.

    """

    def __init__(self, instrument, sampling):
        """
        Parameters
        ----------
        instrument : Imager
            The Imager instance.
        sampling : array-like of shape (n,3) or structured array of shape (n,),
                   or sequence of
            The pointing directions.

        """
        if not isinstance(instrument, Imager):
            raise TypeError(
                "The instrument has an invalid type '{}'.".format(
                    type(instrument).__name__
                )
            )
        Acquisition.__init__(self, instrument, sampling)
        self.object2world = RotationBoresightEquatorialOperator(self.sampling)

    def get_map_header(self, resolution=None):
        """
        Return the FITS header of the smallest map that encompasses the config-
        uration pointings, by taking into account the instrument geometry.

        Parameters
        ----------
        resolution : float
            Sky pixel increment, in arc seconds. The default value is the
            Instrument default_resolution.

        Returns
        -------
        header : astropy.io.Header
            The resulting FITS header.

        """
        if resolution is None:
            resolution = self.instrument.default_resolution
        if hasattr(self.instrument.detector, 'vertex'):
            coords = self.instrument.detector.vertex
        else:
            coords = self.instrument.detector.center
        coords = convex_hull(coords)
        self.instrument.image2object(coords, out=coords)

        valid = ~self.sampling['masked']
        if not np.any(valid):
            raise ValueError(
                'The FITS header cannot be inferred: there is no ' 'valid pointing.'
            )
        sampling = self.sampling[valid]

        # get a dummy header, with correct cd and crval
        ra0, dec0 = barycenter_lonlat(sampling.ra, sampling.dec)
        header = create_fitsheader(
            (1, 1), cdelt=resolution / 3600, crval=(ra0, dec0), crpix=(1, 1)
        )

        # compute coordinate boundaries according to the header's astrometry
        xmin, ymin, xmax, ymax = self._object2xy_minmax(
            coords, sampling, str(header).replace('\n', '')
        )
        ixmin = int(np.round(xmin))
        ixmax = int(np.round(xmax))
        iymin = int(np.round(ymin))
        iymax = int(np.round(ymax))
        nx = ixmax - ixmin + 1
        ny = iymax - iymin + 1

        # move the reference pixel (not the reference value!)
        header = create_fitsheader(
            (nx, ny),
            cdelt=resolution / 3600,
            crval=(ra0, dec0),
            crpix=(-ixmin + 2, -iymin + 2),
        )

        # gather and combine the FITS headers
        headers = self.sky.comm.allgather(header)
        return combine_fitsheader(headers)

    def get_projection_matrix(self, header, npixels_per_sample=0, method=None):
        """
        Return the pointing matrix for a given set of pointings.

        Parameters
        ----------
        header : pyfits.Header
            The map FITS header
        npixels_per_sample : int
            Maximum number of sky pixels intercepted by a detector.
            By setting 0 (the default), the actual value will be determined
            automatically.
        method : string
            'sharp' : the intersection of the sky pixels and the detectors
                      is computed assuming that the transmission outside
                      the detector is zero and one otherwise (sharp edge
                      geometry)
            'nearest' : the value of the sky pixel closest to the detector
                        center is taken as the sample value, assuming
                        surface brightness conservation.

        """
        if method is None:
            if hasattr(self.instrument.detector, 'vertex'):
                method = 'sharp'
            else:
                method = 'nearest'
        method = method.lower()
        choices = ('nearest', 'sharp')
        if method not in choices:
            raise ValueError(
                "Invalid method '" + method + "'. Expected values"
                " are " + strenum(choices) + '.'
            )

        comm = self.sky.comm
        header = gather_fitsheader_if_needed(header, comm=comm)
        shape_input = fitsheader2shape(header)
        if product(shape_input) > np.iinfo(np.int32).max:
            raise RuntimeError(
                'The map is too large: pixel indices cannot be '
                'stored using 32 bits: {0}>{1}'.format(
                    product(shape_input), np.iinfo(np.int32).max
                )
            )

        if method == 'nearest':
            npixels_per_sample = 1

        # allocate memory for the pointing matrix
        ndetectors = len(self.instrument)
        nsamples = len(self.sampling)
        shape = (ndetectors, nsamples, npixels_per_sample)
        matrix = PointingMatrix.empty(shape, shape_input, verbose=False)

        # compute the pointing matrix
        if method == 'sharp':
            coords = self.instrument.image2object(self.instrument.detector.vertex)
            new_npps, outside = self._object2pmatrix_sharp_edges(
                coords, self.sampling, header, matrix, npixels_per_sample
            )
        elif method == 'nearest':
            coords = self.instrument.image2object(self.instrument.detector.center)
            new_npps = 1
            outside = self._object2pmatrix_nearest_neighbour(
                coords, self.sampling, header, matrix
            )
        else:
            raise NotImplementedError()

        matrix.header['method'] = method
        matrix.header['outside'] = bool(outside)
        matrix.header['HIERARCH min_npixels_per_sample'] = new_npps

        return matrix

    def plot(
        self,
        map=None,
        header=None,
        instrument=True,
        new_figure=True,
        percentile=0.01,
        **keywords,
    ):
        """
        map : ndarray of dim 2
            The optional map to be displayed as background.
        header : pyfits.Header
            The optional map's FITS header.
        instrument : boolean
            If true, plot the instrument's footprint on the sky.
        new_figure : boolean
            If true, plot the scan in a new window.
        percentile : float, tuple of two floats
            As a float, percentile of values to be discarded, otherwise,
            percentile of the minimum and maximum values to be displayed.

        """
        image = Acquisition.plot(
            self,
            map=map,
            header=header,
            new_figure=new_figure,
            percentile=percentile,
            **keywords,
        )
        if not instrument:
            return image

        if hasattr(self.sampling, 'masked'):
            p = self.sampling[ifirst(self.sampling.masked, False)]
        else:
            p = self.sampling[0]
        t = RotationBoresightEquatorialOperator(p) * self.instrument.image2object
        f = lambda x: image.projection.topixel(t(x))
        self.instrument.plot(transform=f, autoscale=False)
        return image

    @staticmethod
    def _object2ad(coords, pointing):
        """
        Convert coordinates in the instrument frame into celestial coordinates,
        assuming a pointing direction and a position angle.

        The instrument frame is the frame used for the 'center' or 'corner'
        coordinates, which are fields of the 'detector' attribute.

        Parameters
        ----------
        coords : float array (last dimension is 2)
            Coordinates in the instrument frame in arc seconds.
        pointing : array with flexible dtype including 'ra', 'dec' and 'pa'
            Direction corresponding to (0,0) in the local frame, in degrees.

        Returns
        -------
        coords_converted : float array (last dimension is 2, for Ra and Dec)
            Converted coordinates, in degrees.

        Notes
        -----
        The routine is not accurate at the poles.

        """
        coords = np.array(coords, float, order='c', copy=False)
        shape = coords.shape
        coords = coords.reshape((-1, 2))
        new_shape = (pointing.size,) + coords.shape
        result = np.empty(new_shape, float)
        for r, ra, dec, pa in zip(
            result, pointing['ra'].flat, pointing['dec'].flat, pointing['pa'].flat
        ):
            flib.wcsutils.object2ad(coords.T, r.T, ra, dec, pa)
        result = result.reshape(pointing.shape + shape)
        return result

    @staticmethod
    def _object2xy_minmax(coords, pointing, header):
        """
        Return the minimum and maximum sky pixel coordinate values given a set
        of coordinates specified in the instrument frame.

        """
        coords = np.array(coords, float, order='c', copy=False)
        xmin, ymin, xmax, ymax, status = flib.wcsutils.object2xy_minmax(
            coords.reshape((-1, 2)).T,
            pointing['ra'].ravel(),
            pointing['dec'].ravel(),
            pointing['pa'].ravel(),
            str(header).replace('\n', ''),
        )
        if status != 0:
            raise RuntimeError()
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _object2pmatrix_sharp_edges(
        coords, pointing, header, pmatrix, npixels_per_sample
    ):
        """
        Return the sparse pointing matrix whose values are intersections
        between detectors and map pixels.

        """
        coords = coords.reshape((-1,) + coords.shape[-2:])
        ra = pointing['ra'].ravel()
        dec = pointing['dec'].ravel()
        pa = pointing['pa'].ravel()
        masked = pointing['masked'].view(np.int8).ravel()
        if pmatrix.size == 0:
            # f2py doesn't accept zero-sized opaque arguments
            pmatrix = np.empty(1, np.int64)
        else:
            pmatrix = pmatrix.ravel().view(np.int64)
        header = str(header).replace('\n', '')

        new_npps, out, status = flib.wcsutils.object2pmatrix_sharp_edges(
            coords.T, ra, dec, pa, masked, header, pmatrix, npixels_per_sample
        )
        if status != 0:
            raise RuntimeError()

        return new_npps, out

    @staticmethod
    def _object2pmatrix_nearest_neighbour(coords, pointing, header, pmatrix):
        """
        Return the sparse pointing matrix whose values are intersection between
        detector centers and map pixels.

        """
        coords = coords.reshape((-1, 2))
        area = np.ones(coords.shape[0])
        ra = pointing['ra'].ravel()
        dec = pointing['dec'].ravel()
        pa = pointing['pa'].ravel()
        masked = pointing['masked'].view(np.int8).ravel()
        pmatrix = pmatrix.ravel().view(np.int64)
        header = str(header).replace('\n', '')
        out, status = flib.wcsutils.object2pmatrix_nearest_neighbour(
            coords.T, area, ra, dec, pa, masked, header, pmatrix
        )
        if status != 0:
            raise RuntimeError()

        return out


def _create_scan(
    center, length, step, sampling_period, speed, acceleration, nlegs, angle, dtype
):
    """
    compute the pointing timeline of the instrument reference point
    from the description of a scan map
    Authors: R. Gastaud, P. Chanial

    """
    ra0, dec0 = center

    length = float(length)
    if length <= 0:
        raise ValueError('Input length must be strictly positive.')

    step = float(step)
    if step <= 0:
        raise ValueError('Input step must be strictly positive.')

    sampling_period = float(sampling_period)
    if sampling_period <= 0:
        raise ValueError('Input sampling_period must be strictly positive.')

    speed = float(speed)
    if speed <= 0:
        raise ValueError('Input speed must be strictly positive.')

    acceleration = float(acceleration)
    if acceleration <= 0:
        raise ValueError('Input acceleration must be strictly positive.')

    nlegs = int(nlegs)
    if nlegs <= 0:
        raise ValueError('Input nlegs must be strictly positive.')

    # compute the different times and the total number of points
    # acceleration time at the beginning of a leg, and deceleration time
    # at the end
    # The time needed to turn around is 2 * (extra_time1 + extra_time2)
    extra_time1 = speed / acceleration
    # corresponding length
    extralength = 0.5 * acceleration * extra_time1 * extra_time1
    # Time needed to go from a scan line to the next
    extra_time2 = np.sqrt(step / acceleration)

    # Time needed to go along the scanline at constant speed
    line_time = length / speed
    # Total time for a scanline
    full_line_time = extra_time1 + line_time + extra_time1 + 2 * extra_time2
    # Total duration of the observation
    total_time = full_line_time * nlegs - 2 * extra_time2

    # Number of samples
    nsamples = int(np.ceil(total_time / sampling_period))

    # initialization
    time = np.zeros(nsamples)
    latitude = np.zeros(nsamples)
    longitude = np.zeros(nsamples)
    infos = np.zeros(nsamples, dtype=int)
    line_counters = np.zeros(nsamples, dtype=int)

    # Start of computations, alpha and delta are the longitude and
    # latitide in arc seconds in the referential of the map.
    sign = 1
    delta = -extralength - length / 2
    alpha = -step * (nlegs - 1) / 2
    alpha0 = alpha
    line_counter = 0
    working_time = 0.0

    for i in range(nsamples):
        info = 0

        # check if new line
        if working_time > full_line_time:
            working_time = working_time - full_line_time
            sign = -sign
            line_counter = line_counter + 1
            alpha = -step * (nlegs - 1) / 2 + line_counter * step
            alpha0 = alpha

        # acceleration at the beginning of a scan line to go from 0 to the
        # speed.
        if working_time < extra_time1:
            delta = (
                -sign * (extralength + length / 2)
                + sign * 0.5 * acceleration * working_time * working_time
            )
            info = Pointing.TURNAROUND

        # constant speed
        if working_time >= extra_time1 and working_time < extra_time1 + line_time:
            delta = sign * (-length / 2 + (working_time - extra_time1) * speed)
            info = Pointing.INSCAN

        # Deceleration at then end of the scanline to stop
        if (
            working_time >= extra_time1 + line_time
            and working_time < extra_time1 + line_time + extra_time1
        ):
            dt = working_time - extra_time1 - line_time
            delta = sign * (length / 2 + speed * dt - 0.5 * acceleration * dt**2)
            info = Pointing.TURNAROUND

        # Acceleration to go toward the next scan line
        if (
            working_time >= 2 * extra_time1 + line_time
            and working_time < 2 * extra_time1 + line_time + extra_time2
        ):
            dt = working_time - 2 * extra_time1 - line_time
            alpha = alpha0 + 0.5 * acceleration * dt**2
            info = Pointing.TURNAROUND

        # Deceleration to stop at the next scan line
        if (
            working_time >= 2 * extra_time1 + line_time + extra_time2
            and working_time < full_line_time
        ):
            dt = working_time - 2 * extra_time1 - line_time - extra_time2
            alpha = (
                (alpha0 + step / 2)
                + acceleration * extra_time2 * dt
                - 0.5 * acceleration * dt**2
            )
            info = Pointing.TURNAROUND

        time[i] = i * sampling_period
        infos[i] = info
        latitude[i] = delta
        longitude[i] = alpha
        line_counters[i] = line_counter
        working_time = working_time + sampling_period

    # Convert the longitude and latitude *expressed in degrees) to ra and dec
    ra, dec = _change_coord(ra0, dec0, angle - 90, longitude / 3600, latitude / 3600)

    scan = Pointing((ra, dec, 0.0), time, infos, dtype=dtype)
    header = create_fitsheader(nsamples)
    header.update('ra', ra0)
    header.update('dec', dec0)
    header.update('HIERARCH scan_angle', angle)
    header.update('HIERARCH scan_length', length)
    header.update('HIERARCH scan_nlegs', nlegs)
    header.update('HIERARCH scan_step', step)
    header.update('HIERARCH scan_speed', speed)
    header.update('HIERARCH scan_acceleration', acceleration)
    scan.header = header
    return scan


def _change_coord(ra0, dec0, pa0, lon, lat):
    """
    Transforms the longitude and latitude coordinates expressed in the
    native coordinate system attached to a map into right ascension and
    declination.
    Author: R. Gastaud

    """
    from numpy import arcsin, arctan2, cos, deg2rad, mod, sin, rad2deg

    # Arguments in radian
    lambd = deg2rad(lon)
    beta = deg2rad(lat)
    alpha = deg2rad(ra0)
    delta = deg2rad(dec0)
    eps = deg2rad(pa0)

    # Cartesian coordinates from longitude and latitude
    z4 = sin(beta)
    y4 = sin(lambd) * cos(beta)
    x4 = cos(lambd) * cos(beta)

    # rotation about the x4 axe of -eps
    x3 = x4
    y3 = y4 * cos(eps) + z4 * sin(eps)
    z3 = -y4 * sin(eps) + z4 * cos(eps)

    # rotation about the axis Oy2, angle delta
    x2 = x3 * cos(delta) - z3 * sin(delta)
    y2 = y3
    z2 = x3 * sin(delta) + z3 * cos(delta)

    # rotation about the axis Oz1, angle alpha
    x1 = x2 * cos(alpha) - y2 * sin(alpha)
    y1 = x2 * sin(alpha) + y2 * cos(alpha)
    z1 = z2

    # Compute angles from cartesian coordinates
    # it is the only place where we can get nan with arcsinus
    dec = rad2deg(arcsin(np.clip(z1, -1.0, 1.0)))
    ra = mod(rad2deg(arctan2(y1, x1)), 360.0)

    return ra, dec


class MaskPolicy(object):
    KEEP = 0
    MASK = 1
    REMOVE = 2

    def __init__(self, flags, values, description=None):
        self._description = description
        if isscalarlike(flags):
            if isinstance(flags, str):
                flags = flags.split(',')
            else:
                flags = (flags,)
        if isscalarlike(values):
            if isinstance(values, str):
                values = values.split(',')
            else:
                values = (values,)
        if len(flags) != len(values):
            raise ValueError(
                'The number of policy flags is different from the'
                ' number of policies.'
            )

        self._policy = []
        for flag, value in zip(flags, values):
            if flag[0] == '_':
                raise ValueError(
                    'A policy flag should not start with an under' 'score.'
                )
            value = value.strip().lower()
            choices = ('keep', 'mask', 'remove')
            if value not in choices:
                raise KeyError(
                    'Invalid policy ' + flag + "='" + value + "'. E"
                    "xpected ones are " + strenum(choices) + '.'
                )
            self._policy.append({flag: value})
            setattr(self, flag, value)
        self._policy = tuple(self._policy)

    def __array__(self, dtype=int):
        conversion = {'keep': self.KEEP, 'mask': self.MASK, 'remove': self.REMOVE}
        return np.array(
            [conversion[policy.values()[0]] for policy in self._policy], dtype=dtype
        )

    def __str__(self):
        str = self._description + ': ' if self._description is not None else ''
        str_ = []
        for policy in self._policy:
            str_.append(policy.values()[0] + " '" + policy.keys()[0] + "'")
        str += ', '.join(str_)
        return str
