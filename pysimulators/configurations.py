# Copyrights 2010-2013 Pierre Chanial
# All rights reserved

from __future__ import division

import numpy as np
import time
import types
from pyoperators.utils import (ifirst, isscalar, strelapsed, strenum,
                               strnbytes, strplural)
from pyoperators.utils.mpi import MPI

from .instruments import Instrument
from .pointings import POINTING_DTYPE, Pointing
from .wcsutils import create_fitsheader

__all__ = ['Configuration', 'MaskPolicy']


class Configuration(object):
    """
    The QubicConfiguration class, which represents the instrument and
    pointing setups.

    """
    def __init__(self, instrument, pointing, block_id=None, selection=None):
        """
        Parameters
        ----------
        instrument : str or Instrument
            Instrument name, or an Instrument instance.
        pointing : array-like of shape (n,3) or structured array of shape (n,),
                   or sequence of
            The pointing directions.
        block_id : string or sequence of, optional
           The pointing block identifier.
        selection : integer or sequence of, optional
           The indices of the pointing sequence to be selected to construct
           the pointing configuration.

        """
        if not isinstance(instrument, Instrument):
            raise TypeError("The instrument has an invalid type '{}'.".format(
                            type(instrument).__name__))
        if not isinstance(pointing, (list, tuple)):
            pointing = (pointing,)
        elif isinstance(pointing, types.GeneratorType):
            pointing = tuple(pointing)
        pointing = [np.asanyarray(p) for p in pointing]
        if any(type(p) is not type(pointing[0]) for p in pointing):
            raise TypeError('The input pointings have different types.')
        if any(p.dtype.kind != pointing[0].dtype.kind for p in pointing):
            raise TypeError('The input pointings have different dtype kinds.')
        if pointing[0].dtype.kind == 'V':
            pointing = [np.array(p, copy=False, ndmin=1, subok=True)
                        for p in pointing]
        else:
            if len(pointing) == 3 and all(p.ndim == 0 for p in pointing):
                pointing = [np.hstack(pointing)]
            if any(p.ndim not in (1, 2) or p.shape[-1] != 3 for p in pointing):
                raise ValueError('Invalid pointing dimensions.')
            if len(pointing) > 1 and all(p.ndim == 1 for p in pointing):
                pointing = [np.vstack(pointing)]
            pointing = [np.array(p, copy=False, ndmin=2, subok=True)
                        for p in pointing]
        if selection is None:
            selection = tuple(range(len(pointing)))
        pointing = [pointing[i] for i in selection]
        if block_id is not None:
            block_id = [block_id[i] for i in selection]
        if not isinstance(block_id, (list, tuple, types.NoneType)):
            block_id = (block_id,)
            if any(not isinstance(i, str) for i in block_id):
                raise TypeError('The block id is not a string.')

        self.instrument = instrument
        self.pointing = np.concatenate(pointing).view(type(pointing[0]))
        self.block = self._get_block(pointing, block_id)

    def __str__(self):
        return 'Pointings:\n    {} in {}\n\n'.format(
            self.get_nsamples(), strplural(len(self.block), 'block')) + \
            str(self.instrument)

    __repr__ = __str__

    def get_ndetectors(self):
        return self.instrument.get_ndetectors()
    get_ndetectors.__doc__ = Instrument.get_ndetectors.__doc__

    def get_filter_uncorrelated(self):
        """
        Return the invNtt for uncorrelated detectors.

        """
        raise NotImplementedError()

    def get_map_header(self, resolution=None, downsampling=False):
        """
        Return the FITS header of the smallest map that encompasses
        the observation, by taking into account the instrument geometry.

        Parameters
        ----------
        resolution : float
            Sky pixel increment, in arc seconds. The default value is set
            in Configuration's __init__ method through the default_resolution
            keyword.

        Returns
        -------
        header : pyfits.Header
            The resulting FITS header.

        """
        return self.instrument.get_map_header(self.pointing,
                                              resolution=resolution)

    def get_pointing_matrix(self, header, npixels_per_sample=0, method=None,
                            downsampling=False, section=None,
                            comm=MPI.COMM_WORLD, **keywords):
        """
        Return the pointing matrix for the configuration.

        If the configuration has several blocks, as many pointing matrices
        are returned in a list.

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
        downsampling : boolean
            If True, return a pointing matrix downsampled by the instrument
            fine sampling factor. Otherwise return a pointing matrix sampled
            at the instrument fine sampling factor.
        section : slice
            If specified, return the pointing matrix for a specific slice.
        comm : mpi4py.MPI.Comm
            Map communicator, only used if the input FITS header is local.

        """
        # if section is None, return as many pointing matrices as slices
        if section is None:
            if self.block is not None:
                time0 = time.time()
                pmatrix = [self.get_pointing_matrix(header, npixels_per_sample,
                           method, downsampling, section=s, comm=comm,
                           **keywords) for s in self.block]
                info = [strnbytes(sum(p.nbytes for p in pmatrix))]
                if all(p.shape[-1] == 0 for p in pmatrix):
                    info += 'warning, all detectors fall outside the map'
                else:
                    n = max(p.info['npixels_per_sample_min'] for p in pmatrix)
                    if npixels_per_sample == 0 or npixels_per_sample != n:
                        info += ["set keyword 'npixels_per_sample' to {0} for "
                                 "better performances".format(n)]
                    outside = any(p.info['outside'] for p in pmatrix
                                  if 'outside' in p.info)
                    if outside:
                        info += ['warning, some detectors fall outside the '
                                 'map']
                print(strelapsed(time0, 'Computing the projector') + ' ({0})'.
                      format(', '.join(info)))
                if len(pmatrix) == 1:
                    pmatrix = pmatrix[0]
                return pmatrix
            pointing = self.pointing
        else:
            # otherwise, restrict the pointing to the input section
            if not isinstance(section, slice):
                section = slice(section.start, section.stop)
            pointing = self.pointing[section]
        result = self.instrument.get_pointing_matrix(
            pointing, header, npixels_per_sample, method, downsampling,
            comm=comm, **keywords)
        result.info.update(keywords)
        return result

    def get_nsamples(self):
        """
        Return the number of valid pointings for each block
        They are those for which self.pointing.removed is False.

        """
        return tuple([int(np.sum(~self.pointing[s.start:s.stop].removed))
                      for s in self.block])

    def pack(self, x, masked=False):
        return self.instrument.pack(x, masked=masked)
    pack.__doc__ = Instrument.pack.__doc__

    def unpack(self, x, masked=False):
        return self.instrument.unpack(x, masked=masked)
    unpack.__doc__ = Instrument.unpack.__doc__

    @classmethod
    def create_scan(cls, center, length, step, sampling_period, speed,
                    acceleration, nlegs=3, angle=0, instrument_angle=45,
                    cross_scan=True, dtype=POINTING_DTYPE):
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
        dtype : flexible dtype
            Pointing data type.

        """
        scan = _create_scan(center, length, step, sampling_period, speed,
                            acceleration, nlegs, angle, dtype)
        if cross_scan:
            cross = _create_scan(center, length, step, sampling_period, speed,
                                 acceleration, nlegs, angle + 90, dtype)
            cross.time += scan.time[-1] + sampling_period
            scan, scan.header = (np.hstack([scan, cross]).view(Pointing),
                                 scan.header)

        scan.pa = angle + instrument_angle
        scan.header.update('HIERARCH instrument_angle', instrument_angle)
        return scan

    def plot(self, map=None, header=None, instrument=True, new_figure=True,
             percentile=0.01, **keywords):
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
        if header is None:
            header = getattr(map, 'header', None)
            if header is None:
                header = self.pointing.get_map_header(naxis=1)
        annim = self.pointing[self.block[0].start:self.block[0].stop].plot(
            map=map, header=header, new_figure=new_figure,
            percentile=percentile, **keywords)

        valid = ~self.pointing.removed & ~self.pointing.masked
        if np.max(valid) == 0:
            return

        for s in self.block[1:]:
            self.pointing[s.start:s.stop].plot(header=header, new_figure=False,
                                               **keywords)
        if instrument:
            p = self.pointing[ifirst(valid, True)]
            t = self.instrument.toobject
            f = lambda x: self.instrument._instrument2xy(t(x), p, header) + 1
            self.instrument.plot(f, autoscale=False)

        return annim

    @staticmethod
    def _get_block(pointing, block_id):
        npointings = [p.shape[0] for p in pointing]
        start = np.concatenate([[0], np.cumsum(npointings)[:-1]])
        stop = np.cumsum(npointings)
        block = np.recarray(len(pointing),
                            dtype=[('start', int), ('stop', int),
                                   ('n', int), ('id', 'S29')])
        block.n = npointings
        block.start = start
        block.stop = stop
        block.identifier = block_id if block_id is not None else ''
        return block


def _create_scan(center, length, step, sampling_period, speed, acceleration,
                 nlegs, angle, dtype):
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
    time          = np.zeros(nsamples)
    latitude      = np.zeros(nsamples)
    longitude     = np.zeros(nsamples)
    infos         = np.zeros(nsamples, dtype=int)
    line_counters = np.zeros(nsamples, dtype=int)

    # Start of computations, alpha and delta are the longitude and
    # latitide in arc seconds in the referential of the map.
    sign = 1
    delta = -extralength - length/2
    alpha = -step * (nlegs-1)/2
    alpha0 = alpha
    line_counter = 0
    working_time = 0.

    for i in range(nsamples):
        info = 0

        # check if new line
        if working_time > full_line_time:
            working_time = working_time - full_line_time
            sign = -sign
            line_counter = line_counter + 1
            alpha = -step * (nlegs-1) / 2 + line_counter * step
            alpha0 = alpha

        # acceleration at the beginning of a scan line to go from 0 to the
        # speed.
        if working_time < extra_time1:
            delta = -sign * (extralength + length / 2) + sign * 0.5 * \
                    acceleration * working_time * working_time
            info = Pointing.TURNAROUND

        # constant speed
        if working_time >= extra_time1 and \
           working_time < extra_time1 + line_time:
            delta = sign * (-length / 2 + (working_time - extra_time1) * speed)
            info = Pointing.INSCAN

        # Deceleration at then end of the scanline to stop
        if working_time >= extra_time1 + line_time and \
           working_time < extra_time1 + line_time + extra_time1:
            dt = working_time - extra_time1 - line_time
            delta = sign * (length / 2 + speed * dt -
                    0.5 * acceleration * dt**2)
            info = Pointing.TURNAROUND

        # Acceleration to go toward the next scan line
        if working_time >= 2 * extra_time1 + line_time and \
           working_time < 2 * extra_time1 + line_time + extra_time2:
            dt = working_time - 2 * extra_time1 - line_time
            alpha = alpha0 + 0.5 * acceleration * dt**2
            info = Pointing.TURNAROUND

        # Deceleration to stop at the next scan line
        if working_time >= 2 * extra_time1 + line_time + extra_time2 and \
           working_time < full_line_time:
            dt = working_time - 2 * extra_time1 - line_time - extra_time2
            alpha = (alpha0 + step / 2) + acceleration * extra_time2 * dt - \
                    0.5 * acceleration * dt**2
            info = Pointing.TURNAROUND

        time[i] = i * sampling_period
        infos[i] = info
        latitude[i] = delta
        longitude[i] = alpha
        line_counters[i] = line_counter
        working_time = working_time + sampling_period

    # Convert the longitude and latitude *expressed in degrees) to ra and dec
    ra, dec = _change_coord(ra0, dec0, angle - 90, longitude / 3600,
                            latitude / 3600)

    scan = Pointing((ra, dec, 0.), time, infos, dtype=dtype)
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
    beta  = deg2rad(lat)
    alpha = deg2rad(ra0)
    delta = deg2rad(dec0)
    eps   = deg2rad(pa0)

    # Cartesian coordinates from longitude and latitude
    z4 = sin(beta)
    y4 = sin(lambd)*cos(beta)
    x4 = cos(lambd)*cos(beta)

    # rotation about the x4 axe of -eps
    x3 =  x4
    y3 =  y4*cos(eps) + z4*sin(eps)
    z3 = -y4*sin(eps) + z4*cos(eps)

    # rotation about the axis Oy2, angle delta
    x2 = x3*cos(delta) - z3*sin(delta)
    y2 = y3
    z2 = x3*sin(delta) + z3*cos(delta)

    # rotation about the axis Oz1, angle alpha
    x1 = x2*cos(alpha) - y2*sin(alpha)
    y1 = x2*sin(alpha) + y2*cos(alpha)
    z1 = z2

    # Compute angles from cartesian coordinates
    # it is the only place where we can get nan with arcsinus
    dec = rad2deg(arcsin(np.clip(z1, -1., 1.)))
    ra  = mod(rad2deg(arctan2(y1, x1)), 360.)

    return ra, dec


class MaskPolicy(object):
    KEEP   = 0
    MASK   = 1
    REMOVE = 2

    def __init__(self, flags, values, description=None):
        self._description = description
        if isscalar(flags):
            if isinstance(flags, str):
                flags = flags.split(',')
            else:
                flags = (flags,)
        if isscalar(values):
            if isinstance(values, str):
                values = values.split(',')
            else:
                values = (values,)
        if len(flags) != len(values):
            raise ValueError('The number of policy flags is different from the'
                             ' number of policies.')

        self._policy = []
        for flag, value in zip(flags, values):
            if flag[0] == '_':
                raise ValueError('A policy flag should not start with an under'
                                 'score.')
            value = value.strip().lower()
            choices = ('keep', 'mask', 'remove')
            if value not in choices:
                raise KeyError('Invalid policy ' + flag + "='" + value + "'. E"
                               "xpected ones are " + strenum(choices) + '.')
            self._policy.append({flag: value})
            setattr(self, flag, value)
        self._policy = tuple(self._policy)

    def __array__(self, dtype=int):
        conversion = {'keep': self.KEEP,
                      'mask': self.MASK,
                      'remove': self.REMOVE}
        return np.array([conversion[policy.values()[0]]
                         for policy in self._policy], dtype=dtype)

    def __str__(self):
        str = self._description + ': ' if self._description is not None else ''
        str_ = []
        for policy in self._policy:
            str_.append(policy.values()[0] + " '" + policy.keys()[0] + "'")
        str += ', '.join(str_)
        return str
