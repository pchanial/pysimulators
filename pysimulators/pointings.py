from __future__ import division

try:
    import kapteyn.maputils as km
except ImportError:
    km = None
import numpy as np
import matplotlib.pyplot as mp
from astropy.time import Time
from pyoperators import (
    DifferenceOperator, NormalizeOperator, RadiansOperator,
    Spherical2CartesianOperator)
from pyoperators.memory import empty, zeros
from pyoperators.utils import strenum
from .datatypes import FitsArray, Map
from .quantities import Quantity
from .wcsutils import angle_lonlat, barycenter_lonlat, create_fitsheader

__all__ = ['Pointing', 'PointingEquatorial', 'PointingHorizontal']


class Pointing(FitsArray):
    MANDATORY_NAMES = 'latitude', 'longitude'
    DEFAULT_DATE_OBS = '2000-01-01 00:00:00'
    DEFAULT_DTYPE = [(n, float) for n in MANDATORY_NAMES + ('time',)]
    DEFAULT_SAMPLING_PERIOD = 1  # [s]
    DEFAULT_VELOCITY_UNIT = 'arcsec/s'

    def __new__(cls, x=None, date_obs=None, sampling_period=None, header=None,
                unit=None, derived_units=None, dtype=None, copy=True,
                order='C', subok=False, ndmin=0, **keywords):

        if dtype is None:
            dtype = np.dtype(cls.DEFAULT_DTYPE)
        dtype = np.dtype(dtype)
        if dtype.kind != 'V':
            raise TypeError('The default dtype is not structured.')
        elif any(n not in dtype.names for n in cls.MANDATORY_NAMES):
            raise TypeError(
                'The default structured dtype does not contain the names {0}.'.
                format(strenum(cls.MANDATORY_NAMES)))
        if date_obs is None:
            date_obs = cls.DEFAULT_DATE_OBS
        if isinstance(date_obs, str):
            # XXX astropy.time bug needs []
            date_obs = Time([date_obs], scale='utc')
        elif not isinstance(date_obs, Time):
            raise TypeError('The observation start date is invalid.')
        elif date_obs.is_scalar:  # work around astropy.time bug
            date_obs = Time([str(date_obs)], scale='utc')
        if sampling_period is None:
            sampling_period = cls.DEFAULT_SAMPLING_PERIOD

        if isinstance(x, np.ndarray) and x.dtype.kind == 'V':
            if dtype == x.dtype:
                result = FitsArray.__new__(
                    cls, x, header=header, unit=unit,
                    derived_units=derived_units, dtype=dtype, copy=copy,
                    order=order, subok=True, ndmin=ndmin)
                if not subok and type(x) is not cls:
                    result = result.view(cls)
                result.date_obs = date_obs
                result.sampling_period = sampling_period
                return result

            for n in x.dtype.names:
                if n in dtype.names and n not in keywords:
                    keywords[n] = x[n]

        elif isinstance(x, (list, tuple)):
            if len(x) < len(cls.MANDATORY_NAMES):
                raise ValueError(
                    "The number of inputs '{0}' is incompatible with the numbe"
                    "r of mandatory names '{1}' ({2}).".format(
                        len(x), len(cls.MANDATORY_NAMES), strenum(
                            cls.MANDATORY_NAMES, 'and')))
            if len(x) > len(dtype.names):
                raise ValueError(
                    "The number of inputs '{0}' is larger than the number of n"
                    "ames '{1}' ({2}).".format(
                        len(x), len(dtype.names), strenum(dtype.names, 'and')))
            for n, v in zip(dtype.names, x):
                if n in keywords:
                    raise ValueError(
                        "The name '{0}' is ambiguously specified via arguments"
                        " and keywords.".format(n))
                keywords[n] = v

        elif isinstance(x, dict):
            for k, v in x.items():
                if k not in keywords:
                    keywords[k] = v

        elif x is not None:
            x = np.asarray(x)
            s = x.shape[-1]
            if s < len(cls.MANDATORY_NAMES):
                raise ValueError(
                    "The last dimension of the input '{0}' is incompatible wit"
                    "h the number of mandatory names '{1}' ({2}).".format(
                        s, len(cls.MANDATORY_NAMES), strenum(
                            cls.MANDATORY_NAMES, 'and')))
            if s > len(dtype.names):
                raise ValueError(
                    "The last dimension of the input '{0}' is larger than the "
                    "number of names '{1}' ({2}).".format(
                        s, len(dtype.names), strenum(dtype.names, 'and')))
            for i, n in enumerate(dtype.names):
                if i >= s:
                    break
                if n not in keywords:
                    keywords[n] = x[..., i]

        for k, v in keywords.items():
            keywords[k] = np.asarray(v)

        for n in cls.MANDATORY_NAMES:
            if n not in keywords:
                raise ValueError(
                    "The mandatory name '{0}' is not specified.".format(n))

        try:
            shape = np.broadcast(*keywords.values()).shape
        except ValueError:
            raise ValueError('The pointing inputs do not have the same shape.')

        if 'time' in dtype.names and ('time' not in keywords or
                                      keywords['time'] is None):
            keywords['time'] = (np.arange(shape[-1], dtype=float)
                                if len(shape) > 0 else 0) * sampling_period

        result = FitsArray.zeros(
            shape, header=header, unit=unit, derived_units=derived_units,
            dtype=dtype, order=order)
        result = result.view(type(x) if subok and isinstance(x, cls) else cls)
        for k, v in keywords.items():
            result[k] = v

        return result

    @classmethod
    def empty(cls, shape, dtype=None, date_obs=None, sampling_period=None):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE
        result = cls(empty(shape, dtype), date_obs=date_obs,
                     sampling_period=sampling_period, dtype=dtype, copy=False)
        if 'time' in result.dtype.names:
            result.time = (np.arange(result.shape[-1], dtype=float)
                           if len(result.shape) > 0
                           else 0) * result.sampling_period
        return result

    @classmethod
    def zeros(cls, shape, dtype=None, date_obs=None, sampling_period=None):
        print 'Pointing', cls
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE
        result = cls(zeros(shape, dtype), date_obs=date_obs,
                     sampling_period=sampling_period, dtype=dtype, copy=False)
        if 'time' in result.dtype.names:
            result.time = (np.arange(result.shape[-1], dtype=float)
                           if len(result.shape) > 0
                           else 0) * result.sampling_period
        return result

    def tocartesian(self):
        op = Spherical2CartesianOperator('elevation,azimuth') * \
             RadiansOperator()
        return op(np.array([self.latitude, self.longitude]).T)

    @property
    def velocity(self):
        op = NormalizeOperator() * DifferenceOperator(axis=-2) / \
             self.sampling_period
        velocity = Quantity(op(self.tocartesian()), 'rad/s')
        return velocity.tounit(self.DEFAULT_VELOCITY_UNIT)

    def get_valid(self):
        valid = np.ones(self.shape, bool)
        if 'masked' in self.dtype.names:
            valid &= ~self.masked
        if 'removed' in self.dtype.names:
            valid &= ~self.removed
        return valid


class PointingEquatorial(Pointing):
    MANDATORY_NAMES = 'ra', 'dec'
    DEFAULT_DTYPE = [('ra', float), ('dec', float), ('pa', float),
                     ('time', float)]

    def tocartesian(self):
        op = Spherical2CartesianOperator('azimuth,elevation') * \
             RadiansOperator()
        return op(np.array([self.ra, self.dec]).T)

    def get_map_header(self, resolution=None, naxis=None):
        """
        Returns a FITS header that encompasses all non-removed and
        non-masked pointings.
        """
        mask = self.get_valid()
        ra = self.ra[mask]
        dec = self.dec[mask]
        crval = barycenter_lonlat(ra, dec)
        angles = angle_lonlat((ra, dec), crval)
        angle_max = np.nanmax(angles)
        if angle_max >= 90.:
            print('Warning: some coordinates have an angular distance to the p'
                  'rojection point greater than 90 degrees.')
            angles[angles >= 90] = np.nan
            angle_max = np.nanmax(angles)
        angle_max = np.deg2rad(angle_max)
        if resolution is None:
            cdelt = 2 * np.rad2deg(np.tan(angle_max)) / naxis
        else:
            cdelt = resolution / 3600
            naxis = 2 * np.ceil(np.rad2deg(np.tan(angle_max)) / cdelt)
        return create_fitsheader(2*[naxis], cdelt=cdelt, crval=crval,
                                 crpix=2*[naxis//2+1])

    def plot(self, map=None, header=None, title=None, new_figure=True,
             linewidth=2, percentile=0.01, **kw):
        """
        Plot the pointings' celestial coordinates.

        Parameters
        ----------
        map : ndarray of dim 2
            An optional map to be displayed as background.
        header : pyfits.Header
            The optional map's FITS header.
        title : str
            The figure's title.
        new_figure : boolean
            If true, the display will be done in a new window.
        linewidth : float
            The scan line width.
        percentile : float, tuple of two floats
            As a float, percentile of values to be discarded, otherwise,
            percentile of the minimum and maximum values to be displayed.

        """
        if km is None:
            raise RuntimeError('The kapteyn library is required.')
        invalid = ~self.get_valid()
        ra, dec = self.ra.copy(), self.dec.copy()
        ra[invalid] = np.nan
        dec[invalid] = np.nan
        if np.all(invalid):
            if new_figure:
                raise ValueError('There is no valid coordinates.')
            return

        if header is None:
            header = getattr(map, 'header', None)
        elif map is not None:
            map = Map(map, header=header, copy=False)

        if isinstance(map, Map) and map.has_wcs():
            image = map.imshow(title=title, new_figure=new_figure,
                               percentile=percentile, **kw)
        else:
            if header is None:
                header = self.get_map_header(naxis=1)
            fitsobj = km.FITSimage(externalheader=dict(header))
            if new_figure:
                fig = mp.figure()
                frame = fig.add_axes((0.1, 0.1, 0.8, 0.8))
            else:
                frame = mp.gca()
            if title is not None:
                frame.set_title(title)
            image = fitsobj.Annotatedimage(frame, blankcolor='w')
            image.Graticule()
            image.plot()
            image.interact_toolbarinfo()
            image.interact_writepos()

        mp.show()
        _plot_scan(image, ra, dec, linewidth=linewidth, **kw)
        return image


class PointingHorizontal(Pointing):
    MANDATORY_NAMES = 'azimuth', 'elevation'
    DEFAULT_DTYPE = [('azimuth', float), ('elevation', float),
                     ('pitch', float), ('time', float)]
    DEFAULT_LATITUDE = None
    DEFAULT_LONGITUDE = None

    def __new__(cls, x=None, date_obs=None, sampling_period=None,
                latitude=None, longitude=None, copy=True, **keywords):
        if latitude is None:
            latitude = cls.DEFAULT_LATITUDE
        if latitude is None:
            raise ValueError('The reference latitude is not specified.')
        if longitude is None:
            longitude = cls.DEFAULT_LONGITUDE
        if longitude is None:
            raise ValueError('The reference longitude is not specified.')
        result = Pointing.__new__(
            cls, x=x, date_obs=date_obs, sampling_period=sampling_period,
            copy=copy, **keywords)
        result.latitude = np.asarray(latitude)
        result.longitude = np.asarray(longitude)
        return result

    @classmethod
    def empty(cls, shape, dtype=None, date_obs=None, sampling_period=None,
              latitude=None, longitude=None):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE
        result = cls(empty(shape, dtype), dtype=dtype, date_obs=date_obs,
                     sampling_period=sampling_period,
                     latitude=latitude, longitude=longitude, copy=False)
        if 'time' in result.dtype.names:
            result.time = (np.arange(result.shape[-1], dtype=float)
                           if len(result.shape) > 0
                           else 0) * result.sampling_period
        return result

    @classmethod
    def zeros(cls, shape, dtype=None, date_obs=None, sampling_period=None,
              latitude=None, longitude=None):
        if dtype is None:
            dtype = cls.DEFAULT_DTYPE
        result = cls(zeros(shape, dtype), dtype=dtype, date_obs=date_obs,
                     sampling_period=sampling_period,
                     latitude=latitude, longitude=longitude, copy=False)
        if 'time' in result.dtype.names:
            result.time = (np.arange(result.shape[-1], dtype=float)
                           if len(result.shape) > 0
                           else 0) * result.sampling_period
        return result


class PointingScanEquatorial(PointingEquatorial):
    INSCAN = 0
    TURNAROUND = 1
    OTHER = 2
    DEFAULT_DTYPE = PointingEquatorial.DEFAULT_DTYPE + [('info', int)]


def _plot_scan(image, ra, dec, linewidth=None, **kw):
    x, y = image.topixel(ra, dec)
    p = mp.plot(x, y, linewidth=linewidth, **kw)
    for i in xrange(ra.size):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            mp.plot(x[i], y[i], 'o', color=p[0]._color)
            break
