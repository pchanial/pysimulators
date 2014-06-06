from __future__ import division

try:
    import kapteyn.maputils as km
except ImportError:
    km = None
import numpy as np

try:
    import matplotlib.pyplot as mp
except ImportError:
    pass
from astropy.time import Time
from pyoperators import (
    DifferenceOperator,
    NormalizeOperator,
    Spherical2CartesianOperator,
)
from pyoperators.utils import isscalarlike, tointtuple
from .core import PackedTable
from ..datatypes import Map
from ..operators import (
    SphericalEquatorial2GalacticOperator,
    SphericalHorizontal2EquatorialOperator,
)
from ..quantities import Quantity
from ..wcsutils import angle_lonlat, barycenter_lonlat, create_fitsheader

__all__ = ['Sampling', 'Pointing', 'PointingEquatorial', 'PointingHorizontal']


class Sampling(PackedTable):
    DEFAULT_DATE_OBS = '2000-01-01'
    DEFAULT_PERIOD = 1

    def __init__(self, n, date_obs=None, period=None, **keywords):
        if date_obs is None:
            date_obs = self.DEFAULT_DATE_OBS
        if isinstance(date_obs, str):
            # XXX astropy.time bug needs []
            date_obs = Time([date_obs], scale='utc')
        elif not isinstance(date_obs, Time):
            raise TypeError('The observation start date is invalid.')
        elif date_obs.is_scalar:  # work around astropy.time bug
            date_obs = Time([str(date_obs)], scale='utc')
        if period is None:
            if hasattr(keywords, 'time'):
                period = np.median(np.diff(keywords['time']))
            else:
                period = self.DEFAULT_PERIOD
        time = keywords.pop('time', None)
        PackedTable.__init__(self, n, time=time, **keywords)
        self.date_obs = date_obs
        self.period = float(period)

    def time(self):
        return self.index * self.period


class Pointing(Sampling):
    def __init__(self, *args, **keywords):
        """
        ptg = Pointing(n)
        ptg = Pointing(precession=..., nutation=..., intrinsic_rotation=...)
        ptg = Pointing(precession, nutation, intrinsic_rotation)

        Parameters
        ----------
        n : integer
            The number of samples.
        names : tuple of 3 strings
            The names of the 3 rotation angles. They are stored as Layout special
            attributes. By default, these are the proper Euler angles: precession,
            nutation and intrinsic rotation.
        degrees : boolean, optional
            If true, the input spherical coordinates are assumed to be in
            degrees.
        """
        names = keywords.pop('names', ('precession', 'nutation', 'intrinsic_rotation'))
        degrees = keywords.pop('degrees', False)
        if len(names) != 3:
            raise ValueError('The 3 pointing angles are not named.')

        if len(args) <= 1:
            if (
                names[0] in keywords
                and names[1] not in keywords
                or names[1] in keywords
                and names[0] not in keywords
            ):
                raise ValueError('The pointing is not specified.')
            if names[0] not in keywords:
                keywords[names[0]] = None
            if names[1] not in keywords:
                keywords[names[1]] = None
            if names[2] not in keywords:
                keywords[names[2]] = 0
        elif len(args) <= 3:
            keywords[names[0]] = args[0]
            keywords[names[1]] = args[1]
            keywords[names[2]] = args[2] if len(args) == 3 else 0
        else:
            raise ValueError('Invalid number of arguments.')

        if len(args) == 1:
            if not isscalarlike(args[0]):
                raise ValueError('Invalid number of arguments.')
            shape = tointtuple(args[0])
        else:
            shape = np.broadcast(
                *([keywords[_] for _ in names] + [keywords.get('time', None)])
            ).shape
        if len(shape) == 0:
            shape = (1,)
        elif len(shape) != 1:
            raise ValueError('Invalid dimension for the pointing.')
        Sampling.__init__(
            self, shape, angular=None, cartesian=None, velocity=None, **keywords
        )
        self.names = names
        self.degrees = bool(degrees)

    def angular(self):
        """
        Return the angular coordinates as an (N, 2) ndarray.

        """
        return np.array([getattr(self, self.names[0]), getattr(self, self.names[1])]).T

    def cartesian(self):
        """
        Return the cartesian coordinates.

        """
        raise NotImplementedError()

    def velocity(self):
        op = NormalizeOperator() * DifferenceOperator(axis=-2) / self.period
        velocity = Quantity(op(self.coordinates_cartesian), 'rad/s')
        return velocity.tounit(self.DEFAULT_VELOCITY_UNIT)


#    def get_valid(self):
#        valid = np.ones(self.shape, bool)
#        if 'masked' in self.dtype.names:
#            valid &= ~self.masked
#        if 'removed' in self.dtype.names:
#            valid &= ~self.removed
#        return valid


class PointingEquatorial(Pointing):
    def __init__(self, *args, **keywords):
        """
        ptg = PointingEquatorial(n)
        ptg = PointingEquatorial(ra, dec[, pa])
        ptg = PointingEquatorial(ra=..., dec=...[, pa=...])

        Parameters
        ----------
        n : integer
            The number of samples.
        ra : array-like
            The Right Ascension, in degrees.
        dec : array-like
            The declination, in degrees.
        pa : array-like, optional
            The position angle, in degrees (by default: 0).
        """
        Pointing.__init__(
            self,
            degrees=True,
            names=('ra', 'dec', 'pa'),
            galactic=None,
            *args,
            **keywords,
        )

    def cartesian(self):
        """
        Return the cartesian coordinates in the equatorial referential.

        """
        op = Spherical2CartesianOperator('azimuth,elevation', degrees=self.degrees)
        return op(self.angular)

    def galactic(self):
        """
        Return the angular coordinates in the galactic referential.

        """
        e2g = SphericalEquatorial2GalacticOperator(degrees=True)
        return e2g(self.angular)

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
        if angle_max >= 90.0:
            print(
                'Warning: some coordinates have an angular distance to the p'
                'rojection point greater than 90 degrees.'
            )
            angles[angles >= 90] = np.nan
            angle_max = np.nanmax(angles)
        angle_max = np.deg2rad(angle_max)
        if resolution is None:
            cdelt = 2 * np.rad2deg(np.tan(angle_max)) / naxis
        else:
            cdelt = resolution / 3600
            naxis = 2 * np.ceil(np.rad2deg(np.tan(angle_max)) / cdelt)
        return create_fitsheader(
            2 * [naxis], cdelt=cdelt, crval=crval, crpix=2 * [naxis // 2 + 1]
        )

    def plot(
        self,
        map=None,
        header=None,
        title=None,
        new_figure=True,
        linewidth=2,
        percentile=0.01,
        **kw,
    ):
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
            image = map.imshow(
                title=title, new_figure=new_figure, percentile=percentile, **kw
            )
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
    DEFAULT_LATITUDE = None
    DEFAULT_LONGITUDE = None

    def __init__(self, *args, **keywords):
        """
        ptg = PointingHorizontal(n, latitude=45, longitude=30)
        ptg = PointingHorizontal(azimuth, elevation[, pitch],
                                 latitude=45, longitude=30)
        ptg = PointingHorizontal(n, azimuth=..., elevation=..., pitch=...
                                 latitude=45, longitude=30)

        Parameters
        ----------
        n : integer
            The number of samples.
        azimuth : array-like
            The azimuth, in degrees.
        elevation : array-like
            The elevation, in degrees.
        pitch : array-like, optional
            The pitch angle, in degrees (by default: 0).

        """
        latitude = keywords.pop('latitude', None)
        if latitude is None:
            latitude = self.DEFAULT_LATITUDE
            if latitude is None:
                raise ValueError('The reference latitude is not specified.')
        self.latitude = latitude

        longitude = keywords.pop('longitude', None)
        if longitude is None:
            longitude = self.DEFAULT_LONGITUDE
            if longitude is None:
                raise ValueError('The reference longitude is not specified.')
        self.longitude = longitude

        Pointing.__init__(
            self,
            degrees=True,
            names=('azimuth', 'elevation', 'pitch'),
            equatorial=None,
            galactic=None,
            *args,
            **keywords,
        )

    def cartesian(self):
        """
        Return the cartesian coordinates in the horizontal referential.

        """
        s2c = Spherical2CartesianOperator('azimuth,elevation', degrees=True)
        return s2c(self.angular)

    def equatorial(self):
        """
        Return the angular coordinates in the equatorial referential.

        """
        h2e = SphericalHorizontal2EquatorialOperator(
            'NE', self.time, self.latitude, self.longitude, degrees=True
        )
        return h2e(self.angular, preserve_input=False)

    def galactic(self):
        """
        Return the angular coordinates in the galactic referential.

        """
        h2e = SphericalHorizontal2EquatorialOperator(
            'NE', self.time, self.latitude, self.longitude, degrees=True
        )
        e2g = SphericalEquatorial2GalacticOperator(degrees=True)
        return e2g(h2e)(self.angular, preserve_input=False)


class PointingScanEquatorial(PointingEquatorial):
    INSCAN = 0
    TURNAROUND = 1
    OTHER = 2

    def __init__(self, *args, **keywords):
        PointingEquatorial.__init__(
            self, info=keywords.pop('info', 0), *args, **keywords
        )


def _plot_scan(image, ra, dec, linewidth=None, **kw):
    x, y = image.topixel(ra, dec)
    p = mp.plot(x, y, linewidth=linewidth, **kw)
    for i in xrange(ra.size):
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            mp.plot(x[i], y[i], 'o', color=p[0]._color)
            break