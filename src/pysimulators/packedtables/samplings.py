import numpy as np
from astropy.time import Time, TimeDelta

import pyoperators
from pyoperators import (
    MPI,
    DifferenceOperator,
    NormalizeOperator,
    Spherical2CartesianOperator,
)
from pyoperators.utils import deprecated, isscalarlike, tointtuple
from pyoperators.utils.mpi import as_mpi

from ..datatypes import Map
from ..operators import (
    SphericalEquatorial2GalacticOperator,
    SphericalHorizontal2EquatorialOperator,
)
from ..quantities import Quantity
from .core import PackedTable

__all__ = ['Sampling', 'SamplingSpherical', 'SamplingEquatorial', 'SamplingHorizontal']


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
        masked = keywords.pop('masked', False)
        time = keywords.pop('time', None)
        PackedTable.__init__(self, n, masked=masked, time=time, **keywords)
        self.date_obs = date_obs
        self.period = float(period)

    def time(self):
        return self.index * self.period


class SamplingSpherical(Sampling):
    def __init__(self, *args, **keywords):
        """
        ptg = SamplingSpherical(n)
        ptg = SamplingSpherical(precession=, nutation=, intrinsic_rotation=)
        ptg = SamplingSpherical(precession, nutation, intrinsic_rotation)

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
            self, shape, cartesian=None, spherical=None, velocity=None, **keywords
        )
        self.names = names
        self.degrees = bool(degrees)

    def cartesian(self):
        """
        Return the cartesian coordinates in the equatorial referential.

        """
        return self.spherical2cartesian(self.spherical)

    def spherical(self):
        """
        Return the spherical coordinates as an (N, 2) ndarray.

        """
        return np.array([getattr(self, self.names[0]), getattr(self, self.names[1])]).T

    @property
    def spherical2cartesian(self):
        """
        Return the spherical-to-cartesian transform.

        """
        raise NotImplementedError()

    @property
    def cartesian2spherical(self):
        """
        Return the cartesian-to-spherical transform.

        """
        return self.spherical2cartesian.I

    def velocity(self):
        op = NormalizeOperator() * DifferenceOperator(axis=-2) / self.period
        velocity = Quantity(op(self.cartesian), 'rad/s')
        return velocity.tounit(self.DEFAULT_VELOCITY_UNIT)

    def get_center(self):
        """
        Return the average pointing direction.

        """
        if isscalarlike(self.masked) and self.masked:
            n = 0
            coords = np.zeros((1, 3))
        else:
            coords = self.cartesian
            if not isscalarlike(self.masked):
                valid = 1 - self.masked
                coords *= valid[:, None]
                n = np.sum(valid)
            else:
                n = len(self)
        n = np.asarray(n)
        center = np.sum(coords, axis=0)
        self.comm.Allreduce(MPI.IN_PLACE, as_mpi(n))
        self.comm.Allreduce(MPI.IN_PLACE, as_mpi(center))
        if n == 0:
            raise ValueError('There is no valid pointing.')
        center /= n
        return self.cartesian2spherical(center)


class SamplingEquatorial(SamplingSpherical):
    def __init__(self, *args, **keywords):
        """
        ptg = SamplingEquatorial(n)
        ptg = SamplingEquatorial(ra, dec[, pa])
        ptg = SamplingEquatorial(ra=..., dec=...[, pa=...])

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
        SamplingSpherical.__init__(
            self,
            degrees=True,
            names=('ra', 'dec', 'pa'),
            galactic=None,
            *args,
            **keywords,
        )

    @property
    def spherical2cartesian(self):
        return Spherical2CartesianOperator('azimuth,elevation', degrees=self.degrees)

    @property
    def galactic(self):
        """
        Return the spherical coordinates in the galactic referential.

        """
        e2g = SphericalEquatorial2GalacticOperator(degrees=True)
        return e2g(self.spherical)

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
        import kapteyn.maputils as km
        import matplotlib.pyplot as mp

        if isscalarlike(self.masked) and self.masked or np.all(self.masked):
            if new_figure:
                raise ValueError('There is no valid coordinates.')
            return
        if not isscalarlike(self.masked):
            ra = self.ra.copy()
            dec = self.dec.copy()
            ra[self.masked] = np.nan
            dec[self.masked] = np.nan
        else:
            ra = self.ra
            dec = self.dec

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


class SamplingHorizontal(SamplingSpherical):
    DEFAULT_LATITUDE = None
    DEFAULT_LONGITUDE = None

    def __init__(self, *args, **keywords):
        """
        ptg = SamplingHorizontal(n, latitude=45, longitude=30)
        ptg = SamplingHorizontal(azimuth, elevation[, pitch],
                                 latitude=45, longitude=30)
        ptg = SamplingHorizontal(n, azimuth=..., elevation=..., pitch=...
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

        SamplingSpherical.__init__(
            self,
            degrees=True,
            names=('azimuth', 'elevation', 'pitch'),
            equatorial=None,
            galactic=None,
            *args,
            **keywords,
        )

    @property
    def spherical2cartesian(self):
        return Spherical2CartesianOperator('azimuth,elevation', degrees=self.degrees)

    @property
    def equatorial(self):
        """
        Return the spherical coordinates in the equatorial referential.

        """
        time = self.date_obs + TimeDelta(self.time, format='sec')
        h2e = SphericalHorizontal2EquatorialOperator(
            'NE', time, self.latitude, self.longitude, degrees=True
        )
        return h2e(self.spherical, preserve_input=False)

    @property
    def galactic(self):
        """
        Return the spherical coordinates in the galactic referential.

        """
        time = self.date_obs + TimeDelta(self.time, format='sec')
        h2e = SphericalHorizontal2EquatorialOperator(
            'NE', time, self.latitude, self.longitude, degrees=True
        )
        e2g = SphericalEquatorial2GalacticOperator(degrees=True)
        return e2g(h2e)(self.spherical, preserve_input=False)


class _SamplingEquatorialScan(SamplingEquatorial):
    INSCAN = 0
    TURNAROUND = 1
    OTHER = 2

    def __init__(self, *args, **keywords):
        SamplingEquatorial.__init__(
            self, info=keywords.pop('info', 0), *args, **keywords
        )


def _plot_scan(image, ra, dec, linewidth=None, **kw):
    import matplotlib.pyplot as mp

    x, y = image.topixel(ra, dec)
    p = mp.plot(x, y, linewidth=linewidth, **kw)
    for x_, y_ in np.broadcast(x, y):
        if np.isfinite(x_) and np.isfinite(y_):
            mp.plot(x_, y_, 'o', color=p[0]._color)
            break


if pyoperators.__version__ >= '0.13.6':

    @deprecated
    class PointingSpherical(SamplingSpherical):
        pass

    @deprecated
    class PointingEquatorial(SamplingEquatorial):
        pass

    @deprecated
    class PointingHorizontal(SamplingHorizontal):
        pass

else:
    PointingSpherical = SamplingSpherical
    PointingEquatorial = SamplingEquatorial
    PointingHorizontal = SamplingHorizontal
