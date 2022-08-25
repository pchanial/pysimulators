# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#
"""
The datatypes module contains 3 ndarray subclasses:
    - FitsArray
    - Map
    - Tod
The Map and Tod classes subclass FitsArray.

These classes are useful to load, manipulate and save FITs files.
They also contain specialised display methods.

"""

import io
import os
import pickle
import sys
import time
import uuid
from functools import reduce

import numpy as np
import scipy.stats
from astropy.io import fits as pyfits

from pyoperators import MPI
from pyoperators.memory import empty
from pyoperators.utils import isscalarlike

from .mpiutils import read_fits, write_fits
from .quantities import Quantity
from .wcsutils import create_fitsheader_for, has_wcs

__all__ = ['FitsArray', 'Map', 'Tod']


class FitsArray(Quantity):
    """
    FitsArray(filename|object, header=None, unit=None, derived_units=None,
              dtype=None, copy=True, order='C', subok=False, ndmin=0,
              comm=None)

    An ndarray subclass, whose instances
        - store the FITS header information
        - can be read from and written to FITS files
        - have a specialised matplotlib display and export
        - can be easily displayed using ds9

    Parameters
    ----------
    filename : str
        The FITS file name.
    object : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    header : pyfits.Header, optional
        The FITS header.
    unit : str, optional
        The data unit.
    derived_units : dict, optional
        Dictionary defining non-standard units.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`dtype`, `order`, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.
    comm : mpi4py.Comm, optional
        MPI communicator specifying to which processors the FITS file
        should be distributed.

    """

    _header = None

    def __new__(
        cls,
        data,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        copy=True,
        order='C',
        subok=False,
        ndmin=0,
        comm=None,
    ):

        if isinstance(data, str):

            if comm is None:
                comm = MPI.COMM_SELF

            try:
                buf = pyfits.open(data)['derived_units'].data
                derived_units = pickle.loads(buf.data)
            except (ImportError, KeyError):
                pass

            data, header = read_fits(data, None, comm)

            copy = False
            if unit is None:
                if 'BUNIT' in header:
                    unit = header['BUNIT']
                elif 'QTTY____' in header:
                    unit = header['QTTY____']  # HCSS crap
                    if unit == '1':
                        unit = ''

        elif comm is not None:
            raise ValueError(
                'The MPI communicator can only be set for input F' 'ITS files.'
            )

        # get a new FitsArray instance (or a subclass if subok is True)
        result = Quantity.__new__(
            cls, data, unit, derived_units, dtype, copy, order, True, ndmin
        )
        if not subok and type(result) is not cls:
            result = result.view(cls)

        # copy header attribute
        if header is not None:
            result.header = header
        elif hasattr(data, '_header') and type(data._header) is pyfits.Header:
            if copy:
                result._header = data._header.copy()
            else:
                result._header = data._header
        else:
            result._header = None

        return result

    def __array_finalize__(self, array):
        Quantity.__array_finalize__(self, array)
        self._header = getattr(array, '_header', None)

    def __getattr__(self, name):
        if self.dtype.names and name in self.dtype.names:
            return self[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if self.dtype.names and name in self.dtype.names:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def astype(self, dtype):
        """
        f.astype(t)

        Copy of the FitsArray, cast to a specified type. The FITS keyword
        BITPIX is updated accordingly.

        Parameters
        ----------
        t : str or dtype
            Typecode or data-type to which the array is cast.

        Raises
        ------
        ComplexWarning :
            When casting from complex to float or int. To avoid this,
            one should use ``m.real.astype(t)``.

        Examples
        --------
        >>> f = FitsArray([1, 2, 2.5])
        >>> f.astype(int)
        FitsArray([1 2 2], '')

        """
        result = super().astype(dtype)
        if self._header is not None:
            typename = np.dtype(dtype).name
            result._header = self._header.copy()
            result._header['BITPIX'] = pyfits.DTYPE2BITPIX[typename]
        return result

    @classmethod
    def empty(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            empty(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    @classmethod
    def ones(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.ones(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    @classmethod
    def zeros(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.zeros(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    def has_wcs(self):
        """
        Returns True is the array has a FITS header with a defined World
        Coordinate System.
        """
        return has_wcs(self.header)

    @property
    def header(self):
        if self._header is None and not np.iscomplexobj(self):
            self._header = create_fitsheader_for(self)
        return self._header

    @header.setter
    def header(self, header):
        if header is not None and not isinstance(header, pyfits.Header):
            raise TypeError(
                'Incorrect type for the input header (' + str(type(header)) + ').'
            )
        self._header = header

    def save(self, filename, comm=None):
        """
        Write array as a FITS file.

        Parameters
        ----------
        filename : str
            The FITS file name.
        comm : mpi4py.Comm, optional
            MPI communicator specifying which processors should write
            to the FITS file.
        """

        header = self.header.copy()
        if comm is None:
            comm = MPI.COMM_SELF

        if len(self._unit) != 0:
            header['BUNIT'] = self.unit

        write_fits(filename, self, header, False, None, comm)
        if not self.derived_units:
            return
        if comm is None or comm.rank == 0:
            buf = io.BytesIO()
            pickle.dump(self.derived_units, buf, pickle.HIGHEST_PROTOCOL)
            data = np.frombuffer(buf.getvalue(), np.uint8)
            write_fits(filename, data, None, True, 'derived_units', MPI.COMM_SELF)
        comm.Barrier()

    def imsave(self, filename, colorbar=True, **kw):
        """
        Export to file (JPG, PNG, etc.) the display that would be produced
        by the imshow method.

        Example
        -------
        >>> Map(np.arange(100).reshape((10,10))).imsave('plot.png')
        """
        import matplotlib
        import matplotlib.pyplot as mp

        is_interactive = matplotlib.is_interactive()
        matplotlib.interactive(False)
        dpi = 80.0
        figsize = np.clip(np.max(np.array(self.shape[::-1]) / dpi), 8, 50)
        figsize = (figsize + (2 if colorbar else 0), figsize)
        fig = mp.figure(figsize=figsize, dpi=dpi)
        self.imshow(colorbar=colorbar, new_figure=False, **kw)
        fig = mp.gcf()
        fig.savefig(filename)
        matplotlib.interactive(is_interactive)

    def imshow(
        self,
        mask=None,
        new_figure=True,
        title=None,
        xlabel='',
        ylabel='',
        interpolation='nearest',
        colorbar=True,
        percentile=0,
        **keywords,
    ):
        """
        A graphical display method specialising matplotlib's imshow.

        mask : array-like, optional
            Data mask. True means masked.
        new_figure : boolean, optional
            If True, a new figure if created. Default is True.
        title : str, optional
            The plot title.
        xlabel : str, optional
           Plot X label.
        ylabel : str, optional
           Plot Y label.
        interpolation : str, optional
            Acceptable values are None, 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
            'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
            'bessel', 'mitchell', 'sinc', 'lanczos'
            Default is 'nearest'.
        colorbar : boolean, optional
            If True, plot an intensity color bar next to the display.
            Default is True.
        percentile : float, tuple of two floats in range [0,100]
            As a float, percentile of values to be discarded, otherwise,
            percentile of the minimum and maximum values to be displayed.
        **keywords : additional are keywords passed to matplotlib's imshow.

        """
        import matplotlib.pyplot as mp

        if np.iscomplexobj(self):
            data = abs(self.magnitude)
        else:
            data = self.magnitude.copy()
        if isscalarlike(percentile):
            percentile = percentile / 2, 100 - percentile / 2
        unfinite = ~np.isfinite(self.magnitude)
        if mask is None:
            mask = unfinite
        else:
            mask = np.logical_or(mask, unfinite)
        data_valid = data[~mask]
        minval = scipy.stats.scoreatpercentile(data_valid, percentile[0])
        maxval = scipy.stats.scoreatpercentile(data_valid, percentile[1])
        data[data < minval] = minval
        data[data > maxval] = maxval
        data[mask] = np.nan

        if new_figure:
            fig = mp.figure()
        else:
            fig = mp.gcf()
        fontsize = 12.0 * fig.get_figheight() / 6.125

        image = mp.imshow(data, interpolation=interpolation, **keywords)
        image.set_clim(minval, maxval)

        ax = mp.gca()
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)

        if title is not None:
            mp.title(title, fontsize=fontsize)
        if colorbar:
            colorbar = mp.colorbar()
            for tick in colorbar.ax.get_yticklabels():
                tick.set_fontsize(fontsize)

        mp.draw()
        return image

    def ds9(self, xpamsg=None, origin=None, new=True, **keywords):
        """
        Display the array using ds9.

        The ds9 process will be given an random id. By default, the
        following access point are set before the array is loaded:
            -cmap heat
            -scale scope local
            -scale mode 99.5
        Other access points can be set before the data is loaded though
        the keywords (see examples below).
        After the array is loaded, the map's header is set and the user
        may add other XPA messages through the xpamsg argument or by
        setting them through the returned ds9 instance.

        Parameters
        ----------
        xpamsg : string or tuple of string
            XPA access point message to be set after the array is loaded.
            (see http://hea-www.harvard.edu/RD/ds9/ref/xpa.html).
        origin: string
            Set origin to 'upper' for Y increasing downwards
        new: boolean
            If true, open the array in a new ds9 instance.
        **keywords : string or tuple of string
            Specify more access points to be set before array loading.
            a keyword such as 'height=400' will be appended to the command
            that launches ds9 in the form 'ds9 [...] -height 400'

        Returns
        -------
        The returned object is a ds9 instance. It can be manipulated using
        XPA access points.

        Examples
        --------
        >>> m = Map('myfits.fits')
        >>> d=m.ds9(('zoom to fit', 'saveimage png myfits.png'),
        ...         scale='histequ', cmap='invert yes', height=400)
        >>> d.set('exit')

        """
        try:
            import pyds9
        except ImportError:
            raise ImportError('The library pyds9 has not been installed.')
        import xpa

        id = None
        if not new:
            list = pyds9.ds9_targets()
            if list is not None:
                id = list[-1]

        if id is None:
            if 'cmap' not in keywords:
                keywords['cmap'] = 'heat'

            if 'scale' not in keywords:
                keywords['scale'] = ('scope local', 'mode 99.5')

            if origin is None:
                origin = getattr(self, 'origin', 'lower')
            if origin == 'upper' and 'orient' not in keywords:
                keywords['orient'] = 'y'

            wait = 10

            id_ = 'ds9_' + str(uuid.uuid1())[4:8]

            command = 'ds9 -title ' + id_

            for k, v in keywords.items():
                k = str(k)
                if type(v) is not tuple:
                    v = (v,)
                command += reduce(lambda x, y: str(x) + ' -' + k + ' ' + str(y), v, '')

            os.system(command + ' &')

            # start the xpans name server
            if xpa.xpaaccess('xpans', None, 1) is None:
                _cmd = None
                # look in install directories
                for _dir in sys.path:
                    _fname = os.path.join(_dir, 'xpans')
                    if os.path.exists(_fname):
                        _cmd = _fname + ' -e &'
                if _cmd:
                    os.system(_cmd)

            for i in range(wait):
                list = xpa.xpaaccess(id_, None, 1024)
                if list is not None:
                    break
                time.sleep(1)
            if not list:
                raise ValueError('No active ds9 running for target: %s' % list)

        # get ds9 instance with given id
        d = pyds9.DS9(id_)

        # load array
        input = self.view(np.ndarray)
        if input.dtype.kind in ('b', 'i'):
            input = np.array(input, np.int32, copy=False)
        d.set_np2arr(input.T)

        # load header
        if self.has_wcs():
            d.set('wcs append', str(self.header))

        if xpamsg is not None:
            if isinstance(xpamsg, str):
                xpamsg = (xpamsg,)
            for v in xpamsg:
                d.set(v)

        return d


class Map(FitsArray):
    """
    Map(filename|object, header=None, unit=None, derived_units=None,
        coverage=None, error=None, origin='lower', dtype=None, copy=True,
        order='C', subok=False, ndmin=0, comm=None)

    A FitsArray subclass, whose instances have
        - a 'coverage' and 'error' attribute
        - a specialised imshow method, using WCS information if available
        - an attribute 'origin', which flags whether the map's origin is on
          the bottom-left ('lower') or top-left ('upper').

    Parameters
    ----------
    filename : str
        The FITS file name.
    object : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    header : pyfits.Header, optional
        The FITS header.
    unit : str, optional
        The data unit.
    derived_units : dict, optional
        Dictionary defining non-standard units.
    coverage : ndarray, optional
        The coverage map, of the same shape as the data.
    error : ndarray, optional
        The error map, of the same shape as the data.
    origin : 'lower' or 'upper', optional
        Display convention for the Map. For an top-left origin, use 'upper'.
        For a bottom-left origin, use 'lower'. The default convention is
        bottom-left.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`dtype`, `order`, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.
    comm : mpi4py.Comm, optional
        MPI communicator specifying to which processors the FITS file
        should be distributed.

    """

    coverage = None
    error = None
    origin = None

    def __new__(
        cls,
        data,
        header=None,
        unit=None,
        derived_units=None,
        coverage=None,
        error=None,
        origin=None,
        dtype=None,
        copy=True,
        order='C',
        subok=False,
        ndmin=0,
        comm=None,
    ):

        # get a new Map instance (or a subclass if subok is True)
        result = FitsArray.__new__(
            cls,
            data,
            header,
            unit,
            derived_units,
            dtype,
            copy,
            order,
            True,
            ndmin,
            comm,
        )
        if not subok and type(result) is not cls:
            result = result.view(cls)

        if isinstance(data, str):

            if comm is None:
                comm = MPI.COMM_SELF

            if 'DISPORIG' in result.header:
                if origin is None:
                    origin = result.header['DISPORIG']
            if coverage is None:
                try:
                    coverage, junk = read_fits(data, 'Coverage', comm)
                except KeyError:
                    pass
            if error is None:
                try:
                    error, junk = read_fits(data, 'Error', comm)
                except KeyError:
                    pass

        if origin is not None:
            origin = origin.strip().lower()
            if origin not in ('lower', 'upper'):
                raise ValueError(
                    "Invalid origin '" + origin + "'. Expected va"
                    "lues are 'lower' or 'upper'."
                )
            result.origin = origin

        if coverage is not None:
            result.coverage = np.asanyarray(coverage)
        elif copy and result.coverage is not None:
            result.coverage = result.coverage.copy()

        if error is not None:
            result.error = np.asanyarray(error)
        elif copy and result.error is not None:
            result.error = result.error.copy()

        return result

    def __array_finalize__(self, array):
        FitsArray.__array_finalize__(self, array)
        self.coverage = getattr(array, 'coverage', None)
        self.error = getattr(array, 'error', None)
        self.origin = getattr(array, 'origin', 'lower')

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if not isinstance(item, Map):
            return item
        if item.coverage is not None:
            item.coverage = item.coverage[key]
        if item.error is not None:
            item.error = item.error[key]
        return item

    def astype(self, dtype):
        """
        m.astype(t)

        Copy of the Map, cast to a specified type. The types of the coverage
        and error attribute are also converted.

        Parameters
        ----------
        t : str or dtype
            Typecode or data-type to which the array is cast.

        Raises
        ------
        ComplexWarning :
            When casting from complex to float or int. To avoid this,
            one should use ``m.real.astype(t)``.

        Examples
        --------
        >>> x = Map([1, 2, 2.5], coverage=[0., 1., 1.])
        >>> x2 = x.astype(int)
        >>> x2
        array([1, 2, 2])
        >>> x2.coverage
        array([0, 1, 1])

        """
        result = super().astype(dtype)
        if self.coverage is not None:
            result.coverage = self.coverage.astype(dtype)
        if self.error is not None:
            result.error = self.error.astype(dtype)
        return result

    @classmethod
    def empty(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        coverage=None,
        error=None,
        origin=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            empty(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            coverage=coverage,
            error=error,
            origin=origin,
            copy=False,
            dtype=dtype,
            **keywords,
        )

    @classmethod
    def ones(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        coverage=None,
        error=None,
        origin=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.ones(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            coverage=coverage,
            error=error,
            origin=origin,
            copy=False,
            dtype=dtype,
            **keywords,
        )

    @classmethod
    def zeros(
        cls,
        shape,
        header=None,
        unit=None,
        derived_units=None,
        coverage=None,
        error=None,
        origin=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.zeros(shape, dtype, order),
            header=header,
            unit=unit,
            derived_units=derived_units,
            coverage=coverage,
            error=error,
            origin=origin,
            copy=False,
            dtype=dtype,
            **keywords,
        )

    def imshow(
        self,
        mask=None,
        new_figure=True,
        title=None,
        xlabel='X',
        ylabel='Y',
        interpolation='nearest',
        origin=None,
        colorbar=True,
        percentile=0,
        **keywords,
    ):
        try:
            import kapteyn.maputils as km
        except ImportError:
            km = None
        import matplotlib.pyplot as mp

        if mask is None and self.coverage is not None:
            mask = self.coverage <= 0

        if origin is None:
            origin = self.origin

        # check if the map has no astrometry information
        if not self.has_wcs() or km is None:
            image = super().imshow(
                mask=mask,
                new_figure=new_figure,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                origin=origin,
                colorbar=colorbar,
                percentile=percentile,
                **keywords,
            )
            return image

        if np.iscomplexobj(self):
            data = abs(self.magnitude)
        else:
            data = self.magnitude.copy()
        if isscalarlike(percentile):
            percentile = percentile / 2, 100 - percentile / 2
        unfinite = ~np.isfinite(self.magnitude)
        if mask is None:
            mask = unfinite
        else:
            mask = np.logical_or(mask, unfinite)
        data_valid = data[~mask]
        minval = scipy.stats.scoreatpercentile(data_valid, percentile[0])
        maxval = scipy.stats.scoreatpercentile(data_valid, percentile[1])
        data[data < minval] = minval
        data[data > maxval] = maxval
        data[mask] = np.nan

        # XXX FIX ME
        colorbar = False

        fitsobj = km.FITSimage(externaldata=data, externalheader=dict(self.header))
        if new_figure:
            fig = mp.figure()
            frame = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        else:
            frame = mp.gca()
        fontsize = 12.0 * fig.get_figheight() / 6.125
        for tick in frame.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        for tick in frame.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
        if title is not None:
            frame.set_title(title)
        if colorbar:
            colorbar = mp.colorbar()
            for tick in colorbar.ax.get_yticklabels():
                tick.set_fontsize(fontsize)
        annim = fitsobj.Annotatedimage(frame, blankcolor='w')
        annim.Image(interpolation=interpolation)
        grat = annim.Graticule()
        grat.setp_gratline(visible=False)
        annim.plot()
        annim.interact_imagecolors()
        annim.interact_toolbarinfo()
        annim.interact_writepos()
        if colorbar:
            annim.Colorbar()
        mp.show()
        return annim

    def save(self, filename, comm=None):
        if comm is None:
            comm = MPI.COMM_SELF

        self.header['DISPORIG'] = self.origin, 'Map display convention'
        FitsArray.save(self, filename, comm=comm)
        if self.coverage is not None:
            write_fits(filename, self.coverage, None, True, 'Coverage', comm)
        if self.error is not None:
            write_fits(filename, self.error, None, True, 'Error', comm)

    def _wrap_func(self, func, unit, *args, **kw):
        result = super()._wrap_func(func, unit, *args, **kw)
        if not isinstance(result, np.ndarray):
            return type(self)(result, unit=unit, derived_units=self.derived_units)
        result.coverage = None
        result.error = None
        return result


class Tod(FitsArray):
    """
    Tod(filename|object, header=None, unit=None, derived_units=None, mask=None,
        dtype=None, copy=True, order='C', subok=False, ndmin=0, comm=None)

    A FitsArray subclass for Time Ordered Data. The time is assumed to be
    the last dimension. Tod instances have
        - a 'mask' attribute, of the same shape than the data. A True value
          means that the data sample is masked
        - a specialised imshow method

    Parameters
    ----------
    filename : str
        The FITS file name.
    object : array_like
        An array, any object exposing the array interface, an
        object whose __array__ method returns an array, or any
        (nested) sequence.
    header : pyfits.Header, optional
        The FITS header.
    unit : str, optional
        The data unit.
    derived_units : dict, optional
        Dictionary defining non-standard units.
    mask : boolean ndarray, optional
        The TOD mask, of the same shape as the data. True means masked.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then
        the type will be determined as the minimum type required
        to hold the objects in the sequence.  This argument can only
        be used to 'upcast' the array.  For downcasting, use the
        .astype(t) method.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`dtype`, `order`, etc.).
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.
    comm : mpi4py.Comm, optional
        MPI communicator specifying to which processors the FITS file
        should be distributed.
    """

    _mask = None

    def __new__(
        cls,
        data,
        mask=None,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        copy=True,
        order='C',
        subok=False,
        ndmin=0,
        comm=None,
    ):

        # get a new Tod instance (or a subclass if subok is True)
        result = FitsArray.__new__(
            cls,
            data,
            header,
            unit,
            derived_units,
            dtype,
            copy,
            order,
            True,
            ndmin,
            comm,
        )
        if not subok and type(result) is not cls:
            result = result.view(cls)

        # mask attribute
        if mask is np.ma.nomask:
            mask = None

        if mask is None and isinstance(data, str):

            if comm is None:
                comm = MPI.COMM_SELF

            try:
                mask, junk = read_fits(data, 'mask', comm)
                mask = mask.view(np.bool8)
                copy = False
            except Exception:  # FIXME
                pass

        if mask is None and hasattr(data, 'mask') and data.mask is not np.ma.nomask:
            mask = data.mask

        if mask is not None:
            result._mask = np.array(mask, np.bool8, copy=copy)

        return result

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is None or mask is np.ma.nomask:
            self._mask = None
            return

        # enforce bool8 dtype
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, np.bool8)
        elif mask.dtype.type != np.bool8:
            if mask.dtype.itemsize == 1:
                mask = mask.view(np.bool8)
            else:
                mask = np.asarray(mask, np.bool8)

        # handle the scalar case
        if mask.ndim == 0:
            if self._mask is None:
                func = np.zeros if mask == 0 else np.ones
                self._mask = func(self.shape, dtype=np.bool8)
            else:
                self._mask[:] = mask
            return

        # check shape compatibility
        if self.shape != mask.shape:
            raise ValueError(
                "The input mask has a shape '"
                + str(mask.shape)
                + "' incompatible with that of the Tod '"
                + str(self.shape)
                + "'."
            )

        self._mask = mask

    def __array_finalize__(self, array):
        FitsArray.__array_finalize__(self, array)
        self._mask = getattr(array, 'mask', None)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if not isinstance(item, Tod):
            return item
        if item.mask is not None:
            item.mask = item.mask[key]
        return item

    def reshape(self, newdims, order='C'):
        result = np.ndarray.reshape(self, newdims, order=order)
        if self.mask is not None:
            result.mask = self.mask.reshape(newdims, order=order)
        result.derived_units = self.derived_units.copy()
        return result

    @classmethod
    def empty(
        cls,
        shape,
        mask=None,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            empty(shape, dtype, order),
            mask=mask,
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    @classmethod
    def ones(
        cls,
        shape,
        mask=None,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.ones(shape, dtype, order),
            mask=mask,
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    @classmethod
    def zeros(
        cls,
        shape,
        mask=None,
        header=None,
        unit=None,
        derived_units=None,
        dtype=None,
        order=None,
        **keywords,
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.zeros(shape, dtype, order),
            mask=mask,
            header=header,
            unit=unit,
            derived_units=derived_units,
            dtype=dtype,
            copy=False,
            **keywords,
        )

    def imshow(
        self,
        mask=None,
        xlabel='Sample',
        ylabel='Detector number',
        aspect='auto',
        origin='upper',
        percentile=0,
        **keywords,
    ):
        if mask is None:
            mask = self.mask
        return super().imshow(
            mask=mask,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect=aspect,
            origin=origin,
            percentile=percentile,
            **keywords,
        )

    def __str__(self):
        output = FitsArray.__str__(self)
        if self.ndim == 0:
            return output
        output += ' ('
        if self.ndim > 1:
            output += str(self.shape[-2]) + ' detector'
            if self.shape[-2] > 1:
                output += 's'
            output += ', '
        output += str(self.shape[-1]) + ' sample'
        if self.shape[-1] > 1:
            output += 's'
        return output + ')'

    def save(self, filename, comm=None):
        if comm is None:
            comm = MPI.COMM_SELF

        FitsArray.save(self, filename, comm=comm)
        if self.mask is not None:
            mask = self.mask.view('uint8')
            write_fits(filename, mask, None, True, 'Mask', comm)

    def median(self, axis=None):
        #        result = Tod(median(self, mask=self.mask, axis=axis),
        result = Tod(
            np.median(self, axis=axis),
            header=self.header.copy(),
            unit=self.unit,
            derived_units=self.derived_units,
            dtype=self.dtype,
            copy=False,
        )
        result.mask = np.zeros_like(result, bool)
        #        # median might introduce NaN from the mask, let's remove them
        #        tmf.processing.filter_nonfinite_mask_inplace(result.ravel(),
        #            result.mask.view(np.int8).ravel())
        return result

    median.__doc__ = np.median.__doc__

    def ravel(self, order='c'):
        mask = self.mask.ravel() if self.mask is not None else None
        return Tod(
            self.magnitude.ravel(),
            mask=mask,
            header=self.header.copy(),
            unit=self.unit,
            derived_units=self.derived_units,
            dtype=self.dtype,
            copy=False,
        )

    def sort(self, axis=-1, kind='quicksort', order=None):
        self.magnitude.sort(axis, kind, order)
        self.mask = None

    def _wrap_func(self, func, unit, *args, **kw):
        self_ma = np.ma.MaskedArray(self.magnitude, mask=self.mask, copy=False)
        output = func(self_ma, *args, **kw)
        if isinstance(output, np.ma.core.MaskedConstant):
            return type(self)(
                np.nan,
                mask=True,
                unit=unit,
                derived_units=self.derived_units,
                dtype=self.dtype,
            )
        if not isinstance(output, np.ndarray):
            return type(self)(
                output, unit=unit, derived_units=self.derived_units, dtype=self.dtype
            )
        result = output.view(type(self))
        result.__array_finalize__(self)
        result.mask = output.mask if output.mask is not np.ma.nomask else None
        result.unit = unit
        return result
