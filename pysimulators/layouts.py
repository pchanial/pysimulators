"""
Define Layout classes:

Layout:
  - LayoutSpatial
      - LayoutSpatialGrid
          - LayoutSpatialGridCircles
      - LayoutSpatialVertex
          - LayoutSpatialGridSquares
  - LayoutTemporal

"""
from __future__ import division

import copy
import functools
import inspect

try:
    from matplotlib import pyplot as mp
except:  # pragma: no coverage
    pass
import numpy as np
import types
from astropy.time import Time
from pyoperators.utils import (
    ilast_is_not,
    isalias,
    isclassattr,
    isscalarlike,
    product,
    split,
    tointtuple,
)

from .geometry import create_circle, create_grid, create_grid_squares
from .quantities import Quantity

__all__ = [
    'Layout',
    'LayoutSpatial',
    'LayoutSpatialVertex',
    'LayoutSpatialGrid',
    'LayoutSpatialGridCircles',
    'LayoutSpatialGridSquares',
    'LayoutTemporal',
]


class Layout(object):
    """
    The Layout class represents a set of elements. Characteristics of
    the components can transparently be accessed as packed or unpacked arrays.

    Example
    -------
    Let's consider a 3x3 array of detectors, in which the top-left detector
    does not working. We will define a mask to flag this detector.
    >>> selection = [[True, True, False],
    ...              [True, True, True],
    ...              [True, True, True]]
    >>> gain = [[1.0, 1.0, 1.0],
    ...         [0.9, 1.0, 1.0],
    ...         [0.8, 1.0, 1.0]]
    >>> layout = Layout((3, 3), selection=selection, gain=gain)
    >>> layout.plot()

    Only the values for the selected detectors are stored, in 1-dimensional
    arrays:
    >>> layout.gain
    array([ 1. ,  1. ,  0.9,  1. ,  1. ,  0.8,  1. ,  1. ])

    But the 2-dimensional layout can be recovered:
    >>> layout.all.gain
    array([[ 1. ,  1. ,  nan],
           [ 0.9,  1. ,  1. ],
           [ 0.8,  1. ,  1. ]])

    The number of selected detectors is:
    >>> len(layout)
    8

    and the number of all detectors is:
    >>> len(layout.all)
    9

    Now, let's have a more complex example: an array of detectors made of
    4 identical 3x3 subarrays in which one corner detector is blind and
    for which we will define an indexing scheme.
    The first subarray is placed on the upper right quadrant and the position
    of the other arrays is obtained by rotating the first array by 90, 180 and
    270 degrees.
    A natural indexing would be given by:

    >>> ordering = [[-1, 14, 17,  0,  1, -1],
    ...             [10, 13, 16,  3,  4,  5],
    ...             [ 9, 12, 15,  6,  7,  8],
    ...             [26, 25, 24, 33, 30, 27],
    ...             [23, 22, 21, 34, 31, 28],
    ...             [-1, 19, 18, 35, 32, -1]]

    The following mask only keeps the 2 subarrays on the left:

    >>> selection = [[ True,  True,  True, False, False, False],
    ...              [ True,  True,  True, False, False, False],
    ...              [ True,  True,  True, False, False, False],
    ...              [ True,  True,  True, False, False, False],
    ...              [ True,  True,  True, False, False, False],
    ...              [ True,  True,  True, False, False, False]]
    >>> layout = Layout((6, 6), selection=selection, ordering=ordering)

    Then, the numbering of the layout fields follows the list of selected
    indices stored in:

    >>> print(layout.index)
    [12  6 13  7  1 14  8  2 32 31 26 25 24 20 19 18]

    which are the 1d-collapsed indices of the following array coordinates:

    >>> print [(i // 6, i % 6) for i in layout.index]
    [(2, 0), (1, 0), (2, 1), (1, 1), (0, 1), (2, 2), (1, 2), (0, 2),
     (5, 2), (5, 1), (4, 2), (4, 1), (4, 0), (3, 2), (3, 1), (3, 0)]

    """

    def __init__(self, shape, selection=None, ordering=None, **keywords):
        """
        shape : tuple of int
            The layout unpacked shape. For 2-dimensional layouts, the shape would
            be (nrows, ncolumns).
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            layout is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.

        """
        object.__setattr__(self, 'shape', tointtuple(shape))
        object.__setattr__(self, 'ndim', len(self.shape))
        self._dtype_index = np.int64
        self._index = self._get_index(selection, ordering)
        for k, v in keywords.items():
            self._special_attributes += (k,)
            if v is None and isclassattr(k, type(self)):
                continue
            if not callable(v):
                v = self.pack(v, copy=True)
            setattr(self, k, v)

    _reserved_attributes = ('all', 'index', 'ndim', 'removed', 'shape')
    _special_attributes = ('index', 'removed')

    @property
    def all(self):
        """
        Give access to the unpacked attributes, without creating a cyclic
        reference.
        """
        return LayoutUnpacked(self)

    @property
    def index(self):
        """Return the index as an int array."""
        if self._index is Ellipsis:
            return np.arange(product(self.shape), dtype=self._dtype_index)
        if isinstance(self._index, slice):
            index = self._normalize_slice_no_none(self._index)
            return np.arange(
                index.start, index.stop, index.step, dtype=self._dtype_index
            )
        return self._index

    @property
    def removed(self):
        return np.array(False)

    def _get_index(self, selection, ordering):
        selection = self._normalize_selection(selection, self.shape)

        if ordering is not None:
            ordering = np.asarray(ordering)
            if ordering.shape != self.shape:
                raise ValueError('Invalid ordering dimensions.')

        if ordering is None:
            # assuming row major storage for the ordering
            if selection is Ellipsis or isinstance(selection, slice):
                return selection
            if isinstance(selection, np.ndarray) and selection.dtype == int:
                out = np.asarray(selection, self._dtype_index)
                return self._normalize_int_selection(out, product(self.shape))
            if isinstance(selection, tuple):
                s = selection[0]
                if isinstance(s, slice) and (
                    self.ndim == 1 or len(selection) == 1 and s.step == 1
                ):
                    n = product(self.shape[1:])
                    return slice(s.start * n, s.stop * n, s.step)
                selection = self._selection2bool(selection)

            npacked = np.sum(selection)
            if npacked == 0:
                return np.array([], self._dtype_index)
            out = np.asarray(
                np.argsort(selection.ravel())[-npacked:], self._dtype_index
            )
            return self._normalize_int_selection(out, product(self.shape))

        if selection is not Ellipsis and (
            not isinstance(selection, np.ndarray) or selection.dtype == int
        ):
            selection = self._selection2bool(selection)

        if selection is not Ellipsis:
            ordering = ordering.copy()
            ordering[~selection] = -1

        npacked = np.sum(ordering >= 0)
        if npacked == 0:
            return np.array([], self._dtype_index)
        out = np.asarray(np.argsort(ordering.ravel())[-npacked:], self._dtype_index)
        return self._normalize_int_selection(out, product(self.shape))

    def __len__(self):
        """Return the number of actual elements in the layout."""
        index = self._index
        if index is Ellipsis:
            return product(self.shape)
        if isinstance(index, slice):
            index = self._normalize_slice_no_none(index)
            return (index.stop - index.start) // index.step
        return index.size

    def __setattr__(self, key, value):
        if key in self._reserved_attributes:
            raise AttributeError('The attribute {0!r} is not writeable.'.format(key))
        if (
            value is not None
            and not callable(value)
            and key in self._special_attributes
        ):
            value = np.asanyarray(value)
            if value.ndim > 0 and value.shape[0] != len(self):
                raise ValueError(
                    "The shape '{0}' is invalid. The expected first dimension "
                    "is '{1}'.".format(value.shape, len(self))
                )
        object.__setattr__(self, key, value)

    def __getattribute__(self, key):
        value = object.__getattribute__(self, key)
        if key in object.__getattribute__(self, '_special_attributes') and callable(
            value
        ):
            spec = inspect.getargspec(value)
            nargs = len(spec.args)
            if spec.defaults is not None:
                nargs -= len(spec.defaults)
            if isinstance(value, types.MethodType):
                nargs -= 1
            if nargs == 0:
                return value()
            if nargs == 1:
                return value(self)
            else:
                return functools.partial(value, self)
        return value

    def __getitem__(self, selection):
        selection = self._normalize_selection(selection, (len(self),))
        if selection is Ellipsis:
            return self

        index = self._index
        if index is Ellipsis and (
            isinstance(selection, slice)
            or isinstance(selection, np.ndarray)
            and selection.dtype == int
        ):
            index = selection
        elif isinstance(index, slice) and not isinstance(selection, slice):
            if selection.stop is None:
                n = (selection.start - len(self)) // selection.step
                sstop = selection.start + n * selection.step
            else:
                sstop = selection.stop
            start = index.start + selection.start * index.step
            n = (sstop - selection.start) // selection.step
            stop = start + n * index.step
            if stop < 0:
                stop = None
            index = slice(start, stop, index.step * selection.step)
        else:
            index = self.index
            index = index[selection]

        if isinstance(index, np.ndarray):
            index = self._normalize_int_selection(index, product(self.shape))

        out = copy.copy(self)
        out._index = index
        for k in out._special_attributes:
            if k == 'index':
                continue
            try:
                v = self.__dict__[k]
            except KeyError:
                continue
            if isinstance(v, np.ndarray) and v.ndim > 0:
                setattr(out, k, v[selection])
        return out

    def __str__(self):
        out = type(self).__name__ + '({0}, '.format(self.shape)
        attributes = ['index'] + sorted(
            _ for _ in self._special_attributes if _ not in ('index', 'removed')
        )
        if len(attributes) > 1:
            out += '\n    '
        out += (
            ',\n    '.join('{0}={1}'.format(_, getattr(self, _)) for _ in attributes)
            + ')'
        )
        return out

    __repr__ = __str__

    def pack(self, x, out=None, copy=False):
        """
        Convert a multi-dimensional array into a 1-dimensional array which
        only includes the selected components, potentially ordered according
        to a given ordering.

        Parameters
        ----------
        x : ndarray
            Array to be packed, whose first dimensions are equal to those
            of the layout.
        out : ndarray, optional
            Placeholder for the packed array.
        copy : boolean, optional
            Setting this keyword to True ensures that the output is not a view
            of the input x.

        Returns
        -------
        output : ndarray
            Packed array, whose first dimension is the number of non-removed
            components.

        See Also
        --------
        unpack : inverse method.

        Notes
        -----
        Unless the 'copy' keyword is set to True, this method does not make
        a copy if it simply does a reshape (when all components are selected
        and no ordering is specified).

        """
        if x is None:
            return None
        x = np.asanyarray(x)
        if x.ndim == 0:
            return x
        if x.shape[: self.ndim] != self.shape:
            raise ValueError(
                "Invalid unpacked shape '{0}'. The first dimension(s) must be "
                "'{1}'.".format(x.shape, self.shape)
            )
        packed_shape = (len(self),) + x.shape[self.ndim :]
        flat_shape = (product(self.shape),) + x.shape[self.ndim :]
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != packed_shape:
                raise ValueError(
                    "The output array shape '{0}' is invalid. The expected sha"
                    "pe is '{1}'.".format(out.shape, packed_shape)
                )
        else:
            if not copy:
                if self._index is Ellipsis:
                    return x.reshape(packed_shape)
                elif isinstance(self._index, slice):
                    return x.reshape(flat_shape)[self._index]
            out = np.empty(packed_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        if self._index is Ellipsis:
            out[...] = x.reshape(packed_shape)
            return out
        x_ = x.view(np.ndarray).reshape(flat_shape)
        if isinstance(self._index, slice):
            out[...] = x_[self._index]
        else:
            np.take(x_, self._index, axis=0, out=out)
        return out

    def split(self, n):
        """
        Split the layout in partitioning groups.

        Example
        -------
        >>> layout = Layout((4, 4), selection=[0, 1, 4, 5])
        >>> print(layout.split(2))
        (Layout((4, 4), index=slice(0, 2, 1)),
         Layout((4, 4), index=slice(4, 6, 1)))

        """
        return tuple(self[_] for _ in split(len(self), n))

    def unpack(self, x, out=None, missing_value=None, copy=False):
        """
        Convert a 1-dimensional array into a multi-dimensional array which
        includes the non-selected components, mimicking the multi-dimensional
        layout.

        Parameters
        ----------
        x : ndarray
            Array to be unpacked, whose first dimension is the number of
            selected components.
        out : ndarray, optional
            Placeholder for the unpacked array.
        missing_value : any, optional
            The value to be used for non-selected components.
        copy : boolean, optional
            Setting this keyword to True ensures that the output is not a view
            of the input x.

        Returns
        -------
        output : ndarray
            Unpacked array, whose first dimensions are equal to those of the
            layout.

        See Also
        --------
        pack : inverse method.

        Notes
        -----
        Unless the 'copy' keyword is set to True, this method does not make
        a copy if it simply does a reshape (when all components are selected
        and no ordering is specified).

        """
        if x is None:
            return None
        x = np.array(x, copy=False, ndmin=1, subok=True)
        if x.shape[0] not in (1, len(self)):
            raise ValueError(
                "Invalid input packed shape '{0}'. The expected first dimensio"
                "n is '{1}'.".format(x.shape, len(self))
            )
        unpacked_shape = self.shape + x.shape[1:]
        flat_shape = (product(self.shape),) + x.shape[1:]
        has_out = out is not None
        if has_out:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != unpacked_shape:
                raise ValueError(
                    "The output array shape '{0}' is invalid. The expected sha"
                    "pe is '{1}'.".format(out.shape, unpacked_shape)
                )
        else:
            if self._index is Ellipsis and not copy and x.shape[0] == len(self):
                return x.reshape(unpacked_shape)
            out = np.empty(unpacked_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        if product(self.shape) > len(self):
            if missing_value is None:
                missing_value = self._get_default_missing_value(x.dtype)
            out[...] = missing_value
        elif self._index is Ellipsis and x.shape[0] == len(self):
            out[...] = x.reshape(unpacked_shape)
            return out
        out_ = out.reshape(flat_shape)
        out_[self._index] = x
        if has_out and not isalias(out, out_):
            out[...] = out_.reshape(out.shape)
        return out

    def _get_default_missing_value(self, dtype):
        value = np.empty((), dtype)
        if dtype.kind == 'V':
            for f, (dt, junk) in dtype.fields.items():
                value[f] = self._get_default_missing_value(dt)
            return value
        return {
            'b': False,
            'i': -1,
            'u': 0,
            'f': np.nan,
            'c': np.nan,
            'S': '',
            'U': u'',
            'O': None,
        }[dtype.kind]

    def _normalize_selection(self, selection, shape):
        # return the selection as an array of int or bool, a slice, an Ellipsis
        # or a tuple.
        if isinstance(selection, list) and len(selection) == 0:
            return np.array([], self._dtype_index)
        if (
            selection is None
            or selection is Ellipsis
            or isinstance(selection, tuple)
            and len(selection) == 0
        ):
            return Ellipsis
        if isscalarlike(selection):
            return np.asarray([selection], self._dtype_index)
        if isinstance(selection, slice):
            return self._normalize_slice(selection, product(shape))
        if isinstance(selection, (list, tuple)):
            selection_ = np.asarray(selection)
            if selection_.dtype == object:
                if len(selection) > len(shape):
                    raise ValueError('Invalid selection dimensions.')
                selection = tuple(
                    self._normalize_slice(_, s) if isinstance(_, slice) else _
                    for _, s in zip(selection, shape)
                )
                try:
                    return selection[: ilast_is_not(selection, Ellipsis) + 1]
                except ValueError:
                    return Ellipsis
            else:
                selection = selection_
        elif not isinstance(selection, np.ndarray):
            raise TypeError('Invalid selection.')

        if selection.dtype not in (bool, int):
            raise TypeError('Invalid selection.')
        if selection.dtype == int:
            if selection.ndim != 1:
                raise ValueError('Index selection is not 1-dimensional.')
        elif selection.shape != shape:
            raise ValueError('Invalid boolean selection dimensions.')

        return selection

    def _normalize_int_selection(self, s, n):
        # all elements in the integer selection should be positive
        size = s.size
        if size <= 1:
            return s
        start = s[0]
        step = s[1] - s[0]
        if np.all(s == start + np.arange(size) * step):
            if size == n and step == 1:
                return Ellipsis
            else:
                stop = start + size * step
                if stop < 0:
                    stop = None
                return slice(start, stop, step)
        return s

    def _normalize_slice(self, s, n):
        step = 1 if s.step is None else s.step
        if step == 0:
            raise ValueError('Invalide stride.')
        start = s.start
        if start is None:
            start = 0 if step > 0 else n - 1
        elif start < 0:
            start = n + start
        elif start > n:
            start = n
        stop = s.stop
        if stop is None:
            if step > 0:
                stop = n
        elif stop < 0:
            stop = n + stop
        elif stop > n:
            stop = n
        if step > 0:
            start = min(start, n)
            stop = min(stop, start + n * step)
        else:
            start = min(start, n - 1)
            if stop is not None:
                stop = max(stop, stop + n * step)
        if start == 0 and stop == n and step == 1:
            return Ellipsis
        return slice(start, stop, step)

    def _normalize_slice_no_none(self, s):
        if s.stop is not None:
            return s
        return slice(s.start, -1, s.step)

    def _selection2bool(self, selection):
        out = np.zeros(self.shape, bool)
        out[selection] = True
        return out


class LayoutUnpacked(object):
    def __init__(self, packed):
        object.__setattr__(self, '_packed', packed)
        for k in packed._special_attributes:
            if k == 'removed':
                continue
            object.__setattr__(self, k, None)  # for IPython autocompletion

    @property
    def removed(self):
        out = self._packed.unpack(False, missing_value=True)
        out.flags.writeable = False
        return out

    def __len__(self):
        return product(self._packed.shape)

    def __getattribute__(self, key):
        if key.startswith('_') or key == 'removed':
            return object.__getattribute__(self, key)
        if key not in self._packed._special_attributes:
            raise AttributeError(
                'The unpacked layout has no attribute {0!r}.'.format(key)
            )
        v = getattr(self._packed, key)
        if v is None:
            return v
        if callable(v):
            return lambda *args: self._packed.unpack(v(*args))
        out = self._packed.unpack(v).view()
        out.flags.writeable = False
        return out

    #    def __getitem__(self, selection):
    #        shape = self._packed.shape
    #        selection = self._packed._normalize_selection(selection, shape)
    #        if selection is Ellipsis:
    #            return self
    #        selection = self._packed._selection2bool(selection)
    #        index = self._packed.index
    #        return self._packed[selection[index]]

    def __setattr__(self, key, value):
        p = self._packed
        if key not in p._special_attributes:
            raise AttributeError('{0!r} is not an unpacked attribute.'.format(key))
        if key in p._reserved_attributes:
            raise AttributeError('The attribute {0!r} is not writeable.'.format(key))
        if callable(value):
            raise TypeError('A function cannot be set as an unpacked attribute.')
        object.__setattr__(p, key, p.pack(value, copy=True))


class LayoutSpatial(Layout):
    """
    A class for spatial layout, handling position of components.

    Attributes
    ----------
    shape : tuple of integers
        Tuple containing the layout dimensions
    ndim : int
        The number of layout dimensions.
    center : array of shape (ncomponents, 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    all : object
        Access to the unpacked attributes.

    """

    def __init__(self, shape, **keywords):
        """
        shape : tuple of int
            The layout unpacked shape. For 2-dimensional layouts, the shape would
            be (nrows, ncolumns).
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            layout is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.
        center : array-like, Quantity, optional
            The position of the components. For 2-dimensional layouts, its shape
            must be (nrows, ncolumns, 2). If not provided, these positions are
            derived from the vertex keyword.

        """
        if hasattr(self, 'center'):
            keywords['center'] = None
        elif keywords.get('center', None) is None:
            raise ValueError('The spatial layout is not defined.')
        Layout.__init__(self, shape, **keywords)

    def plot(self, transform=None, **keywords):
        """
        Plot the layout.

        Parameters
        ----------
        transform : function, Operator
            Operator to be used to transform the layout coordinates into
            the data coordinate system.

        """
        self._plot(self.center, transform, **keywords)

    def _plot(self, coords, transform, **keywords):
        if transform is not None:
            coords = transform(coords)

        if coords.ndim == 3:
            a = mp.gca()
            for p in coords:
                a.add_patch(mp.Polygon(p, closed=True, fill=False, **keywords))
            a.autoscale_view()
        elif coords.ndim == 2:
            if 'color' not in keywords:
                keywords['color'] = 'black'
            if 'marker' not in keywords:
                keywords['marker'] = 'o'
            if 'linestyle' not in keywords:
                keywords['linestyle'] = ''
            mp.plot(coords[:, 0], coords[:, 1], **keywords)
        else:
            raise ValueError('Invalid number of dimensions.')

        mp.show()


class LayoutSpatialGrid(LayoutSpatial):
    """
    The components are laid out in a rectangular grid (nrows, ncolumns).
    Their positions are expressed in the (X, Y) referential.

       Y ^
         |
         +--> X

    Before rotation, rows increase along the -Y axis (unless yreflection is
    set to True) and columns increase along the X axis (unless xreflection
    is set to True).

    Example
    -------
    >>> spacing = 0.001
    >>> layout = LayoutSpatialGrid((16, 16), spacing)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        Tuple containing the number of rows and columns of the grid.
    ndim : int
        The number of layout dimensions.
    center : array of shape (len(self), 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    all : object
        Access to the unpacked attributes of shape (nrows, ncolumns, ...)

    """

    def __init__(
        self,
        shape,
        spacing,
        xreflection=False,
        yreflection=False,
        angle=0,
        origin=(0, 0),
        startswith1=False,
        **keywords,
    ):
        """
        shape : tuple of two integers (nrows, ncolumns)
            Number of rows and columns of the grid.
        spacing : float, Quantity
            Physical spacing between components.
        xreflection : boolean, optional
            Reflection along the X-axis (before rotation).
        yreflection : boolean, optional
            Reflection along the Y-axis (before rotation).
        angle : float, optional
            Counter-clockwise rotation angle in degrees (before translation).
        origin : array-like of shape (2,), optional
            The (X, Y) coordinates of the grid center
        startswith1 : boolean, optional
            If True, start column and row indewing with one.
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            layout is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.

        """
        if not isinstance(shape, (list, tuple)):
            raise TypeError('Invalid grid shape.')
        if len(shape) != 2:
            raise ValueError('The layout grid is not 2-dimensional.')

        unit = getattr(spacing, 'unit', '') or getattr(origin, 'unit', '')
        if len(origin) != len(shape):
            raise ValueError('Invalid dimensions of the layout center.')
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            origin = Quantity(origin).tounit(unit)
        center = create_grid(
            shape,
            spacing,
            xreflection=xreflection,
            yreflection=yreflection,
            center=origin,
            angle=angle,
        )
        if unit:
            center = Quantity(center, unit, copy=False)
        self.spacing = spacing
        self.origin = origin
        self.xreflection = xreflection
        self.yreflection = yreflection
        self.angle = angle
        self.startswith1 = startswith1
        self.unit = unit
        LayoutSpatial.__init__(self, shape, center=center, **keywords)
        self._special_attributes += ('column', 'row')

    def column(self):
        return self.index % self.shape[1] + self.startswith1

    def row(self):
        return self.index // self.shape[1] + self.startswith1


class LayoutSpatialGridCircles(LayoutSpatialGrid):
    """
    The circle components are laid out in a rectangular grid (nrows, ncolumns).
    Their positions are expressed in the (X, Y) referential.

       Y ^
         |
         +--> X

    Before rotation, rows increase along the -Y axis (unless yreflection is
    set to True) and columns increase along the X axis (unless xreflection
    is set to True).

    Example
    -------
    >>> spacing = 0.001
    >>> layout = LayoutSpatialGridCircles((16, 16), spacing)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        Tuple containing the number of rows and columns of the grid.
    ndim : int
        The number of layout dimensions.
    center : array of shape (len(self), 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    all : object
        Access to the unpacked attributes of shape (nrows, ncolumns, ...)

    """

    def __init__(self, shape, spacing, radius=None, **keywords):
        """
        shape : tuple of two integers (nrows, ncolumns)
            Number of rows and columns of the grid.
        spacing : float, Quantity
            Physical spacing between components.
        radius : array-like, float
            The component radii.
        xreflection : boolean
            Reflection along the X-axis (before rotation).
        yreflection : boolean
            Reflection along the Y-axis (before rotation).
        angle : float
            Counter-clockwise rotation angle in degrees (before translation).
        origin : array-like of shape (2,)
            The (X, Y) coordinates of the grid center
        startswith1 : boolean, optional
            If True, start column and row indewing with one.
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            layout is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.

        """
        if radius is None:
            radius = spacing / 2
        unit = getattr(spacing, 'unit', '') or getattr(radius, 'unit', '')
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            radius = Quantity(radius, copy=False).tounit(unit)
        LayoutSpatialGrid.__init__(self, shape, spacing, radius=radius, **keywords)

    def plot(self, transform=None, **keywords):
        coords = create_circle(self.radius, center=self.center)
        self._plot(coords, transform, **keywords)

    plot.__doc__ = LayoutSpatialGrid.plot.__doc__


class LayoutSpatialVertex(LayoutSpatial):
    def __init__(self, shape, nvertices, **keywords):
        """
        vertex : array-like of shape self.shape + (nvertices, 2), optional
            The vertices of the components. For 2-dimensional layouts of squares,
            its shape must be (nrows, ncolumns, 4, 2)
        """
        if hasattr(self, 'vertex'):
            keywords['vertex'] = None
        elif keywords.get('vertex', None) is None:
            raise ValueError('The layout vertices are not defined.')
        LayoutSpatial.__init__(self, shape, **keywords)
        self.nvertices = nvertices

    @property
    def center(self):
        try:
            return np.mean(self.vertex, axis=-2)
        except:
            return None

    def plot(self, transform=None, **keywords):
        self._plot(self.vertex, transform, **keywords)

    plot.__doc__ = LayoutSpatialGrid.plot.__doc__


class LayoutSpatialGridSquares(LayoutSpatialVertex):
    """
    The square components are laid out in a rectangular grid (nrows, ncolumns).
    The positions of their corners are expressed in the (X, Y) referential.

       Y ^
         |
         +--> X

    Before rotation, rows increase along the -Y axis (unless yreflection is
    set to True) and columns increase along the X axis (unless xreflection
    is set to True).

    Example
    -------
    >>> spacing = 0.001
    >>> layout = LayoutSpatialGridSquares((8, 8), spacing, filling_factor=0.8)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        See below.
    ndim : int
        The number of layout dimensions.
    center : array of shape (len(self), 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    vertex : array of shape (len(self), 4, 2)
        Corners of the components. The dimensions refers to the component
        number, the corner number counted counter-clockwise starting from
        the top-right and the (X, Y) coordinates.
    all : object
        Access to the unpacked attributes of shape (nrows, ncolumns, ...)

    """

    def __init__(
        self,
        shape,
        spacing,
        filling_factor=1,
        xreflection=False,
        yreflection=False,
        angle=0,
        origin=(0, 0),
        startswith1=False,
        **keywords,
    ):
        """
        shape : tuple of two integers (nrows, ncolumns)
            Number of rows and columns of the grid.
        spacing : float, Quantity
            Physical spacing between components.
        filling_factor : float
            Ratio of the component area over spacing**2.
        xreflection : boolean
            Reflection along the X-axis (before rotation).
        yreflection : boolean
            Reflection along the Y-axis (before rotation).
        angle : float
            Counter-clockwise rotation angle in degrees (before translation).
        origin : array-like of shape (2,)
            The (X, Y) coordinates of the grid center
        startswith1 : boolean, optional
            If True, start column and row indewing with one.
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            layout is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.

        """
        unit = getattr(spacing, 'unit', '') or getattr(origin, 'unit', '')
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            origin = Quantity(origin).tounit(unit)
        vertex = create_grid_squares(
            shape,
            spacing,
            filling_factor=filling_factor,
            xreflection=xreflection,
            yreflection=yreflection,
            center=origin,
            angle=angle,
        )
        if unit:
            vertex = Quantity(vertex, unit, copy=False)
        LayoutSpatialVertex.__init__(self, shape, 4, vertex=vertex, **keywords)
        self.angle = angle
        self.filling_factor = filling_factor
        self.origin = origin
        self.spacing = spacing
        self.startswith1 = startswith1
        self.unit = unit
        self.xreflection = xreflection
        self.yreflection = yreflection


class LayoutTemporal(Layout):
    DEFAULT_DATE_OBS = '2000-01-01'
    DEFAULT_SAMPLING_PERIOD = 1

    def __init__(self, n, date_obs=None, sampling_period=None, **keywords):
        if date_obs is None:
            date_obs = self.DEFAULT_DATE_OBS
        if isinstance(date_obs, str):
            # XXX astropy.time bug needs []
            date_obs = Time([date_obs], scale='utc')
        elif not isinstance(date_obs, Time):
            raise TypeError('The observation start date is invalid.')
        elif date_obs.is_scalar:  # work around astropy.time bug
            date_obs = Time([str(date_obs)], scale='utc')
        if sampling_period is None:
            if hasattr(keywords, 'time'):
                sampling_period = np.median(np.diff(keywords['time']))
            else:
                sampling_period = self.DEFAULT_SAMPLING_PERIOD
        Layout.__init__(self, n, time=keywords.get('time', None), **keywords)
        self.date_obs = date_obs
        self.sampling_period = float(sampling_period)

    def time(self):
        return self.index * self.sampling_period
