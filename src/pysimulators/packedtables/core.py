"""
Define PackedTable classes:

PackedTable:
  - Layout
      - LayoutGrid
          - LayoutGridSquare
  - Sampling
      - Pointing
          - PointingEquatorial
          - PointingHorizontal
  - Scene
      - SceneGrid
      - SceneHealpix

"""

import copy
import functools
import inspect
import types
import warnings
from collections.abc import Callable

import numpy as np

from pyoperators import MPI
from pyoperators.utils import (
    ilast_is_not,
    isalias,
    isclassattr,
    isscalarlike,
    product,
    split,
    tointtuple,
)

from ..warnings import PySimulatorsWarning

__all__ = ['PackedTable']


class PackedTable:
    """
    The PackedTable class gathers information from a set of elements which can
    have a multi-dimensional layout. This information can transparently be
    accessed as packed or unpacked arrays.

    Example
    -------
    Let's consider a 3x3 array of detectors, in which the top-left detector
    is not working. We will define a mask to flag this detector.
    >>> selection = [[True, True, False],
    ...              [True, True, True],
    ...              [True, True, True]]
    >>> gain = [[1.0, 1.2, 1.5],
    ...         [0.9, 1.0, 1.0],
    ...         [0.8, 1.0, 1.0]]
    >>> table = PackedTable((3, 3), selection=selection, gain=gain)

    Only the values for the selected detectors are stored, in 1-dimensional
    arrays:
    >>> table.gain
    array([ 1. ,  1.2,  0.9,  1. ,  1. ,  0.8,  1. ,  1. ])

    But the 2-dimensional table can be recovered:
    >>> table.all.gain
    array([[ 1. ,  1.2,  nan],
           [ 0.9,  1. ,  1. ],
           [ 0.8,  1. ,  1. ]])

    The number of selected detectors is:
    >>> len(table)
    8

    and the number of all detectors is:
    >>> len(table.all)
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
    >>> table = PackedTable((6, 6), selection=selection, ordering=ordering)

    Then, the numbering of the table fields follows the list of selected
    indices stored in:

    >>> print(table.index)
    [12  6 13  7  1 14  8  2 32 31 26 25 24 20 19 18]

    which are the 1d-collapsed indices of the following array coordinates:

    >>> print([(i // 6, i % 6) for i in table.index])
    [(2, 0), (1, 0), (2, 1), (1, 1), (0, 1), (2, 2), (1, 2), (0, 2),
     (5, 2), (5, 1), (4, 2), (4, 1), (4, 0), (3, 2), (3, 1), (3, 0)]

    """

    def __init__(self, shape, selection=None, ordering=None, ndim=None, **keywords):
        """
        shape : tuple of int
            The shape of the unpacked table attributes. For 2-dimensional
            attributes, the shape would be (nrows, ncolumns).
        ndim : int, optional
            The number of splittable (indexable) dimensions. It is the actual
            number of dimensions of the layout. It can be lower than that
            specified by the layout shape, in which case the extra dimensions
            are instructed not to be split.
        selection : array-like of bool or int, slices, optional
            The slices or the integer or boolean selection that specifies
            the selected components (and reject those that are not physically
            present or those not handled by the current MPI process when the
            table is distributed in a parallel processing).
        ordering : array-like of int, optional
            The values in this array specify an ordering of the components. It is
            used to define the 1-dimensional indexing of the packed components.
            A negative value means that the component is removed.

        """
        shape = tointtuple(shape)
        if ndim is None:
            ndim = len(shape)
        elif ndim < 1 or ndim > len(shape):
            raise ValueError(f"Invalid ndim value '{ndim}'.")
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'ndim', ndim)
        object.__setattr__(self, 'comm', MPI.COMM_SELF)
        self._dtype_index = np.int64
        self._index = self._get_index(selection, ordering)

        for k, v in keywords.items():
            self._special_attributes += (k,)
            if v is None and isclassattr(k, type(self)):
                continue
            if not isinstance(v, Callable):
                v = self.pack(v, copy=True)
            setattr(self, k, v)

    _reserved_attributes = ('all', 'comm', 'index', 'ndim', 'removed', 'shape')
    _special_attributes = ('index', 'removed')

    @property
    def all(self):
        """
        Give access to the unpacked attributes, without creating a cyclic
        reference.
        """
        return UnpackedTable(self)

    @property
    def index(self):
        """Return the index as an int array."""
        index = self._index
        if index is Ellipsis:
            return np.arange(self._size_actual, dtype=self._dtype_index)
        if isinstance(index, slice):
            return np.arange(
                index.start, index.stop, index.step, dtype=self._dtype_index
            )
        return index

    @property
    def _indexable(self):
        # slice index with negative stop values cannot be used for indexing
        index = self._index
        if isinstance(index, slice) and index.stop < 0:
            return slice(index.start, None, index.step)
        return index

    @property
    def removed(self):
        return np.array(False)

    @property
    def _shape_actual(self):
        return self.shape[: self.ndim]

    @property
    def _size_actual(self):
        return product(self._shape_actual)

    def _get_index(self, selection, ordering):
        selection = self._normalize_selection(selection, self._shape_actual)

        if ordering is not None:
            ordering = np.asarray(ordering)
            if ordering.shape != self._shape_actual:
                raise ValueError('Invalid ordering dimensions.')

        if ordering is None:
            # assuming row major storage for the ordering
            if selection is Ellipsis or isinstance(selection, slice):
                return selection
            if isinstance(selection, np.ndarray) and selection.dtype == int:
                out = np.asarray(selection, self._dtype_index)
                return self._normalize_int_selection(out, self._size_actual)
            if isinstance(selection, tuple):
                s = selection[0]
                if isinstance(s, slice) and (
                    self.ndim == 1 or len(selection) == 1 and s.step == 1
                ):
                    n = product(self._shape_actual[1:])
                    return slice(s.start * n, s.stop * n, s.step)
                selection = self._selection2bool(selection)

            return self._normalize_int_selection(
                np.asarray(np.where(selection.ravel())[0], dtype=self._dtype_index),
                self._size_actual,
            )

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
        out = np.asarray(
            np.argsort(ordering.ravel(), kind='mergesort')[-npacked:], self._dtype_index
        )
        return self._normalize_int_selection(out, self._size_actual)

    def __len__(self):
        """Return the number of actual elements in the table."""
        index = self._index
        if index is Ellipsis:
            return self._size_actual
        if isinstance(index, slice):
            return (index.stop - index.start) // index.step
        return index.size

    def __setattr__(self, key, value):
        if key in self._reserved_attributes:
            raise AttributeError(f'The attribute {key!r} is not writeable.')
        if (
            value is not None
            and not isinstance(value, Callable)
            and key in self._special_attributes
        ):
            value = np.asanyarray(value)
            if value.ndim > 0 and value.shape[0] != len(self):
                raise ValueError(
                    f"The shape '{value.shape}' is invalid. The expected first "
                    f"dimension is '{len(self)}'."
                )
            elif value.ndim == 0:
                try:
                    old_value = object.__getattribute__(self, key)
                    if old_value is not None and not isinstance(old_value, Callable):
                        old_value[...] = value
                        return
                except AttributeError:
                    pass
        object.__setattr__(self, key, value)

    def __getattribute__(self, key):
        if key == 'packed':
            warnings.warn(
                "Please update your code: the 'packed' attribute is not requi"
                'red anymore: the attributes are already packed. To access th'
                "e unpacked ones, use the 'all' attribute.",
                PySimulatorsWarning,
            )
            return self
        value = object.__getattribute__(self, key)
        if key not in object.__getattribute__(
            self, '_special_attributes'
        ) or not isinstance(value, Callable):
            return value
        spec = inspect.getfullargspec(value)
        nargs = len(spec.args)
        if spec.defaults is not None:
            nargs -= len(spec.defaults)
        if isinstance(value, types.MethodType):
            if nargs == 1:
                return value()
            return value
        if nargs == 0:
            return value()
        if nargs == 1:
            return value(self)
        return functools.partial(value, self)

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
        elif isinstance(index, slice) and isinstance(selection, slice):
            n = (selection.stop - selection.start) // selection.step
            start = index.start + selection.start * index.step
            step = index.step * selection.step
            stop = start + n * step
            index = slice(start, stop, step)
        else:
            index = self.index
            index = index[selection]

        if isinstance(index, np.ndarray):
            index = self._normalize_int_selection(index, self._size_actual)

        out = copy.copy(self)
        out._index = index
        for k in out._special_attributes:
            if k in out._reserved_attributes:
                continue
            try:
                v = self.__dict__[k]
            except KeyError:
                continue
            if isinstance(v, np.ndarray) and v.ndim > 0:
                setattr(out, k, v[selection])
        return out

    def __str__(self):
        out = type(self).__name__ + f'({self._shape_actual}, '
        attributes = ['index'] + sorted(
            _
            for _ in self._special_attributes
            if _ not in ('index', 'removed')
            and not isinstance(
                object.__getattribute__(self, _),
                (types.FunctionType, types.MethodType, type(None)),
            )
        )
        if len(attributes) > 1:
            out += '\n    '
        out += (
            ',\n    '.join(f'{_}={str(getattr(self, _))[:65]}' for _ in attributes)
            + ')'
        )
        return out

    __repr__ = __str__

    def copy(self):
        """
        Return a deep copy of the packed array.

        """
        out = copy.deepcopy(self)
        out.comm = self.comm.Dup()
        return out

    def pack(self, x, out=None, copy=False):
        """
        Convert a multi-dimensional array into a 1-dimensional array which
        only includes the selected components, potentially ordered according
        to a given ordering.

        Parameters
        ----------
        x : ndarray
            Array to be packed, whose first dimensions are equal to those
            of the table attributes.
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
        if x.shape[: self.ndim] != self._shape_actual:
            raise ValueError(
                f"Invalid unpacked shape '{x.shape}'. The first dimension(s) must be "
                f"'{self._shape_actual}'."
            )
        index = self._indexable

        packed_shape = (len(self),) + x.shape[self.ndim :]
        flat_shape = (self._size_actual,) + x.shape[self.ndim :]
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != packed_shape:
                raise ValueError(
                    f"The output array shape '{out.shape}' is invalid. The expected "
                    f"shape is '{packed_shape}'."
                )
        else:
            if not copy:
                if index is Ellipsis:
                    return x.reshape(packed_shape)
                elif isinstance(index, slice):
                    return x.reshape(flat_shape)[index]
            out = np.empty(packed_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        if index is Ellipsis:
            out[...] = x.reshape(packed_shape)
            return out
        x_ = x.view(np.ndarray).reshape(flat_shape)
        if isinstance(index, slice):
            out[...] = x_[index]
        else:
            np.take(x_, index, axis=0, out=out)
        return out

    def scatter(self, comm=None):
        """
        MPI-scatter of the table.

        Parameter
        ---------
        comm : MPI.Comm
            The MPI communicator of the group of processes in which the table
            will be scattered.

        """
        if self.comm.size > 1:
            raise ValueError('The table is already distributed.')
        if comm is None:
            comm = MPI.COMM_WORLD
        if comm.size == 1:
            return self

        selection = split(len(self), comm.size, comm.rank)
        out = self[selection]
        for k in out._special_attributes:
            if k in out._reserved_attributes:
                continue
            try:
                v = out.__dict__[k]
            except KeyError:
                continue
            if isinstance(v, np.ndarray) and v.ndim > 0:
                setattr(out, k, v.copy())
        object.__setattr__(out, 'comm', comm)
        return out

    def gather(self, *args):
        """
        MPI-gather the (already scattered) table or a given array.

        table_global = table_local.gather()
        array_global = table_local.gather(array_local)

        Parameter
        ---------
        array_local : array-like, optional
            If provided, gather the scattered input array, instead of the whole
            table.

        Returns
        -------
        table_global : PackedTable
            The global packed table, whose all special attribute have been
            MPI-gathered.
        array_global : array
            The MPI-gathered input array.

        """

        def func(x):
            x = np.asarray(x)
            out = np.empty((ntot,) + x.shape[1:], x.dtype)
            nbytes = product(x.shape[1:]) * x.itemsize
            self.comm.Allgatherv(
                x.view(np.byte),
                [
                    out.view(np.byte),
                    ([_ * nbytes for _ in counts], [_ * nbytes for _ in offsets]),
                ],
            )
            return out

        ntot = np.array(len(self))
        self.comm.Allreduce(MPI.IN_PLACE, ntot, op=MPI.SUM)
        counts = []
        offsets = [0]
        for s in split(ntot, self.comm.size):
            n = s.stop - s.start
            counts.append(n)
            offsets.append(offsets[-1] + n)
        offsets.pop()

        if len(args) == 1:
            return func(args[0])
        elif len(args) > 1:
            raise TypeError(f'gather takes at most 1 argument ({len(args)} given)')

        out = copy.copy(self)
        out._index = self._normalize_int_selection(
            func(self.index), product(self.shape[: self.ndim])
        )
        for k in out._special_attributes:
            if k in out._reserved_attributes:
                continue
            try:
                v = self.__dict__[k]
            except KeyError:
                continue
            if isinstance(v, np.ndarray) and v.ndim > 0:
                setattr(out, k, func(v))
        return out

    def split(self, n):
        """
        Split the table in partitioning groups.

        Example
        -------
        >>> table = PackedTable((4, 4), selection=[0, 1, 4, 5])
        >>> print(table.split(2))
        (PackedTable((4, 4), index=slice(0, 2, 1)),
         PackedTable((4, 4), index=slice(4, 6, 1)))

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
            table attributes.

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
                f"Invalid input packed shape '{x.shape}'. The expected first dimension "
                f"is '{len(self)}'."
            )
        index = self._indexable

        unpacked_shape = self._shape_actual + x.shape[1:]
        flat_shape = (self._size_actual,) + x.shape[1:]
        has_out = out is not None
        if has_out:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != unpacked_shape:
                raise ValueError(
                    f"The output array shape '{out.shape}' is invalid. The expected "
                    f"shape is '{unpacked_shape}'."
                )
        else:
            if (
                index is Ellipsis
                and not copy
                and x.shape[0] == len(self)
                and self.comm.size == 1
            ):
                return x.reshape(unpacked_shape)
            out = np.empty(unpacked_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        if self._size_actual > len(self):
            if missing_value is None:
                missing_value = self._get_default_missing_value(x.dtype)
            out[...] = missing_value
        elif index is Ellipsis and x.shape[0] == len(self) and self.comm.size == 1:
            out[...] = x.reshape(unpacked_shape)
            return out
        out_ = out.reshape(flat_shape)
        if self.comm.size == 1:
            out_[index] = x
        else:
            ix = self.comm.allgather((index, x))
            for i_, x_ in ix:
                out_[i_] = x_
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
            'U': '',
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
                return slice(start, stop, step)
        return s

    @staticmethod
    def _normalize_slice(s, n):
        step = 1 if s.step is None else s.step
        if step == 0:
            raise ValueError('Invalide stride.')

        start = s.start
        if start is None:
            start = 0 if step > 0 else n - 1
        elif start < 0:
            start = n + start
            if start < 0:
                if step < 0:
                    return slice(0, 0, step)
                start = 0
        elif start >= n:
            if step > 0:
                return slice(n - 1, n - 1, step)
            start = n - 1

        stop = s.stop
        if stop is None:
            stop = n if step > 0 else -1
        elif stop < 0:
            stop = n + stop
        elif stop > n:
            stop = n

        if step > 0:
            extent = max(stop - start, 0)
        else:
            extent = max(start - stop, 0)
        size = int(np.ceil(extent / abs(step)))
        if size == n and step == 1:
            return Ellipsis

        return slice(start, start + size * step, step)

    def _selection2bool(self, selection):
        out = np.zeros(self._shape_actual, bool)
        out[selection] = True
        return out


class UnpackedTable:
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
            raise AttributeError(f'The unpacked table has no attribute {key!r}.')
        v = getattr(self._packed, key)
        if v is None:
            return v
        if isinstance(v, Callable):
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
            raise AttributeError(f'{key!r} is not an unpacked attribute.')
        if key in p._reserved_attributes:
            raise AttributeError(f'The attribute {key!r} is not writeable.')
        if isinstance(value, Callable):
            raise TypeError('A function cannot be set as an unpacked attribute.')
        object.__setattr__(p, key, p.pack(value, copy=True))
