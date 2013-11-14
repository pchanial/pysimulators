from __future__ import division

try:
    from matplotlib import pyplot as mp
except:  # pragma: no coverage
    pass
import numpy as np
from pyoperators.utils import isalias, isscalar, product, strenum, tointtuple

from .geometry import create_circle, create_grid, create_grid_squares
from .quantities import Quantity

__all__ = ['Layout', 'LayoutGrid', 'LayoutGridCircles', 'LayoutGridSquares']


class Layout(object):
    """
    The Layout class represents a set of components. Characteristics of
    the components can transparently be accessed as packed or unpacked arrays.

    Example
    -------
    Let's consider a 3x3 array of detectors, in which the top-left detector
    does not working. We will define a mask to flag this detector.
    >>> removed = [[False, False, True ],
                   [False, False, False],
                   [False, False, False]]
    >>> center = [[[-1.5,  1.5], [0,  1], [1.5,  1.5]],
                  [[-1,    0  ], [0,  0], [1,    0  ]],
                  [[-1.5, -1.5], [0, -1], [1.5, -1.5]]]
    >>> gain = [[1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0],
                [0.8, 1.0, 1.0]]
    >>> layout = Layout((3, 3), removed=removed, center=center, gain=gain)
    >>> layout.plot()
    >>> layout.packed.gain
    array([ 1. ,  1. ,  0.9,  1. ,  1. ,  0.8,  1. ,  1. ])
    >>> len(layout)
    9
    >>> len(layout.packed)
    8

    Now, let's have a more complex example: an array of detectors made of
    4 identical 3x3 subarrays in which one corner detector is blind and
    for which we will define an indexing scheme.
    The first subarray is placed on the upper right quadrant and the position
    of the other arrays is obtained by rotating the first array by 90, 180 and
    270 degrees.
    A natural indexing would be given by:

    >>> index = [[-1, 14, 17,  0,  1, -1],
                 [10, 13, 16,  3,  4,  5],
                 [ 9, 12, 15,  6,  7,  8],
                 [26, 25, 24, 33, 30, 27],
                 [23, 22, 21, 34, 31, 28],
                 [-1, 19, 18, 35, 32, -1]]

    The following mask only keeps the 2 arrays on the left:

    >>> removed = [[False, False, False, True, True, True],
                   [False, False, False, True, True, True],
                   [False, False, False, True, True, True],
                   [False, False, False, True, True, True],
                   [False, False, False, True, True, True],
                   [False, False, False, True, True, True]]
    >>> layout = Layout((6, 6), removed=removed, index=index)

    Then the numbering of the elements in the packed attributes follows
    the list of unpacked indices stored in:

    >>> print layout.packed.index
    [12  6 13  7  1 14  8  2 32 31 26 25 24 20 19 18]

    which are the 1d-collapsed indices of the following array coordinates:

    >>> print [(i // 6, i % 6) for i in layout.packed.index]
    [(2, 0), (1, 0), (2, 1), (1, 1), (0, 1), (2, 2), (1, 2), (0, 2),
     (5, 2), (5, 1), (4, 2), (4, 1), (4, 0), (3, 2), (3, 1), (3, 0)]

    """
    def __init__(self, shape, center=None, vertex=None, removed=False,
                 masked=False, index=None, nvertices=None, **keywords):
        """
    shape : tuple of int
        The layout unpacked shape. For 2-dimensional layouts, the shape is
        (nrows, ncolumns).
    center : array-like, Quantity, optional
        The position of the components. For 2-dimensional layouts, its shape
        must be (nrows, ncolumns, 2). If not provided, these positions are
        derived from the vertex keyword.
    vertex : array-like, Quantity, optional
        The vertices of the components. For 2-dimensional layouts of squares,
        its shape must be (nrows, ncolumns, 4, 2)
    removed : array-like of bool, optional
        The mask that specifies the removed components (either because they
        are not physically present or if they are not handled by the current
        MPI process when the components are distributed in a parallel
        processing). True means removed (numpy.ma convention).
    masked : array-like of bool, optional
        The mask array that specifies the masked components. True means masked
        (numpy.ma convention).
    index : array-like of int, optional
        The values in this array specify the ranks of the components. It is
        used to define a 1-dimensional indexing of the packed components.
        A negative value means that the component is removed.
    nvertices : int, optional
        The component number of vertices. If not specified, it is deduced
        from the last but one dimension of the vertex keyword. If the component
        vertices are given by a function, setting the keyword nvertices spares
        a call to this function.

        """
        shape = tointtuple(shape)
        removed = np.array(removed, dtype=bool)
        if isscalar(removed):
            removed = np.lib.stride_tricks.as_strided(
                removed, shape, len(shape) * (0,))
        elif removed.shape != shape:
            raise ValueError('Invalid shape of the removed attribute.')
        if index is not None:
            index = np.asarray(index, np.int32)
            if index.shape != shape:
                raise ValueError('Invalid shape of the index attribute.')
            if np.any(index < 0):
                removed = removed | (index < 0)  # no |= because of 0 strides
        removed.flags.writeable = False
        self._special_attributes = set()
        self._reserved_attributes = ('ndim', 'shape', 'nvertices', 'removed',
                                     'packed')
        self.shape = shape
        self.ndim = len(self.shape)
        super(Layout, self).__setattr__('removed', removed)
        self.packed = _Packed(self)
        self.setattr_packed('index', self._pack_index(index, removed))
        if vertex is not None:
            center = None
        keywords.update({'center': center, 'vertex': vertex, 'masked': masked})
        for k, v in keywords.items():
            if v is not None:
                self.setattr_unpacked(k, v)

        if vertex is None:
            nvertices = 0
        if nvertices is not None:
            self.nvertices = nvertices
        else:
            self.nvertices = self.vertex.shape[-2]

    def __len__(self):
        """ Return the number of elements in the layout. """
        return product(self.shape)

    def __getattribute__(self, key):
        """
        Some magic is applied here for the special attributes. These special
        attributes can be a function, a scalar or an array, but they are always
        returned as  an ndarray of the same shape as the layout.
        We use the more cumbersome __getattribute__ instead of __getattr__ only
        because we can make the special attributes actual attributes that can
        be inspected by Ipython's autocomplete feature.

        """
        v = super(Layout, self).__getattribute__(key)
        if key == '_special_attributes' or key not in self._special_attributes:
            return v
        if callable(v):
            out = v()
        elif v is None:
            out = self.unpack(getattr(self.packed, key))
        else:
            return v
        return out

    def __setattr__(self, key, value):
        # make removed and special attributes not writeable
        if key == 'removed':
            raise RuntimeError('The removed mask is not writeable.')
        if key == 'center' and self.nvertices > 0:
            raise RuntimeError('The vertices should be set instead of the cent'
                               'ers.')
        if key in getattr(self, '_special_attributes', ()):
            self.setattr_packed(key, self.pack(value))
        else:
            super(Layout, self).__setattr__(key, value)

    def setattr_unpacked(self, key, value):
        """
        Set a special attribute to the layout using an unpacked value.

        A special attribute is an attribute that can be accessed through
        its packed or unpacked representation.

        """
        if key in self._reserved_attributes:
            raise KeyError('A special layout attribute cannot be {0}.'.format(
                           strenum(self._reserved_attributes)))
        self._special_attributes.add(key)
        if key == 'vertex':
            self.setattr_unpacked('center',
                                  lambda: np.mean(self.vertex, axis=-2))

        if callable(value):
            pvalue = lambda: self.pack(getattr(self, key))
        elif value is None:
            pvalue = None
        else:
            value = np.asanyarray(value)
            if isscalar(value):
                pvalue = value
            elif value.shape[:self.ndim] != self.shape:
                raise ValueError(
                    "Invalid input unpacked shape '{0}'. The first dimension(s"
                    ") must be '{1}'.".format(value.shape, self.shape))
            else:
                pvalue = self.pack(value)
                value = None

        super(Layout, self).__setattr__(key, value)
        super(_Packed, self.packed).__setattr__(key, pvalue)

    def setattr_packed(self, key, value):
        """
        Set a special attribute to the layout using a packed value.

        A special attribute is an attribute that can be accessed through
        its packed or unpacked representation.

        """
        if key in self._reserved_attributes:
            raise KeyError('A special layout attribute cannot be {0}.'.format(
                           strenum(self._reserved_attributes)))
        self._special_attributes.add(key)
        if key == 'vertex':
            self.setattr_packed('center',
                                lambda: np.mean(self.packed.vertex, axis=-2))

        if callable(value):
            uvalue = lambda: self.unpack(getattr(self.packed, key))
        elif value is None:
            uvalue = None
        else:
            value = np.asanyarray(value)
            if isscalar(value):
                uvalue = value
            elif value.shape[0] != len(self.packed):
                raise ValueError(
                    "Invalid input packed shape '{0}'. The first dimension mus"
                    "t be '{1}'.".format(value.shape, len(self.packed)))
            else:
                uvalue = None

        super(Layout, self).__setattr__(key, uvalue)
        super(_Packed, self.packed).__setattr__(key, value)

    def pack(self, x, out=None):
        """
        Convert a multi-dimensional array into a 1-dimensional array which
        only includes the non-removed components, potentially ordered according
        to a given indexing.

        Parameters
        ----------
        x : ndarray
            Array to be packed, whose first dimensions are equal to those
            of the layout.
        out : ndarray, optional
            Placeholder for the packed array.

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
        This method does not make a copy if it simply does a reshape (when no
        component is removed and no indexing is specified).

        """
        if x is None:
            return None
        x = np.asanyarray(x)
        if x.shape[:self.ndim] != self.shape:
            raise ValueError("Invalid input dimensions '{0}'.".format(x.shape))
        packed_shape = (len(self.packed),) + x.shape[self.ndim:]
        if out is None:
            if self.packed.index is None:
                return x.reshape(packed_shape)
            out = np.empty(packed_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != packed_shape:
                raise ValueError(
                    "The output array shape '{0}' is invalid. The expected sha"
                    "pe is '{1}'.".format(out.shape, packed_shape))
        if self.index is None:
            out[...] = x.reshape(packed_shape)
            return out
        x_ = x.view(np.ndarray).reshape((len(self),) + packed_shape[1:])
        np.take(x_, self.packed.index, axis=0, out=out)
        return out

    def unpack(self, x, out=None, removed_value=None):
        """
        Convert a 1-dimensional array into a multi-dimensional array which
        includes the removed components, mimicking the multi-dimensional
        layout.

        Parameters
        ----------
        x : ndarray
            Array to be unpacked, whose first dimension is the number of
            non-removed components.
        out : ndarray, optional
            Placeholder for the unpacked array.
        removed_value : any
            The value to be used for removed components.

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
        This method does not make a copy if it simply does a reshape (when no
        component is removed and no indexing is specified).

        """
        if x is None:
            return None
        x = np.asanyarray(x)
        if x.shape[0] != len(self.packed):
            raise ValueError("Invalid input dimensions '{0}'.".format(x.shape))
        unpacked_shape = self.shape + x.shape[1:]
        has_out = out is not None
        if not has_out:
            if self.packed.index is None:
                return x.reshape(unpacked_shape)
            out = np.empty(unpacked_shape, dtype=x.dtype).view(type(x))
            if out.__array_finalize__ is not None:
                out.__array_finalize__(x)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError('The output array is not an ndarray.')
            if out.shape != unpacked_shape:
                raise ValueError(
                    "The output array shape '{0}' is invalid. The expected sha"
                    "pe is '{1}'.".format(out.shape, unpacked_shape))
            out = out.view(np.ndarray)
        if len(self) > len(self.packed):
            if removed_value is None:
                removed_value = self._get_default_removed_value(x.dtype)
            out[...] = removed_value
        elif self.packed.index is None:
            out[...] = x.reshape(unpacked_shape)
            return out
        out_ = out.reshape((len(self),) + x.shape[1:])
        out_[self.packed.index] = x
        if has_out and not isalias(out, out_):
            out[...] = out_.reshape(out.shape)
        return out

    def plot(self, transform=None, **keywords):
        """
        Plot the layout.

        Parameters
        ----------
        transform : Operator
            Operator to be used to transform the input coordinates into
            the data coordinate system.

        """
        if self.nvertices > 0:
            coords = self.packed.vertex
        else:
            coords = self.packed.center
            if hasattr(self, 'radius'):
                coords = create_circle(self.packed.radius, center=coords)

        if transform is not None:
            coords = transform(coords)

        if self.nvertices > 0 or hasattr(self, 'radius'):
            a = mp.gca()
            patches = coords.reshape((-1,) + coords.shape[-2:])
            for p in patches:
                a.add_patch(mp.Polygon(p, closed=True, fill=False, **keywords))
            a.autoscale_view()
        else:
            if 'color' not in keywords:
                keywords['color'] = 'black'
            if 'marker' not in keywords:
                keywords['marker'] = 'o'
            if 'linestyle' not in keywords:
                keywords['linestyle'] = ''
            mp.plot(coords[..., 0], coords[..., 1], **keywords)

        mp.show()

    def _get_default_removed_value(self, dtype):
        value = np.empty((), dtype)
        if dtype.kind == 'V':
            for f, (dt, junk) in dtype.fields.items():
                value[f] = self._get_default_removed_value(dt)
            return value
        return {'b': False,
                'i': -1,
                'u': 0,
                'f': np.nan,
                'c': np.nan,
                'S': '',
                'U': u'',
                'O': None}[dtype.kind]

    @staticmethod
    def _pack_index(index, removed):
        if np.all(removed):
            return np.array([], int)
        if index is None:
            if not np.any(removed):
                return None
            return np.where(~np.ravel(removed))[0]
        index = np.array(index)
        index[removed] = -1
        index = index.ravel()
        npacked = np.sum(index >= 0)
        if npacked == 0:
            return np.array([], int)
        isort = np.argsort(index)
        return isort[-npacked:].copy()


class _Packed(object):
    """ Class used to store unpacked characteristics of the layout. """
    def __init__(self, unpacked):
        if isscalar(unpacked.removed) and not unpacked.removed:
            self._len = len(unpacked)
        else:
            self._len = int(np.sum(~unpacked.removed))

    def __len__(self):
        return self._len

    def __getattribute__(self, key):
        v = super(_Packed, self).__getattribute__(key)
        if key[0] == '_' or not callable(v):
            return v
        return v()

    def __setattr__(self, key, value):
        if key[0] == '_':
            super(_Packed, self).__setattr__(key, value)
            return
        raise KeyError("A packed attribute should be set with the Layout metho"
                       "d 'setattr_packed'.")


class LayoutGrid(Layout):
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
    >>> layout = LayoutGrid((16, 16), spacing)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        Tuple containing the number of rows and columns of the grid.
    ndim : int
        The number of layout dimensions.
    center : array of shape (nrows, ncolumns, 2)
        Position of the component centers. The first dimension refers to the
        layout row, the second one to the layout column. The last dimension's
        two elements are the X and Y coordinates.

    """
    def __init__(self, shape, spacing, xreflection=False, yreflection=False,
                 angle=0, origin=(0, 0), **keywords):
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
    nvertices : int, optional
        The component number of vertices. If not specified, it is deduced
        from the last but one dimension of the vertex keyword. If the component
        vertices are given by a function, setting the keyword nvertices spares
        a call to this function.
    vertex : array-like, Quantity, optional
        The vertices of the components. For 2-dimensional layouts of squares,
        its shape must be (nrows, ncolumns, 4, 2)
    removed : array-like of bool, optional
        The mask that specifies the removed components (either because they
        are not physically present or if they are not handled by the current
        MPI process when the components are distributed in a parallel
        processing). True means removed (numpy.ma convention).
    masked : array-like of bool, optional
        The mask array that specifies the masked components. True means masked
        (numpy.ma convention).
    index : array-like of int, optional
        The values in this array specify the ranks of the components. It is
        used to define a 1-dimensional indexing of the packed components.
        A negative value means that the component is removed.

        """
        def center():
            out = create_grid(
                self.shape, self.spacing, xreflection=self.xreflection,
                yreflection=self.yreflection, center=self.origin,
                angle=self.angle)
            if self._unit:
                out = Quantity(out, self._unit, copy=False)
            return out
        unit = getattr(spacing, 'unit', '') or \
               getattr(origin, 'unit', '')
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            origin = Quantity(origin).tounit(unit)
        if len(origin) != len(shape):
            raise ValueError('Invalid dimensions of the layout center.')
        self.spacing = spacing
        self.origin = origin
        self.xreflection = xreflection
        self.yreflection = yreflection
        self.angle = angle
        self._unit = unit
        Layout.__init__(self, shape, center=center, **keywords)


class LayoutGridCircles(LayoutGrid):
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
    >>> layout = LayoutGridCircles((16, 16), spacing)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        Tuple containing the number of rows and columns of the grid.
    ndim : int
        The number of layout dimensions.
    center : array of shape (nrows, ncolumns, 2)
        Position of the component centers. The first dimension refers to the
        layout row, the second one to the layout column. The last dimension's
        two elements are the X and Y coordinates.

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
    removed : array-like of bool, optional
        The mask that specifies the removed components (either because they
        are not physically present or if they are not handled by the current
        MPI process when the components are distributed in a parallel
        processing). True means removed (numpy.ma convention).
    masked : array-like of bool, optional
        The mask array that specifies the masked components. True means masked
        (numpy.ma convention).
    index : array-like of int, optional
        The values in this array specify the ranks of the components. It is
        used to define a 1-dimensional indexing of the packed components.
        A negative value means that the component is removed.

        """
        if radius is None:
            radius = spacing / 2
        unit = getattr(spacing, 'unit', '') or getattr(radius, 'unit', '')
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            radius = Quantity(radius, copy=False).tounit(unit)
        LayoutGrid.__init__(self, shape, spacing, radius=radius, nvertices=0,
                            **keywords)


class LayoutGridSquares(LayoutGrid):
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
    >>> layout = LayoutGridSquares((16, 16), spacing, filling_factor=0.8)
    >>> layout.plot()

    Attributes
    ----------
    shape : tuple of integers (nrows, ncolumns)
        See below.
    ndim : int
        The number of layout dimensions.
    center : array of shape (nrows, ncolumns, 2)
        Position of the component centers. The first dimension refers to the
        layout row, the second one to the layout column. The last dimension's
        two elements are the X and Y coordinates.
    vertex : array of shape (nrows, ncolumns, 4, 2)
        Corners of the components. The first dimension refers to the
        layout row, the second one to the layout column. The third dimension
        refers to the corner number, which is counted counter-clockwise,
        starting from the top-right one. The last dimension's two elements are
        the X and Y coordinates.

    """
    def __init__(self, shape, spacing, filling_factor=1, **keywords):
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
    removed : array-like of bool, optional
        The mask that specifies the removed components (either because they
        are not physically present or if they are not handled by the current
        MPI process when the components are distributed in a parallel
        processing). True means removed (numpy.ma convention).
    masked : array-like of bool, optional
        The mask array that specifies the masked components. True means masked
        (numpy.ma convention).
    index : array-like of int, optional
        The values in this array specify the ranks of the components. It is
        used to define a 1-dimensional indexing of the packed components.
        A negative value means that the component is removed.

        """
        def vertex():
            out = create_grid_squares(
                self.shape, self.spacing, filling_factor=self.filling_factor,
                xreflection=self.xreflection,
                yreflection=self.yreflection, center=self.origin,
                angle=self.angle)
            if self._unit:
                out = Quantity(out, self._unit, copy=False)
            return out
        self.filling_factor = filling_factor
        LayoutGrid.__init__(self, shape, spacing, vertex=vertex, nvertices=4,
                            **keywords)
