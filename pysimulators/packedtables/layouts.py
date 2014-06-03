from __future__ import absolute_import, division, print_function
try:
    from matplotlib import pyplot as mp
except:  # pragma: no coverage
    pass
import numpy as np
from .core import PackedTable
from ..geometry import create_circle, create_grid, create_grid_squares
from ..quantities import Quantity

__all__ = ['Layout',
           'LayoutGrid',
           'LayoutGridCircles',
           'LayoutGridSquares',
           'LayoutVertex']


class Layout(PackedTable):
    """
    A class for packed table including spatial information of the components.

    Attributes
    ----------
    shape : tuple of integers
        Tuple containing the layout dimensions
    ndim : int
        The actual number of layout dimensions.
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
    ndim : int, optional
        The number of splittable (indexable) dimensions. It is the actual
        number of dimensions of the layout. It can be lower than that
        specified by the layout shape, in which case the extra dimensions
        are instructed not to be split.
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
        PackedTable.__init__(self, shape, **keywords)

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
    center : array of shape (len(self), 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    all : object
        Access to the unpacked attributes of shape (nrows, ncolumns, ...)

    """
    def __init__(self, shape, spacing, xreflection=False, yreflection=False,
                 angle=0, origin=(0, 0), startswith1=False, **keywords):
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
        center = create_grid(shape, spacing, xreflection=xreflection,
                             yreflection=yreflection, center=origin,
                             angle=angle)
        if unit:
            center = Quantity(center, unit, copy=False)
        self.spacing = spacing
        self.origin = origin
        self.xreflection = xreflection
        self.yreflection = yreflection
        self.angle = angle
        self.startswith1 = startswith1
        self.unit = unit
        Layout.__init__(self, shape, center=center, **keywords)
        self._special_attributes += ('column', 'row')

    def column(self):
        return self.index % self.shape[1] + self.startswith1

    def row(self):
        return self.index // self.shape[1] + self.startswith1


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
        LayoutGrid.__init__(self, shape, spacing, radius=radius, **keywords)

    def plot(self, transform=None, **keywords):
        coords = create_circle(self.radius, center=self.center)
        self._plot(coords, transform, **keywords)
    plot.__doc__ = LayoutGrid.plot.__doc__


class LayoutVertex(Layout):
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
        Layout.__init__(self, shape, **keywords)
        self.nvertices = nvertices

    @property
    def center(self):
        try:
            return np.mean(self.vertex, axis=-2)
        except:
            return None

    def plot(self, transform=None, **keywords):
        self._plot(self.vertex, transform, **keywords)
    plot.__doc__ = Layout.plot.__doc__


class LayoutGridSquares(LayoutVertex):
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
    >>> layout = LayoutGridSquares((8, 8), spacing, filling_factor=0.8)
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
    def __init__(self, shape, spacing, filling_factor=1, xreflection=False,
                 yreflection=False, angle=0, origin=(0, 0), startswith1=False,
                 **keywords):
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
            shape, spacing, filling_factor=filling_factor,
            xreflection=xreflection, yreflection=yreflection, center=origin,
            angle=angle)
        if unit:
            vertex = Quantity(vertex, unit, copy=False)
        LayoutVertex.__init__(self, shape, 4, vertex=vertex, **keywords)
        self.angle = angle
        self.filling_factor = filling_factor
        self.origin = origin
        self.spacing = spacing
        self.startswith1 = startswith1
        self.unit = unit
        self.xreflection = xreflection
        self.yreflection = yreflection
