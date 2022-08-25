import numpy as np

from ..geometry import create_circle, create_grid, create_grid_squares
from ..quantities import Quantity
from .core import PackedTable

__all__ = ['Layout', 'LayoutGrid', 'LayoutGridSquares']


class Layout(PackedTable):
    """
    A class for packed table including spatial information of the components.

    Attributes
    ----------
    shape : tuple of integers
        Tuple containing the layout dimensions.
    ndim : int
        The actual number of layout dimensions.
    center : array of shape (ncomponents, 2)
        Position of the component centers. The last dimension stands for
        the (X, Y) coordinates.
    vertex : array of shape (ncomponents, nvertices, 2), optional
        Position of the component vertices. This attribute is not present if
        it has not been specified. The last dimension stands for
        the (X, Y) coordinates.
    nvertices : int, optional
        Number of vertices if the vertex attribute is present.
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
            The position of the components. For 2-dimensional layouts, the expected
            shape is (nrows, ncolumns, 2). If not provided, these positions are
            derived from the vertex keyword, if present.
        vertex : array-like, Quantity, optional
            The position of the component vertices. For 2-dimensional layouts,
            the expected shape is (nrows, ncolumns, nvertices, 2). The last
            dimension stands for the (X, Y) coordinates.

        """
        if hasattr(self, 'vertex') and 'vertex' not in keywords:
            keywords['vertex'] = None
        if hasattr(self, 'center') and 'center' not in keywords:
            keywords['center'] = None
        if 'center' not in keywords and 'vertex' in keywords:
            keywords['center'] = lambda s: np.mean(s.vertex, axis=-2)
        if 'center' not in keywords:
            raise ValueError('The spatial layout is not defined.')
        PackedTable.__init__(self, shape, **keywords)
        if hasattr(self, 'vertex'):
            self.nvertices = self.vertex.shape[-2]

    def plot(
        self,
        autoscale=True,
        transform=None,
        edgecolor=None,
        facecolor=None,
        fill=None,
        **keywords,
    ):
        """
        Plot the layout.

        Parameters
        ----------
        autoscale : boolean
            If true, the axes of the plot will be updated to match the
            boundaries of the detectors.
        transform : callable, Operator
            Operator to be used to transform the layout coordinates into
            the data coordinate system.

        """
        import matplotlib.pyplot as mp

        if hasattr(self, 'vertex'):
            coords = self.vertex[..., :2]
        elif hasattr(self, 'radius'):
            coords = create_circle(self.radius, center=self.center[..., :2])
        else:
            coords = self.center[..., :2]

        if transform is not None:
            coords = transform(coords)

        if coords.ndim == 3:
            if fill is None and facecolor is not None:
                fill = True
            a = mp.gca()
            try:
                from pyoperators.utils import zip_broadcast

                def isnumber(x):
                    try:
                        float(x)
                        return True
                    except (TypeError, ValueError):
                        return False

                # special treatment for RGB tuples
                if isinstance(edgecolor, tuple) and all(isnumber(_) for _ in edgecolor):
                    edgecolor = [edgecolor]
                if isinstance(facecolor, tuple) and all(isnumber(_) for _ in facecolor):
                    facecolor = [facecolor]
                for c, ec, fc in zip_broadcast(
                    coords, edgecolor, facecolor, iter_str=False
                ):
                    a.add_patch(
                        mp.Polygon(
                            c,
                            closed=True,
                            edgecolor=ec,
                            facecolor=fc,
                            fill=fill,
                            **keywords,
                        )
                    )
            except ImportError:
                # for PyOperators < 0.13.1
                for c in coords:
                    a.add_patch(
                        mp.Polygon(
                            c,
                            closed=True,
                            edgecolor=edgecolor,
                            facecolor=facecolor,
                            fill=fill,
                            **keywords,
                        )
                    )
            if autoscale:
                a.autoscale_view()
        elif coords.ndim == 2:
            if 'color' not in keywords:
                keywords['color'] = 'black'
            if 'marker' not in keywords:
                keywords['marker'] = 'o'
            if 'linestyle' not in keywords:
                keywords['linestyle'] = ''
            mp.plot(
                coords[:, 0],
                coords[:, 1],
                scalex=autoscale,
                scaley=autoscale,
                **keywords,
            )
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

    def __init__(
        self,
        shape,
        spacing,
        xreflection=False,
        yreflection=False,
        angle=0,
        origin=(0, 0),
        startswith1=False,
        _z=None,
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
        _z : float, optional, don't use it yet
            The third dimension of the component' centers or vertices.
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
        if len(origin) != len(shape):
            raise ValueError('Invalid dimensions of the layout center.')

        unit = (
            getattr(spacing, 'unit', '')
            or getattr(origin, 'unit', '')
            or 'radius' in keywords
            and getattr(keywords['radius'], 'unit', '')
        )
        if unit:
            spacing = Quantity(spacing).tounit(unit)
            origin = Quantity(origin).tounit(unit)
            if 'radius' in keywords:
                keywords['radius'] = Quantity(keywords['radius']).tounit(unit)
        if 'vertex' not in keywords:
            center = create_grid(
                shape,
                spacing,
                xreflection=xreflection,
                yreflection=yreflection,
                center=origin,
                angle=angle,
            )
            if _z is not None:
                center = np.concatenate([center, np.full_like(center[..., :1], _z)], -1)
            keywords['center'] = center
        if unit:
            if 'center' in keywords:
                keywords['center'] = Quantity(keywords['center'], unit, copy=False)
            if 'vertex' in keywords:
                keywords['vertex'] = Quantity(keywords['vertex'], unit, copy=False)
        self.spacing = spacing
        self.origin = origin
        self.xreflection = xreflection
        self.yreflection = yreflection
        self.angle = angle
        self.startswith1 = startswith1
        self.unit = unit
        Layout.__init__(self, shape, **keywords)
        self._special_attributes += ('column', 'row')

    def column(self):
        return self.index % self.shape[1] + self.startswith1

    def row(self):
        return self.index // self.shape[1] + self.startswith1


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
        LayoutGrid.__init__(
            self,
            shape,
            spacing,
            vertex=vertex,
            xreflection=xreflection,
            yreflection=yreflection,
            angle=angle,
            origin=origin,
            startswith1=startswith1,
            **keywords,
        )
        self.filling_factor = filling_factor
