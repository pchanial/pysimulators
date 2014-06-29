from __future__ import division

import numpy as np
from pyoperators import (
    BlockDiagonalOperator, IdentityOperator, MinMaxOperator, RoundOperator,
    To1dOperator, asoperator)
from pyoperators.utils import strenum, product

from .core import PackedTable
from .. import _flib as flib
from ..operators import ProjectionOperator
from ..sparse import FSRMatrix
from ..wcsutils import fitsheader2shape, WCSToPixelOperator

__all__ = ['Scene', 'SceneGrid']


class Scene(PackedTable):
    """
    Class to handle plane pixelisation and 1-d indexing.

    """
    def __init__(self, shape, topixel=None, ndim=None, **keywords):
        """
    Parameters
    ----------
    shape : tuple of integers
        The surface shape.
    ndim : int, optional
        The number of splittable (indexable) dimensions. It is the actual
        number of dimensions of the layout. It can be lower than that
        specified by the layout shape, in which case the extra dimensions
        are instructed not to be split.
        ndim : int
    topixel : Operator, None
        World-to-pixel coordinate transform.

        """
        PackedTable.__init__(self, shape, ndim=ndim)
        if topixel is None:
            topixel = IdentityOperator(self.shape[:self.ndim])
        else:
            topixel = asoperator(topixel)
        self.topixel = topixel
        self.toworld = topixel.I
        for k, v in keywords.items():
            setattr(self, k, v)


class SceneGrid(Scene):
    """
    Class for 2-dimensional scenes.

    """
    def __init__(self, shape, topixel=None, to1d=None, origin='upper',
                 startswith1=False, **keywords):
        """
        Parameters
        ----------
        shape : tuple of integers
            The surface shape.
        startswith1 : bool
            If True, columns and row starts with 1 instead of 0.
        topixel : Operator, optional
            World-to-pixel coordinate transform.
        to1d : Operator, optional
            Nd-to-1d pixel index transform.

        """
        origins = 'upper', 'lower'
        if not isinstance(origin, str):
            raise TypeError('Invalid origin.')
        origin = origin.lower()
        if origin not in origins:
            raise ValueError(
                'Invalid origin {0!r}. Expected values are {1}.'.format(
                    origin, strenum(origins)))
        Scene.__init__(self, shape, topixel=topixel, to1d=to1d, **keywords)
        if self.ndim != 2:
            raise ValueError('The scene is not 2-dimensional.')
        self.origin = origin
        self.startswith1 = bool(startswith1)
        if to1d is not None:
            to1d = asoperator(to1d)
            self.to1d = to1d
            self.toNd = to1d.I

    @classmethod
    def fromfits(cls, header, **keywords):
        """
        Return a Scene grid described by the WCS of a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS header that defines the WCS transform.

        """
        shape = fitsheader2shape(header)
        topixel = WCSToPixelOperator(header)
        to1d = To1dOperator(shape[::-1], order='f')
        scene = cls(shape, topixel, to1d, origin='lower', **keywords)
        scene.header = header
        return scene

    def column(self):
        return self.index % self.shape[1] + self.startswith1

    def row(self):
        return self.index // self.shape[1] + self.startswith1

    def get_integration_operator(self, polygons, ncolmax=0, **keywords):
        """
        Return the Projection Operator that spatially integrates
        the plane pixel values enclosed by a sequence of polygons.

        Parameters
        ----------
        polygons : array-like of shape (..., nvertices, 2)
            The vertices defining the polygons inside which the spatial
            integration is done, in pixel coordinate units.
        ncolmax : integer
            The maximum number of surface pixel intersected by the polygons.
            If set to zero, or less than the required value, a two-pass
            computation will be performed to determine it.

        """
        itype = np.dtype(np.int64)
        dtype = np.dtype(np.float64)
        shape = polygons.shape[:-2]
        matrix = FSRMatrix((product(shape), product(self.shape)), dtype=dtype,
                           dtype_index=itype, ncolmax=ncolmax)

        if matrix.data.size == 0:
            # f2py doesn't accept zero-sized opaque arguments
            data = np.empty(1, np.int8)
        else:
            data = matrix.data.ravel().view(np.int8)

        nvertices = polygons.shape[-2]
        func = 'matrix_polygon_integration_i{0}_v{1}'.format(itype.itemsize,
                                                             dtype.itemsize)
        try:
            min_ncolmax, outside = getattr(flib.projection, func)(
                polygons.reshape(-1, nvertices, 2).T, self.shape[1],
                self.shape[0], data, ncolmax)
        except AttributeError:
            raise TypeError(
                'The projection matrix cannot be created with types: {0} and {'
                '1}.'.format(dtype, itype))

        if min_ncolmax > ncolmax:
            return self.get_integration_operator(polygons, ncolmax=min_ncolmax,
                                                 **keywords)

        out = ProjectionOperator(matrix, shapein=self.shape, shapeout=shape,
                                 **keywords)
        out.method = 'polygon integration'
        out.outside = bool(outside)
        out.min_ncolmax = min_ncolmax
        return out
