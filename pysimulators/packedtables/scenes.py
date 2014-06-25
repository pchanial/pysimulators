from __future__ import division

import numpy as np
from pyoperators import (
    BlockDiagonalOperator, IdentityOperator, MinMaxOperator, RoundOperator,
    To1dOperator, asoperator)
from pyoperators.utils import strenum

from .core import PackedTable
from .. import _flib as flib
from ..operators import PointingMatrix, ProjectionInMemoryOperator
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
    def fromfits(cls, header):
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
        scene = cls(shape, topixel, to1d, origin='lower')
        scene.header = header
        return scene

    def column(self):
        return self.index % self.shape[1] + self.startswith1

    def row(self):
        return self.index // self.shape[1] + self.startswith1

    def get_integration_operator(self, vertices, npixels_per_sample=0):
        """
        Return the Projection Operator that spatially integrates
        the plane pixel values enclosed by a sequence of vertices.

        Parameters
        ----------
        vertices : array-like of shape (..., nvertices, 2)
            The vertices defining the polygons inside which the spatial
            integration is done, in the surface world coordinate units.
        npixels_per_sample : integer
            The maximum number of surface pixel intersected by the polygons.
            If set to zero, or less than the required value, a two-pass
            computation will be performed to determine it.x

        """
        xy = self.topixel(vertices)
        roi_operator = BlockDiagonalOperator(
            [RoundOperator('rhtpi'), RoundOperator('rhtmi')],
            new_axisin=-2)(MinMaxOperator(axis=-2, new_axisout=-2))
        roi = roi_operator(xy).astype(np.int32)

        shape = vertices.shape[:-2] + (npixels_per_sample,)
        pmatrix = PointingMatrix.empty(shape, self.shape, verbose=False)

        if pmatrix.size == 0:
            # f2py doesn't accept zero-sized opaque arguments
            pmatrix_ = np.empty(1, np.int64)
        else:
            pmatrix_ = pmatrix.ravel().view(np.int64)

        nvertices = vertices.shape[-2]
        new_npps, outside = flib.pointingmatrix.roi2pmatrix_cartesian(
            roi.reshape((-1, 2, 2)).T, xy.reshape((-1, nvertices, 2)).T,
            npixels_per_sample, self.shape[1], self.shape[0], pmatrix_)

        if new_npps > npixels_per_sample:
            shape = vertices.shape[:-2] + (new_npps,)
            pmatrix = PointingMatrix.empty(shape, self.shape)
            pmatrix_ = pmatrix.ravel().view(np.int64)
            flib.pointingmatrix.roi2pmatrix_cartesian(
                roi.reshape((-1, 2, 2)).T, xy.reshape((-1, nvertices, 2)).T,
                new_npps, self.shape[1], self.shape[0], pmatrix_)

        pmatrix.header['method'] = 'sharp'
        pmatrix.header['outside'] = bool(outside)
        pmatrix.header['HIERARCH min_npixels_per_sample'] = new_npps

        return ProjectionInMemoryOperator(pmatrix)
