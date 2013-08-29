from __future__ import division

import numpy as np
from pyoperators.utils import tointtuple
from pyoperators import (BlockDiagonalOperator, MinMaxOperator, RoundOperator,
                         To1dOperator, asoperator)

from . import _flib as flib
from .operators import PointingMatrix, ProjectionInMemoryOperator
from .wcsutils import fitsheader2shape, WCSToPixelOperator

__all__ = ['DiscreteSurface']


class DiscreteSurface(object):
    """
    Class to handle plane pixelisation and 1-d indexing.

    """
    def __init__(self, shape, topixel, to1d):
        """
        Parameters
        ----------
        shape : tuple of integers
            The surface shape.
        topixel : Operator
            World-to-pixel coordinate transform.
        to1d : Operator
            Nd-to-1d pixel index transform.
        """
        self.shape = tointtuple(shape)
        topixel = asoperator(topixel)
        self.topixel = topixel
        self.toworld = topixel.I
        to1d = asoperator(to1d)
        self.to1d = to1d
        self.toNd = to1d.I

    @classmethod
    def fromfits(cls, header):
        """
        Return a DiscreteSurface described by the WCS of a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS header that defines the WCS transform.

        """
        shape = fitsheader2shape(header)
        topixel = WCSToPixelOperator(header)
        to1d = To1dOperator(shape[::-1], order='f')
        return cls(shape, topixel, to1d)

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
            [RoundOperator('rhtpi'), RoundOperator('rhtmi')], new_axisin=-2) *\
            MinMaxOperator(axis=-2, new_axisout=-2)
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
