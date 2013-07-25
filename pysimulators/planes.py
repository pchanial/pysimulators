from __future__ import division

import numpy as np
from pyoperators import BlockDiagonalOperator, MinMaxOperator, RoundOperator
from pyoperators.utils import product

from . import _flib as flib
from .acquisitionmodels import PointingMatrix, ProjectionInMemoryOperator
from .wcsutils import fitsheader2shape, WCSToPixelOperator

__all__ = ['Plane']


class Plane(object):
    """Class to handle plane pixelisation."""

    def __init__(self, header):
        self.header = header
        self.topixel = WCSToPixelOperator(header)
        self.toworld = self.topixel.I

    #        self.topixel1d = To1dOperator(fitsheader2shape(header))

    def get_spatial_integration_operator(self, vertices, npixels_per_sample=0):
        """
        Return the Projection Operator that spatially integrates
        the plane pixel values enclosed by a sequence of vertices.

        Parameters
        ----------
        vertices : array-like
            The vertices inside which the spatial integration is done,
            in the plane world coordinate units.

        """
        xy = self.topixel(vertices)
        roi_operator = BlockDiagonalOperator(
            [RoundOperator('rhtpi'), RoundOperator('rhtmi')], new_axisin=-2
        ) * MinMaxOperator(axis=-2, new_axisout=-2)
        roi = roi_operator(xy).astype(np.int32)

        shape = vertices.shape[:-2] + (npixels_per_sample,)
        shape_input = fitsheader2shape(self.header)
        pmatrix = PointingMatrix.empty(shape, shape_input, verbose=False)

        if pmatrix.size == 0:
            # f2py doesn't accept zero-sized opaque arguments
            pmatrix_ = np.empty(1, np.int64)
        else:
            pmatrix_ = pmatrix.ravel().view(np.int64)

        nvertices = product(vertices.shape[:-2])
        new_npps, outside = flib.pointingmatrix.roi2pmatrix_cartesian(
            roi.reshape((-1, 2, 2)).T,
            xy.reshape((-1, nvertices, 2)).T,
            npixels_per_sample,
            shape_input[1],
            shape_input[0],
            pmatrix_,
        )

        if new_npps > npixels_per_sample:
            shape = vertices.shape[:-2] + (new_npps,)
            pmatrix = PointingMatrix.empty(shape, shape_input)
            pmatrix_ = pmatrix.ravel().view(np.int64)
            flib.pointingmatrix.roi2pmatrix_cartesian(
                roi.reshape((-1, 2, 2)).T,
                xy.reshape((-1, nvertices, 2)).T,
                new_npps,
                shape_input[1],
                shape_input[0],
                pmatrix_,
            )

        pmatrix.header['method'] = 'sharp'
        pmatrix.header['outside'] = bool(outside)
        pmatrix.header['HIERARCH min_npixels_per_sample'] = new_npps

        return ProjectionInMemoryOperator(pmatrix)
