import numpy as np

from pyoperators import (
    IdentityOperator,
    MPIDistributionIdentityOperator,
    To1dOperator,
    asoperator,
)
from pyoperators.memory import empty
from pyoperators.utils import product, strenum

from .. import _flib as flib
from ..operators import ProjectionOperator
from ..sparse import FSRMatrix
from ..wcsutils import WCSToPixelOperator, fitsheader2shape
from .core import PackedTable

__all__ = ['Scene', 'SceneGrid']


class Scene(PackedTable):
    """
    Class to handle plane pixelisation and 1-d indexing.

    """

    def __init__(self, shape, topixel=None, ndim=None, dtype=float, **keywords):
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
        topixel : Operator, optional
            World-to-pixel coordinate transform.

        """
        PackedTable.__init__(self, shape, ndim=ndim)
        if topixel is None:
            topixel = IdentityOperator(self.shape[: self.ndim])
        else:
            topixel = asoperator(topixel)
        self.dtype = np.dtype(dtype)
        self.topixel = topixel
        self.toworld = topixel.I
        for k, v in keywords.items():
            setattr(self, k, v)

    def empty(self):
        """
        Return a new array as described by the scene, without initializing
        entries.

        """
        return empty(self.shape, self.dtype)

    def ones(self):
        """
        Return a new array as described by the scene, filled with ones.

        """
        out = self.empty()
        out[...] = 1
        return out

    def zeros(self):
        """
        Return a new array as described by the scene, filled with zeros.

        """
        out = self.empty()
        out[...] = 0
        return out

    def get_distribution_operator(self, comm):
        """
        Distribute a global scene, of which each MPI process has a copy, to the
        MPI processes.

        It is a block column operator whose blocks are identities distributed
        across the MPI processes.

                       |1   O|
        MPI rank 0 --> |  .  |
                       |O   1|
                       +-----+
                       |1   O|
        MPI rank 1 --> |  .  |
                       |O   1|
                       +-----+
                       |1   O|
        MPI rank 2 --> |  .  |
                       |O   1|

        For an MPI process, the direct method is the Identity and the transpose
        method is a reduction.

        """
        if self.comm.size > 1:
            raise ValueError('The scene is already distributed.')
        return MPIDistributionIdentityOperator(comm)


class SceneGrid(Scene):
    """
    Class for 2-dimensional scenes.

    """

    def __init__(
        self,
        shape,
        topixel=None,
        to1d=None,
        origin='upper',
        startswith1=False,
        **keywords,
    ):
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
                f'Invalid origin {origin!r}. Expected values are {strenum(origins)}.'
            )
        Scene.__init__(self, shape, topixel=topixel, **keywords)
        if self.ndim != 2:
            raise ValueError('The scene is not 2-dimensional.')
        self.origin = origin
        self.startswith1 = bool(startswith1)
        if to1d is not None:
            to1d = asoperator(to1d)
            self.toNd = to1d.I
        else:
            self.toNd = None
        self.to1d = to1d

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

    def get_integration_operator(
        self, polygons, ncolmax=0, dtype=np.float64, dtype_index=None, **keywords
    ):
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
        dtype : dtype, optional
            The float datatype of the matrix elements.
        dtype_index : dtype, optional
            The integer datatype of the matrix. The defaultis np.int32 or
            np.int64 depending on the number of elements of the scene.

        """
        if len(self) != len(self.all):
            raise NotImplementedError('The scene has removed elements.')
        dtype = np.dtype(dtype)
        if dtype_index is None:
            if len(self) <= np.iinfo(np.int32).max:
                dtype_index = np.int32
            else:
                dtype_index = np.int64
        dtype_index = np.dtype(dtype_index)
        if str(dtype_index) not in ('int32', 'int64') or str(dtype) not in (
            'float32',
            'float64',
        ):
            raise TypeError(
                f'The projection matrix cannot be created with an index type '
                f'{dtype_index} and a value type {dtype}'
            )

        polygons = np.array(polygons, dtype=dtype, copy=False)
        shape = polygons.shape[:-2]
        matrix = FSRMatrix(
            (product(shape), product(self.shape)),
            dtype=dtype,
            dtype_index=dtype_index,
            ncolmax=ncolmax,
        )

        if matrix.data.size == 0:
            # f2py doesn't accept zero-sized opaque arguments
            data = np.empty(1, np.int8)
        else:
            data = matrix.data.ravel().view(np.int8)

        nvertices = polygons.shape[-2]
        isize = dtype_index.itemsize
        rsize = dtype.itemsize
        func = f'matrix_polygon_integration_i{isize}_r{rsize}'
        min_ncolmax, outside = getattr(flib.projection, func)(
            polygons.reshape(-1, nvertices, 2).T,
            self.shape[1],
            self.shape[0],
            data,
            ncolmax,
        )

        if min_ncolmax > ncolmax:
            return self.get_integration_operator(
                polygons,
                ncolmax=min_ncolmax,
                dtype=dtype,
                dtype_index=dtype_index,
                **keywords,
            )

        out = ProjectionOperator(matrix, shapein=self.shape, shapeout=shape, **keywords)
        out.method = 'polygon integration'
        out.outside = bool(outside)
        out.min_ncolmax = min_ncolmax
        return out
