from __future__ import division

import numpy as np

from matplotlib import pyplot as mp
from pyoperators import I, asoperator
from pyoperators.utils import strshape
from pyoperators.utils.mpi import MPI

from . import _flib as flib
from .datatypes import Map

__all__ = ['Instrument', 'Imager']


class Instrument(object):
    """
    Class storing information about the instrument.

    Attributes
    ----------
    name
    nvertices : int
        The number of individual detector vertices (4 for square detectors).
        The method 'get_vertices' has to be implemented to return the detector
        vertices. If the detectors cannot be represented by polygons, use
        nvertices equal to 0 and implement the method 'get_centers'.
    comm
    detector (nvertices, removed, masked, get_corners, get_vertices)
    object_plane (toworld, topixel, topixel1d)
    image_plane (toworld, topixel, topixel1d)

    """
    def __init__(self, name, shape, removed=None, masked=None, nvertices=4,
                 default_resolution=None, origin='upper', dtype=None,
                 comm=MPI.COMM_WORLD):

        self.name = str(name)
        self.nvertices = nvertices
        self.default_resolution = default_resolution
        self.comm = comm

        shape = tuple(shape)
        dtype_default = [('masked', np.bool8), ('removed', np.bool8)]

        if removed is not None:
            if removed.shape != shape:
                raise ValueError('The input specifying the removed detectors h'
                                 'as an incompatible shape.')
        else:
            removed = False

        if masked is not None:
            if masked.shape != shape:
                raise ValueError('The input specifying the masked detectors ha'
                                 's an incompatible shape.')
        else:
            masked = False

        if dtype is None:
            dtype = dtype_default

        self.detector = Map.zeros(shape, dtype=dtype, origin=origin)
        self.detector.masked = masked
        self.detector.removed = removed

    def get_ndetectors(self, masked=False):
        """
        Return the number of valid detectors.

        Parameters
        ----------
        masked : boolean
              If False, the valid detectors are those that are not removed.
              Otherwise, they are those that not removed and not masked.
        """
        if masked:
            return int(np.sum(~self.detector.removed & ~self.detector.masked))
        return int(np.sum(~self.detector.removed))

    def get_valid_detectors(self, masked=False):
        """
        Return valid detectors subscripts.

        Parameters
        ----------
        masked : boolean
              If False, the valid detectors are those that are not removed.
              Otherwise, they are those that not removed and not masked.
        """
        mask = ~self.detector.removed
        if masked:
            mask &= ~self.detector.masked
        if np.sum(mask) == self.detector.size:
            return Ellipsis
        return mask

    def pack(self, x, masked=False):
        """
        Convert an ndarray which only includes the valid detectors into
        another ndarray which contains all the detectors under the control
        of the detector mask.

        Parameters
        ----------
        x : ndarray
              Array to be packed, whose first dimensions are equal to those
              of the detector attribute.
        masked : boolean
              If False, the valid detectors are those that are not removed.
              Otherwise, they are those that not removed and not masked.

        Returns
        -------
        output : ndarray
              Packed array, whose first dimension is the number of valid
              detectors.

        See Also
        --------
        unpack : inverse method.

        Notes
        -----
        This method does not necessarily make a copy.

        """
        if isinstance(x, dict):
            output = x.copy()
            for k, v in output.items():
                try:
                    output[k] = self.pack(v, masked=masked)
                except (ValueError, TypeError):
                    output[k] = v
            return output
        if not isinstance(x, np.ndarray):
            raise TypeError('The input is not an ndarray.')
        if x.ndim < self.detector.ndim or \
           x.shape[:self.detector.ndim] != self.detector.shape:
            raise ValueError("The shape of the argument '{0}' is incompatible "
                             "with that of the detectors '{1}'.".format(
                             strshape(x.shape), strshape(self.detector.shape)))
        index = self.get_valid_detectors(masked=masked)
        if index is Ellipsis:
            new_shape = (-1,) + x.shape[self.detector.ndim:]
            output = x.reshape(new_shape)
        else:
            output = x[index]
        if type(x) != np.ndarray:
            output = output.view(type(x))
            for k, v in x.__dict__.items():
                try:
                    output.__dict__[k] = self.pack(v, masked=masked)
                except (ValueError, TypeError):
                    output.__dict__[k] = v
        return output

    def unpack(self, x, masked=False):
        """
        Convert an ndarray which only includes the valid detectors into
        another ndarray which contains all the detectors under the control
        of the detector mask.

        Parameters
        ----------
        x : ndarray
              Array to be unpacked, whose first dimension is equal to the
              number of valid detectors.
        masked : boolean
              If False, the valid detectors are those that are not removed.
              Otherwise, they are those that not removed and not masked.

        Returns
        -------
        output : ndarray
              Unpacked array, whose first dimensions are those of the detector
              attribute.

        See Also
        --------
        pack : inverse method.

        Notes
        -----
        This method does not necessarily make a copy.

        """
        if isinstance(x, dict):
            output = x.copy()
            for k, v in output.items():
                try:
                    output[k] = self.unpack(v, masked=masked)
                except (ValueError, TypeError):
                    output[k] = v
            return output
        if not isinstance(x, np.ndarray):
            raise TypeError('The input is not an ndarray.')

        n = self.get_ndetectors(masked=masked)
        if x.ndim == 0 or n != x.shape[0]:
            raise ValueError("The shape of the argument '{0}' is incompatible "
                             "with the number of valid detectors '{1}'."
                             .format(strshape(x.shape), n))

        index = self.get_valid_detectors(masked=masked)
        new_shape = self.detector.shape + x.shape[1:]
        if index is Ellipsis:
            return x.reshape(new_shape)
        output = np.zeros(new_shape, dtype=x.dtype)
        output[index] = x
        if type(x) != np.ndarray:
            output = output.view(type(x))
            for k, v in x.__dict__.items():
                try:
                    output.__dict__[k] = self.unpack(v, masked=masked)
                except (ValueError, TypeError):
                    output.__dict__[k] = v
        return output

    def get_centers(self):
        """
        Return the coordinates of the detector centers in the image plane.

        """
        if self.nvertices == 0:
            raise NotImplementedError(
                'The instrument geometry is not defined.')
        vertices = self.get_vertices()
        return np.mean(vertices, axis=-2)

    def get_vertices(self):
        """
        Return the coordinates of the detector vertices in the image plane.

        """
        if self.nvertices == 0:
            raise RuntimeError('Detectors have no vertices.')
        raise NotImplementedError('The instrument geometry is not defined.')

    @staticmethod
    def create_grid(shape, size, filling_factor=1, xreflection=False,
                    yreflection=False, rotation=0, xcenter=0, ycenter=0,
                    out=None):
        """
        Return the physical positions of the corners of square detectors in a
        matrix of shape (nrows, ncolumns).

           Y ^
             |
             +--> X
        Before rotation, rows increase along the Y axis (unless yreflection is
        set to True) and columns along the X axis (unless xreflection is set
        to True).

        Parameters
        ----------
        shape : tuple of two integers (nrows, ncolumns)
            Number of rows and columns of the grid.
        size : float
            Detector size, in the same units as the focal distance.
        filling_factor : float
            Fraction of the detector surface that transmits light.
        xreflection : boolean
            Reflection along the x-axis (before rotation).
        yreflection : boolean
            Reflection along the y-axis (before rotation).
        rotation : float
            Counter-clockwise rotation in degrees (before translation).
        xcenter : float
            X coordinate of the grid center.
        ycenter : float
            Y coordinate of the grid center.
        out : array of shape: (nrows, ncolumns, 4, 2), optional
            Placeholder of the output corners.

        Returns
        -------
        corners : array of shape (nrows, ncolumns, 4, 2)
            Corners of the detectors. The first dimension refers to the
            detector row, the second one to the column. The third dimension
            refers to the corner number couterclockwise, starting from the
            bottom-left one. The last dimension's two elements are the X and Y
            coordinates.

        """
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError('The grid must have two dimensions.')
        shape_corners = shape + (4, 2)
        if out is not None:
            if out.shape != shape_corners:
                raise ValueError("The output array has a shape '{0}' incompati"
                                 "ble with that expected '{1}'.".format(
                                 out.shape, shape_corners))
        else:
            out = np.empty(shape_corners)
        flib.geometry.create_grid_squares(
            size, filling_factor, xreflection, yreflection, rotation, xcenter,
            ycenter, out.T)
        return out

    def plot(self, transform=None, autoscale=True, **keywords):
        """
        Plot instrument footprint.

        Parameters
        ----------
        transform : Operator
            Operator to be used to transform the input coordinates into
            the data coordinate system.
        autoscale : boolean
            If true, the axes of the plot will be updated to match the
            boundaries of the detectors.

        Example
        -------
        # overlay the detector grid on the observation pointings
        obs = MyObservation(...)
        annim = obs.pointing.plot()
        transform = lambda x: obs.instrument._instrument2xy(x, obs.pointing[0],
                              annim.hdr)
        obs.instrument.plot(transform, autoscale=False)

        """
        a = mp.gca()

        if transform is None:
            transform = I
        else:
            transform = asoperator(transform)

        if self.nvertices > 0:
            coords = self.get_vertices()
        else:
            coords = self.get_centers()

        transform(coords, out=coords)

        if self.nvertices > 0:
            patches = coords.reshape((-1,) + coords.shape[-2:])
            for p in patches:
                a.add_patch(mp.Polygon(p, closed=True, fill=False, **keywords))
        else:
            if 'color' not in keywords:
                keywords['color'] = 'black'
            if 'marker' not in keywords:
                keywords['marker'] = 'o'
            if 'linestyle' not in keywords:
                keywords['linestyle'] = ''
            mp.plot(coords[..., 0], coords[..., 1], **keywords)

        if autoscale:
            mp.autoscale()

        mp.show()


class Imager(Instrument):
    """
    An Imager is an Instrument for which a relationship between the object
    plane and image plane world coordinates does exist (unlike an inter-
    ferometer).

    Attributes
    ----------
    object2image : Operator
        Transform from object plane to image plane coordinates.
    image2object : Operator
        Transform from image plane to object plane coordinates.

    """
    def __init__(self, name, shape, removed=None, masked=None, nvertices=4,
                 default_resolution=None, origin='upper', dtype=None,
                 comm=MPI.COMM_WORLD, image2object=None, object2image=None):
        if image2object is None and object2image is None:
            raise ValueError('Neither the image2object nor the object2image tr'
                             'ansforms are speficied.')
        Instrument.__init__(self, name, shape, removed=removed, masked=masked,
                            nvertices=nvertices,
                            default_resolution=default_resolution,
                            origin=origin, dtype=dtype, comm=comm)
        if object2image is not None:
            self.object2image = asoperator(object2image)
            self.image2object = self.object2image.I
        else:
            self.image2object = asoperator(image2object)
            self.object2image = self.image2object.I
