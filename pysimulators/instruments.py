from __future__ import division

from matplotlib import pyplot as mp
from pyoperators import I, asoperator
from pyoperators.utils.mpi import MPI

from .layouts import Layout

__all__ = ['Instrument', 'Imager']


class Instrument(object):
    """
    Class storing information about the instrument.

    Attributes
    ----------
    name : str
        The instrument configuration name.
    layout : Layout
        The detector layout.
    commin : mpi4py.MPI.COMM
        The MPI communicator for the input map.
    commout : mpi4py.MPI.COMM
        The MPI communicator for the output time-ordered data.

    """
    def __init__(self, name, layout, default_resolution=None,
                 commin=MPI.COMM_WORLD, commout=MPI.COMM_WORLD):
        self.name = str(name)
        self.detector = layout
        self.default_resolution = default_resolution
        self.commin = commin
        self.commout = commout

    def pack(self, x):
        return self.detector.pack(x)
    pack.__doc__ = Layout.pack.__doc__

    def unpack(self, x):
        return self.detector.unpack(x)
    unpack.__doc__ = Layout.unpack.__doc__

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

        if self.detector.nvertices > 0:
            coords = self.detector.vertex
        else:
            coords = self.detector.center

        transform(coords, out=coords)

        if self.detector.nvertices > 0:
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
    An Imager is an Instrument for which a relationship between the world
    coordinates of the object plane and the image plane does exist (unlike
    an interferometer).

    Attributes
    ----------
    object2image : Operator
        Transform from object plane to image plane coordinates.
    image2object : Operator
        Transform from image plane to object plane coordinates.

    """
    def __init__(self, name, layout, default_resolution=None,
                 commin=MPI.COMM_WORLD, commout=MPI.COMM_WORLD,
                 image2object=None, object2image=None):
        if image2object is None and object2image is None:
            raise ValueError('Neither the image2object nor the object2image tr'
                             'ansforms are speficied.')
        Instrument.__init__(self, name, layout,
                            default_resolution=default_resolution,
                            commin=commin, commout=commout)
        if object2image is not None:
            self.object2image = asoperator(object2image)
            self.image2object = self.object2image.I
        else:
            self.image2object = asoperator(image2object)
            self.object2image = self.image2object.I
