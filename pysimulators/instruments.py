from __future__ import division
import copy

try:
    from matplotlib import pyplot as mp
except ImportError:
    pass
from pyoperators import I, asoperator
from pyoperators.utils import split
from pyoperators.utils.mpi import MPI

from .packedtables import Layout

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

    """

    def __init__(self, name, layout):
        self.name = str(name)
        self.detector = layout

    def __getitem__(self, selection):
        """
        Shallow copy of the Instrument for the selected deep-copied
        non-removed detectors.

        """
        out = copy.copy(self)
        out.detector = self.detector[selection]
        return out

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.detector)

    def pack(self, x):
        return self.detector.pack(x)

    pack.__doc__ = Layout.pack.__doc__

    def unpack(self, x):
        return self.detector.unpack(x)

    unpack.__doc__ = Layout.unpack.__doc__

    def scatter(self, comm=None):
        """
        MPI-scatter of the instrument.

        Parameter
        ---------
        comm : MPI.Comm
            The MPI communicator of the group of processes in which
            the instrument will be scattered.

        """
        if self.detector.comm.size > 1:
            raise ValueError('The instrument is already distributed.')
        if comm is None:
            comm = MPI.COMM_WORLD
        out = copy.copy(self)
        out.detector = out.detector.scatter(comm)
        return out

    def split(self, n):
        """
        Split the instrument in partitioning groups.

        Example
        -------
        >>> instr = Instrument('instr', Layout((4, 4)))
        >>> [len(_) for _ in instr.split(2)]
        [8, 8]

        """
        return tuple(self[_] for _ in split(len(self), n))

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

    def __init__(self, name, layout, image2object=None, object2image=None):
        if image2object is None and object2image is None:
            raise ValueError(
                'Neither the image2object nor the object2image tr'
                'ansforms are speficied.'
            )
        Instrument.__init__(self, name, layout)
        if object2image is not None:
            self.object2image = asoperator(object2image)
            self.image2object = self.object2image.I
        else:
            self.image2object = asoperator(image2object)
            self.object2image = self.image2object.I
