from __future__ import absolute_import, division, print_function
from pyoperators.utils import strenum
from .operators import Spherical2HealpixOperator
from ...packedtables import Scene

__all__ = ['SceneHealpix']


class SceneHealpix(Scene):
    """
    Class for Healpix scenes.

    """
    def __init__(self, nside, kind='I', nest=False,
                 _convention='zenith,azimuth', **keywords):
        """
        nside : int
            Value of the map resolution parameter.*
        kind : 'I', 'QU' or 'IQU'
            The kind of sky: intensity-only, Q and U Stokes parameters only or
            intensity, Q and U.
        _convention : string, optional
            One of the following spherical coordinate conventions:
            'zenith,azimuth', 'azimuth,zenith', 'elevation,azimuth' and
            'azimuth,elevation'.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            the ring scheme.

        """
        kinds = 'I', 'QU', 'IQU'
        if not isinstance(kind, str):
            raise TypeError(
                'Invalid type {0!r} for the scene kind. Expected type is strin'
                'g.'.format(type(kind).__name__))
        kind = kind.upper()
        if kind not in kinds:
            raise ValueError('Invalid scene kind {0!r}. Expected kinds are: {1'
                             '}.'.format(kind, strenum(kinds)))
        nside = int(nside)
        topixel = Spherical2HealpixOperator(nside, _convention, nest)
        shape = (12 * nside**2,)
        if kind != 'I':
            shape += (len(kind),)
        Scene.__init__(self, shape, topixel, ndim=1, **keywords)
        self.kind = kind
        self.nside = nside
        self.nest = bool(nest)
