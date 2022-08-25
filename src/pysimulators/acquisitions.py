# Copyrights 2010-2013 Pierre Chanial
# All rights reserved


import gc
import operator
from copy import copy

import numpy as np

from pyoperators import (
    MPI,
    BlockColumnOperator,
    BlockDiagonalOperator,
    CompositionOperator,
)
from pyoperators.memory import empty
from pyoperators.utils import isscalarlike, operation_assignment, split, strenum

from .instruments import Instrument
from .packedtables import Layout, Sampling, Scene

__all__ = ['Acquisition', 'MaskPolicy']


class Acquisition:
    """
    The Acquisition class, which combines the instrument, sampling and scene
    models.

    """

    def __init__(
        self,
        instrument,
        sampling,
        scene,
        block=None,
        max_nbytes=None,
        nprocs_instrument=None,
        nprocs_sampling=None,
        comm=None,
    ):
        """
        Parameters
        ----------
        instrument : Instrument
            The Instrument instance.
        sampling : Sampling
            The sampling information (pointings, etc.)
        scene : Scene
            Discretization of the observed scene.
        block : tuple of slices, optional
            Partition of the samplings.
        max_nbytes : int or None, optional
            Maximum number of bytes to be allocated for the acquisition's
            operator.
        nprocs_instrument : int
            For a given sampling slice, number of procs dedicated to
            the instrument.
        nprocs_sampling : int
            For a given detector slice, number of procs dedicated to
            the sampling.
        comm : mpi4py.MPI.Comm
            The acquisition's MPI communicator. Note that it is transformed
            into a 2d cartesian communicator before being stored as the 'comm'
            attribute. The following relationship must hold:
                comm.size = nprocs_instrument * nprocs_sampling

        """
        if not isinstance(instrument, Instrument):
            raise TypeError(
                f'The instrument input has an invalid type '
                f'{type(instrument).__name__!r}.'
            )
        if not isinstance(sampling, Sampling):
            raise TypeError(
                f'The sampling input has an invalid type {type(instrument).__name__!r}.'
            )
        if not isinstance(scene, Scene):
            raise TypeError(
                f'The scene input has an invalid type {type(scene).__name__!r}.'
            )

        if comm is None:
            comm = MPI.COMM_WORLD
        if nprocs_instrument is None and nprocs_sampling is None:
            nprocs_sampling = comm.size
        if nprocs_instrument is None:
            if nprocs_sampling < 1 or nprocs_sampling > comm.size:
                raise ValueError(
                    f"Invalid value for nprocs_sampling '{nprocs_sampling}'."
                )
            nprocs_instrument = comm.size // nprocs_sampling
        elif nprocs_sampling is None:
            if nprocs_instrument < 1 or nprocs_sampling > comm.size:
                raise ValueError(
                    f"Invalid value for nprocs_instrument '{nprocs_instrument}'."
                )
            nprocs_sampling = comm.size // nprocs_instrument
        if nprocs_instrument * nprocs_sampling != comm.size:
            raise ValueError('Invalid MPI distribution of the acquisition.')

        commgrid = comm.Create_cart([nprocs_sampling, nprocs_instrument], reorder=True)

        comm_instrument = commgrid.Sub([False, True])
        comm_sampling = commgrid.Sub([True, False])

        self.scene = scene
        self.instrument = instrument.scatter(comm_instrument)
        self.sampling = sampling.scatter(comm_sampling)
        self.comm = commgrid
        self.block = block
        if block is None:
            self.block = (slice(0, len(self.sampling)),)
            if max_nbytes is not None:
                nbytes = self.get_operator_nbytes()
                if nbytes > max_nbytes:
                    nblocks = int(np.ceil(nbytes / max_nbytes))
                    self.block = tuple(split(len(self.sampling), nblocks))
        elif not isinstance(block, (list, tuple)) or any(
            not isinstance(b, slice) for b in block
        ):
            raise TypeError(f"Invalid block argument: '{block}'.")

    _operator = None

    def __getitem__(self, x):
        """
        Restrict the acquisition model to a set of detectors, samplings or
        scene pixels.

        new_acq = acq[selection_instrument, selection_sampling,
                      selection_scene]

        Example
        -------
        >>> acq = MyAcquisition()
        Restrict to the first 10 detectors:
        >>> new_acq = acq[:10]
        Restrict to the first 10 samplings:
        >>> new_acq = acq[:, :10]
        Restrict to the first 10 pixels of the scene:
        >>> new_acq = acq[..., :10]
        """
        out = copy(self)
        if not isinstance(x, tuple):
            out.instrument = self.instrument[x]
            return out
        if len(x) == 2 and x[0] is Ellipsis:
            x = Ellipsis, Ellipsis, x[1]
        if len(x) > 3:
            raise ValueError('Invalid selection.')
        x = x + (3 - len(x)) * (Ellipsis,)
        if x[2] is not Ellipsis and (
            not isinstance(x[2], slice) or x[2] == slice(None)
        ):
            self._operator = None
            gc.collect()
        out.instrument = self.instrument[x[0]]
        out.sampling = self.sampling[x[1]]  # XXX FIX BLOCKS!!!
        out.scene = self.scene[x[2]]
        return out

    def __str__(self):
        return f'{self.instrument}\nSamplings: {len(self.sampling)}'

    __repr__ = __str__

    def pack(self, x, out=None, copy=False):
        return self.instrument.detector.pack(x)

    pack.__doc__ = Layout.pack.__doc__

    def unpack(self, x, out=None, missing_value=None, copy=False):
        return self.instrument.detector.unpack(x)

    unpack.__doc__ = Layout.unpack.__doc__

    def get_observation(
        self, x, noiseless=False, out=None, operation=operation_assignment
    ):
        """
        Return out=H(x)+n, the observation according to the acquisition model.

        Parameters
        ----------
        x : float array
            The input values of the Scene.
        noiseless : boolean, optional
            If True, no noise is added to the observation.
        out : float array
            Buffer of shape len(instrument)xlen(sampling) for the output
            observation.
        operation : function
            get_observation(x, out=out, operation=operation) is equivalent to
            operation(out, get_observation(x))

        """
        H = self.get_operator()
        out = H(x, out, operation=operation)
        if not noiseless:
            self.get_noise(out, operation=operator.iadd)
        return out

    def get_operator(self):
        """
        Return the acquisition model H as an operator.

        """
        if self._operator is None:
            self._operator = CompositionOperator(
                [
                    BlockColumnOperator(
                        [
                            self.instrument.get_operator(self.sampling[b], self.scene)
                            for b in self.block
                        ],
                        axisin=1,
                    ),
                    self.scene.get_distribution_operator(self.comm),
                ]
            )
        return self._operator

    def get_operator_nbytes(self):
        """
        Return the number of bytes required to store the acquisition model
        as an operator.

        """
        n = self[0, 0].get_operator().nbytes
        return n * len(self.instrument) * len(self.sampling)

    def get_invntt_operator(self):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        """
        return BlockDiagonalOperator(
            [self.instrument.get_invntt_operator(self.sampling[b]) for b in self.block],
            axisin=1,
        )

    def get_noise(self, out=None, operation=operation_assignment):
        """
        Return the noise realization according the instrument's noise model.

        Parameters
        ----------
        out : ndarray, optional
            Placeholder for the output noise.

        """
        if out is None:
            if operation is not operation_assignment:
                raise ValueError('The output buffer is not specified.')
            out = empty((len(self.instrument), len(self.sampling)))
        for b in self.block:
            self.instrument.get_noise(
                self.sampling[b], out=out[:, b], operation=operation
            )
        return out

    def plot(self, map=None, header=None, new_figure=True, percentile=0.01, **keywords):
        """
        map : ndarray of dim 2
            The optional map to be displayed as background.
        header : pyfits.Header
            The optional map's FITS header.
        new_figure : boolean
            If true, plot the scan in a new window.
        percentile : float, tuple of two floats
            As a float, percentile of values to be discarded, otherwise,
            percentile of the minimum and maximum values to be displayed.

        """
        if header is None:
            header = getattr(map, 'header', None)
            if header is None:
                header = self.sampling.get_map_header(naxis=1)
        annim = self.sampling.plot(
            map=map,
            header=header,
            new_figure=new_figure,
            percentile=percentile,
            **keywords,
        )
        return annim


class MaskPolicy:
    KEEP = 0
    MASK = 1
    REMOVE = 2

    def __init__(self, flags, values, description=None):
        self._description = description
        if isscalarlike(flags):
            if isinstance(flags, str):
                flags = flags.split(',')
            else:
                flags = [flags]
        flags = [_.strip() for _ in flags]
        if isscalarlike(values):
            if isinstance(values, str):
                values = values.split(',')
            else:
                values = [values]
        values = [_.strip().lower() for _ in values]
        if len(flags) != len(values):
            raise ValueError(
                'The number of policy flags is different from the'
                ' number of policies.'
            )

        self._policy = []
        for flag, value in zip(flags, values):
            if flag[0] == '_':
                raise ValueError(
                    'A policy flag should not start with an under' 'score.'
                )
            choices = 'keep', 'mask', 'remove'
            if value not in choices:
                raise KeyError(
                    f'Invalid policy {flag}={value!r}. Expected values are '
                    f'{strenum(choices)}.'
                )
            setattr(self, flag, value)
        self._flags = flags

    def __array__(self, dtype=int):
        conversion = {'keep': self.KEEP, 'mask': self.MASK, 'remove': self.REMOVE}
        return np.array(
            [conversion[getattr(self, _)] for _ in self._flags], dtype=dtype
        )

    def __str__(self):
        s = self._description + ': ' if self._description is not None else ''
        s += ', '.join(f'{_}={getattr(self, _)!r}' for _ in self._flags)
        return s
