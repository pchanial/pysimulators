# Copyrights 2010-2013 Pierre Chanial
# All rights reserved

from __future__ import division

import astropy.io.fits as pyfits
import glob
import numpy as np
import os
import re
from pyoperators import (
    BlockColumnOperator, BlockDiagonalOperator, SymmetricBandToeplitzOperator,
    MPI)
from ...acquisitions import Acquisition
from ...datatypes import Map, Tod
from ...instruments import Instrument
from ...packedtables import PackedTable
from ...operators import PointingMatrix, ProjectionInMemoryOperator
from ...wcsutils import create_fitsheader

__all__ = ['MadMap1Observation']


class MadMap1Observation(Acquisition):
    """
    Class for the handling of an observation in the MADMAP1 format.

    """
    def __init__(self, name, ndetectors, todfilename, invnttfilename,
                 mapmaskfilename, bigendian=False, missing_value=None,
                 commin=MPI.COMM_WORLD, commout=MPI.COMM_WORLD):

        if commin.size > 1 or commout.size > 1:
            raise NotImplementedError('The parallelisation of the TOD is not i'
                                      'mplemented')

        # Get information from files
        filters = _read_filters(invnttfilename, bigendian=bigendian)
        ncorrelations = filters[0]['data'].size
        nblocks = len(filters) // ndetectors
        ns = [f['last'] - f['first'] + 1 for f in filters[::ndetectors]]
        filters = np.array([f['data'] for f in filters]).reshape(
            (nblocks, ndetectors, ncorrelations))
        self._filters = filters

        m = re.search(r'(?P<filename>.*)\[(?P<extname>\w+)\]$',
                      mapmaskfilename)
        if m is None:
            mask = pyfits.open(mapmaskfilename)[0].data
        else:
            filename = m.group('filename')
            extname = m.group('extname')
            mask = pyfits.open(filename)[str(extname)].data #XXX Python3
        if mask is None:
            raise IOError('HDU ' + mapmaskfilename + ' has no data.')
        mapmask = np.zeros(mask.shape, dtype=bool)
        if missing_value is None:
            mapmask[mask != 0] = True
        elif np.isnan(missing_value):
            mapmask[np.isnan(mask)] = True
        elif np.isinf(missing_value):
            mapmask[np.isinf(mask)] = True
        else:
            mapmask[mask == missing_value] = True

        with open(todfilename) as f:
            ldtype = '>i8' if bigendian else '<i8'
            first, last, npps, npixels_in_map = np.fromfile(f, ldtype, count=4)
        if npixels_in_map != np.sum(~mapmask):
            raise ValueError('The map mask is not compatible with the number o'
                             'f map pixels in the TOD file.')

        # Store instrument information
        layout = PackedTable(ndetectors)
        self.instrument = Instrument(name, layout, commin=commin,
                                     commout=commout)

        # Store observation information
        class MadMap1ObservationInfo(object):
            pass
        self.info = MadMap1ObservationInfo()
        self.info.todfilename = todfilename
        self.info.invnttfilename = invnttfilename
        self.info.mapmaskfilename = mapmaskfilename
        self.info.npixels_per_sample = npps
        self.info.npixels_in_map = npixels_in_map
        self.info.ncorrelations = ncorrelations
        self.info.bigendian = bigendian
        self.info.missing_value = missing_value
        self.info.mapmask = mapmask

        # Store block information
        self.block = np.recarray(
            nblocks, dtype=[('n', int), ('start', int), ('stop', int)])
        self.block.n = ns
        self.block.start = np.cumsum([0] + ns)[:-1]
        self.block.stop = np.cumsum(ns)

        # Store pointing information
        self.pointing = np.recarray(np.sum(ns), [('removed', np.bool_)])
        self.pointing.removed = False

    def get_map_header(self, **keywords):
        return create_fitsheader(np.sum(~self.info.mapmask), dtype=float)

    def get_invntt_operator(self, fftw_flag='FFTW_MEASURE', nthreads=None):
        ops = [SymmetricBandToeplitzOperator(
            (self.get_ndetectors(), b.n), self._filters,
            fftw_flag=fftw_flag, nthreads=nthreads) for b in self.block]
        return BlockDiagonalOperator(ops, axisin=-1)

    def get_projection_operator(self):
        header = self.get_map_header()
        junk, pmatrix = self._read_tod_pmatrix()
        return BlockColumnOperator(
            [ProjectionInMemoryOperator(
                p, attrin={'header': header}, classin=Map, classout=Tod)
                for p in pmatrix], axisout=-1)

    def get_operator(self, aslist=False):
        operands = [self.get_projection_operator()]
        if aslist:
            return operands
        return operands[0]

    def get_tod(self, unit=None):
        """
        Method to get the Tod from this observation.

        """
        tod, junk = self._read_tod_pmatrix()
        if unit is not None:
            tod.unit = unit
        return tod

    def get_filter_uncorrelated(self, **keywords):
        """
        Method to get the invNtt for uncorrelated detectors.

        """
        return self._filters

    def _read_tod_pmatrix(self):
        ndetectors = self.get_ndetectors()
        nsamples = np.sum(self.get_nsamples())
        npps = self.info.npixels_per_sample
        filesize = os.stat(self.info.todfilename).st_size
        if filesize != 32 + (npps + 1) * ndetectors * nsamples * 8:
            raise ValueError("Invalid size '{0}' for file '{1}'.".format(
                             filesize, self.info.todfilename))
        tod = Tod.empty((ndetectors, nsamples))
        pmatrix = []
        with open(self.info.todfilename) as f:
            f.seek(32)
            idtype = '>i4' if self.info.bigendian else '<i4'
            fdtype = '>f4' if self.info.bigendian else '<f4'
            ddtype = '>f8' if self.info.bigendian else '<f8'
            pdtype = [('tod', ddtype), ('matrix', [('value', fdtype),
                      ('index', idtype)], (npps,))]
            for b in self.block:
                data = np.fromfile(f, pdtype, count=b.n * ndetectors)
                data = data.reshape((ndetectors, b.n))
                tod[:, b.start:b.stop] = data['tod']
                p = PointingMatrix.empty((ndetectors, b.n, npps),
                                         self.info.npixels_in_map)
                p[...] = data['matrix']
                pmatrix.append(p)
        return tod, pmatrix


def _read_one_filter(filename, bigendian=False):
    dtype = '>i8' if bigendian else '<i8'
    first, last, ncorr = np.fromfile(filename, dtype, count=3)
    data = np.fromfile(filename, dtype.replace('i', 'f'), count=4+ncorr)[3:]
    return {'first': first, 'last': last, 'data': data}


def _read_filters(filename, bigendian=False):
    files = glob.glob(filename + '.*')
    regex = re.compile(filename + '\.[0-9]+$')
    files = sorted((f for f in files if regex.match(f)),
                   key=lambda s: int(s[-s[::-1].index('.'):]))
    filters = tuple(_read_one_filter(f, bigendian=bigendian) for f in files)
    ncorrelations = filters[0]['data'].size
    if any(f['data'].size != ncorrelations for f in filters):
        raise ValueError('Blocks do not have the same correlation lengths.')
    return filters
