# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#

import os
import sys

import numpy as np
from astropy.io import fits as pyfits

from pyoperators.utils import omp_num_threads, product, split, strshape
from pyoperators.utils.mpi import DTYPE_MAP, MPI, combine

from .wcsutils import create_fitsheader_for, has_wcs

__all__ = []


def gather_fitsheader(header, comm=MPI.COMM_WORLD):
    """
    Combine headers of local arrays into a global one.

    """
    if 'NAXIS' not in header:
        raise KeyError("The FITS header does not contain the 'NAXIS' keyword.")
    if 'NAXIS1' not in header:
        raise KeyError('Scalar FITS headers cannot be gathered.')
    naxis = str(header['NAXIS'])
    nlocal = header['NAXIS' + naxis]
    nglobal = combine(nlocal, comm=comm)
    s = split(nglobal, comm.size, comm.rank)
    header = header.copy()
    header['NAXIS' + naxis] = nglobal
    if 'CRPIX' + naxis in header:
        header['CRPIX' + naxis] += s.start
    return header


def gather_fitsheader_if_needed(header, comm=MPI.COMM_WORLD):
    """
    Combine headers of local arrays into a global one unless the input is
    global.

    """
    if 'NAXIS' not in header:
        raise KeyError("The FITS header does not contain the 'NAXIS' keyword.")
    if 'NAXIS1' not in header:
        raise KeyError('Scalar FITS headers cannot be combined.')
    if comm.size == 1:
        return header
    if not has_wcs(header):
        raise ValueError(
            'The input FITS header does not define a world coordi' 'nate system.'
        )
    naxis = header['NAXIS']
    required_same = ['CRVAL1', 'CRVAL2', 'CRTYPE1', 'CRTYPE2', 'CDELT1', 'CDELT2']
    required_same += ['NAXIS' + str(i + 1) for i in range(naxis - 1)]
    required_same += ['CRPIX' + str(i + 1) for i in range(naxis - 1)]
    headers = comm.allgather(header)
    for keyword in required_same:
        if keyword not in header:
            continue
        values = [h[keyword] for h in headers]
        if any(v != headers[0][keyword] for v in values):
            raise ValueError(
                f'The FITS keyword {keyword!r} has different values across MPI '
                f'processes: {values}'
            )

    keyword = 'CRPIX' + str(naxis)
    if all(h[keyword] == headers[0][keyword] for h in headers):
        return header
    return gather_fitsheader(header, comm=comm)


def scatter_fitsheader(header, comm=MPI.COMM_WORLD):
    """
    Return the header of local arrays given that of the global array.
    """
    if 'NAXIS' not in header:
        raise KeyError("The FITS header does not contain the 'NAXIS' keyword.")
    if 'NAXIS1' not in header:
        raise KeyError('Scalar FITS headers cannot be split.')
    axis = str(header['NAXIS'])
    nglobal = header['NAXIS' + axis]
    s = split(nglobal, comm.size, comm.rank)
    header = header.copy()
    header['NAXIS' + axis] = s.stop - s.start
    if 'CRPIX' + axis in header:
        header['CRPIX' + axis] -= s.start
    return header


def distribute_observation(detectors, observations, rank=None, comm=MPI.COMM_WORLD):
    size = comm.size
    if size == 1:
        return detectors.copy(), list(observations)

    if rank is None:
        rank = comm.Get_rank()
    nthreads = omp_num_threads()
    ndetectors = np.sum(~detectors)
    nobservations = len(observations)

    # number of observations. They should approximatively be of the same length
    nx = nobservations

    # number of detectors, grouped by the number of cpu cores
    ny = int(np.ceil(ndetectors / nthreads))

    # we start with the minimum blocksize and increase it until we find a
    # configuration that covers all the observations
    blocksize = int(np.ceil(nx * ny / size))
    while True:
        # by looping over x first, we favor larger numbers of detectors and
        # fewer numbers of observations per processor, to minimise inter-
        # processor communication in case of correlations between
        # detectors
        for xblocksize in range(1, blocksize + 1):
            if blocksize / xblocksize != blocksize // xblocksize:
                continue
            yblocksize = int(blocksize // xblocksize)
            nx_block = int(np.ceil(nx / xblocksize))
            ny_block = int(np.ceil(ny / yblocksize))
            if nx_block * ny_block <= size:
                break
        if nx_block * ny_block <= size:
            break
        blocksize += 1

    ix = rank // ny_block
    iy = rank % ny_block

    # check that the processor has something to do
    if ix >= nx_block:
        idetector = slice(0, 0)
        iobservation = slice(0, 0)
    else:
        idetector = slice(iy * yblocksize * nthreads, (iy + 1) * yblocksize * nthreads)
        iobservation = slice(ix * xblocksize, (ix + 1) * xblocksize)

    detectors_ = detectors.copy()
    igood = np.where(~detectors_.ravel())[0]
    detectors_.ravel()[igood[0 : idetector.start]] = True
    detectors_.ravel()[igood[idetector.stop :]] = True
    observations_ = observations[iobservation]

    return detectors_, observations_


def read_fits(filename, extname, comm):
    """
    Read and distribute a FITS file into local arrays.

    Parameters
    ----------
    filename : str
        The FITS file name.
    extname : str
        The FITS extension name. Use None to read the first HDU with data.
    comm : mpi4py.Comm
        The MPI communicator of the local arrays.
    """

    # check if the file name is the same for all MPI jobs
    files = comm.allgather(filename + str(extname))
    all_equal = all([f == files[0] for f in files])
    if comm.size > 1 and not all_equal:
        raise ValueError('The file name is not the same for all MPI jobs.')

    # get primary hdu or extension
    fits = pyfits.open(filename)
    if extname is not None:
        hdu = fits[extname]
    else:
        ihdu = 0
        while True:
            try:
                hdu = fits[ihdu]
            except IndexError:
                raise OSError('The FITS file has no data.')
            if hdu.header['NAXIS'] == 0:
                ihdu += 1
                continue
            if hdu.data is not None:
                break

    header = hdu.header
    n = header['NAXIS' + str(header['NAXIS'])]
    s = split(n, comm.size, comm.rank)
    output = pyfits.Section(hdu)[s]

    if not output.dtype.isnative:
        output = output.byteswap().newbyteorder('=')

    # update the header
    header['NAXIS' + str(header['NAXIS'])] = s.stop - s.start
    try:
        if header['CTYPE1'] == 'RA---TAN' and header['CTYPE2'] == 'DEC--TAN':
            header['CRPIX2'] -= s.start
    except KeyError:
        pass
    comm.Barrier()

    return output, header


def write_fits(filename, data, header, extension, extname, comm):
    """
    Collectively write local arrays into a single FITS file.

    Parameters
    ----------
    filename : str
        The FITS file name.
    data : ndarray
        The array to be written.
    header : pyfits.Header
        The data FITS header. None can be set, in which case a minimal FITS
        header will be inferred from the data.
    extension : boolean
        If True, the data will be written as an extension to an already
        existing FITS file.
    extname : str
        The FITS extension name. Use None to write the primary HDU.
    comm : mpi4py.Comm
        The MPI communicator of the local arrays. Use MPI.COMM_SELF if the data
        are not meant to be combined into a global array. Make sure that the
        MPI processes are not executing this routine with the same file name.

    """
    # check if the file name is the same for all MPI jobs
    files = comm.allgather(filename + str(extname))
    all_equal = all(f == files[0] for f in files)
    if comm.size > 1 and not all_equal:
        raise ValueError('The file name is not the same for all MPI jobs.')
    ndims = comm.allgather(data.ndim)
    if any(n != ndims[0] for n in ndims):
        raise ValueError(
            f'The arrays have an incompatible number of dimensions: {ndims}.'
        )
    ndim = ndims[0]
    shapes = comm.allgather(data.shape)
    if any(s[1:] != shapes[0][1:] for s in shapes):
        raise ValueError(f"The arrays have incompatible shapes: '{strshape(shapes)}'.")

    # get header
    if header is None:
        header = create_fitsheader_for(data, extname=extname)
    else:
        header = header.copy()
    if extname is not None:
        header['extname'] = extname

    # we remove the file first to avoid an annoying pyfits informative message
    if not extension:
        if comm.rank == 0:
            try:
                os.remove(filename)
            except OSError:
                pass

    # case without MPI communication
    if comm.size == 1:
        if not extension:
            hdu = pyfits.PrimaryHDU(data, header)
            hdu.writeto(filename, overwrite=True)
        else:
            pyfits.append(filename, data, header)
        return

    # get global/local parameters
    nglobal = sum(s[0] for s in shapes)
    s = split(nglobal, comm.size, comm.rank)
    nlocal = s.stop - s.start
    if data.shape[0] != nlocal:
        msg = '' if comm.rank > 0 else f' Shapes are: {shapes}.'
        raise ValueError(
            f"On rank {comm.rank}, the local array shape '{data.shape}' is invalid. "
            f"The first dimension does not match the expected local number '{nlocal}' "
            f"given the global number '{nglobal}'.{msg}"
        )

    # write FITS header
    if comm.rank == 0:
        header['NAXIS' + str(ndim)] = nglobal
        shdu = pyfits.StreamingHDU(filename, header)
        data_loc = shdu._datLoc
        shdu.close()
    else:
        data_loc = None
    data_loc = comm.bcast(data_loc)

    # get a communicator excluding the processes which have no work to do
    # (Create_subarray does not allow 0-sized subarrays)
    chunk = product(data.shape[1:])
    rank_nowork = min(comm.size, nglobal)
    group = comm.Get_group()
    group.Incl(list(range(rank_nowork)))
    newcomm = comm.Create(group)

    # collectively write data
    if comm.rank < rank_nowork:
        # mpi4py 1.2.2: pb with viewing data as big endian KeyError '>d'
        if (
            sys.byteorder == 'little'
            and data.dtype.byteorder == '='
            or data.dtype.byteorder == '<'
        ):
            data = data.byteswap()
        data = data.newbyteorder('=')
        mtype = DTYPE_MAP[data.dtype]
        ftype = mtype.Create_subarray(
            [nglobal * chunk], [nlocal * chunk], [s.start * chunk]
        )
        ftype.Commit()
        f = MPI.File.Open(
            newcomm, filename, amode=MPI.MODE_APPEND | MPI.MODE_WRONLY | MPI.MODE_CREATE
        )
        f.Set_view(data_loc, mtype, ftype, 'native', MPI.INFO_NULL)
        f.Write_all(data)
        f.Close()
        ftype.Free()
    newcomm.Free()

    # pad FITS file with zeros
    if comm.rank == 0:
        datasize = nglobal * chunk * data.dtype.itemsize
        BLOCK_SIZE = 2880
        padding = BLOCK_SIZE - (datasize % BLOCK_SIZE)
        with open(filename, 'a') as f:
            if f.tell() - data_loc != datasize:
                raise RuntimeError('Unexpected file size.')
            f.write(padding * '\0')

    comm.Barrier()
