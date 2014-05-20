"""
Patch PyOperator's SparseOperator to include the following formats:
    - Fixed Sparse Column (FSCMatrix)
    - Fixed Sparse Row (FSRMatrix)
    - Fixed Sparse Column Rotation 2d (FSCRotation2dMatrix)
    - Fixed Sparse Row Rotation 2d (FSRRotation2dMatrix)
    - Fixed Sparse Column Rotation 3d (FSCRotation3dMatrix)
    - Fixed Sparse Row Rotation 3d (FSRRotation3dMatrix)

"""
from __future__ import division

import numpy as np
import operator
import pyoperators
import scipy.sparse as sp
from pyoperators import operation_assignment
from pyoperators.memory import empty
from pyoperators.utils import isscalarlike, product
from pysimulators._flib import sparse as fsp

__all__ = []


class _FSMatrix(object):
    def __init__(self, shape, sparse_axis, n, data=None, dtype=None,
                 dtype_index=None, dtype_names=('value',), block_size=1,
                 verbose=False):
        if not isinstance(shape, tuple):
            raise TypeError("Invalid shape '{0}'.".format(shape))
        if len(shape) != 2:
            raise ValueError("The number of dimensions is not 2.")
        straxes = ('row', 'column')
        if sparse_axis == 1:
            straxes = straxes[::-1]
        if data is None:
            if n is None:
                raise ValueError('The maximum number of non-zero {0}s per {1} '
                                 'is not specified.'.format(*straxes))
            shape_data = (shape[1-sparse_axis] // block_size, n)
            if dtype is None:
                dtype = float
            if dtype_index is None:
                dtype_index = int
            dtype_data = [('index', dtype_index)] + [(name, dtype)
                                                     for name in dtype_names]
            data = empty(shape_data, dtype_data, verbose=verbose).view(
                np.recarray)
        elif data.dtype.names != ('index',) + dtype_names:
            raise TypeError('The fields of the structured array are invalid.')
        elif any(s % block_size != 0 for s in shape):
            raise ValueError(
                "The shape of the matrix '{0}' is not a multiple of '{1}'.".
                format(shape, block_size))
        elif shape[1-sparse_axis] // block_size != product(data.shape[:-1]):
            raise ValueError(
                "The shape of the matrix '{0}' is incompatible with that of th"
                "e structured array '{1}'.".format(shape, data.shape))
        elif n not in (None, data.shape[-1]):
            raise ValueError(
                "The n{0}max keyword value '{1}' is incompatible with the shap"
                "e of the structured array '{2}'.".format(
                    straxes[0][:3], n, data.shape))
        elif any(data[name].dtype != data[dtype_names[0]].dtype
                 for name in dtype_names):
            raise TypeError(
                'The data fields of the structured array do not have the same '
                'dtype.')
        else:
            data = data.view(np.recarray)
            if n is None:
                n = data.shape[-1]
            dtype = data[dtype_names[0]].dtype
        self.dtype = np.dtype(dtype)
        self.data = data
        self.ndim = 2
        self.shape = shape
        self.block_size = block_size
        setattr(self, 'n' + straxes[0][:3] + 'max', n)

    def _matvec(self, fname, v, out, nmax):
        v = np.asarray(v)
        if v.size != self.shape[1]:
            raise ValueError(
                "The input array has an invalid size '{0}'. The expected size "
                "is '{1}'.".format(v.size, self.shape[1]))
        if out is None:
            out = np.zeros(self.shape[0],
                           np.find_common_type([self.dtype, v.dtype], []))
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif out.size != self.shape[0]:
            raise ValueError(
                "The output array has an invalid size '{0}'. The expected size"
                " is '{1}'.".format(out.size, self.shape[0]))
        elif not out.flags.contiguous:
            raise ValueError('The output array is not contiguous.')

        di = self.data.index.dtype
        ds = self.dtype
        dv = v.dtype
        if ds.type not in (np.float32, np.float64) or \
           di.type not in (np.int32, np.int64):
            return out, False

        out = out.ravel()
        if dv.kind != 'f' or dv.type in (np.float16, np.float128) or \
           dv.type is np.float32 and ds.type is np.float64:
            v = v.astype(self.dtype)
            dv = self.dtype
        if out.dtype != dv:
            out_ = np.empty(self.shape[0], dtype=dv)
            out_[...] = out
        else:
            out_ = out
        f = '{0}_matvec_i{1}_m{2}_v{3}'.format(
            fname, di.itemsize, ds.itemsize, dv.itemsize)
        func = getattr(fsp, f)
        m = self.data.ravel().view(np.int8)
        func(m, v.ravel(), out_, nmax)
        if out.dtype != dv:
            out[...] = out_
        return out, True

    def __mul__(self, other):
        if isscalarlike(other):
            data = self.data.copy()
            data.value *= other
            return type(self)(self.shape, data=data)
        other = np.asarray(other)
        if other.ndim == 1:
            return self._matvec(other)
        return NotImplemented

    def __rmul__(self, other):
        if isscalarlike(other):
            return self * other
        other = np.asarray(other)
        if other.ndim == 1:
            return self._transpose() * other
        return NotImplemented

    __array_priority__ = 10.1


class FSCMatrix(_FSMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column.

    """
    def __init__(self, shape, data=None, nrowmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSMatrix.__init__(self, shape, 0, nrowmax, data=data, dtype=dtype,
                           dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsc', v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out = out.ravel()
        v = v.ravel()
        if data.index.dtype.kind == 'u':
            for i in xrange(self.shape[1]):
                for b in data[i]:
                    out[b.index] += b.value * v[i]
        else:
            for i in xrange(self.shape[1]):
                for b in data[i]:
                    if b.index >= 0:
                        out[b.index] += b.value * v[i]
        return out

    def _transpose(self):
        return FSRMatrix(self.shape[::-1], data=self.data)


class FSRMatrix(_FSMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row.

    """
    def __init__(self, shape, data=None, ncolmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSMatrix.__init__(self, shape, 1, ncolmax, data=data, dtype=dtype,
                           dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsr', v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out = out.ravel()
        v = v.ravel()
        if data.index.dtype.kind == 'u':
            for i in xrange(self.shape[0]):
                b = data[i]
                out[i] += np.sum(b.value * v[b.index])
        else:
            for i in xrange(self.shape[0]):
                b = data[i]
                b = b[b.index >= 0]
                out[i] += np.sum(b.value * v[b.index])
        return out

    def _transpose(self):
        return FSCMatrix(self.shape[::-1], data=self.data)


class _FSRotation2dMatrix(_FSMatrix):
    def __init__(self, shape, sparse_axis, n, data=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSMatrix.__init__(
            self, shape, sparse_axis, n, data=data, dtype=dtype,
            dtype_index=dtype_index, dtype_names=('r11', 'r21'),
            block_size=2, verbose=verbose)

    def __mul__(self, other):
        if isscalarlike(other):
            data = self.data.copy()
            data.r11 *= other
            data.r21 *= other
            return type(self)(self.shape, data=data)
        other = np.asarray(other)
        if other.ndim == 1:
            return self._matvec(other)
        return NotImplemented


class FSCRotation2dMatrix(_FSRotation2dMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. Each element of the sparse matrix is a 2x2 rotation
    matrix.

    """
    def __init__(self, shape, data=None, nrowmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSRotation2dMatrix.__init__(
            self, shape, 0, nrowmax, data=data, dtype=dtype,
            dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsc_rot2d', v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in xrange(self.shape[1] // 2):
            for b in data[i]:
                if b.index >= 0:
                    out_[b.index, 0] +=  b.r11 * v[i, 0] + b.r21 * v[i, 1]
                    out_[b.index, 1] += -b.r21 * v[i, 0] + b.r11 * v[i, 1]
        return out

    def _transpose(self):
        return FSRRotation2dMatrix(self.shape[::-1], data=self.data)


class FSRRotation2dMatrix(_FSRotation2dMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. Each element of the sparse matrix is a 2x2 rotation matrix.

    """
    def __init__(self, shape, data=None, ncolmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSRotation2dMatrix.__init__(
            self, shape, 1, ncolmax, data=data, dtype=dtype,
            dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsr_rot2d', v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in xrange(self.shape[0] // 2):
            b = data[i]
            b = b[b.index >= 0]
            out_[i, 0] += np.sum(b.r11 * v[b.index, 0] -
                                 b.r21 * v[b.index, 1])
            out_[i, 1] += np.sum(b.r21 * v[b.index, 0] +
                                 b.r11 * v[b.index, 1])
        return out

    def _transpose(self):
        return FSCRotation2dMatrix(self.shape[::-1], data=self.data)


class _FSRotation3dMatrix(_FSMatrix):
    def __init__(self, shape, sparse_axis, n, data=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSMatrix.__init__(
            self, shape, sparse_axis, n, data=data, dtype=dtype,
            dtype_index=dtype_index, dtype_names=('r11', 'r22', 'r32'),
            block_size=3, verbose=verbose)

    def __mul__(self, other):
        if isscalarlike(other):
            data = self.data.copy()
            data.r11 *= other
            data.r22 *= other
            data.r32 *= other
            return type(self)(self.shape, data=data)
        other = np.asarray(other)
        if other.ndim == 1:
            return self._matvec(other)
        return NotImplemented


class FSCRotation3dMatrix(_FSRotation3dMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. Each element of the sparse matrix is a 3x3 block
    whose transpose performs a rotation about the first component axis.

    """
    def __init__(self, shape, data=None, nrowmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSRotation3dMatrix.__init__(
            self, shape, 0, nrowmax, data=data, dtype=dtype,
            dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsc_rot3d', v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape(-1, self.nrowmax)
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in xrange(self.shape[1] // 3):
            for b in data[i]:
                if b.index >= 0:
                    out_[b.index, 0] +=  b.r11 * v[i, 0]
                    out_[b.index, 1] +=  b.r22 * v[i, 1] + b.r32 * v[i, 2]
                    out_[b.index, 2] += -b.r32 * v[i, 1] + b.r22 * v[i, 2]
        return out

    def _transpose(self):
        return FSRRotation3dMatrix(self.shape[::-1], data=self.data)


class FSRRotation3dMatrix(_FSRotation3dMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. Each element of the sparse matrix is a 3x3 block
    performing a rotation about the first component axis.

    """
    def __init__(self, shape, data=None, ncolmax=None, dtype=None,
                 dtype_index=None, verbose=False):
        _FSRotation3dMatrix.__init__(
            self, shape, 1, ncolmax, data=data, dtype=dtype,
            dtype_index=dtype_index, verbose=verbose)

    def _matvec(self, v, out=None):
        out, done = _FSMatrix._matvec(self, 'fsr_rot3d', v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in xrange(self.shape[0] // 3):
            b = data[i]
            b = b[b.index >= 0]
            out_[i, 0] += np.sum(b.r11 * v[b.index, 0])
            out_[i, 1] += np.sum(b.r22 * v[b.index, 1] -
                                 b.r32 * v[b.index, 2])
            out_[i, 2] += np.sum(b.r32 * v[b.index, 1] +
                                 b.r22 * v[b.index, 2])
        return out_

    def _transpose(self):
        return FSCRotation3dMatrix(self.shape[::-1], data=self.data)


class SparseOperator(pyoperators.linear.SparseBase):
    def __init__(self, arg, dtype=None, **keywords):
        if sp.issparse(arg):
            self.__class__ = pyoperators.linear.SparseOperator
            self.__init__(arg, dtype=None, **keywords)
            return
        if not isinstance(arg, (FSCMatrix, FSRMatrix,
                                FSCRotation2dMatrix, FSRRotation2dMatrix,
                                FSCRotation3dMatrix, FSRRotation3dMatrix)):
            raise TypeError('The input sparse matrix type is not recognised.')
        if isinstance(arg, (FSCRotation2dMatrix, FSRRotation2dMatrix,
                            FSCRotation3dMatrix, FSRRotation3dMatrix)):
            n = arg.block_size
            if 'shapein' not in keywords:
                keywords['shapein'] = (arg.shape[1] // n, n)
            if 'shapeout' not in keywords:
                keywords['shapeout'] = (arg.shape[0] // n, n)

        def direct(input, output, operation=operation_assignment):
            if operation is operation_assignment:
                output[...] = 0
            elif operation is not operator.iadd:
                raise ValueError(
                    'Invalid reduction operation: {0}.'.format(operation))
            self.matrix._matvec(input, output)

        pyoperators.linear.SparseBase.__init__(
            self, arg, dtype=dtype, direct=direct, **keywords)
        self.set_rule('T', lambda s: SparseOperator(s.matrix._transpose(),
                                                    dtype=s.dtype))

pyoperators.SparseOperator = SparseOperator
