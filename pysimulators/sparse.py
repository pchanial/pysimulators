"""
Patch PyOperator's SparseOperator to include another four formats:
    - Fixed Sparse Column (FSCMatrix)
    - Fixed Sparse Row (FSRMatrix)
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
from pyoperators.utils import isscalar, product
from pysimulators._flib import sparse as fsp

__all__ = []


class _FSMatrix(object):
    def __init__(self, shape, sparse_axis, n, data=None, dtype=None,
                 dtype_index=None, dtype_names=('value',), block_size=1):
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
            data = empty(shape_data, dtype_data, verbose=True).view(
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
        setattr(self, 'n' + straxes[0][:3] + 'max', n)

    def __mul__(self, other):
        if isscalar(other):
            data = self.data.copy()
            data.value *= other
            return type(self)(self.shape, data=data)
        other = np.asarray(other)
        if other.ndim == 1:
            return self._matvec(other)
        return NotImplemented

    def __rmul__(self, other):
        if isscalar(other):
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
                 dtype_index=None):
        _FSMatrix.__init__(self, shape, 0, nrowmax, data=data, dtype=dtype,
                           dtype_index=dtype_index)

    def _matvec(self, v, out=None):
        v = np.asarray(v).ravel()
        if v.shape != (self.shape[1],):
            raise ValueError(
                "The input array has an invalid shape '{0}'. The expected shap"
                "e is '{1}'.".format(v.shape, self.shape[1]))
        if out is None:
            out = np.zeros(self.shape[0],
                           np.find_common_type([self.dtype, v.dtype], []))
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif out.size != self.shape[0]:
            raise ValueError(
                "The output array has an invalid shape '{0}'. The expected sha"
                "pe is '{1}'.".format(out.shape, self.shape[0]))
        else:
            out = out.ravel()

        tm = self.dtype.type
        tv = v.dtype.type
        to = out.dtype.type
        ti = self.data.index.dtype.type
        m = self.data.ravel().view(np.int8)
        if tm is np.float64 and to is np.float64:
            v = v.astype(self.dtype)
            if ti is np.int64:
                fsp.fsc_matvec_i8_m8_v8(m, v, out, self.nrowmax)
                return out
            elif ti is np.int32:
                fsp.fsc_matvec_i4_m8_v8(m, v, out, self.nrowmax)
                return out
        elif tm is np.float32:
            if tv is np.float64 and to is np.float64:
                if ti is np.int64:
                    fsp.fsc_matvec_i8_m4_v8(m, v, out, self.nrowmax)
                    return out
                elif ti is np.int32:
                    fsp.fsc_matvec_i4_m4_v8(m, v, out, self.nrowmax)
                    return out
            elif tv is np.float32 and to is np.float32:
                if ti is np.int64:
                    fsp.fsc_matvec_i8_m4_v4(m, v, out, self.nrowmax)
                    return out
                elif ti is np.int32:
                    fsp.fsc_matvec_i4_m4_v4(m, v, out, self.nrowmax)
                    return out
        data = self.data.reshape((-1, self.data.shape[-1]))
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
                 dtype_index=None):
        _FSMatrix.__init__(self, shape, 1, ncolmax, data=data, dtype=dtype,
                           dtype_index=dtype_index)

    def _matvec(self, v, out=None):
        v = np.asarray(v).ravel()
        if v.shape != (self.shape[1],):
            raise ValueError(
                "The input array has an invalid shape '{0}'. The expected shap"
                "e is '{1}'.".format(v.shape, self.shape[1]))
        if out is None:
            out = np.zeros(self.shape[0],
                           np.find_common_type([self.dtype, v.dtype], []))
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif out.size != self.shape[0]:
            raise ValueError(
                "The output array has an invalid shape '{0}'. The expected sha"
                "pe is '{1}'.".format(out.shape, self.shape[0]))
        else:
            out = out.ravel()

        tm = self.dtype.type
        tv = v.dtype.type
        to = out.dtype.type
        ti = self.data.index.dtype.type
        m = self.data.ravel().view(np.int8)
        if tm is np.float64 and to is np.float64:
            v = v.astype(self.dtype)
            if ti is np.int64:
                fsp.fsr_matvec_i8_m8_v8(m, v, out, self.ncolmax)
                return out
            elif ti is np.int32:
                fsp.fsr_matvec_i4_m8_v8(m, v, out, self.ncolmax)
                return out
        elif tm is np.float32:
            if tv is np.float64 and to is np.float64:
                if ti is np.int64:
                    fsp.fsr_matvec_i8_m4_v8(m, v, out, self.ncolmax)
                    return out
                elif ti is np.int32:
                    fsp.fsr_matvec_i4_m4_v8(m, v, out, self.ncolmax)
                    return out
            elif tv is np.float32 and to is np.float32:
                if ti is np.int64:
                    fsp.fsr_matvec_i8_m4_v4(m, v, out, self.ncolmax)
                    return out
                elif ti is np.int32:
                    fsp.fsr_matvec_i4_m4_v4(m, v, out, self.ncolmax)
                    return out

        data = self.data.reshape((-1, self.data.shape[-1]))
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


class _FSRotation3dMatrix(_FSMatrix):
    def __init__(self, shape, sparse_axis, n, data=None, dtype=None,
                 dtype_index=None):
        _FSMatrix.__init__(
            self, shape, sparse_axis, n, data=data, dtype=dtype,
            dtype_index=dtype_index, dtype_names=('r11', 'r22', 'r32'),
            block_size=3)

    def __mul__(self, other):
        if isscalar(other):
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
    whose transpose perform a rotation along the first component axis.

    """
    def __init__(self, shape, data=None, nrowmax=None, dtype=None,
                 dtype_index=None):
        _FSRotation3dMatrix.__init__(self, shape, 0, nrowmax, data=data,
                                     dtype=dtype, dtype_index=dtype_index)

    def _matvec(self, v, out=None):
        v = np.asarray(v).reshape((-1, 3))
        if v.shape != (self.shape[1] // 3, 3):
            raise ValueError(
                "The input array has an invalid shape '{0}'. The expected shap"
                "e is '{1}'.".format(v.shape, (self.shape[1] // 3, 3)))
        if out is None:
            out = np.zeros((self.shape[0] // 3, 3),
                           np.find_common_type([self.dtype, v.dtype], []))
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif out.size != self.shape[0]:
            raise ValueError(
                "The output array has an invalid shape '{0}'. The expected sha"
                "pe is '{1}'.".format(out.shape, (self.shape[0] // 3, 3)))
        else:
            out = out.reshape((-1, 3))
        out_ = out.ravel()

        tm = self.dtype.type
        tv = v.dtype.type
        to = out.dtype.type
        ti = self.data.index.dtype.type
        m = self.data.ravel().view(np.int8)
        if tm is np.float64 and to is np.float64:
            v = v.astype(self.dtype)
            if ti is np.int64:
                fsp.fsc_rot3d_matvec_i8_m8_v8(m, v.T, out.T, self.nrowmax)
                return out_
            elif ti is np.int32:
                fsp.fsc_rot3d_matvec_i4_m8_v8(m, v.T, out.T, self.nrowmax)
                return out_
        elif tm is np.float32:
            if tv is np.float64 and to is np.float64:
                if ti is np.int64:
                    fsp.fsc_rot3d_matvec_i8_m4_v8(m, v.T, out.T, self.nrowmax)
                    return out_
                elif ti is np.int32:
                    fsp.fsc_rot3d_matvec_i4_m4_v8(m, v.T, out.T, self.nrowmax)
                    return out_
            elif tv is np.float32 and to is np.float32:
                if ti is np.int64:
                    fsp.fsc_rot3d_matvec_i8_m4_v4(m, v.T, out.T, self.nrowmax)
                    return out_
                elif ti is np.int32:
                    fsp.fsc_rot3d_matvec_i4_m4_v4(m, v.T, out.T, self.nrowmax)
                    return out_

        data = self.data.reshape((-1, self.data.shape[-1]))
        for i in xrange(self.shape[1] // 3):
            for b in data[i]:
                if b.index >= 0:
                    out[b.index, 0] +=  b.r11 * v[i, 0]
                    out[b.index, 1] +=  b.r22 * v[i, 1] + b.r32 * v[i, 2]
                    out[b.index, 2] += -b.r32 * v[i, 1] + b.r22 * v[i, 2]
        return out_

    def _transpose(self):
        return FSRRotation3dMatrix(self.shape[::-1], data=self.data)


class FSRRotation3dMatrix(_FSRotation3dMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. Each element of the sparse matrix is a 3x3 block
    performing a rotation along the first component axis.

    """
    def __init__(self, shape, data=None, ncolmax=None, dtype=None,
                 dtype_index=None):
        _FSRotation3dMatrix.__init__(self, shape, 1, ncolmax, data=data,
                                     dtype=dtype, dtype_index=dtype_index)

    def _matvec(self, v, out=None):
        v = np.asarray(v).reshape((-1, 3))
        if v.shape != (self.shape[1] // 3, 3):
            raise ValueError(
                "The input array has an invalid shape '{0}'. The expected shap"
                "e is '{1}'.".format(v.shape, (self.shape[1] // 3, 3)))
        if out is None:
            out = np.zeros((self.shape[0] // 3, 3),
                           np.find_common_type([self.dtype, v.dtype], []))
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif out.size != self.shape[0]:
            raise ValueError(
                "The output array has an invalid shape '{0}'. The expected sha"
                "pe is '{1}'.".format(out.shape, (self.shape[0] // 3, 3)))
        else:
            out = out.reshape((-1, 3))
        out_ = out.ravel()

        tm = self.dtype.type
        tv = v.dtype.type
        to = out.dtype.type
        ti = self.data.index.dtype.type
        m = self.data.ravel().view(np.int8)
        if tm is np.float64 and to is np.float64:
            v = v.astype(self.dtype)
            if ti is np.int64:
                fsp.fsr_rot3d_matvec_i8_m8_v8(m, v.T, out.T, self.ncolmax)
                return out_
            elif ti is np.int32:
                fsp.fsr_rot3d_matvec_i4_m8_v8(m, v.T, out.T, self.ncolmax)
                return out_
        elif tm is np.float32:
            if tv is np.float64 and to is np.float64:
                if ti is np.int64:
                    fsp.fsr_rot3d_matvec_i8_m4_v8(m, v.T, out.T, self.ncolmax)
                    return out_
                elif ti is np.int32:
                    fsp.fsr_rot3d_matvec_i4_m4_v8(m, v.T, out.T, self.ncolmax)
                    return out_
            elif tv is np.float32 and to is np.float32:
                if ti is np.int64:
                    fsp.fsr_rot3d_matvec_i8_m4_v4(m, v.T, out.T, self.ncolmax)
                    return out_
                elif ti is np.int32:
                    fsp.fsr_rot3d_matvec_i4_m4_v4(m, v.T, out.T, self.ncolmax)
                    return out_

        data = self.data.reshape((-1, self.data.shape[-1]))
        for i in xrange(self.shape[0] // 3):
            b = data[i]
            b = b[b.index >= 0]
            out[i, 0] += np.sum(b.r11 * v[b.index, 0])
            out[i, 1] += np.sum(b.r22 * v[b.index, 1] -
                                b.r32 * v[b.index, 2])
            out[i, 2] += np.sum(b.r32 * v[b.index, 1] +
                                b.r22 * v[b.index, 2])
        return out_

    def _transpose(self):
        return FSCRotation3dMatrix(self.shape[::-1], data=self.data)


class SparseOperator(pyoperators.core.SparseBase):
    def __init__(self, arg, dtype=None, **keywords):
        if sp.issparse(arg):
            self.__class__ = pyoperators.core.SparseOperator
            self.__init__(arg, dtype=None, **keywords)
            return
        if not isinstance(arg, (FSCMatrix, FSRMatrix, FSCRotation3dMatrix,
                                FSRRotation3dMatrix)):
            raise TypeError('The input sparse matrix type is not recognised.')
        if isinstance(arg, (FSCRotation3dMatrix, FSRRotation3dMatrix)):
            if 'shapein' not in keywords:
                keywords['shapein'] = (arg.shape[1] // 3, 3)
            if 'shapeout' not in keywords:
                keywords['shapeout'] = (arg.shape[0] // 3, 3)

        def direct(input, output, operation=operation_assignment):
            if operation is operation_assignment:
                output[...] = 0
            elif operation is not operator.iadd:
                raise ValueError(
                    'Invalid reduction operation: {0}.'.format(operation))
            self.matrix._matvec(input, output)

        pyoperators.core.SparseBase.__init__(
            self, arg, dtype=dtype, direct=direct, **keywords)
        self.set_rule('T', lambda s: SparseOperator(s.matrix._transpose(),
                                                    dtype=s.dtype))

pyoperators.SparseOperator = SparseOperator
