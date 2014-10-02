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
from pyoperators.linear import SparseBase
from pyoperators.memory import empty
from pyoperators.utils import isscalarlike, product, tointtuple
from pysimulators._flib import sparse as fsp

__all__ = []


class _FSMatrix(object):
    def __init__(
        self,
        flib_id,
        shape,
        sparse_axis,
        n,
        data=None,
        dtype=None,
        dtype_index=None,
        dtype_names=('value',),
        block_size=1,
        verbose=False,
    ):
        self._flib_id = flib_id
        if not isinstance(shape, tuple):
            raise TypeError("Invalid shape '{0}'.".format(shape))
        if len(shape) != 2:
            raise ValueError("The number of dimensions is not 2.")
        straxes = ('row', 'column')
        if sparse_axis == 1:
            straxes = straxes[::-1]
        if data is None:
            if n is None:
                raise ValueError(
                    'The maximum number of non-zero {0}s per {1} '
                    'is not specified.'.format(*straxes)
                )
            shape_data = (shape[1 - sparse_axis] // block_size, n)
            if dtype is None:
                dtype = float
            if dtype_index is None:
                dtype_index = int
            dtype_data = [('index', dtype_index)] + [
                (name, dtype) for name in dtype_names
            ]
            data = empty(shape_data, dtype_data, verbose=verbose).view(np.recarray)
        elif data.dtype.names != ('index',) + dtype_names:
            raise TypeError('The fields of the structured array are invalid.')
        elif any(s % block_size != 0 for s in shape):
            raise ValueError(
                "The shape of the matrix '{0}' is not a multiple of '{1}'.".format(
                    shape, block_size
                )
            )
        elif shape[1 - sparse_axis] // block_size != product(data.shape[:-1]):
            raise ValueError(
                "The shape of the matrix '{0}' is incompatible with that of th"
                "e structured array '{1}'.".format(shape, data.shape)
            )
        elif n not in (None, data.shape[-1]):
            raise ValueError(
                "The n{0}max keyword value '{1}' is incompatible with the shap"
                "e of the structured array '{2}'.".format(straxes[0][:3], n, data.shape)
            )
        elif any(
            data[name].dtype != data[dtype_names[0]].dtype for name in dtype_names
        ):
            raise TypeError(
                'The data fields of the structured array do not have the same ' 'dtype.'
            )
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

    def _matvec(self, v, out, nmax):
        v = np.asarray(v).ravel()
        self._validatein(v.shape)
        if out is None:
            out = np.zeros(
                v.size * self.shape[0] // self.shape[1],
                np.find_common_type([self.dtype, v.dtype], []),
            )
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output array is not an ndarray.')
        elif not out.flags.contiguous:
            raise ValueError('The output array is not contiguous.')
        elif out.size != v.size * self.shape[0] // self.shape[1]:
            raise ValueError(
                "The output size '{0}' is incompatible with the number of rows"
                " of the sparse matrix '{1}'.".format(out.size, self.shape[0])
            )
        else:
            out = out.ravel()
            self._validateout(out.shape)

        di = self.data.index.dtype
        ds = self.dtype
        dv = v.dtype
        if str(ds) not in ('float32', 'float64') or str(di) not in ('int32', 'int64'):
            return v, out, False

        if (
            dv.kind != 'f'
            or dv.type in (np.float16, np.float128)
            or dv.type is np.float32
            and ds.type is np.float64
        ):
            v = v.astype(self.dtype)
            dv = self.dtype
        if out.dtype != dv or not out.flags.contiguous:
            out_ = np.empty(out.size, dtype=dv)
            out_[...] = out
        else:
            out_ = out
        flib_id = self._flib_id
        if isinstance(self, (FSCMatrix, FSRMatrix)):
            block_size = v.size // self.shape[1]
            if block_size > 1:
                flib_id += '_nd'
        f = '{0}_matvec_i{1}_m{2}_v{3}'.format(
            flib_id, di.itemsize, ds.itemsize, dv.itemsize
        )
        func = getattr(fsp, f)
        m = self.data.ravel().view(np.int8)
        if flib_id.endswith('_nd'):
            func(m, v, out_, nmax, self.shape[1], self.shape[0], block_size)
        else:
            func(m, v, out_, nmax)
        if out.dtype != dv:
            out[...] = out_
        return v, out, True

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

    def copy(self):
        return type(self)(self.shape, self.data.copy())

    def _reshapein(self, shape):
        block_size = product(shape) // self.shape[1]
        if block_size > 1 or shape[-1] == 1:
            return self.shape[0], block_size
        return self.shape[0]

    def _reshapeout(self, shape):
        block_size = product(shape) // self.shape[0]
        if block_size > 1 or shape[-1] == 1:
            return self.shape[1], block_size
        return self.shape[1]

    def _validatein(self, shape):
        if not isinstance(self, (FSCMatrix, FSRMatrix)) or self.block_size > 1:
            if product(shape) == self.shape[1]:
                return
        else:
            block_size = product(shape) // self.shape[1]
            if product(shape) % self.shape[1] == 0 and (
                block_size == 1 or len(shape) == 1 or shape[-1] == block_size
            ):
                return
        raise ValueError(
            "The shape '{0}' is incompatible with the number of columns of the"
            " sparse matrix '{1}'.".format(shape, self.shape[1])
        )

    def _validateout(self, shape):
        if not isinstance(self, (FSCMatrix, FSRMatrix)) or self.block_size > 1:
            if product(shape) == self.shape[0]:
                return
        else:
            block_size = product(shape) // self.shape[0]
            if product(shape) % self.shape[0] == 0 and (
                block_size == 1 or len(shape) == 1 or shape[-1] == block_size
            ):
                return
        raise ValueError(
            "The shape '{0}' is incompatible with the number of rows of the sp"
            "arse matrix '{1}'.".format(shape, self.shape[0])
        )

    __array_priority__ = 10.1


class FSCMatrix(_FSMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. This format can also be used for block homothety matrices.

    """

    def __init__(
        self,
        shape,
        data=None,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        block_size=1,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            'fsc',
            shape,
            0,
            nrowmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            block_size=block_size,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        block_size = max(self.block_size, v.size // self.shape[1])
        v = v.reshape(-1, block_size)
        out_ = out.reshape(-1, block_size)
        if data.index.dtype.kind == 'u':
            for i in xrange(self.shape[1]):
                for b in data[i]:
                    out_[b.index] += b.value * v[i]
        else:
            for i in xrange(self.shape[1]):
                for b in data[i]:
                    if b.index >= 0:
                        out_[b.index] += b.value * v[i]
        return out

    def _transpose(self):
        return FSRMatrix(self.shape[::-1], block_size=self.block_size, data=self.data)


class FSRMatrix(_FSMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. This format can also be used for block homothety matrices.

    """

    def __init__(
        self,
        shape,
        data=None,
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        block_size=1,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            'fsr',
            shape,
            1,
            ncolmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            block_size=block_size,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        block_size = max(self.block_size, v.size // self.shape[1])
        v = v.reshape(-1, block_size)
        out_ = out.reshape(-1, block_size)
        if data.index.dtype.kind == 'u':
            for i in xrange(self.shape[0]):
                b = data[i]
                out_[i] += np.sum(b.value[:, None] * v[b.index], axis=0)
        else:
            for i in xrange(self.shape[0]):
                b = data[i]
                b = b[b.index >= 0]
                out_[i] += np.sum(b.value[:, None] * v[b.index], axis=0)
        return out

    def _transpose(self):
        return FSCMatrix(self.shape[::-1], block_size=self.block_size, data=self.data)


class _FSRotation2dMatrix(_FSMatrix):
    def __init__(
        self,
        flib_id,
        shape,
        sparse_axis,
        n,
        data=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            flib_id,
            shape,
            sparse_axis,
            n,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            dtype_names=('r11', 'r21'),
            block_size=2,
            verbose=verbose,
        )

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

    def __init__(
        self,
        shape,
        data=None,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSRotation2dMatrix.__init__(
            self,
            'fsc_rot2d',
            shape,
            0,
            nrowmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in xrange(self.shape[1] // 2):
            for b in data[i]:
                if b.index >= 0:
                    out_[b.index, 0] += b.r11 * v[i, 0] + b.r21 * v[i, 1]
                    out_[b.index, 1] += -b.r21 * v[i, 0] + b.r11 * v[i, 1]
        return out

    def _transpose(self):
        return FSRRotation2dMatrix(self.shape[::-1], data=self.data)


class FSRRotation2dMatrix(_FSRotation2dMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. Each element of the sparse matrix is a 2x2 rotation matrix.

    """

    def __init__(
        self,
        shape,
        data=None,
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSRotation2dMatrix.__init__(
            self,
            'fsr_rot2d',
            shape,
            1,
            ncolmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in xrange(self.shape[0] // 2):
            b = data[i]
            b = b[b.index >= 0]
            out_[i, 0] += np.sum(b.r11 * v[b.index, 0] - b.r21 * v[b.index, 1])
            out_[i, 1] += np.sum(b.r21 * v[b.index, 0] + b.r11 * v[b.index, 1])
        return out

    def _transpose(self):
        return FSCRotation2dMatrix(self.shape[::-1], data=self.data)


class _FSRotation3dMatrix(_FSMatrix):
    def __init__(
        self,
        flib_id,
        shape,
        sparse_axis,
        n,
        data=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            flib_id,
            shape,
            sparse_axis,
            n,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            dtype_names=('r11', 'r22', 'r32'),
            block_size=3,
            verbose=verbose,
        )

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

    def __init__(
        self,
        shape,
        data=None,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSRotation3dMatrix.__init__(
            self,
            'fsc_rot3d',
            shape,
            0,
            nrowmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape(-1, self.nrowmax)
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in xrange(self.shape[1] // 3):
            for b in data[i]:
                if b.index >= 0:
                    out_[b.index, 0] += b.r11 * v[i, 0]
                    out_[b.index, 1] += b.r22 * v[i, 1] + b.r32 * v[i, 2]
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

    def __init__(
        self,
        shape,
        data=None,
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        verbose=False,
    ):
        _FSRotation3dMatrix.__init__(
            self,
            'fsr_rot3d',
            shape,
            1,
            ncolmax,
            data=data,
            dtype=dtype,
            dtype_index=dtype_index,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in xrange(self.shape[0] // 3):
            b = data[i]
            b = b[b.index >= 0]
            out_[i, 0] += np.sum(b.r11 * v[b.index, 0])
            out_[i, 1] += np.sum(b.r22 * v[b.index, 1] - b.r32 * v[b.index, 2])
            out_[i, 2] += np.sum(b.r32 * v[b.index, 1] + b.r22 * v[b.index, 2])
        return out_

    def _transpose(self):
        return FSCRotation3dMatrix(self.shape[::-1], data=self.data)


class SparseOperator(SparseBase):
    def __init__(
        self,
        arg,
        block_shapein=None,
        block_shapeout=None,
        shapein=None,
        shapeout=None,
        dtype=None,
        **keywords,
    ):
        if sp.issparse(arg):
            self.__class__ = pyoperators.linear.SparseOperator
            self.__init__(arg, dtype=None, **keywords)
            return
        if not isinstance(
            arg,
            (
                FSCMatrix,
                FSRMatrix,
                FSCRotation2dMatrix,
                FSRRotation2dMatrix,
                FSCRotation3dMatrix,
                FSRRotation3dMatrix,
            ),
        ):
            raise TypeError('The input sparse matrix type is not recognised.')
        n = arg.block_size
        if block_shapein is None:
            if shapein is not None:
                shapein = tointtuple(shapein)
                if n == 1 and product(shapein) == arg.shape[1]:
                    block_shapein = shapein
                else:
                    block_shapein = shapein[:-1]
            else:
                block_shapein = (arg.shape[1] // n,)
        if shapein is None and n > 1:
            shapein = block_shapein + (n,)
        if block_shapeout is None:
            if shapeout is not None:
                shapeout = tointtuple(shapeout)
                if n == 1 and product(shapeout) == arg.shape[0]:
                    block_shapeout = shapeout
                else:
                    block_shapeout = shapeout[:-1]
            else:
                block_shapeout = (arg.shape[0] // n,)
        if shapeout is None and n > 1:
            shapeout = block_shapeout + (n,)

        self.block_shapein = block_shapein
        self.block_shapeout = block_shapeout
        pyoperators.linear.SparseBase.__init__(
            self, arg, dtype=dtype, shapein=shapein, shapeout=shapeout, **keywords
        )
        self.set_rule(
            'T',
            lambda s: SparseOperator(
                s.matrix._transpose(),
                block_shapein=s.block_shapeout,
                block_shapeout=s.block_shapein,
                dtype=s.dtype,
            ),
        )

    def direct(self, input, output, operation=operation_assignment):
        if operation is operation_assignment:
            output[...] = 0
        elif operation is not operator.iadd:
            raise ValueError('Invalid reduction operation: {0}.'.format(operation))
        self.matrix._matvec(input, output)

    def reshapein(self, shape):
        return self.block_shapeout + shape[len(self.block_shapein) :]

    def reshapeout(self, shape):
        return self.block_shapein + shape[len(self.block_shapeout) :]

    def validatein(self, shape):
        n = self.matrix.block_size
        if n > 1 and shape[-1] != n:
            raise ValueError(
                "The last dimension of input shape '{0}' is not a multiple of "
                "'{1}'.".format(shape, n)
            )
        block_shape = shape[: len(self.block_shapein)]
        if block_shape != self.block_shapein:
            raise ValueError(
                "Invalid input shape '{0}'. The expected input block shape is "
                "'{1}'".format(shape, self.block_shapein)
            )
        self.matrix._validatein(shape)

    def validateout(self, shape):
        n = self.matrix.block_size
        if n > 1 and shape[-1] != n:
            raise ValueError(
                "The last dimension of output shape '{0}' is not a multiple of"
                " '{1}'.".format(shape, n)
            )
        block_shape = shape[: len(self.block_shapeout)]
        if block_shape != self.block_shapeout:
            raise ValueError(
                "Invalid output shape '{0}'. The expected output block shape i"
                "s '{1}'".format(shape, self.block_shapeout)
            )
        self.matrix._validateout(shape)


pyoperators.SparseOperator = SparseOperator
