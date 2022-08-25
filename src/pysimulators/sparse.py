"""
Patch PyOperator's SparseOperator to include the following formats:
    - Fixed Sparse Column (FSCMatrix)
    - Fixed Sparse Row (FSRMatrix)
    - Fixed Sparse Column Rotation 2d (FSCRotation2dMatrix)
    - Fixed Sparse Row Rotation 2d (FSRRotation2dMatrix)
    - Fixed Sparse Column Rotation 3d (FSCRotation3dMatrix)
    - Fixed Sparse Row Rotation 3d (FSRRotation3dMatrix)

"""

import operator

import numpy as np
import scipy.sparse as sp

import pyoperators
from pyoperators import CompositionOperator, DiagonalOperator, operation_assignment
from pyoperators.linear import SparseBase
from pyoperators.memory import empty
from pyoperators.utils import ilast, isscalarlike, product, tointtuple
from pysimulators._flib import sparse as fsp

__all__ = []


class _FSMatrix:
    def __init__(
        self,
        flib_id,
        shape,
        block_shape,
        nmax,
        sparse_axis,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        self._flib_id = flib_id
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"Invalid shape '{shape}'.")
        if len(shape) != 2:
            raise ValueError('The number of dimensions is not 2.')
        if len(block_shape) != 2:
            raise ValueError('The number of dimensions of the blocks is not 2.')
        straxes = ('row', 'column')
        if sparse_axis == 1:
            straxes = straxes[::-1]
        if any(s % b != 0 for s, b in zip(shape, block_shape)):
            raise ValueError(
                f"The shape of the matrix '{shape}' is incompatible with blocks of "
                f"shape '{block_shape}'."
            )

        if data is None:
            if nmax is None:
                raise ValueError(
                    f'The maximum number of non-zero {straxes[0]}s per {straxes[1]} '
                    f'is not specified.'
                )
            shape_data = (shape[1 - sparse_axis] // block_shape[0], nmax)
            if dtype is None:
                dtype = float
            if dtype_index is None:
                dtype_index = int
            dtype_data = self._get_dtype_data(dtype, dtype_index, block_shape)
            data = empty(shape_data, dtype_data, verbose=verbose)

        elif 'index' not in data.dtype.names:
            raise TypeError('The structured array has no field index.')

        elif len(data.dtype.names) == 1:
            raise TypeError('The structured array has no data field.')

        elif (
            product(data.shape[:-1]) * block_shape[1 - sparse_axis]
            != shape[1 - sparse_axis]
        ):
            raise ValueError(
                f"The shape of the matrix '{shape}' is incompatible with that of the "
                f"structured array '{data.shape}'."
            )

        elif nmax not in (None, data.shape[-1]):
            raise ValueError(
                f"The n{straxes[0][:3]}max keyword value '{nmax}' is incompatible with "
                f"the shape of the structured array '{data.shape}'."
            )

        else:
            dtype_index = data.dtype['index']
            dtype = data.dtype[1]
            if dtype.type == np.void:
                dtype = dtype.subdtype[0].type
            expected = self._get_dtype_data(dtype, dtype_index, block_shape)
            if data.dtype != expected:
                raise TypeError(
                    f'The input dtype {data.dtype} is invalid. Expected dtype is '
                    f'{expected}.'
                )
            if nmax is None:
                nmax = data.shape[-1]

        self.dtype = np.dtype(dtype)
        self.data = data.view(np.recarray)
        self.ndim = 2
        self.shape = tuple(shape)
        self.block_shape = tuple(block_shape)
        setattr(self, 'n' + straxes[0][:3] + 'max', int(nmax))

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
                f"The output size '{out.size}' is incompatible with the number of rows"
                f" of the sparse matrix '{self.shape[0]}'."
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
            or dv.type is np.float16
            or hasattr(np, 'float128')
            and dv.type is np.float128
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
            extra_size = v.size // self.shape[1]
            if extra_size > 1:
                flib_id += '_homothety'
        f = f'{flib_id}_matvec_i{di.itemsize}_r{ds.itemsize}_v{dv.itemsize}'
        func = getattr(fsp, f)

        m = self.data.ravel().view(np.int8)
        if flib_id.endswith('_homothety'):
            func(m, v, out_, nmax, self.shape[1], self.shape[0], extra_size)
        elif flib_id.endswith('_block'):
            func(
                m,
                v,
                out_,
                nmax,
                self.shape[1] // self.block_shape[1],
                self.shape[0] // self.block_shape[0],
                self.block_shape[0],
                self.block_shape[1],
            )
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
        return type(self)(self.shape, data=self.data.copy())

    def _reshapein(self, shape):
        extra_size = product(shape) // self.shape[1]
        if extra_size > 1 or shape[-1] == 1:
            return self.shape[0], extra_size
        return self.shape[0]

    def _reshapeout(self, shape):
        extra_size = product(shape) // self.shape[0]
        if extra_size > 1 or shape[-1] == 1:
            return self.shape[1], extra_size
        return self.shape[1]

    def _validatein(self, shape):
        size = product(shape)
        if not isinstance(self, (FSCMatrix, FSRMatrix)):
            if size == self.shape[1]:
                return
        elif size % self.shape[1] == 0:
            return
        raise ValueError(
            f"The shape '{shape}' is incompatible with the number of columns of the "
            f"sparse matrix '{self.shape[1]}'."
        )

    def _validateout(self, shape):
        size = product(shape)
        if not isinstance(self, (FSCMatrix, FSRMatrix)):
            if size == self.shape[0]:
                return
        elif size % self.shape[0] == 0:
            return
        raise ValueError(
            f"The shape '{shape}' is incompatible with the number of rows of the sparse"
            f" matrix '{self.shape[0]}'."
        )

    def _get_dtype_data(self, dtype, dtype_index, block_shape):
        return [('index', dtype_index), ('value', dtype)]

    __array_priority__ = 10.1


class FSCMatrix(_FSMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. This format can also be used for block homothety matrices.

    """

    def __init__(
        self,
        shape,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            'fsc',
            shape,
            (1, 1),
            nrowmax,
            0,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        extra_size = v.size // self.shape[1]
        v = v.reshape(-1, extra_size)
        out_ = out.reshape(-1, extra_size)
        if data.index.dtype.kind == 'u':
            for i in range(self.shape[1]):
                for b in data[i]:
                    out_[b.index] += b.value * v[i]
        else:
            for i in range(self.shape[1]):
                for b in data[i]:
                    if b.index >= 0:
                        out_[b.index] += b.value * v[i]
        return out

    def _transpose(self):
        return FSRMatrix(self.shape[::-1], data=self.data)


class FSRMatrix(_FSMatrix):
    """
    Fixed Sparse Row format, in which the number of non-zero columns is fixed
    for each row. This format can also be used for block homothety matrices.

    """

    def __init__(
        self,
        shape,
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            'fsr',
            shape,
            (1, 1),
            ncolmax,
            1,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        extra_size = v.size // self.shape[1]
        v = v.reshape(-1, extra_size)
        out_ = out.reshape(-1, extra_size)
        if data.index.dtype.kind == 'u':
            for i in range(self.shape[0]):
                b = data[i]
                out_[i] += np.sum(b.value[:, None] * v[b.index], axis=0)
        else:
            for i in range(self.shape[0]):
                b = data[i]
                b = b[b.index >= 0]
                out_[i] += np.sum(b.value[:, None] * v[b.index], axis=0)
        return out

    def _transpose(self):
        return FSCMatrix(self.shape[::-1], data=self.data)


class _FSBlockMatrix(_FSMatrix):
    def __init__(
        self,
        flib_id,
        shape,
        block_shape,
        nrowmax,
        sparse_axis,
        dtype,
        dtype_index,
        data,
        verbose,
    ):
        kind = flib_id[:3].upper()
        if data is None:
            if block_shape is None:
                raise TypeError('The block shape is not specified.')
            block_shape = tointtuple(block_shape)
            if len(block_shape) != 2:
                raise ValueError('The number of dimensions of the blocks is not 2.')
        if block_shape == (1, 1):
            raise ValueError(f'For block size of (1, 1), use the {kind}Matrix format.')
        if any(_ not in (1, 2, 3) for _ in block_shape):
            raise NotImplementedError(
                f'The sparse format {kind}BlockMatrix is not implemented for blocks '
                f"of shape '{block_shape}.'"
            )
        _FSMatrix.__init__(
            self,
            flib_id,
            shape,
            block_shape,
            nrowmax,
            sparse_axis,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )


class FSCBlockMatrix(_FSBlockMatrix):
    """
    Fixed Sparse Column Block format, in which the number of non-zero block
    rows is fixed for each block column.

    """

    def __init__(
        self,
        shape,
        block_shape=None,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        if data is not None:
            block_shape = data.dtype['value'].shape[::-1]
        _FSBlockMatrix.__init__(
            self,
            'fsc_block',
            shape,
            block_shape,
            nrowmax,
            0,
            dtype,
            dtype_index,
            data,
            verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, self.block_shape[0])
        v = v.reshape(-1, self.block_shape[1])
        for j in range(self.shape[1] // self.block_shape[1]):
            for b in data[j]:
                i = b['index']
                if i >= 0:
                    out_[i] += np.dot(b['value'].T, v[j])
        return out

    def _transpose(self):
        return FSRBlockMatrix(self.shape[::-1], data=self.data)

    def _get_dtype_data(self, dtype, dtype_index, block_shape):
        return [('index', dtype_index), ('value', dtype, block_shape[::-1])]


class FSRBlockMatrix(_FSBlockMatrix):
    """
    Fixed Sparse Row Block format, in which the number of non-zero block
    columns is fixed for each block row.

    """

    def __init__(
        self,
        shape,
        block_shape=None,
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        if data is not None:
            block_shape = data.dtype['value'].shape
        _FSMatrix.__init__(
            self,
            'fsr_block',
            shape,
            block_shape,
            ncolmax,
            1,
            dtype,
            dtype_index,
            data,
            verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, self.block_shape[0])
        v = v.reshape(-1, self.block_shape[1])
        for i in range(self.shape[0] // self.block_shape[0]):
            b = data[i]
            b = b[b.index >= 0]
            out_[i] += np.sum(np.einsum('...ij,...j->...i', b.value, v[b.index]), 0)
        return out

    def _transpose(self):
        return FSCBlockMatrix(self.shape[::-1], data=self.data)

    def _get_dtype_data(self, dtype, dtype_index, block_shape):
        return [('index', dtype_index), ('value', dtype, block_shape)]


class _FSRotation2dMatrix(_FSMatrix):
    def __init__(
        self,
        flib_id,
        shape,
        nmax,
        sparse_axis,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            flib_id,
            shape,
            (2, 2),
            nmax,
            sparse_axis,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
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

    def _get_dtype_data(self, dtype, dtype_index, block_shape):
        return [('index', dtype_index), ('r11', dtype), ('r21', dtype)]


class FSCRotation2dMatrix(_FSRotation2dMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. Each element of the sparse matrix is a 2x2 rotation
    matrix.

    """

    def __init__(
        self,
        shape,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSRotation2dMatrix.__init__(
            self,
            'fsc_rot2d',
            shape,
            nrowmax,
            0,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in range(self.shape[1] // 2):
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
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSRotation2dMatrix.__init__(
            self,
            'fsr_rot2d',
            shape,
            ncolmax,
            1,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 2)
        v = v.reshape(-1, 2)
        for i in range(self.shape[0] // 2):
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
        nmax,
        sparse_axis,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSMatrix.__init__(
            self,
            flib_id,
            shape,
            (3, 3),
            nmax,
            sparse_axis,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
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

    def _get_dtype_data(self, dtype, dtype_index, block_shape):
        return [('index', dtype_index), ('r11', dtype), ('r22', dtype), ('r32', dtype)]


class FSCRotation3dMatrix(_FSRotation3dMatrix):
    """
    Fixed Sparse Column format, in which the number of non-zero rows is fixed
    for each column. Each element of the sparse matrix is a 3x3 block
    whose transpose performs a rotation about the first component axis.

    """

    def __init__(
        self,
        shape,
        nrowmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSRotation3dMatrix.__init__(
            self,
            'fsc_rot3d',
            shape,
            nrowmax,
            0,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.nrowmax)
        if done:
            return out

        data = self.data.reshape(-1, self.nrowmax)
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in range(self.shape[1] // 3):
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
        ncolmax=None,
        dtype=None,
        dtype_index=None,
        data=None,
        verbose=False,
    ):
        _FSRotation3dMatrix.__init__(
            self,
            'fsr_rot3d',
            shape,
            ncolmax,
            1,
            dtype=dtype,
            dtype_index=dtype_index,
            data=data,
            verbose=verbose,
        )

    def _matvec(self, v, out=None):
        v, out, done = _FSMatrix._matvec(self, v, out, self.ncolmax)
        if done:
            return out

        data = self.data.reshape((-1, self.data.shape[-1]))
        out_ = out.reshape(-1, 3)
        v = v.reshape(-1, 3)
        for i in range(self.shape[0] // 3):
            b = data[i]
            b = b[b.index >= 0]
            out_[i, 0] += np.sum(b.r11 * v[b.index, 0])
            out_[i, 1] += np.sum(b.r22 * v[b.index, 1] - b.r32 * v[b.index, 2])
            out_[i, 2] += np.sum(b.r32 * v[b.index, 1] + b.r22 * v[b.index, 2])
        return out_

    def _transpose(self):
        return FSCRotation3dMatrix(self.shape[::-1], data=self.data)


class SparseOperator(SparseBase):
    def __init__(self, arg, shapein=None, shapeout=None, dtype=None, **keywords):
        if sp.issparse(arg):
            self.__class__ = pyoperators.linear.SparseOperator
            self.__init__(arg, dtype=None, **keywords)
            return
        if not isinstance(
            arg,
            (
                FSCMatrix,
                FSRMatrix,
                FSCBlockMatrix,
                FSRBlockMatrix,
                FSCRotation2dMatrix,
                FSRRotation2dMatrix,
                FSCRotation3dMatrix,
                FSRRotation3dMatrix,
            ),
        ):
            raise TypeError('The input sparse matrix type is not recognised.')
        if isinstance(arg, (FSCMatrix, FSRMatrix)):
            if shapein is None:
                bshapein = keywords.pop('broadcastable_shapein', (arg.shape[1],))
            else:
                shapein = tointtuple(shapein)
                test = np.cumprod(shapein) == arg.shape[1]
                try:
                    bshapein = shapein[: ilast(test, lambda x: x) + 1]
                except ValueError:
                    bshapein = (arg.shape[1],)
            self.broadcastable_shapein = bshapein
            if shapeout is None:
                bshapeout = keywords.pop('broadcastable_shapeout', (arg.shape[0],))
            else:
                shapeout = tointtuple(shapeout)
                test = np.cumprod(shapeout) == arg.shape[0]
                try:
                    bshapeout = shapeout[: ilast(test, lambda x: x) + 1]
                except ValueError:
                    bshapeout = (arg.shape[0],)
            self.broadcastable_shapeout = bshapeout
        else:
            bs = arg.block_shape
            if shapein is None:
                if bs[1] == 1:
                    shapein = arg.shape[1]
                else:
                    shapein = arg.shape[1] // bs[1], bs[1]
            if shapeout is None:
                if bs[0] == 1:
                    shapeout = arg.shape[0]
                else:
                    shapeout = arg.shape[0] // bs[0], bs[0]
        pyoperators.linear.SparseBase.__init__(
            self, arg, dtype=dtype, shapein=shapein, shapeout=shapeout, **keywords
        )
        self.set_rule('T', self._rule_transpose)
        self.set_rule(('T', '.'), self._rule_pTp, CompositionOperator)

    def direct(self, input, output, operation=operation_assignment):
        if operation is operation_assignment:
            output[...] = 0
        elif operation is not operator.iadd:
            raise ValueError(f'Invalid reduction operation: {operation}.')
        self.matrix._matvec(input, output)

    def todense(self, shapein=None, shapeout=None, inplace=False):
        shapein, shapeout = self._validate_shapes(shapein, shapeout)
        if shapein is None:
            raise ValueError(
                "The operator's input shape is not explicit. Spec"
                "ify it with the 'shapein' keyword."
            )
        extra_size = product(shapein) // self.matrix.shape[1]
        if self.shapein is None and extra_size > 1:
            shapein = shapein[:-1]
            broadcasting = True
        else:
            broadcasting = False
        out = SparseBase.todense(self, shapein=shapein, inplace=inplace)
        if broadcasting:
            out = np.kron(out, np.eye(extra_size))
        return out

    def reshapein(self, shape):
        if isinstance(self.matrix, (FSCMatrix, FSRMatrix)):
            return (
                self.broadcastable_shapeout + shape[len(self.broadcastable_shapein) :]
            )
        return self.shapeout

    def reshapeout(self, shape):
        if isinstance(self.matrix, (FSCMatrix, FSRMatrix)):
            return (
                self.broadcastable_shapein + shape[len(self.broadcastable_shapeout) :]
            )
        return self.shapein

    def validatein(self, shape):
        size = product(shape)
        if isinstance(self.matrix, (FSCMatrix, FSRMatrix)):
            if size % self.matrix.shape[1] != 0:
                raise ValueError(
                    f"The input shape '{shape}' is incompatible with the expected "
                    f"number of column '{self.matrix.shape[1]}'."
                )
        self.matrix._validatein(shape)

    def validateout(self, shape):
        size = product(shape)
        if isinstance(self.matrix, (FSCMatrix, FSRMatrix)):
            if size % self.matrix.shape[0] != 0:
                raise ValueError(
                    f"The output shape '{shape}' is incompatible with the expected "
                    f"number of rows '{self.matrix.shape[0]}'."
                )
        self.matrix._validateout(shape)

    def corestrict(self, mask, inplace=False):
        """
        Corestrict the operator to a subspace defined by a mask
        (True means that the element is kept).

        """
        if not isinstance(
            self.matrix, (FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix)
        ):
            raise NotImplementedError(
                f'Corestriction is not implemented for {type(self.matrix).__name__} '
                f'sparse storage.'
            )
        nrow = np.sum(mask)
        mask_ = np.repeat(mask, self.matrix.block_shape[0])
        out = self.copy()
        out.matrix = type(self.matrix)(
            (nrow * self.matrix.block_shape[0], self.matrix.shape[1]),
            data=self.matrix.data[mask_],
        )
        if isinstance(self.matrix, FSRMatrix):
            out.broadcastable_shapeout = (nrow,)
            if self.shapeout is not None:
                ndims_out = len(self.broadcastable_shapeout)
                out.shapeout = (nrow,) + self.shapeout[ndims_out:]
        else:
            out.shapeout = (nrow, out.matrix.block_shape[0])
        if inplace:
            self.delete()
        return out

    def restrict(self, mask, inplace=False):
        """
        Restrict the operator to a subspace defined by a mask
        (True means that the element is kept). Indices are renumbered in-place
        if the inplace keyword is set to True.

        """
        if not isinstance(
            self.matrix,
            (FSRMatrix, FSRBlockMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix),
        ):
            raise NotImplementedError(
                f'Restriction is not implemented for {type(self.matrix).__name__} '
                f'sparse storage.'
            )
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise TypeError('The mask is not boolean.')
        if isinstance(self.matrix, FSRMatrix):
            block_shapein = self.broadcastable_shapein
        else:
            block_shapein = self.shapein[:-1]
        if mask.shape != block_shapein:
            raise ValueError(
                f"Invalid shape '{mask.shape}'. Expected value is '{block_shapein}'."
            )

        if inplace:
            matrix = self.matrix
        else:
            matrix = self.matrix.copy()
        itype = matrix.data.dtype['index']
        if itype.type in (np.int8, np.int16, np.int32, np.int64):
            f = f'fsr_restrict_i{itype.itemsize}'
            func = getattr(fsp, f)
            block_shape = matrix.block_shape[0]
            ncol = func(
                matrix.data.view(np.int8).ravel(),
                mask.ravel(),
                matrix.ncolmax,
                matrix.shape[0] // block_shape,
                matrix.data.strides[-1],
            )
        else:
            ncol = np.sum(mask)
            new_index = empty(mask.shape, itype)
            new_index[...] = -1
            new_index[mask] = np.arange(ncol, dtype=itype)
            undef = matrix.data.index < 0
            matrix.data.index = new_index[matrix.data.index]
            matrix.data.index[undef] = -1
        out = self.copy()
        matrix.shape = matrix.shape[0], ncol * matrix.block_shape[1]
        out.matrix = matrix
        if isinstance(self.matrix, FSRMatrix):
            out.broadcastable_shapein = (ncol,)
            if self.shapein is not None:
                ndims_in = len(self.broadcastable_shapein)
                out.shapein = (ncol,) + self.shapein[ndims_in:]
        else:
            out.shapein = (ncol, out.matrix.block_shape[1])
        if inplace:
            self.delete()
        return out

    @staticmethod
    def _rule_transpose(self):
        if isinstance(self.matrix, (FSRMatrix, FSCMatrix)):
            keywords = {
                'broadcastable_shapein': self.broadcastable_shapeout,
                'broadcastable_shapeout': self.broadcastable_shapein,
            }
        else:
            keywords = {}
        return SparseOperator(self.matrix._transpose(), **keywords)

    @staticmethod
    def _rule_pTp(selfT, self):
        if (
            not isinstance(
                self.matrix, (FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix)
            )
            or self.matrix.ncolmax != 1
            or self.shapein is None
        ):
            return
        di = self.matrix.data.index.dtype
        dr = self.matrix.dtype
        dv = self.dtype
        if (
            di not in (np.int32, np.int64)
            or dr not in (np.float32, np.float64)
            or dv not in (np.float32, np.float64)
        ):
            return
        id_ = self.matrix._flib_id
        f = f'fsc_{id_}_ncolmax1_i{di.itemsize}_r{dr.itemsize}_v{dv.itemsize}'
        try:
            func = getattr(fsp, f)
        except AttributeError:
            return

        m = self.matrix.data.ravel().view(np.int8)
        data = np.zeros(self.shapein, dtype=self.dtype)
        func(m, data.ravel())
        return DiagonalOperator(data)


pyoperators.SparseOperator = SparseOperator
