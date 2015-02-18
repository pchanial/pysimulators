from __future__ import division

import numpy as np
import operator
from numpy.testing import assert_equal
from pyoperators import MaskOperator, PackOperator, Rotation2dOperator
from pyoperators.core import DeletedOperator
from pyoperators.utils.testing import assert_same, assert_is_type
from pysimulators.operators import ProjectionOperator
from pysimulators.sparse import FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix

clss = FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix
ftypes = np.float16, np.float32, np.float64, np.float128
itypes = np.int8, np.int16, np.int32, np.int64
index1 = [2, -1, 2, 1, 3, -1]
index2 = [-1, 3, 1, 1, 0, -1]
value1 = [0, 0, 0, 1.1, 0.3, 10]
value2 = [1, 2.2, 0.5, 1, 1, 10]
rotation1 = Rotation2dOperator(np.arange(6) * 30, degrees=True, dtype=np.float128)
rotation2 = Rotation2dOperator(np.arange(6, 12) * 30, degrees=True, dtype=np.float128)


def min_dtype(t1, t2):
    return min(t1, t2, key=lambda x: x().itemsize)


def _get_projection_fsr(itype, ftype, stype=None):
    if stype is None:
        stype = ftype
    dtype = [('index', itype), ('value', ftype)]
    data = np.recarray((6, 2), dtype=dtype)
    data[..., 0].index, data[..., 1].index = index1, index2
    data[..., 0].value, data[..., 1].value = value1, value2
    return ProjectionOperator(FSRMatrix((6, 5), data=data), dtype=stype)


def _get_projection_fsrrot2d(itype, ftype, stype=None):
    if stype is None:
        stype = ftype
    dtype = [('index', itype), ('r11', ftype), ('r21', ftype)]
    data = np.recarray((6, 2), dtype=dtype)
    data[..., 0].index, data[..., 1].index = index1, index2
    data[..., 0].r11 = value1 * rotation1.data[:, 0, 0]
    data[..., 1].r11 = value2 * rotation2.data[:, 0, 0]
    data[..., 0].r21 = value1 * rotation1.data[:, 1, 0]
    data[..., 1].r21 = value2 * rotation2.data[:, 1, 0]
    return ProjectionOperator(
        FSRRotation2dMatrix((6 * 2, 5 * 2), data=data), dtype=stype
    )


def _get_projection_fsrrot3d(itype, ftype, stype=None):
    if stype is None:
        stype = ftype
    dtype = [('index', itype), ('r11', ftype), ('r22', ftype), ('r32', ftype)]
    data = np.recarray((6, 2), dtype=dtype)
    data[..., 0].index, data[..., 1].index = index1, index2
    data[..., 0].r11, data[..., 1].r11 = value1, value2
    data[..., 0].r22 = value1 * rotation1.data[:, 0, 0]
    data[..., 1].r22 = value2 * rotation2.data[:, 0, 0]
    data[..., 0].r32 = value1 * rotation1.data[:, 1, 0]
    data[..., 1].r32 = value2 * rotation2.data[:, 1, 0]
    return ProjectionOperator(
        FSRRotation3dMatrix((6 * 3, 5 * 3), data=data), dtype=stype
    )


_get_projection = {
    FSRMatrix: _get_projection_fsr,
    FSRRotation2dMatrix: _get_projection_fsrrot2d,
    FSRRotation3dMatrix: _get_projection_fsrrot3d,
}


def test_kernel():
    expected = [False, False, True, False, True]

    def func(cls, itype, ftype):
        proj = _get_projection[cls](itype, ftype)
        actual = proj.canonical_basis_in_kernel()
        assert_equal(actual, expected)
        out = np.empty(5, bool)
        actual = proj.canonical_basis_in_kernel(out=out)
        assert_equal(actual, expected)
        kernel0 = [True, True, True, False, False]
        out = np.array(kernel0)
        proj.canonical_basis_in_kernel(out=out, operation=operator.iand)
        assert_equal(out, np.array(kernel0) & expected)
        kernel0 = [False, True, False, False, True]
        out = np.array(kernel0)
        proj.canonical_basis_in_kernel(out=out, operation=operator.ior)
        assert_equal(out, np.array(kernel0) | expected)

    for cls in clss:
        for itype in itypes:
            for ftype in ftypes:
                yield func, cls, itype, ftype


def test_pT1():
    def func(cls, itype, ftype, vtype):
        expected_ = np.asarray(expected, min_dtype(ftype, vtype))
        proj = _get_projection[cls](itype, ftype, vtype)
        pT1 = proj.pT1()
        assert_same(pT1, expected_)
        pT1 = proj.pT1(out=pT1)
        assert_same(pT1, expected_)
        pT1[...] = 2
        proj.pT1(out=pT1, operation=operator.iadd)
        assert_same(pT1, 2 + expected_)

    proj_ref = _get_projection[FSRMatrix](np.int32, np.float128)
    expected = proj_ref.T(np.ones(proj_ref.matrix.shape[0], np.float128))
    for cls in clss:
        for itype in itypes:
            for ftype in ftypes:
                for vtype in ftypes:
                    yield func, cls, itype, ftype, vtype


def test_pTx_pT1():
    input_fsr = np.array([1.1, 2, -1.3, 2, -3, 4.1], dtype=np.float128)
    input_fsrrot2d = input_fsr[:, None] * Rotation2dOperator(
        np.arange(6) * 30, dtype=np.float128
    )([1, 0])
    input_fsrrot3d = np.array(
        [input_fsr, np.random.random_sample(6), np.random.random_sample(6)]
    ).T
    inputs = {
        FSRMatrix: input_fsr,
        FSRRotation2dMatrix: input_fsrrot2d,
        FSRRotation3dMatrix: input_fsrrot3d,
    }

    def func(cls, itype, ftype, vtype):
        proj = _get_projection[cls](itype, ftype, vtype)
        input = np.asarray(inputs[cls], vtype)
        expectedx_ = np.asarray(expectedx, min_dtype(ftype, vtype))
        expected1_ = np.asarray(expected1, min_dtype(ftype, vtype))
        pTx, pT1 = proj.pTx_pT1(input)
        assert_same(pTx, expectedx_)
        assert_same(pT1, expected1_)
        proj.pTx_pT1(input, out=(pTx, pT1))
        assert_same(pTx, expectedx_)
        assert_same(pT1, expected1_)
        pTx[...] = 1
        pT1[...] = 2
        proj.pTx_pT1(input, out=(pTx, pT1), operation=operator.iadd)
        assert_same(pTx, 1 + expectedx_)
        assert_same(pT1, 2 + expected1_)

    proj_ref = _get_projection[FSRMatrix](np.int32, np.float128)
    expectedx = proj_ref.T(input_fsr)
    expected1 = proj_ref.T(np.ones(proj_ref.matrix.shape[0], np.float128))
    for cls in clss[0], clss[2]:
        for itype in itypes:
            for ftype in ftypes:
                for vtype in ftypes:
                    yield func, cls, itype, ftype, vtype


def test_restrict():
    restriction = np.array([True, False, False, True, True])
    kernel = [False, False, True, False, True]

    def func(cls, itype, ftype, inplace):
        proj_ref = _get_projection[cls](itype, ftype)
        proj_ = _get_projection[cls](itype, ftype)
        proj = proj_.restrict(restriction, inplace=inplace)
        if inplace:
            assert_is_type(proj_, DeletedOperator)
        if cls is not FSRMatrix:

            def pack(v):
                return v[restriction, :]

        else:
            pack = PackOperator(restriction)
        masking = MaskOperator(~restriction, broadcast='rightward')
        block_size = proj_ref.matrix.block_shape[1]
        shape = (5,) + ((block_size,) if block_size > 1 else ())
        x = np.arange(5 * block_size).reshape(shape) + 1
        assert_equal(proj_ref(masking(x)), proj(pack(x)))
        pack = PackOperator(restriction)
        assert_equal(pack(kernel), proj.canonical_basis_in_kernel())

    for cls in clss:
        for itype in itypes:
            for ftype in ftypes:
                for inplace in False, True:
                    yield func, cls, itype, ftype, inplace
