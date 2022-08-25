import operator

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyoperators import MaskOperator, PackOperator, Rotation2dOperator
from pyoperators.core import DeletedOperator
from pyoperators.utils.testing import assert_same
from pysimulators.operators import ProjectionOperator
from pysimulators.sparse import FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix

from .common import BIGGEST_FLOAT_TYPE, FLOAT_TYPES, SINT_TYPES

CLASSES = FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix
index1 = [2, -1, 2, 1, 3, -1]
index2 = [-1, 3, 1, 1, 0, -1]
value1 = [0, 0, 0, 1.1, 0.3, 10]
value2 = [1, 2.2, 0.5, 1, 1, 10]
rotation1 = Rotation2dOperator(
    np.arange(6) * 30, degrees=True, dtype=BIGGEST_FLOAT_TYPE
)
rotation2 = Rotation2dOperator(
    np.arange(6, 12) * 30, degrees=True, dtype=BIGGEST_FLOAT_TYPE
)


def min_dtype(t1, t2):
    return min(t1, t2, key=lambda x: x().itemsize)


def get_projection(cls, itype, ftype, stype=None):
    if stype is None:
        stype = ftype

    if cls is FSRMatrix:
        dtype = [('index', itype), ('value', ftype)]
        data = np.recarray((6, 2), dtype=dtype)
        data[..., 0].index, data[..., 1].index = index1, index2
        data[..., 0].value, data[..., 1].value = value1, value2
        matrix = FSRMatrix((6, 5), data=data)
        return ProjectionOperator(matrix, dtype=stype)

    if cls is FSRRotation2dMatrix:
        dtype = [('index', itype), ('r11', ftype), ('r21', ftype)]
        data = np.recarray((6, 2), dtype=dtype)
        data[..., 0].index, data[..., 1].index = index1, index2
        data[..., 0].r11 = value1 * rotation1.data[:, 0, 0]
        data[..., 1].r11 = value2 * rotation2.data[:, 0, 0]
        data[..., 0].r21 = value1 * rotation1.data[:, 1, 0]
        data[..., 1].r21 = value2 * rotation2.data[:, 1, 0]
        matrix = FSRRotation2dMatrix((6 * 2, 5 * 2), data=data)
        return ProjectionOperator(matrix, dtype=stype)

    if cls is FSRRotation3dMatrix:
        dtype = [('index', itype), ('r11', ftype), ('r22', ftype), ('r32', ftype)]
        data = np.recarray((6, 2), dtype=dtype)
        data[..., 0].index, data[..., 1].index = index1, index2
        data[..., 0].r11, data[..., 1].r11 = value1, value2
        data[..., 0].r22 = value1 * rotation1.data[:, 0, 0]
        data[..., 1].r22 = value2 * rotation2.data[:, 0, 0]
        data[..., 0].r32 = value1 * rotation1.data[:, 1, 0]
        data[..., 1].r32 = value2 * rotation2.data[:, 1, 0]
        matrix = FSRRotation3dMatrix((6 * 3, 5 * 3), data=data)
        return ProjectionOperator(matrix, dtype=stype)

    raise


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_kernel(cls, itype, ftype):
    expected = [False, False, True, False, True]

    proj = get_projection(cls, itype, ftype)
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


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
def test_pT1(cls, itype, ftype, vtype):
    proj_ref = get_projection(FSRMatrix, np.int32, BIGGEST_FLOAT_TYPE)
    expected = proj_ref.T(np.ones(proj_ref.matrix.shape[0], BIGGEST_FLOAT_TYPE))
    expected_ = np.asarray(expected, min_dtype(ftype, vtype))

    proj = get_projection(cls, itype, ftype, vtype)
    pT1 = proj.pT1()
    assert_same(pT1, expected_)
    pT1 = proj.pT1(out=pT1)
    assert_same(pT1, expected_)
    pT1[...] = 2
    proj.pT1(out=pT1, operation=operator.iadd)
    assert_same(pT1, 2 + expected_)


@pytest.mark.parametrize('cls', [FSRMatrix, FSRRotation3dMatrix])
@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
def test_pTx_pT1(cls, itype, ftype, vtype):
    input_fsr = np.array([1.1, 2, -1.3, 2, -3, 4.1], dtype=BIGGEST_FLOAT_TYPE)
    # input_fsrrot2d = input_fsr[:, None] * Rotation2dOperator(
    #    np.arange(6) * 30, dtype=BIGGEST_FLOAT_TYPE
    # )([1, 0])
    input_fsrrot3d = np.array(
        [input_fsr, np.random.random_sample(6), np.random.random_sample(6)]
    ).T
    inputs = {
        FSRMatrix: input_fsr,
        # FSRRotation2dMatrix: input_fsrrot2d,
        FSRRotation3dMatrix: input_fsrrot3d,
    }

    def func(cls, itype, ftype, vtype):
        proj = get_projection(cls, itype, ftype, vtype)
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

    proj_ref = get_projection(FSRMatrix, np.int32, BIGGEST_FLOAT_TYPE)
    expectedx = proj_ref.T(input_fsr)
    expected1 = proj_ref.T(np.ones(proj_ref.matrix.shape[0], BIGGEST_FLOAT_TYPE))
    func(cls, itype, ftype, vtype)


@pytest.mark.parametrize('cls', CLASSES)
@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('inplace', [False, True])
def test_restrict(cls, itype, ftype, inplace):
    restriction = np.array([True, False, False, True, True])
    kernel = [False, False, True, False, True]

    proj_ref = get_projection(cls, itype, ftype)
    proj_ = get_projection(cls, itype, ftype)
    proj = proj_.restrict(restriction, inplace=inplace)
    if inplace:
        assert type(proj_) is DeletedOperator
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
