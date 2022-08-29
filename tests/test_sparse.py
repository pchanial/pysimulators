import itertools

import numpy as np
import pytest
import scipy.sparse

import pyoperators
from pyoperators import (
    CompositionOperator,
    DiagonalOperator,
    Rotation2dOperator,
    Rotation3dOperator,
)
from pyoperators.utils.testing import assert_same
from pysimulators.sparse import (
    FSCMatrix,
    FSCRotation2dMatrix,
    FSCRotation3dMatrix,
    FSRBlockMatrix,
    FSRMatrix,
    FSRRotation2dMatrix,
    FSRRotation3dMatrix,
    SparseOperator,
)

from .common import BIGGEST_FLOAT_TYPE, FLOAT_TYPES, INT_TYPES, SINT_TYPES


def min_type(t1, t2):
    return min(np.dtype(t1), np.dtype(t2)).type


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
@pytest.mark.parametrize('block_size', [1, 2])
def test_fsc1(itype, ftype, vtype, block_size):
    input = [1, 2, 1, 1, 1, 1]
    index = [3, 2, 2, 1, 2, -1]
    index_u = [3, 2, 2, 1, 2, 3]
    value = [1, 1, 0.5, 1, 2, 10]
    expected = [0, 1, 4.5, 1]
    expected_u = [0, 1, 4.5, 11]

    if np.dtype(itype).kind != 'u':
        ind = index
        exp = expected
    else:
        ind = index_u
        exp = expected_u
    input_ = np.array(input, vtype)
    if block_size == 2:
        input_ = np.array([input_, input_]).T.ravel()
        exp = np.array([exp, exp]).T.ravel()
    dtype = [('index', itype), ('value', ftype)]
    matrix = np.recarray((6, 1), dtype=dtype)
    matrix[..., 0].index = ind
    matrix[..., 0].value = value
    op = FSCMatrix((4, 6), data=matrix)
    out = op * np.array(input_, vtype)
    assert_same(out, exp)
    out[...] = 0
    op._matvec(np.array(input_, vtype), out=out)
    assert_same(out, exp)


def get_fsc2_mat(itype, ftype):
    if np.dtype(itype).kind != 'u':
        index = [[3, 2, 2, 1, 2, -1], [-1, 3, 1, 1, 0, -1]]
    else:
        index = [[3, 2, 2, 1, 2, 3], [3, 3, 1, 1, 0, 3]]
    value = [[1, 1, 0.5, 1, 2, 10], [1, 2, 0.5, 1, 1, 10]]
    dtype = [('index', itype), ('value', ftype)]
    matrix = np.recarray((6, 2), dtype=dtype)
    matrix[..., 0].index, matrix[..., 1].index = index
    matrix[..., 0].value, matrix[..., 1].value = value
    return FSCMatrix((4, 6), data=matrix)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
@pytest.mark.parametrize('block_size', [1, 2])
def test_fsc2(itype, ftype, vtype, block_size):
    input_fsc = np.asarray([1, 2, 1, 1, 1, 1], vtype)
    input_fsr = np.asarray([1, 2, 1, 4], vtype)
    if np.dtype(itype).kind != 'u':
        expected = np.asarray([1, 2.5, 4.5, 5])
    else:
        expected = np.asarray([1, 2.5, 4.5, 26])

    if block_size == 2:
        input_fsc = np.array([input_fsc, input_fsc]).T.ravel()
        input_fsr = np.array([input_fsr, input_fsr]).T.ravel()
        expected = np.array([expected, expected]).T.ravel()

    mat = get_fsc2_mat(itype, ftype)
    out = mat * input_fsc
    assert_same(out, expected)
    out[...] = 0
    mat._matvec(input_fsc, out=out)
    assert_same(out, expected)
    out = input_fsr * mat
    assert_same(out, FSRMatrix(mat.shape[::-1], data=mat.data) * input_fsr)
    out = (3 * mat) * input_fsc
    assert_same(out, 3 * expected)
    out = (mat * 3) * input_fsc
    assert_same(out, 3 * expected)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('shapein', [(), (3,)])
def test_fsc2_dense(itype, ftype, shapein):
    mat = get_fsc2_mat(itype, ftype)
    op = SparseOperator(mat)
    todense = op.todense(shapein=(6,) + shapein)
    assert_same(todense.T, op.T.todense(shapeout=(6,) + shapein))
    op2 = SparseOperator(mat, shapeout=(2, 2) + shapein, shapein=(3, 2) + shapein)
    assert_same(op2.todense(), todense)


def test_fsc3():
    mat = FSCMatrix((3, 4), 10)
    assert mat.data.shape == (4, 10)
    assert mat.data.index.dtype == int
    assert mat.data.value.dtype == float

    mat = FSCMatrix((3, 4), 10, dtype=np.float16, dtype_index=np.int8)
    assert mat.data.index.dtype == np.int8
    assert mat.data.value.dtype == np.float16


def test_fsc_error():
    data = np.zeros((3, 4), [('index', int), ('value', float)])
    data_wrong1 = np.empty((3, 4), [('index_', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('value', float), ('index', int)])

    with pytest.raises(TypeError):
        FSCMatrix(3)

    with pytest.raises(ValueError):
        FSCMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSCMatrix((3, 8), data=data)

    with pytest.raises(TypeError):
        FSCMatrix((8, 3), data=data_wrong1)

    with pytest.raises(TypeError):
        FSCMatrix((8, 3), data=data_wrong2)

    with pytest.raises(ValueError):
        FSCMatrix((8, 3))

    mat = FSCMatrix((8, 3), data=data)
    mat._matvec(np.ones(3))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(7))

    with pytest.raises(TypeError):
        mat._matvec(np.ones(3), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones(3), out=np.zeros(7))


def get_fsr1_mat(itype, ftype):
    if np.dtype(itype).kind != 'u':
        ind = [3, 2, 2, 1, 2, -1]
    else:
        ind = [3, 2, 2, 1, 2, 3]
    value = [1, 1, 0.5, 1, 2, 10]
    dtype = [('index', itype), ('value', ftype)]

    matrix = np.recarray((6, 1), dtype=dtype)
    matrix[..., 0].index = ind
    matrix[..., 0].value = value
    return FSRMatrix((6, 4), data=matrix)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
@pytest.mark.parametrize('block_size', [1, 2])
def test_fsr1(itype, ftype, vtype, block_size):
    input = [1, 2, 3, 4]
    expected = [4, 3, 1.5, 2, 6, 0]
    expected_u = [4, 3, 1.5, 2, 6, 40]

    if np.dtype(itype).kind != 'u':
        exp = expected
    else:
        exp = expected_u
    input_ = np.array(input, vtype)
    if block_size == 2:
        input_ = np.array([input_, input_]).T.ravel()
        exp = np.array([exp, exp]).T.ravel()
    op = get_fsr1_mat(itype, ftype)
    out = op * input_
    assert_same(out, exp)
    out[...] = 0
    op._matvec(input_, out=out)
    assert_same(out, exp)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_fsr1_dense(itype, ftype):
    mat = get_fsr1_mat(itype, ftype)
    op = SparseOperator(mat, shapein=4)
    todense = op.todense()
    pTp = op.T * op
    if (itype, ftype) not in (
        (np.int32, np.float32),
        (np.int32, np.float64),
        (np.int64, np.float32),
        (np.int64, np.float64),
    ):
        assert type(pTp) is CompositionOperator
        return
    assert type(pTp) is DiagonalOperator
    assert_same(pTp.todense(), np.dot(todense.T, todense))


def get_fsr2_mat(itype, ftype):
    if np.dtype(itype).kind != 'u':
        index = [[3, 2, 2, 1, 2, -1], [-1, 3, 1, 1, 0, -1]]
    else:
        index = [[3, 2, 2, 1, 2, 3], [3, 3, 1, 1, 0, 3]]
    value = [[1, 1, 0.5, 1, 2, 10], [1, 2, 0.5, 1, 1, 10]]
    dtype = [('index', itype), ('value', ftype)]
    matrix = np.recarray((6, 2), dtype=dtype)
    matrix[..., 0].index, matrix[..., 1].index = index
    matrix[..., 0].value, matrix[..., 1].value = value
    return FSRMatrix((6, 4), data=matrix)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('vtype', FLOAT_TYPES)
@pytest.mark.parametrize('block_size', [1, 2])
def test_fsr2(itype, ftype, vtype, block_size):
    input_fsr = [1, 2, 3, 4]
    input_fsc = [1, 2, 3, 4, 5, 6]
    expected = [4, 11, 2.5, 4, 7, 0]
    expected_u = [8, 11, 2.5, 4, 7, 80]

    input_fsc_ = np.asarray(input_fsc, vtype)
    input_fsr_ = np.asarray(input_fsr, vtype)
    if np.dtype(itype).kind != 'u':
        exp = np.asarray(expected)
    else:
        exp = np.asarray(expected_u)
    if block_size == 2:
        input_fsc_ = np.array([input_fsc_, input_fsc_]).T.ravel()
        input_fsr_ = np.array([input_fsr_, input_fsr_]).T.ravel()
        exp = np.array([exp, exp]).T.ravel()
    mat = get_fsr2_mat(itype, ftype)
    out = mat * input_fsr_
    assert_same(out, exp)
    out[...] = 0
    mat._matvec(input_fsr_, out=out)
    assert_same(out, exp)
    out = input_fsc_ * mat
    assert_same(out, FSCMatrix(mat.shape[::-1], data=mat.data) * input_fsc_)
    out = (3 * mat) * input_fsr_
    assert_same(out, 3 * exp)
    out = (mat * 3) * input_fsr_
    assert_same(out, 3 * exp)


@pytest.mark.parametrize('itype', INT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize('shapein', [(), (3,)])
def test_fsr2_dense(itype, ftype, shapein):
    mat = get_fsr2_mat(itype, ftype)
    op = SparseOperator(mat)
    todense = op.todense(shapein=(4,) + shapein)
    assert_same(todense.T, op.T.todense(shapeout=(4,) + shapein))
    if shapein is None:
        return
    op2 = SparseOperator(mat, shapein=(2, 2) + shapein, shapeout=(3, 2) + shapein)
    assert_same(op2.todense(), todense)


def test_fsr3():
    mat = FSRMatrix((4, 3), 10)
    assert mat.data.shape == (4, 10)
    assert mat.data.index.dtype == int
    assert mat.data.value.dtype == float

    mat = FSRMatrix((4, 3), 10, dtype=np.float16, dtype_index=np.int8)
    assert mat.data.index.dtype == np.int8
    assert mat.data.value.dtype == np.float16


def test_fsr_error():
    data = np.zeros((3, 4), [('index', int), ('value', float)])
    data_wrong1 = np.empty((3, 4), [('index_', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('value', float), ('index', int)])

    with pytest.raises(TypeError):
        FSRMatrix(3)

    with pytest.raises(ValueError):
        FSRMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSRMatrix((8, 3), data=data)

    with pytest.raises(TypeError):
        FSRMatrix((3, 8), data=data_wrong1)

    with pytest.raises(TypeError):
        FSRMatrix((3, 8), data=data_wrong2)

    with pytest.raises(ValueError):
        FSRMatrix((3, 8), 5, data=data)

    with pytest.raises(ValueError):
        FSRMatrix((3, 8))

    mat = FSRMatrix((3, 8), data=data)
    mat._matvec(np.ones(8))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(7))

    with pytest.raises(TypeError):
        mat._matvec(np.ones(8), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones(8), out=np.zeros(4))


def get_block_mat(itype, ftype, inner_block_shape):
    outer_block_shape = (6, 4)
    indices = [[3, 2, 2, 1, 2, -1], [-1, 3, 1, 1, 0, -1]]
    shape = [i * o for i, o in zip(inner_block_shape, outer_block_shape)]
    mat = FSRBlockMatrix(
        shape, inner_block_shape, len(indices), dtype=ftype, dtype_index=itype
    )
    for k, index in enumerate(indices):
        mat.data[:, k].index = index
    for i in range(outer_block_shape[0]):
        for k in range(len(indices)):
            mat.data[i, k]['value'][...] = np.random.randint(-10, 11, inner_block_shape)

    dense = np.zeros(shape, dtype=ftype)
    for i, d in enumerate(mat.data):
        a1, z1 = i * inner_block_shape[0], (i + 1) * inner_block_shape[0]
        for d_ in d:
            j = d_['index']
            if j < 0:
                continue
            a2, z2 = j * inner_block_shape[1], (j + 1) * inner_block_shape[1]
            dense[a1:z1, a2:z2] += d_['value']
    return mat, dense


@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
@pytest.mark.parametrize(
    'inner_block_shape',
    itertools.islice(itertools.product([1, 2, 3], repeat=2), 1, None),
)
def test_block(itype, ftype, inner_block_shape):
    np.random.seed(0)

    mat_fsr, dense = get_block_mat(itype, ftype, inner_block_shape)
    mat_fsc = mat_fsr._transpose()
    input_fsc = np.random.random_sample(mat_fsc.shape[1])
    input_fsr = np.random.random_sample(mat_fsr.shape[1])

    assert_same(mat_fsr.block_shape, inner_block_shape)
    assert_same(mat_fsc.block_shape, inner_block_shape[::-1])
    op = SparseOperator(mat_fsc)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense.T)
    assert_same(op.T.todense(), dense)

    ref = mat_fsc._matvec(input_fsc.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsc._matvec(input_fsc.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)), atol=10)

    ref = (2 * (mat_fsc * input_fsc)).astype(ftype)
    assert_same((mat_fsc * 2) * input_fsc, ref, atol=10)
    assert_same((2 * mat_fsc) * input_fsc, ref, atol=10)
    assert_same(input_fsr * mat_fsc, mat_fsr * input_fsr, atol=10)

    op = SparseOperator(mat_fsr)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense)
    assert_same(op.T.todense(), dense.T)

    ref = mat_fsr._matvec(input_fsr.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsr._matvec(input_fsr.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)), atol=10)

    ref = (2 * (mat_fsr * input_fsr)).astype(ftype)
    assert_same((mat_fsr * 2) * input_fsr, ref, atol=10)
    assert_same((2 * mat_fsr) * input_fsr, ref, atol=10)
    assert_same(input_fsc * mat_fsr, mat_fsc * input_fsc, atol=10)


def fill_rot2d(array, dense, index, value, angle, n):
    for i, (j, v, a) in enumerate(zip(index, value, angle)):
        array[i, n].index = j
        if j == -1:
            array[i, n].r11 = 0
            array[i, n].r21 = 0
            continue
        r = v * Rotation2dOperator(a, degrees=True).todense(shapein=2)
        array[i, n].r11 = r[0, 0]
        array[i, n].r21 = r[1, 0]
        dense[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] += r


@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_rot2d(itype, ftype):
    input_fsc = np.arange(6 * 2, dtype=ftype)
    input_fsr = np.arange(4 * 2, dtype=ftype)
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    angle1 = [1, 10, 20, 30, 40, -10]
    angle2 = [0, 30, -20, 1, 1, 10]

    array = np.recarray(
        (6, 2), dtype=[('index', itype), ('r11', ftype), ('r21', ftype)]
    )
    dense = np.zeros((6 * 2, 4 * 2), dtype=ftype)
    fill_rot2d(array, dense, index1, value1, angle1, 0)
    fill_rot2d(array, dense, index2, value2, angle2, 1)

    mat_fsc = FSCRotation2dMatrix(dense.shape[::-1], data=array)
    mat_fsr = FSRRotation2dMatrix(dense.shape, data=array)

    op = SparseOperator(mat_fsc)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense.T)
    assert_same(op.T.todense(), dense)
    ref = mat_fsc._matvec(input_fsc.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsc._matvec(input_fsc.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)))
    ref = 2 * (mat_fsc * input_fsc)
    assert_same((mat_fsc * 2) * input_fsc, ref)
    assert_same((2 * mat_fsc) * input_fsc, ref)
    assert_same(input_fsr * mat_fsc, mat_fsr * input_fsr)

    op = SparseOperator(mat_fsr)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense)
    assert_same(op.T.todense(), dense.T)
    ref = mat_fsr._matvec(input_fsr.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsr._matvec(input_fsr.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)))
    ref = 2 * (mat_fsr * input_fsr)
    assert_same((mat_fsr * 2) * input_fsr, ref)
    assert_same((2 * mat_fsr) * input_fsr, ref)
    assert_same(input_fsc * mat_fsr, mat_fsc * input_fsc)


@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_rot2d_dense(itype, ftype):
    index1 = [3, 2, 2, 1, 2, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    angle1 = [1, 10, 20, 30, 40, -10]

    array = np.recarray(
        (6, 1), dtype=[('index', itype), ('r11', ftype), ('r21', ftype)]
    )
    dense = np.zeros((6 * 2, 4 * 2), dtype=ftype)
    fill_rot2d(array, dense, index1, value1, angle1, 0)
    op = SparseOperator(FSRRotation2dMatrix(dense.shape, data=array))
    pTp = op.T * op
    if (itype, ftype) not in (
        (np.int32, np.float32),
        (np.int32, np.float64),
        (np.int64, np.float32),
        (np.int64, np.float64),
    ):
        assert type(pTp) is CompositionOperator
        return
    assert type(pTp) is DiagonalOperator
    assert_same(pTp.todense(), np.dot(dense.T, dense), atol=1)


def test_fsc_rot2d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float), ('r21', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r21', float), ('index', int)])
    data_wrong3 = np.empty(
        (3, 4), dtype=[('index', int), ('r11', float), ('r21', np.float32)]
    )

    with pytest.raises(TypeError):
        FSCRotation2dMatrix(2)

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((6, 16), data=data)

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((7, 16), data=data)

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((17, 6), data=data)

    with pytest.raises(TypeError):
        FSCRotation2dMatrix((16, 6), data=data_wrong1)

    with pytest.raises(TypeError):
        FSCRotation2dMatrix((16, 6), data=data_wrong2)

    with pytest.raises(TypeError):
        FSCRotation2dMatrix((16, 6), data=data_wrong3)

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((16, 6), 5, data=data)

    with pytest.raises(ValueError):
        FSCRotation2dMatrix((16, 6))

    mat = FSCRotation2dMatrix((16, 6), data=data)
    with pytest.raises(ValueError):
        mat._matvec(np.ones(4))

    mat._matvec(np.ones((3, 2)))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(17))

    with pytest.raises(ValueError):
        mat._matvec(np.ones((8, 2)))

    with pytest.raises(TypeError):
        mat._matvec(np.ones((3, 2)), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones((3, 2)), out=np.zeros(7))


def test_fsr_rot2d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float), ('r21', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r21', float), ('index', int)])
    data_wrong3 = np.empty(
        (3, 4), dtype=[('index', int), ('r11', float), ('r21', np.float32)]
    )

    with pytest.raises(TypeError):
        FSRRotation2dMatrix(3)

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((16, 6), data=data)

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((7, 16), data=data)

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((6, 17), data=data)

    with pytest.raises(TypeError):
        FSRRotation2dMatrix((6, 16), data=data_wrong1)

    with pytest.raises(TypeError):
        FSRRotation2dMatrix((6, 16), data=data_wrong2)

    with pytest.raises(TypeError):
        FSRRotation2dMatrix((6, 16), data=data_wrong3)

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((6, 16), 5, data=data)

    with pytest.raises(ValueError):
        FSRRotation2dMatrix((6, 16))

    mat = FSRRotation2dMatrix((6, 16), data=data)
    with pytest.raises(ValueError):
        mat._matvec(np.ones(4))

    mat._matvec(np.ones((8, 2)))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(17))

    with pytest.raises(ValueError):
        mat._matvec(np.ones((3, 2)))

    with pytest.raises(TypeError):
        mat._matvec(np.ones((8, 2)), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones((8, 2)), out=np.zeros(7))


def fill_rot3d(array, dense, index, value, angle, n):
    for i, (j, v, a) in enumerate(zip(index, value, angle)):
        array[i, n].index = j
        if j == -1:
            array[i, n].r11 = 0
            array[i, n].r22 = 0
            array[i, n].r32 = 0
            continue
        r = v * Rotation3dOperator('X', a, degrees=True).todense(shapein=3)
        array[i, n].r11 = r[0, 0]
        array[i, n].r22 = r[1, 1]
        array[i, n].r32 = r[2, 1]
        dense[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += r


@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_rot3d(itype, ftype):
    input_fsc = np.arange(6 * 3, dtype=ftype)
    input_fsr = np.arange(4 * 3, dtype=ftype)
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    angle1 = [1, 10, 20, 30, 40, -10]
    angle2 = [0, 30, -20, 1, 1, 10]

    array = np.recarray(
        (6, 2),
        dtype=[('index', itype), ('r11', ftype), ('r22', ftype), ('r32', ftype)],
    )
    dense = np.zeros((6 * 3, 4 * 3), dtype=ftype)
    fill_rot3d(array, dense, index1, value1, angle1, 0)
    fill_rot3d(array, dense, index2, value2, angle2, 1)

    mat_fsc = FSCRotation3dMatrix(dense.shape[::-1], data=array)
    mat_fsr = FSRRotation3dMatrix(dense.shape, data=array)

    op = SparseOperator(mat_fsc)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense.T)
    assert_same(op.T.todense(), dense)
    ref = mat_fsc._matvec(input_fsc.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsc._matvec(input_fsc.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)))
    ref = 3 * (mat_fsc * input_fsc)
    assert_same((mat_fsc * 3) * input_fsc, ref)
    assert_same((3 * mat_fsc) * input_fsc, ref)
    assert_same(input_fsr * mat_fsc, mat_fsr * input_fsr)

    op = SparseOperator(mat_fsr)
    assert op.matrix.dtype == ftype
    assert op.dtype == ftype
    assert_same(op.todense(), dense)
    assert_same(op.T.todense(), dense.T)
    ref = mat_fsr._matvec(input_fsr.astype(BIGGEST_FLOAT_TYPE))
    for ftype2 in FLOAT_TYPES:
        out = np.zeros_like(ref, ftype2)
        mat_fsr._matvec(input_fsr.astype(ftype2), out)
        assert_same(out, ref.astype(min_type(ftype, ftype2)))
    ref = 3 * (mat_fsr * input_fsr)
    assert_same((mat_fsr * 3) * input_fsr, ref, rtol=10)
    assert_same((3 * mat_fsr) * input_fsr, ref, rtol=10)
    assert_same(input_fsc * mat_fsr, mat_fsc * input_fsc, rtol=10)


@pytest.mark.parametrize('itype', SINT_TYPES)
@pytest.mark.parametrize('ftype', FLOAT_TYPES)
def test_rot3d_dense(itype, ftype):
    index1 = [3, 2, 2, 1, 2, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    angle1 = [1, 10, 20, 30, 40, -10]

    array = np.recarray(
        (6, 1),
        dtype=[('index', itype), ('r11', ftype), ('r22', ftype), ('r32', ftype)],
    )
    dense = np.zeros((6 * 3, 4 * 3), dtype=ftype)
    fill_rot3d(array, dense, index1, value1, angle1, 0)
    op = SparseOperator(FSRRotation3dMatrix(dense.shape, data=array))
    pTp = op.T * op
    if (itype, ftype) not in (
        (np.int32, np.float32),
        (np.int32, np.float64),
        (np.int64, np.float32),
        (np.int64, np.float64),
    ):
        assert type(pTp) is CompositionOperator
        return
    assert type(pTp) is DiagonalOperator
    assert_same(pTp.todense(), np.dot(dense.T, dense), atol=1)


def test_fsc_rot3d_error():
    data = np.zeros(
        (3, 4), dtype=[('index', int), ('r11', float), ('r22', float), ('r32', float)]
    )
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty(
        (3, 4), [('r11', float), ('r22', float), ('r32', float), ('index', int)]
    )
    data_wrong3 = np.empty(
        (3, 4),
        dtype=[('index', int), ('r11', float), ('r22', np.float32), ('r32', float)],
    )
    with pytest.raises(TypeError):
        FSCRotation3dMatrix(3)

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((9, 24), data=data)

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((10, 24), data=data)

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((25, 9), data=data)

    with pytest.raises(TypeError):
        FSCRotation3dMatrix((24, 9), data=data_wrong1)

    with pytest.raises(TypeError):
        FSCRotation3dMatrix((24, 9), data=data_wrong2)

    with pytest.raises(TypeError):
        FSCRotation3dMatrix((24, 9), data=data_wrong3)

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((24, 9), 5, data=data)

    with pytest.raises(ValueError):
        FSCRotation3dMatrix((24, 9))

    mat = FSCRotation3dMatrix((24, 9), data=data)
    with pytest.raises(ValueError):
        mat._matvec(np.ones(4))

    mat._matvec(np.ones((3, 3)))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(25))

    with pytest.raises(ValueError):
        mat._matvec(np.ones((8, 3)))

    with pytest.raises(TypeError):
        mat._matvec(np.ones((3, 3)), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones((3, 3)), out=np.zeros(7))


def test_fsr_rot3d_error():
    data = np.zeros(
        (3, 4), dtype=[('index', int), ('r11', float), ('r22', float), ('r32', float)]
    )
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty(
        (3, 4), [('r11', float), ('r22', float), ('r32', float), ('index', int)]
    )
    data_wrong3 = np.empty(
        (3, 4),
        dtype=[('index', int), ('r11', float), ('r22', np.float32), ('r32', float)],
    )

    with pytest.raises(TypeError):
        FSRRotation3dMatrix(3)

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((2, 3, 4))

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((24, 9), data=data)

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((10, 24), data=data)

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((9, 25), data=data)

    with pytest.raises(TypeError):
        FSRRotation3dMatrix((9, 24), data=data_wrong1)

    with pytest.raises(TypeError):
        FSRRotation3dMatrix((9, 24), data=data_wrong2)

    with pytest.raises(TypeError):
        FSRRotation3dMatrix((9, 24), data=data_wrong3)

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((9, 24), 5, data=data)

    with pytest.raises(ValueError):
        FSRRotation3dMatrix((9, 24))

    mat = FSRRotation3dMatrix((9, 24), data=data)
    with pytest.raises(ValueError):
        mat._matvec(np.ones(4))

    mat._matvec(np.ones((8, 3)))
    with pytest.raises(ValueError):
        mat._matvec(np.ones(25))

    with pytest.raises(ValueError):
        mat._matvec(np.ones((3, 3)))

    with pytest.raises(TypeError):
        mat._matvec(np.ones((8, 3)), out=1)

    with pytest.raises(ValueError):
        mat._matvec(np.ones((8, 3)), out=np.zeros(7))


def test_sparse_operator():
    with pytest.raises(TypeError):
        SparseOperator(np.zeros((10, 3)))
    sp = SparseOperator(scipy.sparse.csc_matrix((4, 6)))
    assert type(sp) is pyoperators.linear.SparseOperator

    data = FSRMatrix((10, 3), ncolmax=1)
    with pytest.raises(ValueError):
        SparseOperator(data, shapein=4)
    with pytest.raises(ValueError):
        SparseOperator(data, shapeout=11)

    data = FSRRotation2dMatrix((12, 6), ncolmax=1)
    with pytest.raises(ValueError):
        SparseOperator(data, shapein=(5, 2))
    with pytest.raises(ValueError):
        SparseOperator(data, shapeout=(7,))

    data = FSRRotation3dMatrix((12, 6), ncolmax=1)
    with pytest.raises(ValueError):
        SparseOperator(data, shapein=(5, 3))
    with pytest.raises(ValueError):
        SparseOperator(data, shapeout=(7,))
