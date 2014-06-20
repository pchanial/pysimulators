from __future__ import division

import numpy as np
import pyoperators
import scipy.sparse
from numpy.testing import assert_equal, assert_raises
from pyoperators import Rotation2dOperator, Rotation3dOperator
from pyoperators.utils.testing import assert_is_type, assert_same
from pysimulators.sparse import (
    FSCMatrix, FSRMatrix, FSCRotation2dMatrix, FSRRotation2dMatrix,
    FSCRotation3dMatrix, FSRRotation3dMatrix, SparseOperator)

ftypes = np.float16, np.float32, np.float64, np.float128
iutypes = (np.uint8, np.uint16, np.uint32, np.uint64)
itypes = (np.int8, np.int16, np.int32, np.int64)


def test_fsc1():
    input = [1, 2, 1, 1, 1, 1]
    index = [3, 2, 2, 1, 2, -1]
    index_u = [3, 2, 2, 1, 2, 3]
    value = [1, 1, 0.5, 1, 2, 10]
    expected = [0, 1, 4.5, 1]
    expected_u = [0, 1, 4.5, 11]

    def func(itype, ftype, vtype):
        if np.dtype(itype).kind != 'u':
            ind = index
            exp = expected
        else:
            ind = index_u
            exp = expected_u
        dtype = [('index', itype), ('value', ftype)]
        matrix = np.recarray((6, 1), dtype=dtype)
        matrix[..., 0].index = ind
        matrix[..., 0].value = value
        op = FSCMatrix((4, 6), matrix)
        out = op * np.array(input, vtype)
        assert_same(out, exp)
        out[...] = 0
        op._matvec(np.array(input, vtype), out=out)
        assert_same(out, exp)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            for vtype in ftypes:
                yield func, itype, ftype, vtype


def test_fsc2():
    input_fsc = [1, 2, 1, 1, 1, 1]
    input_fsr = [1, 2, 1, 4]
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    index1_u = [3, 2, 2, 1, 2, 3]
    index2_u = [3, 3, 1, 1, 0, 3]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    expected = [1, 2.5, 4.5, 5]
    expected_u = [1, 2.5, 4.5, 26]

    def get_mat(itype, ftype):
        if np.dtype(itype).kind != 'u':
            ind = index1, index2
        else:
            ind = index1_u, index2_u
        dtype = [('index', itype), ('value', ftype)]
        matrix = np.recarray((6, 2), dtype=dtype)
        matrix[..., 0].index, matrix[..., 1].index = ind
        matrix[..., 0].value, matrix[..., 1].value = value1, value2
        return FSCMatrix((4, 6), matrix)

    def func1(itype, ftype, vtype):
        input_fsc_ = np.asarray(input_fsc, vtype)
        input_fsr_ = np.asarray(input_fsr, vtype)
        if np.dtype(itype).kind != 'u':
            exp = np.asarray(expected)
        else:
            exp = np.asarray(expected_u)
        mat = get_mat(itype, ftype)
        out = mat * input_fsc_
        assert_same(out, exp)
        out[...] = 0
        mat._matvec(input_fsc_, out=out)
        assert_same(out, exp)
        out = input_fsr_ * mat
        assert_same(out, FSRMatrix(mat.shape[::-1], mat.data) * input_fsr_)
        out = (3 * mat) * input_fsc_
        assert_same(out, 3 * exp)
        out = (mat * 3) * input_fsc_
        assert_same(out, 3 * exp)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            for vtype in ftypes:
                yield func1, itype, ftype, vtype

    def func2(itype, ftype):
        mat = get_mat(itype, ftype)
        op = SparseOperator(mat)
        todense = op.todense()
        assert_same(todense.T, op.T.todense())
        op2 = SparseOperator(mat, shapeout=(2, 2), shapein=(3, 2))
        assert_same(op2.todense(), todense)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            yield func2, itype, ftype


def test_fsc3():
    mat = FSCMatrix((3, 4), nrowmax=10)
    assert_equal(mat.data.shape, (4, 10))
    assert_equal(mat.data.index.dtype, int)
    assert_equal(mat.data.value.dtype, float)

    mat = FSCMatrix((3, 4), nrowmax=10, dtype=np.float16, dtype_index=np.int8)
    assert_equal(mat.data.index.dtype, np.int8)
    assert_equal(mat.data.value.dtype, np.float16)


def test_fsc_error():
    data = np.zeros((3, 4), [('index', int), ('value', float)])
    data_wrong1 = np.empty((3, 4), [('index_', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('value', float), ('index', int)])
    assert_raises(TypeError, FSCMatrix, 3)
    assert_raises(ValueError, FSCMatrix, (2, 3, 4))
    assert_raises(ValueError, FSCMatrix, (3, 8), data)
    assert_raises(TypeError, FSCMatrix, (8, 3), data_wrong1)
    assert_raises(TypeError, FSCMatrix, (8, 3), data_wrong2)
    assert_raises(ValueError, FSCMatrix, (8, 3), data, nrowmax=5)
    FSCMatrix((8, 3), data, nrowmax=4)
    assert_raises(ValueError, FSCMatrix, (8, 3))
    mat = FSCMatrix((8, 3), data)
    mat._matvec(np.ones(3))
    assert_raises(ValueError, mat._matvec, np.ones(7))
    assert_raises(TypeError, mat._matvec, np.ones(3), out=1)
    assert_raises(ValueError, mat._matvec, np.ones(3), out=np.zeros(7))


def test_fsr1():
    input = [1, 2, 3, 4]
    index = [3, 2, 2, 1, 2, -1]
    index_u = [3, 2, 2, 1, 2, 3]
    value = [1, 1, 0.5, 1, 2, 10]
    expected = [4, 3, 1.5, 2, 6, 0]
    expected_u = [4, 3, 1.5, 2, 6, 40]

    def func(itype, ftype, vtype):
        if np.dtype(itype).kind != 'u':
            ind = index
            exp = expected
        else:
            ind = index_u
            exp = expected_u
        dtype = [('index', itype), ('value', ftype)]
        matrix = np.recarray((6, 1), dtype=dtype)
        matrix[..., 0].index = ind
        matrix[..., 0].value = value
        op = FSRMatrix((6, 4), matrix)
        out = op * np.array(input, vtype)
        assert_same(out, exp)
        out[...] = 0
        op._matvec(np.array(input, vtype), out=out)
        assert_same(out, exp)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            for vtype in ftypes:
                yield func, itype, ftype, vtype


def test_fsr2():
    input_fsr = [1, 2, 3, 4]
    input_fsc = [1, 2, 3, 4, 5, 6]
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    index1_u = [3, 2, 2, 1, 2, 3]
    index2_u = [3, 3, 1, 1, 0, 3]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    expected = [4, 11, 2.5, 4, 7, 0]
    expected_u = [8, 11, 2.5, 4, 7, 80]

    def get_mat(itype, ftype):
        if np.dtype(itype).kind != 'u':
            ind = index1, index2
        else:
            ind = index1_u, index2_u
        dtype = [('index', itype), ('value', ftype)]
        matrix = np.recarray((6, 2), dtype=dtype)
        matrix[..., 0].index, matrix[..., 1].index = ind
        matrix[..., 0].value, matrix[..., 1].value = value1, value2
        return FSRMatrix((6, 4), matrix)

    def func1(itype, ftype, vtype):
        input_fsc_ = np.asarray(input_fsc, vtype)
        input_fsr_ = np.asarray(input_fsr, vtype)
        if np.dtype(itype).kind != 'u':
            exp = np.asarray(expected)
        else:
            exp = np.asarray(expected_u)
        mat = get_mat(itype, ftype)
        out = mat * input_fsr_
        assert_same(out, exp)
        out[...] = 0
        mat._matvec(input_fsr_, out=out)
        assert_same(out, exp)
        out = input_fsc_ * mat
        assert_same(out, FSCMatrix(mat.shape[::-1], mat.data) * input_fsc_)
        out = (3 * mat) * input_fsr_
        assert_same(out, 3 * exp)
        out = (mat * 3) * input_fsr_
        assert_same(out, 3 * exp)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            for vtype in ftypes:
                yield func1, itype, ftype, vtype

    def func2(itype, ftype):
        mat = get_mat(itype, ftype)
        op = SparseOperator(mat)
        todense = op.todense()
        assert_same(todense.T, op.T.todense())
        op2 = SparseOperator(mat, shapein=(2, 2), shapeout=(3, 2))
        assert_same(op2.todense(), todense)
    for itype in iutypes + itypes:
        for ftype in ftypes:
            yield func2, itype, ftype


def test_fsr3():
    mat = FSRMatrix((4, 3), ncolmax=10)
    assert_equal(mat.data.shape, (4, 10))
    assert_equal(mat.data.index.dtype, int)
    assert_equal(mat.data.value.dtype, float)

    mat = FSRMatrix((4, 3), ncolmax=10, dtype=np.float16, dtype_index=np.int8)
    assert_equal(mat.data.index.dtype, np.int8)
    assert_equal(mat.data.value.dtype, np.float16)



def test_fsr_error():
    data = np.zeros((3, 4), [('index', int), ('value', float)])
    data_wrong1 = np.empty((3, 4), [('index_', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('value', float), ('index', int)])
    assert_raises(TypeError, FSRMatrix, 3)
    assert_raises(ValueError, FSRMatrix, (2, 3, 4))
    assert_raises(ValueError, FSRMatrix, (8, 3), data)
    assert_raises(TypeError, FSRMatrix, (3, 8), data_wrong1)
    assert_raises(TypeError, FSRMatrix, (3, 8), data_wrong2)
    assert_raises(ValueError, FSRMatrix, (3, 8), data, ncolmax=5)
    FSRMatrix((3, 8), data, ncolmax=4)
    assert_raises(ValueError, FSRMatrix, (3, 8))
    mat = FSRMatrix((3, 8), data)
    mat._matvec(np.ones(8))
    assert_raises(ValueError, mat._matvec, np.ones(7))
    assert_raises(TypeError, mat._matvec, np.ones(8), out=1)
    assert_raises(ValueError, mat._matvec, np.ones(8), out=np.zeros(4))


def test_rot2d():
    input_fsc = np.arange(6*2.)
    input_fsr = np.arange(4*2.)
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    angle1 = [1, 10, 20, 30, 40, -10]
    angle2 = [0, 30, -20, 1, 1, 10]

    def fill(array, dense, index, value, angle, n):
        for i, (j, v, a) in enumerate(zip(index, value, angle)):
            array[i, n].index = j
            if j == -1:
                continue
            r = v * Rotation2dOperator(a, degrees=True).todense(shapein=2)
            array[i, n].r11 = r[0, 0]
            array[i, n].r21 = r[1, 0]
            dense[2*i:2*i+2, 2*j:2*j+2] += r

    def func(itype, ftype):
        array = np.recarray((6, 2), dtype=[('index', itype), ('r11', ftype),
                                           ('r21', ftype)])
        dense = np.zeros((6*2, 4*2), dtype=ftype)
        fill(array, dense, index1, value1, angle1, 0)
        fill(array, dense, index2, value2, angle2, 1)

        mat_fsc = FSCRotation2dMatrix(dense.shape[::-1], array)
        mat_fsr = FSRRotation2dMatrix(dense.shape, array)

        op = SparseOperator(mat_fsc)
        assert_equal(op.matrix.dtype, ftype)
        assert_equal(op.dtype, ftype)
        assert_same(op.todense(), dense.T)
        assert_same(op.T.todense(), dense)
        ref = mat_fsc._matvec(input_fsc.astype(np.float128))
        for ftype2 in ftypes[:-1]:
            out = np.zeros_like(ref, ftype2)
            mat_fsc._matvec(input_fsc.astype(ftype2), out)
            if np.__version__ >= '1.8':
                assert_same(out, ref.astype(min(ftype, ftype2)))
            else:
                assert_same(out, ref.astype(min(ftype, ftype2,
                                                key=lambda x: x().itemsize)))
        ref = (2 * (mat_fsc * input_fsc)).astype(ftype)
        assert_same((mat_fsc * 2) * input_fsc, ref)
        assert_same((2 * mat_fsc) * input_fsc, ref)
        assert_same(input_fsr * mat_fsc, mat_fsr * input_fsr)

        op = SparseOperator(mat_fsr)
        assert_equal(op.matrix.dtype, ftype)
        assert_equal(op.dtype, ftype)
        assert_same(op.todense(), dense)
        assert_same(op.T.todense(), dense.T)
        ref = mat_fsr._matvec(input_fsr.astype(np.float128))
        for ftype2 in ftypes[:-1]:
            out = np.zeros_like(ref, ftype2)
            mat_fsr._matvec(input_fsr.astype(ftype2), out)
            if np.__version__ >= '1.8':
                assert_same(out, ref.astype(min(ftype, ftype2)))
            else:
                assert_same(out, ref.astype(min(ftype, ftype2,
                                                key=lambda x: x().itemsize)))
        ref = (2 * (mat_fsr * input_fsr)).astype(ftype)
        assert_same((mat_fsr * 2) * input_fsr, ref)
        assert_same((2 * mat_fsr) * input_fsr, ref)
        assert_same(input_fsc * mat_fsr, mat_fsc * input_fsc)

    for itype in itypes:
        for ftype in ftypes:
            yield func, itype, ftype


def test_fsc_rot2d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float),
                                   ('r21', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r21', float),
                                    ('index', int)])
    data_wrong3 = np.empty((3, 4), dtype=[('index', int), ('r11', float),
                                          ('r21', np.float32)])
    assert_raises(TypeError, FSCRotation2dMatrix, 2)
    assert_raises(ValueError, FSCRotation2dMatrix, (2, 3, 4))
    assert_raises(ValueError, FSCRotation2dMatrix, (6, 16), data)
    assert_raises(ValueError, FSCRotation2dMatrix, (7, 16), data)
    assert_raises(ValueError, FSCRotation2dMatrix, (17, 6), data)
    assert_raises(TypeError, FSCRotation2dMatrix, (16, 6), data_wrong1)
    assert_raises(TypeError, FSCRotation2dMatrix, (16, 6), data_wrong2)
    assert_raises(TypeError, FSCRotation2dMatrix, (16, 6), data_wrong3)
    assert_raises(ValueError, FSCRotation2dMatrix, (16, 6), data, nrowmax=5)
    assert_raises(ValueError, FSCRotation2dMatrix, (16, 6))
    mat = FSCRotation2dMatrix((16, 6), data)
    assert_raises(ValueError, mat._matvec, np.ones(4))
    mat._matvec(np.ones((3, 2)))
    assert_raises(ValueError, mat._matvec, np.ones(17))
    assert_raises(ValueError, mat._matvec, np.ones((8, 2)))
    assert_raises(TypeError, mat._matvec, np.ones((3, 2)), out=1)
    assert_raises(ValueError, mat._matvec, np.ones((3, 2)), out=np.zeros(7))


def test_fsr_rot2d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float),
                                   ('r21', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r21', float),
                                    ('index', int)])
    data_wrong3 = np.empty((3, 4), dtype=[('index', int), ('r11', float),
                                          ('r21', np.float32)])
    assert_raises(TypeError, FSRRotation2dMatrix, 3)
    assert_raises(ValueError, FSRRotation2dMatrix, (2, 3, 4))
    assert_raises(ValueError, FSRRotation2dMatrix, (16, 6), data)
    assert_raises(ValueError, FSRRotation2dMatrix, (7, 16), data)
    assert_raises(ValueError, FSRRotation2dMatrix, (6, 17), data)
    assert_raises(TypeError, FSRRotation2dMatrix, (6, 16), data_wrong1)
    assert_raises(TypeError, FSRRotation2dMatrix, (6, 16), data_wrong2)
    assert_raises(TypeError, FSRRotation2dMatrix, (6, 16), data_wrong3)
    assert_raises(ValueError, FSRRotation2dMatrix, (6, 16), data, ncolmax=5)
    assert_raises(ValueError, FSRRotation2dMatrix, (6, 16))
    mat = FSRRotation2dMatrix((6, 16), data)
    assert_raises(ValueError, mat._matvec, np.ones(4))
    mat._matvec(np.ones((8, 2)))
    assert_raises(ValueError, mat._matvec, np.ones(17))
    assert_raises(ValueError, mat._matvec, np.ones((3, 2)))
    assert_raises(TypeError, mat._matvec, np.ones((8, 2)), out=1)
    assert_raises(ValueError, mat._matvec, np.ones((8, 2)), out=np.zeros(7))


def test_rot3d():
    input_fsc = np.arange(6*3.)
    input_fsr = np.arange(4*3.)
    index1 = [3, 2, 2, 1, 2, -1]
    index2 = [-1, 3, 1, 1, 0, -1]
    value1 = [1, 1, 0.5, 1, 2, 10]
    value2 = [1, 2, 0.5, 1, 1, 10]
    angle1 = [1, 10, 20, 30, 40, -10]
    angle2 = [0, 30, -20, 1, 1, 10]

    def fill(array, dense, index, value, angle, n):
        for i, (j, v, a) in enumerate(zip(index, value, angle)):
            array[i, n].index = j
            if j == -1:
                continue
            r = v * Rotation3dOperator('X', a, degrees=True).todense(shapein=3)
            array[i, n].r11 = r[0, 0]
            array[i, n].r22 = r[1, 1]
            array[i, n].r32 = r[2, 1]
            dense[3*i:3*i+3, 3*j:3*j+3] += r

    def func(itype, ftype):
        array = np.recarray((6, 2), dtype=[('index', itype), ('r11', ftype),
                                           ('r22', ftype), ('r32', ftype)])
        dense = np.zeros((6*3, 4*3), dtype=ftype)
        fill(array, dense, index1, value1, angle1, 0)
        fill(array, dense, index2, value2, angle2, 1)

        mat_fsc = FSCRotation3dMatrix(dense.shape[::-1], array)
        mat_fsr = FSRRotation3dMatrix(dense.shape, array)

        op = SparseOperator(mat_fsc)
        assert_equal(op.matrix.dtype, ftype)
        assert_equal(op.dtype, ftype)
        assert_same(op.todense(), dense.T)
        assert_same(op.T.todense(), dense)
        ref = mat_fsc._matvec(input_fsc.astype(np.float128))
        for ftype2 in ftypes[:-1]:
            out = np.zeros_like(ref, ftype2)
            mat_fsc._matvec(input_fsc.astype(ftype2), out)
            if np.__version__ >= '1.8':
                assert_same(out, ref.astype(min(ftype, ftype2)))
            else:
                assert_same(out, ref.astype(min(ftype, ftype2,
                                                key=lambda x: x().itemsize)))
        ref = (3 * (mat_fsc * input_fsc)).astype(ftype)
        assert_same((mat_fsc * 3) * input_fsc, ref)
        assert_same((3 * mat_fsc) * input_fsc, ref)
        assert_same(input_fsr * mat_fsc, mat_fsr * input_fsr)

        op = SparseOperator(mat_fsr)
        assert_equal(op.matrix.dtype, ftype)
        assert_equal(op.dtype, ftype)
        assert_same(op.todense(), dense)
        assert_same(op.T.todense(), dense.T)
        ref = mat_fsr._matvec(input_fsr.astype(np.float128))
        for ftype2 in ftypes[:-1]:
            out = np.zeros_like(ref, ftype2)
            mat_fsr._matvec(input_fsr.astype(ftype2), out)
            if np.__version__ >= '1.8':
                assert_same(out, ref.astype(min(ftype, ftype2)))
            else:
                assert_same(out, ref.astype(min(ftype, ftype2,
                                                key=lambda x: x().itemsize)))
        ref = (3 * (mat_fsr * input_fsr)).astype(ftype)
        assert_same((mat_fsr * 3) * input_fsr, ref)
        assert_same((3 * mat_fsr) * input_fsr, ref)
        assert_same(input_fsc * mat_fsr, mat_fsc * input_fsc)

    for itype in itypes:
        for ftype in ftypes:
            yield func, itype, ftype


def test_fsc_rot3d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float),
                                   ('r22', float), ('r32', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r22', float),
                                    ('r32', float), ('index', int)])
    data_wrong3 = np.empty((3, 4), dtype=[('index', int), ('r11', float),
                                          ('r22', np.float32), ('r32', float)])
    assert_raises(TypeError, FSCRotation3dMatrix, 3)
    assert_raises(ValueError, FSCRotation3dMatrix, (2, 3, 4))
    assert_raises(ValueError, FSCRotation3dMatrix, (9, 24), data)
    assert_raises(ValueError, FSCRotation3dMatrix, (10, 24), data)
    assert_raises(ValueError, FSCRotation3dMatrix, (25, 9), data)
    assert_raises(TypeError, FSCRotation3dMatrix, (24, 9), data_wrong1)
    assert_raises(TypeError, FSCRotation3dMatrix, (24, 9), data_wrong2)
    assert_raises(TypeError, FSCRotation3dMatrix, (24, 9), data_wrong3)
    assert_raises(ValueError, FSCRotation3dMatrix, (24, 9), data, nrowmax=5)
    assert_raises(ValueError, FSCRotation3dMatrix, (24, 9))
    mat = FSCRotation3dMatrix((24, 9), data)
    assert_raises(ValueError, mat._matvec, np.ones(4))
    mat._matvec(np.ones((3, 3)))
    assert_raises(ValueError, mat._matvec, np.ones(25))
    assert_raises(ValueError, mat._matvec, np.ones((8, 3)))
    assert_raises(TypeError, mat._matvec, np.ones((3, 3)), out=1)
    assert_raises(ValueError, mat._matvec, np.ones((3, 3)), out=np.zeros(7))


def test_fsr_rot3d_error():
    data = np.zeros((3, 4), dtype=[('index', int), ('r11', float),
                                   ('r22', float), ('r32', float)])
    data_wrong1 = np.empty((3, 4), [('index', int), ('value', float)])
    data_wrong2 = np.empty((3, 4), [('r11', float), ('r22', float),
                                    ('r32', float), ('index', int)])
    data_wrong3 = np.empty((3, 4), dtype=[('index', int), ('r11', float),
                                          ('r22', np.float32), ('r32', float)])
    assert_raises(TypeError, FSRRotation3dMatrix, 3)
    assert_raises(ValueError, FSRRotation3dMatrix, (2, 3, 4))
    assert_raises(ValueError, FSRRotation3dMatrix, (24, 9), data)
    assert_raises(ValueError, FSRRotation3dMatrix, (10, 24), data)
    assert_raises(ValueError, FSRRotation3dMatrix, (9, 25), data)
    assert_raises(TypeError, FSRRotation3dMatrix, (9, 24), data_wrong1)
    assert_raises(TypeError, FSRRotation3dMatrix, (9, 24), data_wrong2)
    assert_raises(TypeError, FSRRotation3dMatrix, (9, 24), data_wrong3)
    assert_raises(ValueError, FSRRotation3dMatrix, (9, 24), data, ncolmax=5)
    assert_raises(ValueError, FSRRotation3dMatrix, (9, 24))
    mat = FSRRotation3dMatrix((9, 24), data)
    assert_raises(ValueError, mat._matvec, np.ones(4))
    mat._matvec(np.ones((8, 3)))
    assert_raises(ValueError, mat._matvec, np.ones(25))
    assert_raises(ValueError, mat._matvec, np.ones((3, 3)))
    assert_raises(TypeError, mat._matvec, np.ones((8, 3)), out=1)
    assert_raises(ValueError, mat._matvec, np.ones((8, 3)), out=np.zeros(7))


def test_sparse_operator():
    assert_raises(TypeError, SparseOperator, np.zeros((10, 3)))
    sp = SparseOperator(scipy.sparse.csc_matrix((4, 6)))
    assert_is_type(sp, pyoperators.linear.SparseOperator)

    data = FSRMatrix((10, 3), ncolmax=1)
    assert_raises(ValueError, SparseOperator, data, shapein=4)
    assert_raises(ValueError, SparseOperator, data, shapeout=11)

    data = FSRRotation2dMatrix((12, 6), ncolmax=1)
    assert_raises(ValueError, SparseOperator, data, shapein=(5, 2))
    assert_raises(ValueError, SparseOperator, data, shapeout=(7,))

    data = FSRRotation3dMatrix((12, 6), ncolmax=1)
    assert_raises(ValueError, SparseOperator, data, shapein=(5, 3))
    assert_raises(ValueError, SparseOperator, data, shapeout=(7,))
