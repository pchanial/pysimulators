import numpy as np
import scipy
import scipy.constants

from numpy.testing import assert_equal, assert_raises
from pyoperators import (
    Operator, CompositionOperator, BlockDiagonalOperator,
    MultiplicationOperator, decorators)
from pyoperators.utils import all_eq, isscalar
from pyoperators.utils.testing import assert_is_instance
from pysimulators.operators import (
    BlackBodyOperator, PowerLawOperator, RollOperator, block_diagonal)


def test_partitioning_chunk():

    @block_diagonal('value', 'mykey', axisin=0)
    @decorators.square
    class MyOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey

        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    @block_diagonal('value', 'mykey', axisin=0)
    @decorators.square
    class MySupOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey

    class MySubOp(MySupOp):
        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    arg1 = [1, 2, 3, 4, 5]
    arg3 = ['a', 'b', 'c', 'd']

    def func(cls, n, v, k):
        n1 = 1 if isscalar(v) else len(v)
        n2 = 1 if isscalar(k) else len(k)
        nn = max(n1, n2) if n is None else 1 if isscalar(n) else len(n)
        if not isscalar(v) and not isscalar(k) and n1 != n2:
            # the partitioned arguments do not have the same length
            assert_raises(ValueError, lambda: cls(arg1, v, arg3, mykey=k,
                                                  partitionin=n))
            return
        if nn != max(n1, n2) and (not isscalar(v) or not isscalar(k)):
            # the partition is incompatible with the partitioned arguments
            return #XXX test assert_raises(ValueError)

        op = cls(arg1, v, arg3, mykey=k, partitionin=n)
        if nn == 1:
            v = v if isscalar(v) else v[0]
            k = k if isscalar(k) else k[0]
            func2(cls, op, v, k)
        else:
            assert op.__class__ is BlockDiagonalOperator
            assert len(op.operands) == nn
            if n is None:
                assert op.partitionin == nn * (None,)
                assert op.partitionout == nn * (None,)
                return
            v = nn * [v] if isscalar(v) else v
            k = nn * [k] if isscalar(k) else k
            for op_, v_, k_ in zip(op.operands, v, k):
                func2(cls, op_, v_, k_)
            expected = np.hstack(n_*[v_] for n_, v_ in zip(n, v))
            input = np.ones(np.sum(n))
            output = op(input)
            assert_equal(output, expected)

    def func2(cls, op, v, k):
        assert op.__class__ is cls
        assert op.arg1 is arg1
        assert op.value is v
        assert hasattr(op, 'arg3')
        assert op.arg3 is arg3
        assert op.mykey is k
        input = np.ones(1)
        output = op(input)
        assert_equal(output, v)

    for c in (MyOp, MySubOp):
        for n in (None, 2, (2,), (4, 2)):
            for v in (2., (2.,), (2., 3)):
                for k in (0., (0.,), (0., 1.)):
                    yield func, c, n, v, k


def test_partitioning_stack():

    @block_diagonal('value', 'mykey', new_axisin=0)
    @decorators.square
    class MyOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey

        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    @block_diagonal('value', 'mykey', new_axisin=0)
    @decorators.square
    class MySupOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey

    class MySubOp(MySupOp):
        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    arg1 = [1, 2, 3, 4, 5]
    arg3 = ['a', 'b', 'c', 'd']

    def func(cls, v, k):
        n1 = 1 if isscalar(v) else len(v)
        n2 = 1 if isscalar(k) else len(k)
        nn = max(n1, n2)
        if not isscalar(v) and not isscalar(k) and n1 != n2:
            # the partitioned arguments do not have the same length
            assert_raises(ValueError, lambda: cls(arg1, v, arg3, mykey=k))
            return

        op = cls(arg1, v, arg3, mykey=k)
        if nn == 1:
            v = v if isscalar(v) else v[0]
            k = k if isscalar(k) else k[0]
            func2(cls, op, v, k)
        else:
            assert op.__class__ is BlockDiagonalOperator
            assert len(op.operands) == nn
            v = nn * [v] if isscalar(v) else v
            k = nn * [k] if isscalar(k) else k
            for op_, v_, k_ in zip(op.operands, v, k):
                func2(cls, op_, v_, k_)
            input = np.ones(nn)
            output = op(input)
            assert_equal(output, v)

    def func2(cls, op, v, k):
        assert op.__class__ is cls
        assert op.arg1 is arg1
        assert op.value is v
        assert hasattr(op, 'arg3')
        assert op.arg3 is arg3
        assert op.mykey is k
        input = np.ones(1)
        output = op(input)
        assert_equal(output, v)

    for c in (MyOp, MySubOp):
        for v in (2., (2.,), (2., 3)):
            for k in (0., (0.,), (0., 1.)):
                yield func, c, v, k

def test_blackbody():

    def bb(w,T):
        c = 2.99792458e8
        h = 6.626068e-34
        k = 1.380658e-23
        nu = c/w
        return 2*h*nu**3/c**2 / (np.exp(h*nu/(k*T))-1)

    w = np.arange(90., 111) * 1e-6
    T = 15.
    flux = bb(w, T) / bb(100e-6, T)
    ops = [BlackBodyOperator(T, wavelength=wave, wavelength0=100e-6)
           for wave in w]
    flux2 = [op(1.) for op in ops]
    assert all_eq(flux, flux2)

    w, T = np.ogrid[90:111, 15:20]
    w = w * 1.e-6
    flux = bb(w, T) / bb(100e-6, T)
    ops = [BlackBodyOperator(T.squeeze(), wavelength=wave[0],
                             wavelength0=100e-6)
           for wave in w]
    flux2 = np.array([op(np.ones(T.size)) for op in ops])
    assert all_eq(flux, flux2)


def test_power_law():
    c = scipy.constants.c
    nu = c / 120e-6
    nu0 = c / 100e-6
    values = np.arange(11)

    def func(alpha):
        expected = (nu/nu0)**alpha * values
        op = PowerLawOperator(alpha, nu, nu0)
        assert all_eq(op(values), expected)
    for alpha in (-1, np.arange(11)-5):
        yield func, alpha

    def func2(cls):
        op = cls([2, PowerLawOperator(1, nu, nu0)])
        assert_is_instance(op, PowerLawOperator)
    for cls in CompositionOperator, MultiplicationOperator:
        yield func2, cls


def test_roll():
    shape = np.arange(2, 6)
    v = np.arange(2*3*4*5).reshape(shape)
    for n in range(4):
        for axis in ((0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2),
                     (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3),
                     (1, 2, 3), (1, 2, 3), (0, 1, 2, 3)):
            expected = v.copy()
            for a in axis:
                expected = np.roll(expected, n, a)
            result = RollOperator(axis=axis, n=n)(v)
            yield assert_equal, result, expected
