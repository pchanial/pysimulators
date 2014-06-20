#coding: utf-8
from __future__ import division

import numpy as np
import scipy
import scipy.constants
from astropy.coordinates.angles import Angle
from astropy.time import Time
from numpy.testing import assert_allclose, assert_equal, assert_raises
from pyoperators import (
    Operator, Cartesian2SphericalOperator, CompositionOperator,
    BlockDiagonalOperator, IdentityOperator, MultiplicationOperator,
    Spherical2CartesianOperator, flags)
from pyoperators.utils import all_eq, isscalarlike, product
from pyoperators.utils.testing import (
    assert_is_instance, assert_is_type, assert_same)
from pysimulators.operators import (
    BlackBodyOperator, ConvolutionTruncatedExponentialOperator,
    PowerLawOperator, RollOperator, block_diagonal,
    CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianGalactic2EquatorialOperator,
    CartesianHorizontal2EquatorialOperator,
    SphericalEquatorial2GalacticOperator,
    SphericalEquatorial2HorizontalOperator,
    SphericalGalactic2EquatorialOperator,
    SphericalHorizontal2EquatorialOperator)


def test_partitioning_chunk():

    @block_diagonal('value', 'mykey', axisin=0)
    @flags.square
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
    @flags.square
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
        n1 = 1 if isscalarlike(v) else len(v)
        n2 = 1 if isscalarlike(k) else len(k)
        nn = max(n1, n2) if n is None else 1 if isscalarlike(n) else len(n)
        if not isscalarlike(v) and not isscalarlike(k) and n1 != n2:
            # the partitioned arguments do not have the same length
            assert_raises(ValueError, lambda: cls(arg1, v, arg3, mykey=k,
                                                  partitionin=n))
            return
        if nn != max(n1, n2) and (not isscalarlike(v) or not isscalarlike(k)):
            # the partition is incompatible with the partitioned arguments
            return #XXX test assert_raises(ValueError)

        op = cls(arg1, v, arg3, mykey=k, partitionin=n)
        if nn == 1:
            v = v if isscalarlike(v) else v[0]
            k = k if isscalarlike(k) else k[0]
            func2(cls, op, v, k)
        else:
            assert op.__class__ is BlockDiagonalOperator
            assert len(op.operands) == nn
            if n is None:
                assert op.partitionin == nn * (None,)
                assert op.partitionout == nn * (None,)
                return
            v = nn * [v] if isscalarlike(v) else v
            k = nn * [k] if isscalarlike(k) else k
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
    @flags.square
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
    @flags.square
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
        n1 = 1 if isscalarlike(v) else len(v)
        n2 = 1 if isscalarlike(k) else len(k)
        nn = max(n1, n2)
        if not isscalarlike(v) and not isscalarlike(k) and n1 != n2:
            # the partitioned arguments do not have the same length
            assert_raises(ValueError, lambda: cls(arg1, v, arg3, mykey=k))
            return

        op = cls(arg1, v, arg3, mykey=k)
        if nn == 1:
            v = v if isscalarlike(v) else v[0]
            k = k if isscalarlike(k) else k[0]
            func2(cls, op, v, k)
        else:
            assert op.__class__ is BlockDiagonalOperator
            assert len(op.operands) == nn
            v = nn * [v] if isscalarlike(v) else v
            k = nn * [k] if isscalarlike(k) else k
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

    def bb(w, T):
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


def test_convolution_truncated_exponential():
    tau = (2, 3, 0)
    r = ConvolutionTruncatedExponentialOperator(tau, shapein=(3, 10))
    a = np.ones((3, 10))
    b = r(a)
    assert np.allclose(a, b)

    a[:, 1:] = 0
    b = r(a)
    assert_same(b[:2, :9], [np.exp(-np.arange(9) / _) for _ in tau[:2]])
    assert_same(b[2], a[2])
    assert_same(r.T.todense(), r.todense().T)
    assert_same(r.T.todense(inplace=True), r.todense(inplace=True).T)


def test_convolution_truncated_exponential_morphing():
    def func1(tau):
        assert_is_type(ConvolutionTruncatedExponentialOperator(tau),
                       IdentityOperator)
    for tau in 0, [0], (0,), np.array(0), np.array([0, 0]):
        yield func1, tau

    def func2(tau):
        assert_is_type(ConvolutionTruncatedExponentialOperator(tau),
                       ConvolutionTruncatedExponentialOperator)
    for tau in [0, 1], (0, 1), np.array([0, 1]):
        yield func2, tau


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


def test_equ2gal():
    equ2gal = CartesianEquatorial2GalacticOperator()
    gal2equ = CartesianGalactic2EquatorialOperator()

    assert equ2gal.I is equ2gal.T
    assert gal2equ.I is gal2equ.T
    assert_same(equ2gal.todense(shapein=3), equ2gal.data)
    assert_same(gal2equ.todense(shapein=3), equ2gal.data.T)

    shapes = (3, (2, 3), (4, 2, 3))

    def func(op, shape):
        vec = np.arange(product(shape)).reshape(shape)
        vec_ = vec.reshape((-1, 3))
        expected = np.empty(shape)
        expected_ = expected.reshape((-1, 3))
        for i in range(expected.size // 3):
            expected_[i] = op(vec_[i])
        actual = op(vec)
        assert_same(actual, expected)

    for op in (equ2gal, gal2equ):
        for shape in shapes:
            yield func, op, shape

    assert_is_instance(equ2gal(gal2equ), IdentityOperator)
    assert_is_instance(gal2equ(equ2gal), IdentityOperator)


def test_equ2hor():
    lat = 52
    lon = -64
    date = Time('1980-04-22 14:36:51.67', scale='ut1')
    E2h = CartesianEquatorial2HorizontalOperator
    gst = E2h._jd2gst(date.jd)
    lst = E2h._gst2lst(gst, lon)

    # Duffett-Smith §21
    assert_allclose(gst, 4.668119)
    assert_allclose(lst, 0.401453, rtol=1e-6)

    ra = Angle(lst - 5.862222, unit='hour').radian
    dec = Angle((23, 13, 10), unit='degree').radian
    s2c = Spherical2CartesianOperator('azimuth,elevation')
    op = CartesianEquatorial2HorizontalOperator('NE', date, lat, lon)
    incoords = s2c([ra, dec])
    outcoords = op(incoords)
    assert_same(op.I(outcoords), incoords)
    az, el = np.degrees(s2c.I(outcoords))

    # Duffett-Smith §25
    assert_allclose(az % 360, 283.271027)
    assert_allclose(el, 19.334345, rtol=1e-6)


def test_spherical():
    ops = ((SphericalEquatorial2GalacticOperator,
            CartesianEquatorial2GalacticOperator),
           (SphericalEquatorial2HorizontalOperator,
            CartesianEquatorial2HorizontalOperator),
           (SphericalGalactic2EquatorialOperator,
            CartesianGalactic2EquatorialOperator),
           (SphericalHorizontal2EquatorialOperator,
            CartesianHorizontal2EquatorialOperator))
    dirs_za = ((0, 0), (20, 0), (130, 0), (10, 20), (20, 190),
               ((0, 0), (20, 0), (130, 0), (10, 20), (20, 130)),
               (((0, 0), (20, 200), (130, 300)),))
    dirs_az = ((0, 0), (0, 20), (0, 130), (20, 10), (190, 20),
               ((0, 0), (0, 20), (0, 130), (20, 10), (130, 20)),
               (((0, 0), (200, 20), (300, 130)),))
    dirs_ea = ((90, 0), (70, 0), (-40, 0), (80, 20), (70, 190),
               ((90, 0), (70, 0), (-40, 0), (80, 20), (70, 130)),
               (((90, 0), (70, 200), (-40, 300)),))
    dirs_ae = ((0, 90), (0, 70), (0, -40), (20, 80), (190, 70),
               ((0, 90), (0, 70), (0, -40), (20, 80), (130, 70)),
               (((0, 90), (200, 70), (300, -40)),))
    shapes = ((), (), (), (), (), (5,), (1, 3))

    def func(cls_sph, cls_car, cin, cout, v, s, d):
        if 'Horizontal' in str(cls_sph):
            args = ('NE', Time('1980-04-22 14:36:51.67', scale='ut1'),
                    100.1, -80)
        else:
            args = ()
        op_sph = cls_sph(*args, conventionin=cin, conventionout=cout,
                         degrees=d)
        actual = op_sph(v)
        assert_equal(actual.shape, s + (2,))
        if d:
            v = np.radians(v)
        expected = Cartesian2SphericalOperator(cout)(
            cls_car(*args)(Spherical2CartesianOperator(cin)(v)))
        if d:
            np.degrees(expected, expected)
        assert_same(actual, expected)

    for cls_sph, cls_car in ops:
        for cin, vs in (('zenith,azimuth', dirs_za),
                        ('azimuth,zenith', dirs_az),
                        ('elevation,azimuth', dirs_ea),
                        ('azimuth,elevation', dirs_ae)):
            for cout in ('zenith,azimuth',
                         'azimuth,zenith',
                         'elevation,azimuth',
                         'azimuth,elevation'):
                for v, s in zip(vs, shapes):
                    for d in (False, True):
                        yield func, cls_sph, cls_car, cin, cout, v, s, d
