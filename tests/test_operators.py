import numpy as np
import pytest
import scipy
import scipy.constants
from astropy.coordinates.angles import Angle
from astropy.time import Time
from numpy.testing import assert_allclose, assert_equal

from pyoperators import (
    BlockDiagonalOperator,
    Cartesian2SphericalOperator,
    CompositionOperator,
    IdentityOperator,
    MultiplicationOperator,
    Operator,
    Spherical2CartesianOperator,
    flags,
)
from pyoperators.utils import isscalarlike, product
from pyoperators.utils.testing import assert_same
from pysimulators.operators import (
    BlackBodyOperator,
    CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianGalactic2EquatorialOperator,
    CartesianHorizontal2EquatorialOperator,
    ConvolutionTruncatedExponentialOperator,
    PowerLawOperator,
    RollOperator,
    SphericalEquatorial2GalacticOperator,
    SphericalEquatorial2HorizontalOperator,
    SphericalGalactic2EquatorialOperator,
    SphericalHorizontal2EquatorialOperator,
    block_diagonal,
)


@block_diagonal('value', 'key', axisin=0)
@flags.square
class PartitionChunkOp(Operator):
    def __init__(self, arg1, value, arg3, key=None, **keywords):
        Operator.__init__(self, **keywords)
        self.arg1 = arg1
        self.value = value
        self.arg3 = arg3
        self.key = key

    def direct(self, input, output):
        output[...] = self.value * input


class PartitionChunkSubOp(PartitionChunkOp):
    pass


@pytest.mark.parametrize('cls', [PartitionChunkOp, PartitionChunkSubOp])
@pytest.mark.parametrize('partitionin', [None, 2, (2,), (4, 2)])
@pytest.mark.parametrize('value', [2.0, (2.0,), (2.0, 3)])
@pytest.mark.parametrize('key', [0.0, (0.0,), (0.0, 1.0)])
def test_partitioning_chunk(cls, partitionin, value, key):
    def assert_partioning_chunk(cls, op, value, key):
        assert op.__class__ is cls
        assert op.arg1 is arg1
        assert op.value is value
        assert hasattr(op, 'arg3')
        assert op.arg3 is arg3
        assert op.key is key
        input = np.ones(1)
        output = op(input)
        assert_equal(output, value)

    arg1 = [1, 2, 3, 4, 5]
    arg3 = ['a', 'b', 'c', 'd']

    n1 = 1 if isscalarlike(value) else len(value)
    n2 = 1 if isscalarlike(key) else len(key)
    nn = (
        max(n1, n2)
        if partitionin is None
        else 1
        if isscalarlike(partitionin)
        else len(partitionin)
    )
    if n1 != n2 and not isscalarlike(value) and not isscalarlike(key):
        # the partitioned arguments do not have the same length
        with pytest.raises(
            ValueError,
            match='The partition variables do not have the same number of elements',
        ):
            cls(arg1, value, arg3, key=key, partitionin=partitionin)
        return
    if nn != max(n1, n2) and (not isscalarlike(value) or not isscalarlike(key)):
        # the partition is incompatible with the partitioned arguments
        with pytest.raises(
            ValueError,
            match='The specified partitioning is incompatible with the number of elements in the partition variables.',
        ):
            cls(arg1, value, arg3, key=key, partitionin=partitionin)
        return

    op = cls(arg1, value, arg3, key=key, partitionin=partitionin)
    if nn == 1:
        value = value if isscalarlike(value) else value[0]
        key = key if isscalarlike(key) else key[0]
        assert_partioning_chunk(cls, op, value, key)
        return

    assert op.__class__ is BlockDiagonalOperator
    assert len(op.operands) == nn
    if partitionin is None:
        assert op.partitionin == nn * (None,)
        assert op.partitionout == nn * (None,)
        return
    value = nn * [value] if isscalarlike(value) else value
    key = nn * [key] if isscalarlike(key) else key
    for op_, v_, k_ in zip(op.operands, value, key):
        assert_partioning_chunk(cls, op_, v_, k_)
    expected = np.hstack([n_ * [v_] for n_, v_ in zip(partitionin, value)])
    input = np.ones(np.sum(partitionin))
    output = op(input)
    assert_equal(output, expected)


@block_diagonal('value', 'key', new_axisin=0)
@flags.square
class PartitionStackOp(Operator):
    def __init__(self, arg1, value, arg3, key=None, **keywords):
        Operator.__init__(self, **keywords)
        self.arg1 = arg1
        self.value = value
        self.arg3 = arg3
        self.key = key

    def direct(self, input, output):
        output[...] = self.value * input


class PartitionStackSubOp(PartitionStackOp):
    pass


@pytest.mark.parametrize('cls', [PartitionStackOp, PartitionStackSubOp])
@pytest.mark.parametrize('value', [2.0, (2.0,), (2.0, 3)])
@pytest.mark.parametrize('key', [0.0, (0.0,), (0.0, 1.0)])
def test_partitioning_stack(cls, value, key):
    def assert_partition_stack(cls, op, v, k):
        assert op.__class__ is cls
        assert op.arg1 is arg1
        assert op.value is v
        assert hasattr(op, 'arg3')
        assert op.arg3 is arg3
        assert op.key is k
        input = np.ones(1)
        output = op(input)
        assert_equal(output, v)

    arg1 = [1, 2, 3, 4, 5]
    arg3 = ['a', 'b', 'c', 'd']

    n1 = 1 if isscalarlike(value) else len(value)
    n2 = 1 if isscalarlike(key) else len(key)
    nn = max(n1, n2)
    if not isscalarlike(value) and not isscalarlike(key) and n1 != n2:
        # the partitioned arguments do not have the same length
        with pytest.raises(ValueError):
            cls(arg1, value, arg3, key=key)
        return

    op = cls(arg1, value, arg3, key=key)
    if nn == 1:
        value = value if isscalarlike(value) else value[0]
        key = key if isscalarlike(key) else key[0]
        assert_partition_stack(cls, op, value, key)
        return

    assert op.__class__ is BlockDiagonalOperator
    assert len(op.operands) == nn
    value = nn * [value] if isscalarlike(value) else value
    key = nn * [key] if isscalarlike(key) else key
    for op_, v_, k_ in zip(op.operands, value, key):
        assert_partition_stack(cls, op_, v_, k_)
    input = np.ones(nn)
    output = op(input)
    assert_equal(output, value)


def ref_blackbody(w, T):
    c = scipy.constants.c
    h = scipy.constants.h
    k = scipy.constants.k
    nu = c / w
    return 2 * h * nu**3 / c**2 / (np.exp(h * nu / (k * T)) - 1)


def test_blackbody_scalar():

    w = np.arange(90.0, 111) * 1e-6
    T = 15.0
    flux = ref_blackbody(w, T) / ref_blackbody(100e-6, T)
    ops = [BlackBodyOperator(T, wavelength=wave, wavelength0=100e-6) for wave in w]
    flux2 = [op(1.0) for op in ops]
    assert_same(flux, flux2)


def test_blackbody_array():
    w, T = np.ogrid[90:111, 15:20]
    w = w * 1.0e-6
    flux = ref_blackbody(w, T) / ref_blackbody(100e-6, T)
    ops = [
        BlackBodyOperator(T.squeeze(), wavelength=wave[0], wavelength0=100e-6)
        for wave in w
    ]
    flux2 = np.array([op(np.ones(T.size)) for op in ops])
    assert_same(flux, flux2)


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


@pytest.mark.parametrize('tau', [0, [0], (0,), np.array(0), np.array([0, 0])])
def test_convolution_truncated_exponential_morphing(tau):
    op = ConvolutionTruncatedExponentialOperator(tau)
    assert type(op) is IdentityOperator


@pytest.mark.parametrize('tau', [[0, 1], (0, 1), np.array([0, 1])])
def test_convolution_truncated_exponential_not_morphing(tau):
    op = ConvolutionTruncatedExponentialOperator(tau)
    assert type(op) is ConvolutionTruncatedExponentialOperator


@pytest.mark.parametrize('alpha', [-1, np.arange(11) - 5])
def test_power_law(alpha):
    c = scipy.constants.c
    nu = c / 120e-6
    nu0 = c / 100e-6
    values = np.arange(11)

    expected = (nu / nu0) ** alpha * values
    op = PowerLawOperator(alpha, nu, nu0)
    assert_same(op(values), expected)


@pytest.mark.parametrize('cls', [CompositionOperator, MultiplicationOperator])
def test_power_law_absorb_scalar(cls):
    op = cls([2, PowerLawOperator(1, 2.0, 3.0)])
    assert isinstance(op, PowerLawOperator)


@pytest.mark.parametrize('n', range(4))
@pytest.mark.parametrize(
    'axis',
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (1, 2, 3),
        (0, 1, 2, 3),
    ],
)
def test_roll(n, axis):
    shape = np.arange(2, 6)
    v = np.arange(2 * 3 * 4 * 5).reshape(shape)
    expected = v.copy()
    for a in axis:
        expected = np.roll(expected, n, a)
    result = RollOperator(axis=axis, n=n)(v)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    'op',
    [CartesianEquatorial2GalacticOperator(), CartesianGalactic2EquatorialOperator()],
)
@pytest.mark.parametrize('shape', [3, (2, 3), (4, 2, 3)])
def test_equ2gal(op, shape):
    assert op.I is op.T
    assert_same(op.todense(shapein=3), op.data)

    vec = np.arange(product(shape)).reshape(shape)
    vec_ = vec.reshape((-1, 3))
    expected = np.empty(shape)
    expected_ = expected.reshape((-1, 3))
    for i in range(expected.size // 3):
        expected_[i] = op(vec_[i])
    actual = op(vec)
    assert_same(actual, expected)


def test_equ2gal_rules():
    equ2gal = CartesianEquatorial2GalacticOperator()
    gal2equ = CartesianGalactic2EquatorialOperator()

    assert_same(gal2equ.todense(shapein=3), equ2gal.data.T)
    assert_same(equ2gal.todense(shapein=3), gal2equ.data.T)

    assert isinstance(equ2gal(gal2equ), IdentityOperator)
    assert isinstance(gal2equ(equ2gal), IdentityOperator)


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
    dec = Angle('23d13m10s').radian
    s2c = Spherical2CartesianOperator('azimuth,elevation')
    op = CartesianEquatorial2HorizontalOperator('NE', date, lat, lon)
    incoords = s2c([ra, dec])
    outcoords = op(incoords)
    assert_same(op.I(outcoords), incoords)
    az, el = np.degrees(s2c.I(outcoords))

    # Duffett-Smith §25
    assert_allclose(az % 360, 283.271027)
    assert_allclose(el, 19.334345, rtol=1e-6)


@pytest.mark.parametrize(
    'cls_sph, cls_car',
    [
        (SphericalEquatorial2GalacticOperator, CartesianEquatorial2GalacticOperator),
        (
            SphericalEquatorial2HorizontalOperator,
            CartesianEquatorial2HorizontalOperator,
        ),
        (SphericalGalactic2EquatorialOperator, CartesianGalactic2EquatorialOperator),
        (
            SphericalHorizontal2EquatorialOperator,
            CartesianHorizontal2EquatorialOperator,
        ),
    ],
)
@pytest.mark.parametrize(
    'conventionin, dirs, shape',
    [
        ('zenith,azimuth', (0, 0), ()),
        ('zenith,azimuth', (20, 0), ()),
        ('zenith,azimuth', (130, 0), ()),
        ('zenith,azimuth', (10, 20), ()),
        ('zenith,azimuth', (20, 190), ()),
        ('zenith,azimuth', [(0, 0), (20, 0), (130, 0), (10, 20), (20, 130)], (5,)),
        ('zenith,azimuth', [[(0, 0), (20, 200), (130, 300)]], (1, 3)),
        ('azimuth,zenith', (0, 0), ()),
        ('azimuth,zenith', (0, 20), ()),
        ('azimuth,zenith', (0, 130), ()),
        ('azimuth,zenith', (20, 10), ()),
        ('azimuth,zenith', (190, 20), ()),
        ('azimuth,zenith', [(0, 0), (0, 20), (0, 130), (20, 10), (130, 20)], (5,)),
        ('azimuth,zenith', [[(0, 0), (200, 20), (300, 130)]], (1, 3)),
        ('elevation,azimuth', (90, 0), ()),
        ('elevation,azimuth', (70, 0), ()),
        ('elevation,azimuth', (-40, 0), ()),
        ('elevation,azimuth', (80, 20), ()),
        ('elevation,azimuth', (70, 190), ()),
        ('elevation,azimuth', [(90, 0), (70, 0), (-40, 0), (80, 20), (70, 130)], (5,)),
        ('elevation,azimuth', [[(90, 0), (70, 200), (-40, 300)]], (1, 3)),
        ('azimuth,elevation', (0, 90), ()),
        ('azimuth,elevation', (0, 70), ()),
        ('azimuth,elevation', (0, -40), ()),
        ('azimuth,elevation', (20, 80), ()),
        ('azimuth,elevation', (190, 70), ()),
        ('azimuth,elevation', [(0, 90), (0, 70), (0, -40), (20, 80), (130, 70)], (5,)),
        ('azimuth,elevation', [[(0, 90), (200, 70), (300, -40)]], (1, 3)),
    ],
)
@pytest.mark.parametrize(
    'conventionout',
    [
        'zenith,azimuth',
        'azimuth,zenith',
        'elevation,azimuth',
        'azimuth,elevation',
    ],
)
@pytest.mark.parametrize('degrees', [False, True])
def test_spherical(cls_sph, cls_car, conventionin, dirs, shape, conventionout, degrees):
    if 'Horizontal' in str(cls_sph):
        args = ('NE', Time('1980-04-22 14:36:51.67', scale='ut1'), 100.1, -80)
    else:
        args = ()
    op_sph = cls_sph(
        *args, conventionin=conventionin, conventionout=conventionout, degrees=degrees
    )
    actual = op_sph(dirs)
    assert_equal(actual.shape, shape + (2,))
    if degrees:
        dirs = np.radians(dirs)
    expected = Cartesian2SphericalOperator(conventionout)(
        cls_car(*args)(Spherical2CartesianOperator(conventionin)(dirs))
    )
    if degrees:
        np.degrees(expected, expected)
    assert_same(actual, expected)
