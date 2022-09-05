import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from pyoperators import CompositionOperator, IdentityOperator
from pyoperators.utils.testing import assert_same
from pysimulators.interfaces.healpy import (
    Cartesian2HealpixOperator,
    Healpix2CartesianOperator,
    Healpix2SphericalOperator,
    HealpixConvolutionGaussianOperator,
    HealpixLaplacianOperator,
    Spherical2HealpixOperator,
)

hp = pytest.importorskip('healpy')

NSIDE = 512


@pytest.mark.parametrize(
    'vec, shape',
    [
        ((1, 0, 0), ()),
        ((0, 1, 0), ()),
        ((0, 0, 1), ()),
        (((1, 0, 0), (0, 1, 0), (0, 0, 1)), (3,)),
        ([(1, 0, 0), (0, 1, 0), (0, 0, 1)], (3,)),
        ((((1, 0, 0), (0, 1, 0)),), (1, 2)),
        ([[(1, 0, 0), (0, 1, 0)]], (1, 2)),
    ],
)
@pytest.mark.parametrize('nest', [False, True])
def test_cartesian_healpix(vec, shape, nest):

    c2h = Cartesian2HealpixOperator(NSIDE, nest=nest)
    h2c = Healpix2CartesianOperator(NSIDE, nest=nest)
    a = h2c(c2h(vec))
    assert_equal(a.shape, shape + (3,))
    assert_allclose(a, vec, atol=1e-1)


def test_cartesian_healpix_rules():
    op = Cartesian2HealpixOperator(NSIDE)
    assert type(op.I) is Healpix2CartesianOperator
    assert op.I.nside == op.nside
    assert (
        type(Cartesian2HealpixOperator(NSIDE)(Healpix2CartesianOperator(2 * NSIDE)))
        is CompositionOperator
    )

    assert (
        type(Cartesian2HealpixOperator(NSIDE)(Healpix2CartesianOperator(NSIDE)))
        is IdentityOperator
    )


@pytest.mark.parametrize('shapein', [(), (1,), (2,), (3,)])
@pytest.mark.parametrize('shapeout', [(), (1,), (2,), (3,)])
def test_cartesian_healpix_invalid_shape(shapein, shapeout):
    op = Cartesian2HealpixOperator(NSIDE)
    input = np.zeros(shapein)
    output = np.zeros(shapeout)

    if shapein == (3,) and shapeout == ():
        op(input, output)
        return

    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('nest', [False, True])
@pytest.mark.parametrize(
    'vec, shape',
    [
        (0, ()),
        (100, ()),
        (12 * NSIDE**2 - 1, ()),
        ([0, 100, 12 * NSIDE**2 - 1], (3,)),
        ([[0, 100]], (1, 2)),
    ],
)
def test_healpix_cartesian(nest, vec, shape):
    h2c = Healpix2CartesianOperator(NSIDE, nest=nest)
    c2h = Cartesian2HealpixOperator(NSIDE, nest=nest)
    a = c2h(h2c(vec))
    assert a.shape == shape
    assert_allclose(a, vec)


def test_healpix_cartesian_rules():
    op = Healpix2CartesianOperator(NSIDE)
    assert type(op.I) is Cartesian2HealpixOperator
    assert op.I.nside == op.nside
    assert (
        type(Healpix2CartesianOperator(NSIDE)(Cartesian2HealpixOperator(2 * NSIDE)))
        is CompositionOperator
    )

    assert (
        type(Healpix2CartesianOperator(NSIDE)(Cartesian2HealpixOperator(NSIDE)))
        is IdentityOperator
    )


@pytest.mark.parametrize('shapein', [(), (1,), (2,), (3,)])
@pytest.mark.parametrize('shapeout', [(), (1,), (2,), (3,)])
def test_healpix_cartesian_invalid_shape(shapein, shapeout):
    op = Healpix2CartesianOperator(NSIDE)
    input = np.zeros(shapein)
    output = np.zeros(shapeout)

    if shapein == () and shapeout == (3,):
        op(input, output)
        return

    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('pixel', [-1, 12 * NSIDE**2])
def test_healpix_cartesian_invalid_pixel(pixel):
    op = Healpix2CartesianOperator(NSIDE)
    with pytest.warns(RuntimeWarning):
        op(pixel)


@pytest.mark.parametrize(
    'convention, dirs, dirs_za, shape',
    [
        ('azimuth,zenith', (0, 10), (10, 0), ()),
        ('azimuth,zenith', (0, 20), (20, 0), ()),
        ('azimuth,zenith', (0, 130), (130, 0), ()),
        ('azimuth,zenith', (20, 10), (10, 20), ()),
        ('azimuth,zenith', (130, 20), (20, 130), ()),
        (
            'azimuth,zenith',
            [(0, 10), (0, 20), (0, 130), (20, 10), (130, 20)],
            [(10, 0), (20, 0), (130, 0), (10, 20), (20, 130)],
            (5,),
        ),
        (
            'azimuth,zenith',
            [[(0, 10), (200, 20), (300, 130)]],
            [[(10, 0), (20, 200), (130, 300)]],
            (1, 3),
        ),
        ('elevation,azimuth', (80, 0), (10, 0), ()),
        ('elevation,azimuth', (70, 0), (20, 0), ()),
        ('elevation,azimuth', (-40, 0), (130, 0), ()),
        ('elevation,azimuth', (80, 20), (10, 20), ()),
        ('elevation,azimuth', (70, 130), (20, 130), ()),
        (
            'elevation,azimuth',
            [(80, 0), (70, 0), (-40, 0), (80, 20), (70, 130)],
            [(10, 0), (20, 0), (130, 0), (10, 20), (20, 130)],
            (5,),
        ),
        (
            'elevation,azimuth',
            [[(80, 0), (70, 200), (-40, 300)]],
            [[(10, 0), (20, 200), (130, 300)]],
            (1, 3),
        ),
        ('azimuth,elevation', (0, 80), (10, 0), ()),
        ('azimuth,elevation', (0, 70), (20, 0), ()),
        ('azimuth,elevation', (0, -40), (130, 0), ()),
        ('azimuth,elevation', (20, 80), (10, 20), ()),
        ('azimuth,elevation', (130, 70), (20, 130), ()),
        (
            'azimuth,elevation',
            [(0, 80), (0, 70), (0, -40), (20, 80), (130, 70)],
            [(10, 0), (20, 0), (130, 0), (10, 20), (20, 130)],
            (5,),
        ),
        (
            'azimuth,elevation',
            [[(0, 80), (200, 70), (300, -40)]],
            [[(10, 0), (20, 200), (130, 300)]],
            (1, 3),
        ),
    ],
)
def test_spherical_healpix(convention, dirs, dirs_za, shape):
    op_ref = Spherical2HealpixOperator(NSIDE, 'zenith,azimuth')
    refs = op_ref(np.radians(dirs_za))

    s2h = Spherical2HealpixOperator(NSIDE, convention)
    h2s = Healpix2SphericalOperator(NSIDE, convention)
    assert_allclose(s2h(np.radians(dirs)), refs)
    a = np.degrees(h2s(s2h(np.radians(dirs))))
    assert_equal(a.shape, shape + (2,))
    assert_allclose(a, dirs, rtol=1e-2, atol=0.5)


@pytest.mark.parametrize('convention1', Spherical2HealpixOperator.CONVENTIONS)
@pytest.mark.parametrize('convention2', Spherical2HealpixOperator.CONVENTIONS)
@pytest.mark.parametrize('nside', [NSIDE, 2 * NSIDE])
@pytest.mark.parametrize('nest', [False, True])
def test_spherical_healpix_rules(convention1, convention2, nside, nest):
    op = Spherical2HealpixOperator(NSIDE, convention1, nest=True)
    assert type(op.I) is Healpix2SphericalOperator
    assert op.I.convention == op.convention

    op2 = Healpix2SphericalOperator(nside, convention2, nest=nest)
    if convention2 == op.convention and nside == op.nside and nest == op.nest:
        assert type(op(op2)) is IdentityOperator
    else:
        assert type(op(op2)) is CompositionOperator


def test_spherical_healpix_invalid_convention():
    with pytest.raises(ValueError):
        Spherical2HealpixOperator(NSIDE, 'bla')


@pytest.mark.parametrize('shapein', [(), (1,), (2,), (3,)])
@pytest.mark.parametrize('shapeout', [(), (1,), (2,), (3,)])
def test_spherical_healpix_invalid_shape(shapein, shapeout):
    op = Spherical2HealpixOperator(NSIDE, 'zenith,azimuth')
    input = np.zeros(shapein)
    output = np.zeros(shapeout)

    if shapein == (2,) and shapeout == ():
        op(input, output)
        return

    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('convention', Healpix2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize('nest', [False, True])
@pytest.mark.parametrize(
    'vec, shape',
    [
        (0, ()),
        (100, ()),
        (12 * NSIDE**2 - 1, ()),
        ([0, 100, 12 * NSIDE**2 - 1], (3,)),
        ([[0, 100]], (1, 2)),
    ],
)
def test_healpix_spherical(convention, nest, vec, shape):
    h2s = Healpix2SphericalOperator(NSIDE, convention, nest=nest)
    s2h = Spherical2HealpixOperator(NSIDE, convention, nest=nest)
    a = s2h(h2s(vec))
    assert a.shape == shape
    assert_allclose(a, vec)


@pytest.mark.parametrize('convention1', Healpix2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize('convention2', Healpix2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize('nside', [NSIDE, 2 * NSIDE])
@pytest.mark.parametrize('nest', [False, True])
def test_healpix_spherical_rules(convention1, convention2, nside, nest):
    op = Healpix2SphericalOperator(NSIDE, convention1, nest=True)
    assert type(op.I) is Spherical2HealpixOperator
    assert op.I.convention == op.convention

    op2 = Spherical2HealpixOperator(nside, convention2, nest=nest)
    if convention2 == op.convention and nside == op.nside and nest == op.nest:
        assert type(op(op2)) is IdentityOperator
    else:
        assert type(op(op2)) is CompositionOperator


def test_healpix_spherical_invalid_convention():
    with pytest.raises(ValueError):
        Healpix2SphericalOperator(NSIDE, 'bla')


@pytest.mark.parametrize('convention', Healpix2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize('shapein', [(), (1,), (2,), (3,)])
@pytest.mark.parametrize('shapeout', [(), (1,), (2,), (3,)])
def test_healpix_spherical_invalid_shapes(convention, shapein, shapeout):
    op = Healpix2SphericalOperator(NSIDE, 'zenith,azimuth')
    input = np.zeros(shapein)
    output = np.zeros(shapeout)

    if shapein == () and shapeout == (2,):
        op(input, output)
        return

    with pytest.raises(ValueError):
        op(input, output)


@pytest.mark.parametrize('convention', Healpix2SphericalOperator.CONVENTIONS)
@pytest.mark.parametrize('pixel', [-1, 12 * NSIDE**2])
def test_healpix_spherical_invalid_pixel(convention, pixel):
    op = Healpix2SphericalOperator(NSIDE, convention)
    with pytest.warns(RuntimeWarning):
        op(pixel)


def test_healpix_convolution():
    nside = 16
    keywords = {'fwhm': np.radians(30), 'iter': 2, 'lmax': 8, 'use_weights': False}

    input = np.arange(12 * nside**2)
    op = HealpixConvolutionGaussianOperator(**keywords)

    for i in input, np.repeat(input[:, None], 3, 1):
        expected = np.transpose(hp.smoothing(i.T, **keywords))
        assert_same(op(i), expected)

    if hp.__version__ <= '1.8.6':  # healpy #259
        return

    op = HealpixConvolutionGaussianOperator(pol=False, **keywords)
    input_ = np.arange(12 * nside**2)
    input = np.array([input_, input_, input_]).T
    expected_ = hp.smoothing(input_, **keywords)
    expected = np.array([expected_, expected_, expected_]).T
    assert_same(op(input), expected)


def test_healpix_convolution_morph():
    op = HealpixConvolutionGaussianOperator(fwhm=0)
    assert type(op) is IdentityOperator
    op = HealpixConvolutionGaussianOperator(sigma=0)
    assert type(op) is IdentityOperator


@pytest.mark.parametrize(
    'keywords', [{'iter': 2}, {'fwhm': -1}, {'sigma': -1}, {'fwhm': 1, 'sigma': 1}]
)
def test_healpix_convolution_errors(keywords):
    with pytest.raises(ValueError):
        HealpixConvolutionGaussianOperator(**keywords)


def test_healpix_laplacian():
    nside = 1
    ipix = 4
    npix = 12
    map = np.zeros(npix)
    map[ipix] = 1
    L = HealpixLaplacianOperator(nside)
    assert L.flags.square
    assert L.flags.symmetric
    h2 = np.array(4 * np.pi / npix)
    expected = [1, 0, 0, 1, -20, 4, 0, 4, 1, 0, 0, 1] / (6 * h2)
    assert_same(L(map), expected)
    assert_same(
        L(np.repeat(map, 5).reshape(12, 5)), expected[:, None], broadcasting=True
    )
