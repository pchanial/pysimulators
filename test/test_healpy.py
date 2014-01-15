from __future__ import division

import healpy as hp
import itertools
import numpy as np
from nose import SkipTest
from numpy.testing import assert_allclose, assert_equal
from pyoperators import IdentityOperator, CompositionOperator
from pyoperators.utils.testing import (
    assert_is_type, assert_raises, assert_same)
from pysimulators.interfaces.healpy import (
    Cartesian2HealpixOperator, Healpix2CartesianOperator,
    Spherical2HealpixOperator, Healpix2SphericalOperator,
    HealpixConvolutionGaussianOperator)

NSIDE = 512


def test_cartesian_healpix():
    vecs = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (((1, 0, 0), (0, 1, 0)),))
    shapes = ((), (), (), (3,), (1, 2))

    def func(n, v, s):
        c2h = Cartesian2HealpixOperator(NSIDE, nest=n)
        h2c = Healpix2CartesianOperator(NSIDE, nest=n)
        a = h2c(c2h(v))
        assert_equal(a.shape, s + (3,))
        assert_allclose(a, v, atol=1e-1)
    for n in False, True:
        for v, s in zip(vecs, shapes):
            yield func, n, v, s


def test_cartesian_healpix_rules():
    op = Cartesian2HealpixOperator(NSIDE)
    assert_is_type(op.I, Healpix2CartesianOperator)
    assert_equal(op.I.nside, op.nside)
    assert_is_type(Cartesian2HealpixOperator(NSIDE) *
                   Healpix2CartesianOperator(2 * NSIDE), CompositionOperator)
    assert_is_type(Cartesian2HealpixOperator(NSIDE) *
                   Healpix2CartesianOperator(NSIDE), IdentityOperator)


def test_cartesian_healpix_error():
    op = Cartesian2HealpixOperator(NSIDE)

    def func(i, o):
        if i.shape == (3,) and o.shape == ():
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_healpix_cartesian():
    vecs = (0, 100, 12*NSIDE**2-1,
            (0, 100, 12*NSIDE**2-1),
            ((0, 100),))
    shapes = ((), (), (), (3,), (1, 2))

    def func(n, v, s):
        h2c = Healpix2CartesianOperator(NSIDE, nest=n)
        c2h = Cartesian2HealpixOperator(NSIDE, nest=n)
        a = c2h(h2c(v))
        assert_equal(a.shape, s)
        assert_equal(a, v)
    for n in False, True:
        for v, s in zip(vecs, shapes):
            yield func, n, v, s


def test_healpix_cartesian_rules():
    op = Healpix2CartesianOperator(NSIDE)
    assert_is_type(op.I, Cartesian2HealpixOperator)
    assert_equal(op.I.nside, op.nside)
    assert_is_type(Healpix2CartesianOperator(NSIDE) *
                   Cartesian2HealpixOperator(2 * NSIDE), CompositionOperator)
    assert_is_type(Healpix2CartesianOperator(NSIDE) *
                   Cartesian2HealpixOperator(NSIDE), IdentityOperator)


def test_healpix_cartesian_error():
    op = Healpix2CartesianOperator(NSIDE)

    def func(i, o):
        if i.shape == () and o.shape == (3,):
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_spherical_healpix():
    dirs_za = ((10, 0), (20, 0), (130, 0), (10, 20), (20, 130),
               ((10, 0), (20, 0), (130, 0), (10, 20), (20, 130)),
               (((10, 0), (20, 200), (130, 300)),))
    dirs_az = ((0, 10), (0, 20), (0, 130), (20, 10), (130, 20),
               ((0, 10), (0, 20), (0, 130), (20, 10), (130, 20)),
               (((0, 10), (200, 20), (300, 130)),))
    dirs_ea = ((80, 0), (70, 0), (-40, 0), (80, 20), (70, 130),
               ((80, 0), (70, 0), (-40, 0), (80, 20), (70, 130)),
               (((80, 0), (70, 200), (-40, 300)),))
    dirs_ae = ((0, 80), (0, 70), (0, -40), (20, 80), (130, 70),
               ((0, 80), (0, 70), (0, -40), (20, 80), (130, 70)),
               (((0, 80), (200, 70), (300, -40)),))
    shapes = ((), (), (), (), (), (5,), (1, 3))

    op_ref = Spherical2HealpixOperator(NSIDE, 'zenith,azimuth')
    refs = [op_ref(np.radians(v)) for v in dirs_za]

    def func(c, v, s, r):
        s2h = Spherical2HealpixOperator(NSIDE, c)
        h2s = Healpix2SphericalOperator(NSIDE, c)
        assert_allclose(s2h(np.radians(v)), r)
        a = np.degrees(h2s(s2h(np.radians(v))))
        assert_equal(a.shape, s + (2,))
        assert_allclose(a, v, rtol=1e-2, atol=0.5)
    for c, d in (('zenith,azimuth', dirs_za),
                 ('azimuth,zenith', dirs_az),
                 ('elevation,azimuth', dirs_ea),
                 ('azimuth,elevation', dirs_ae)):
        for v, s, r in zip(d, shapes, refs):
            yield func, c, v, s, r


def test_spherical_healpix_rules():
    def func(op, nside, c, nest):
        op2 = Healpix2SphericalOperator(nside, c, nest=nest)
        if c == op.convention and nside == op.nside and nest == op.nest:
            assert_is_type(op * op2, IdentityOperator)
        else:
            assert_is_type(op * op2, CompositionOperator)
    for c in Spherical2HealpixOperator.CONVENTIONS:
        op = Spherical2HealpixOperator(NSIDE, c, nest=True)
        assert_is_type(op.I, Healpix2SphericalOperator)
        assert_equal(op.I.convention, op.convention)
        for nside in NSIDE, 2*NSIDE:
            for c2 in Spherical2HealpixOperator.CONVENTIONS:
                for nest in False, True:
                    yield func, op, nside, c2, nest


def test_spherical_healpix_error():
    assert_raises(ValueError, Spherical2HealpixOperator, NSIDE, 'bla')
    op = Spherical2HealpixOperator(NSIDE, 'zenith,azimuth')

    def func(i, o):
        if i.shape == (2,) and o.shape == ():
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_healpix_spherical():
    vecs = (0, 100, 12*NSIDE**2-1,
            (0, 100, 12*NSIDE**2-1),
            ((0, 100),))
    shapes = ((), (), (), (3,), (1, 2))

    def func(c, n, v, s):
        h2s = Healpix2SphericalOperator(NSIDE, c, nest=n)
        s2h = Spherical2HealpixOperator(NSIDE, c, nest=n)
        a = s2h(h2s(v))
        assert_equal(a.shape, s)
        assert_equal(a, v)
    for c in Healpix2SphericalOperator.CONVENTIONS:
        for n in False, True:
            for v, s in zip(vecs, shapes):
                yield func, c, n, v, s


def test_healpix_spherical_rules():
    def func(op, nside, c, nest):
        op2 = Spherical2HealpixOperator(nside, c, nest=nest)
        if c == op.convention and nside == op.nside and nest == op.nest:
            assert_is_type(op * op2, IdentityOperator)
        else:
            assert_is_type(op * op2, CompositionOperator)
    for c in Healpix2SphericalOperator.CONVENTIONS:
        op = Healpix2SphericalOperator(NSIDE, c, nest=True)
        assert_is_type(op.I, Spherical2HealpixOperator)
        assert_equal(op.I.convention, op.convention)
        for nside in NSIDE, 2*NSIDE:
            for c2 in Healpix2SphericalOperator.CONVENTIONS:
                for nest in False, True:
                    yield func, op, nside, c2, nest


def test_healpix_spherical_error():
    assert_raises(ValueError, Healpix2SphericalOperator, NSIDE, 'bla')
    op = Healpix2SphericalOperator(NSIDE, 'zenith,azimuth')

    def func(i, o):
        if i.shape == () and o.shape == (2,):
            op(i, o)
            return
        assert_raises(ValueError, op.__call__, i, o)
    for i, o in itertools.product((np.array(1.), np.zeros(2), np.zeros(3)),
                                  (np.array(1.), np.zeros(2), np.zeros(3))):
        yield func, i, o


def test_healpix_convolution():
    nside = 16
    input = np.arange(12 * nside**2)
    keywords = {'fwhm': np.radians(30),
                'iter': 2,
                'lmax': 8,
                'use_weights': True,
                'regression': True}
    expected = hp.smoothing(input, **keywords)

    op = HealpixConvolutionGaussianOperator(**keywords)
    assert_same(op(input), expected)

    input = np.array([input, input]).T
    assert_same(op(input), np.array([expected, expected]).T)
