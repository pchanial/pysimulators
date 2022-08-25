"""
This module provides utilities for the cartesian 2-d geometry.

The coordinates of the geometric object provided by this module refer
to a cartesian referential (X, Y). To state it explicitly, for a given array
of coordinates a, a[..., 0] refers to the X-axis and a[..., 1] to the Y-axis.
           Y ^
             |
             +--> X
Angles are counted counter-clockwise.

"""

import numpy as _np
import scipy.spatial as _spatial

from . import _flib


def convex_hull(points):
    """
    Return the convex hull of 2-dimensional points. The points in the convex
    hull are sorted counter-clockwise.

    Parameters
    ----------
    points : array-like of shape (..., 2)
        The 2-dimensional points for which the convex hull is computed.

    """
    points = _np.atleast_1d(points)
    if points.shape[-1] != 2:
        raise ValueError(
            f"Invalid last dimension of 2-dimensional points '{points.shape[-1]}'."
        )
    points = points.reshape((-1, 2))

    def arctan2(v):
        return (_np.arctan2(v[:, 1], v[:, 0]) + 2 * _np.pi) % (2 * _np.pi)

    ihull = list(set(_spatial.Delaunay(points).convex_hull.ravel()))
    hull = points[ihull, :]
    return hull[_np.argsort(arctan2(hull - hull.mean(axis=0))), :]


def create_circle(radius, out=None, center=(0, 0), n=72, dtype=float):
    """
    Return coordinates of a circle.

    Parameters
    ----------
    radius : float
        The circle radius.
    out : array of shape radius.shape + (2,), optional
        Placeholder for the output coordinates.
    center : array-like of size 2, optional
        The (X, Y) coordinates of the circle center.
    n : integer
        Number of vertices, optional
    dtype : dtype, optional
        Coordinate data type.

    """
    return create_regular_polygon(n, radius, out=out, center=center, dtype=dtype)


def create_grid(
    shape,
    spacing,
    out=None,
    xreflection=False,
    yreflection=False,
    center=(0, 0),
    angle=0,
):
    """
    The grid nodes are laid out as a rectangular matrix (nrows, ncolumns)
    in the (X, Y) referential. Before rotation, rows increase along the -Y axis
    (unless yreflection is set to True) and columns increase along the X axis
    (unless xreflection is set to True).

    Parameters
    ----------
    shape : tuple of two integers (nrows, ncolumns)
        Number of rows and columns of the grid.
    spacing : float, Quantity
        Physical spacing between grid nodes.
    out : array of shape (shape[0], shape[1], 2), optional
        Placeholder for the output coordinates.
    xreflection : boolean, optional
        Reflection along the X-axis before rotation and translation.
    yreflection : boolean, optional
        Reflection along the Y-axis before rotation and translation.
    center : array-like of shape (2,), optional
        The (X, Y) coordinates of the grid center
    angle : float, optional
        Counter-clockwise rotation in degrees around the grid center.

    """
    if out is None:
        out = _np.empty(shape + (2,))
    _flib.geometry.create_grid(
        spacing, xreflection, yreflection, angle, center[0], center[1], out.T
    )
    return out


def create_grid_squares(
    shape,
    spacing,
    out=None,
    filling_factor=1,
    xreflection=False,
    yreflection=False,
    center=(0, 0),
    angle=0,
):
    """
    The square centers are laid out as a rectangular matrix (nrows, ncolumns)
    in the (X, Y) referential. Before rotation, rows increase along the -Y axis
    (unless yreflection is set to True) and columns increase along the X axis
    (unless xreflection is set to True).

    Parameters
    ----------
    shape : tuple of two integers (nrows, ncolumns)
        Number of rows and columns of the grid.
    spacing : float, Quantity
        Physical spacing between square centers.
    out : array of shape (shape[0], shape[1], 4, 2), optional
        Placeholder for the output coordinates.
    filling_factor : float, optional
        Fraction of the area of the squares relative to spacing**2.
    xreflection : boolean, optional
        Reflection along the X-axis before rotation and translation.
    yreflection : boolean, optional
        Reflection along the Y-axis before rotation and translation.
    center : array-like of shape (2,), optional
        The (X, Y) coordinates of the grid center
    angle : float, optional
        Counter-clockwise rotation in degrees around the grid center.

    """
    if out is None:
        out = _np.empty(shape + (4, 2))
    _flib.geometry.create_grid_squares(
        spacing,
        filling_factor,
        xreflection,
        yreflection,
        angle,
        center[0],
        center[1],
        out.T,
    )
    return out


def create_rectangle(size, out=None, center=(0, 0), angle=0, dtype=float):
    """
    Return coordinates of a rectangle.

    Parameters
    ----------
    size : array-like of size 2
        The rectangle dimensions along the X and Y axis.
    out : array of shape size.shape[:-1] + (4, 2), optional
        Placeholder for the output coordinates.
    center : array-like of size 2, optional
        The (X, Y) coordinates of the rectangle center.
    angle : float, optional
        Angle of counter-clockwise rotation around the center in degrees.
    dtype : dtype, optional
        Coordinate data type.

    """
    size = _np.asarray(size, dtype=dtype)
    if size.shape[-1] != 2:
        raise ValueError('Invalid rectangle dimensions.')
    if out is None:
        out = _np.empty(size.shape[:-1] + (4, 2), dtype)
    out[...] = (
        _np.asarray([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]], dtype)
        * size[..., None, :]
    )
    if angle != 0:
        rotate(out, angle, out=out)
    out += center
    return out


def create_regular_polygon(n, radius, out=None, center=(0, 0), angle=0, dtype=float):
    """
    Return coordinates of a regular polygon.

    Parameters
    ----------
    n : integer
        Number of vertices, optional
    radius : float
        The center-to-vertex distance.
    out : array of shape radius.shape + (2,), optional
        Placeholder for the output coordinates.
    center : array-like of size 2, optional
        The (X, Y) coordinates of the circle center.
    angle : float, optional
        Counter-clockwise rotation in degrees around center.
    dtype : dtype, optional
        Coordinate data type.

    """
    radius = _np.asarray(radius, dtype)
    center = _np.asarray(center, dtype)
    if out is None:
        b = _np.broadcast(radius[..., None], center)
        out = _np.empty(b.shape[:-1] + (n, 2), dtype)
    pi = 4 * _np.arctan(_np.asarray(1, dtype))
    a = 2 * pi / n * _np.arange(n, dtype=dtype) + _np.radians(angle)
    out[...] = (
        radius[..., None, None] * _np.asarray([_np.cos(a), _np.sin(a)]).T
        + center[..., None, :]
    )
    return out


def create_square(size, out=None, center=(0, 0), angle=0, dtype=float):
    """
    Return coordinates of a square.

    Parameters
    ----------
    size : float
        The square size.
    out : array of shape size.shape + (4, 2), optional
        Placeholder for the output coordinates.
    center : array-like of size 2, optional
        The (X, Y) coordinates of the square center.
    angle : float, optional
        Angle of counter-clockwise rotation around the center in degrees.
    dtype : dtype, optional
        Coordinate data type.

    """
    size = _np.asarray(size, dtype=dtype)
    if out is None:
        out = _np.empty(size.shape + (4, 2), dtype)
    out[...] = (
        _np.asarray([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]], dtype)
        * size[..., None, None]
    )
    if angle != 0:
        rotate(out, angle, out=out)
    out += center
    return out


def segment_polygon(vertices, max_distance, dtype=float):
    """
    Segment the edges of a polygon, ensuring that all segments have a size
    lower than a given distance. Such segmentation can be useful to better
    represent a polygon after a spatial transform.

    Parameters
    ----------
    vertices : array-like, last dimension is 2
        The (X,Y) coordinates of the vertices.
    min_distance : float
        The maximum size of the segments.
    dtype : dtype, optional
        If provided, forces the calculation to use the data type specified.

    """
    vertices = _np.array(vertices, dtype)
    if vertices.ndim != 2:
        raise ValueError('Invalid dimension of the input vertices.')
    vertices_ = _np.vstack([vertices, vertices[0]])
    ds = [_np.hypot(c[0], c[1]) for c in _np.diff(vertices_, axis=0)]
    ns = [int(_np.ceil(d / max_distance)) for d in ds]
    coords = _np.empty((_np.sum(ns), 2), vertices.dtype)
    i = 0
    for v0, v1, n in zip(vertices_[:-1], vertices_[1:], ns):
        alpha = _np.arange(n) / n
        coords[i : i + n, :] = _np.outer(1 - alpha, v0) + _np.outer(alpha, v1)
        i += n
    return coords


def rotate(coords, angle, out=None):
    """
    Rotate one or more points around the origin.

    Parameters
    ----------
    coords : array-like, last dimension is 2
        The (X, Y) coordinates of the points.
    angle : float
        Angle of counter-clockwise rotation in degrees.
    out : ndarray, optional
        If provided, the calculation is done into this array.

    """
    from pyoperators.utils import isalias

    coords = _np.asarray(coords)
    if out is None:
        out = _np.empty_like(coords)
    if (
        coords.dtype.char == 'd'
        and coords.flags.c_contiguous
        and out.flags.c_contiguous
    ):
        if isalias(coords, out):
            _flib.geometry.rotate_2d_inplace(out.reshape((-1, 2)).T, angle)
        else:
            _flib.geometry.rotate_2d(
                coords.reshape((-1, 2)).T, out.reshape((-1, 2)).T, angle
            )
        return out
    if coords.dtype.itemsize > 8:
        angle = _np.asarray(angle, coords.dtype)
    angle = _np.radians(angle)
    m = _np.asarray(
        [[_np.cos(angle), -_np.sin(angle)], [_np.sin(angle), _np.cos(angle)]],
        coords.dtype,
    )
    if isalias(coords, out):
        coords = coords.copy()
    return _np.dot(coords, m.T, out)


def surface_simple_polygon(coords, out=None, dtype=float):
    """
    Return the surface of convex polygons.

    Parameters
    ----------
    coords : array-like of shape (..., nvertices, 2)
        The vertices coordinates.
    out : ndarray, optional
        Placeholder of the polygon surfaces.

    Example
    -------
    >>> from pysimulators.geometry import (
    ...     create_square, create_regular_polygon, surface_simple_polygon)
    >>> c = create_square([1, 2])
    >>> surface_simple_polygon(c)
    array([ 1.,  4.])
    >>> surface_simple_polygon(create_regular_polygon(6, 1))

    """
    coords = _np.array(coords, dtype=float, order='c')
    if out is None:
        out = _np.empty(coords.shape[:-2], dtype)
    if coords.ndim < 3:
        coords = coords[None, ...]
    if out.ndim == 0:
        out_ = out.reshape(1)
    else:
        out_ = out
    _flib.geometry.surface_simple_polygon(coords.T, out_.T)
    if out.ndim == 0:
        out = out[()]
    return out
