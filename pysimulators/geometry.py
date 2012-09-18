from __future__ import division

def create_circle(radius, center=[0,0], n=72, dtype=float):
    """
    Return coordinates of a circle.

    Parameters
    ----------
    radius : float
        The circle radius.
    center : array-like of size 2, optional
        The (X,Y) coordinates of the circle center.
    n : integer
        Number of vertices, optional
    dtype : dtype, optional
        Coordinate data type.

    """
    import numpy as np
    a = 2 * np.pi / n * np.arange(n, dtype=dtype)
    coords = np.empty((n,2), dtype)
    coords[:,0] = radius * np.cos(a)
    coords[:,1] = radius * np.sin(a)
    coords += center
    return coords

def create_rectangle(size, center=[0,0], angle=0, dtype=float):
    """
    Return coordinates of a rectangle.

    Parameters
    ----------
    size : array-like of size 2
        The rectangle dimensions along the X and Y axis.
    center : array-like of size 2, optional
        The (X,Y) coordinates of the rectangle center.
    angle : float, optional
        Angle of counter-clockwise rotation around the center in degrees.
    dtype : dtype, optional
        Coordinate data type.

    """
    import numpy as np
    if len(size) != 2:
        raise ValueError('Invalid rectangle dimensions.')
    size = np.array(size, dtype=dtype)
    coords = np.empty((4,2), dtype)
    coords[:,0] = [1,-1,-1,1] * size[[0]] / 2
    coords[:,1] = [1,1,-1,-1] * size[[1]] / 2
    if angle != 0:
        _rotate(coords, angle, out=coords)
    coords += center
    return coords

def create_square(size, center=[0,0], angle=0, dtype=float):
    """
    Return coordinates of a square.

    Parameters
    ----------
    size : float
        The square size.
    center : array-like of size 2, optional
        The (X,Y) coordinates of the square center.
    angle : float, optional
        Angle of counter-clockwise rotation around the center in degrees.
    dtype : dtype, optional
        Coordinate data type.

    """
    return create_rectangle([size,size], center, angle, dtype)

def segment_polygon(vertices, max_distance, dtype=float):
    """
    Segment the edges of a polygon, ensuring that all segments have a size
    lower than a given distance.

    Parameters
    ----------
    vertices : array-like, last dimension is 2
        The (X,Y) coordinates of the vertices.
    min_distance : float
        The maximum size of the segments.
    dtype : dtype, optional
        If provided, forces the calculation to use the data type specified.

    """
    import numpy as np
    from itertools import izip
    vertices = np.array(vertices, dtype)
    if vertices.ndim != 2:
        raise ValueError('Invalid dimension of the input vertices.')
    vertices_ = np.vstack([vertices, vertices[0]])
    ds = [np.hypot(c[0],c[1]) for c in np.diff(vertices_, axis=0)]
    ns = [int(np.ceil(d / max_distance)) for d in ds]
    coords = np.empty((np.sum(ns),2), vertices.dtype)
    i = 0
    for v0, v1, n in izip(vertices_[:-1], vertices_[1:], ns):
        alpha = np.arange(n) / n
        coords[i:i+n,:] = np.outer(1-alpha, v0) + np.outer(alpha, v1)
        i += n
    return coords

def _rotate(coords, angle, out=None, dtype=float):
    """
    Rotate one or more points.

    Parameters
    ----------
    coords : array-like, last dimension is 2
        The (X,Y) coordinates of the points.
    angle : float
        Angle of counter-clockwise rotation in degrees.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : dtype, optional
        If provided, forces the calculation to use the data type specified.

    """
    import numpy as np
    coords = np.array(coords, dtype)
    if out is None:
        out = np.empty_like(coords)
    coords = coords.reshape((-1,2))
    theta = np.deg2rad(angle)
    m = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    np.einsum('ij,nj->ni', m, coords, out=out.reshape((-1,2)))
    return out
