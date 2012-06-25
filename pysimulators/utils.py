# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#

from __future__ import division

import numpy as np

from matplotlib import pyplot
from pyoperators.utils import product
from pyoperators.utils.mpi import MPI, filter_comm

#from . import tamasisfortran as tmf

__all__ = [ 
    'hs',
    'minmax',
    'plot_tod',
    'profile',
]

def all_eq(a, b, rtol=None, atol=0.):
    return not any_neq(a, b, rtol=rtol, atol=atol)

def any_neq(a, b, rtol=None, atol=0.):
    """
    Returns True if two arrays are element-wise equal within a tolerance.
    Differs from numpy.allclose in two aspects: the default rtol values (10^-7 
    and 10^-14 for single and double floats or complex) and the treatment of 
    NaN values (do not return False if the two arrays have element-wise
    both NaN value)
    """

    # for dictionaries, look up the items
    if isinstance(a, dict):
        if not isinstance(b, dict):
            print('First argument is a dict and the second one is not.')
            return True
        if set([k for k in a]) != set([k for k in b]):
            print('Arguments are dictionaries of different items.')
            return True
        for k in a:
            if any_neq(a[k], b[k]):
                print('Arguments are dictionaries of different values')
                return True
        return False

    # restrict comparisons to some classes
    class_ok = (np.ndarray, int, float, complex, list, tuple, str, unicode)
    if not isinstance(a, class_ok) or not isinstance(b, class_ok):
        return not isinstance(a, type(b)) and a is not None and \
               not isinstance(b, type(a)) and b is not None

    a = np.asanyarray(a)
    b = np.asanyarray(b)

    # get common base class to give some slack
    cls = type(b)
    while True:
        if isinstance(a, cls):
            break
        cls = cls.__base__

    a = a.view(cls)
    b = b.view(cls)
    
    akind = a.dtype.kind
    bkind = b.dtype.kind

    aattr = get_attributes(a)
    battr = get_attributes(b)
    if set(aattr) != set(battr):
        print('The arguments do not have the same attributes.')
        return True

    for k in aattr:
        if any_neq(getattr(a,k), getattr(b,k), rtol, atol):
            print("The argument attributes '" + k + "' differ.")
            return True

    # then compare the values of the array
    if akind not in 'biufc' or bkind not in 'biufc':
        if akind != bkind:
            print('The argument data types are incompatible.')
            return True
        if akind == 'S':
            result = np.any(a != b)
            if result:
                print('String arguments differ.')
            return result
        if akind == 'V':
            if set(a.dtype.names) != set(b.dtype.names):
                print('The names of the argument dtypes are different.')
                return True
            result = False
            for name in a.dtype.names:
                result_ = any_neq(a[name], b[name])
                if result_:
                    print("Values for '{0}' are different.".format(name))
                result = result or result_
            return result
        else:
            raise NotImplementedError('Kind ' + akind + ' is not implemented.')
    
    if akind in 'biu' and bkind in 'biu':
        result = np.any(a != b)
        if result:
            print('Integer arguments differ.')
        return result
    
    if rtol is None:
        asize = a.dtype.itemsize // 2 if akind == 'c' else a.dtype.itemsize
        bsize = b.dtype.itemsize // 2 if bkind == 'c' else b.dtype.itemsize
        precision = min(asize, bsize)
        if precision == 8:
            rtol = 1.e-14
        elif precision == 4:
            rtol = 1.e-7
        else:
            raise NotImplementedError('The inputs are not in single or double' \
                                      ' precision.')

    if a.ndim > 0 and b.ndim > 0 and a.shape != b.shape:
        print('Argument shapes differ.')
        return True

    mask = np.isnan(a)
    if np.any(mask != np.isnan(b)):
        print('Argument NaNs differ.')
        return True
    if np.all(mask):
        return False

    result = abs(a-b) > rtol * np.maximum(abs(a), abs(b)) + atol
    if not np.isscalar(result):
        result = np.any(result[~mask])

    if result:
        factor = np.nanmax(abs(a-b) / (rtol * np.maximum(abs(a),abs(b)) + atol))
        print('Argument data difference exceeds tolerance by factor ' + \
              str(factor) + '.')

    return result


#-------------------------------------------------------------------------------


def assert_all_eq(a, b, rtol=None, atol=0., msg=None):
    assert all_eq(a, b, rtol=rtol, atol=atol), msg


#-------------------------------------------------------------------------------


def get_attributes(obj):
    try:
        attributes = [ k for k in obj.__dict__ ]
    except AttributeError:
        attributes = []
    if hasattr(obj.__class__, '__mro__'):
        for cls in obj.__class__.__mro__:
            for slot in cls.__dict__.get('__slots__', ()):
                if hasattr(obj, slot):
                    attributes.append(slot)
    return sorted(attributes)


#-------------------------------------------------------------------------------


def get_type(data):
    """Returns input's data type."""
    data_ = np.asarray(data)
    type_ = data_.dtype.type.__name__
    if type_[-1] == '_':
        type_ = type_[0:-1]
    if type_ != 'object':
        return type_
    return type(data).__name__


#-------------------------------------------------------------------------------


def hs(arg):
    """
    Display the attributes of an object (except methods and those starting
    with an underscore) or an ndarray with composite dtype alongside the
    names of the records. The display is truncated so that each name fits
    in one line.
    """
    import inspect
    if isinstance(arg, np.ndarray):
        names = arg.dtype.names
        if names is None:
            print(arg)
            return
        print(str(arg.size) + ' element' + ('s' if arg.size > 1 else ''))
    else:
        members = inspect.getmembers(arg, lambda x: not inspect.ismethod(x) \
                                     and not inspect.isbuiltin(x))
        members = [x for x in members if x[0][0] != '_']
        names = [x[0] for x in members]

    length = np.max(list(map(len, names)))
    lnames = np.array([names[i].ljust(length)+': ' for i in range(len(names))])
    for name, lname in zip(names, lnames):
        value = str(getattr(arg, name))[0:72-length-2].replace('\n', ' ')
        if len(value) == 72-length-2:
            value = value[0:-3] + '...'
        print(lname+value)


#-------------------------------------------------------------------------------


def isscalar(data):
    """Hack around np.isscalar oddity"""
    return data.ndim == 0 if isinstance(data, np.ndarray) else np.isscalar(data)


#-------------------------------------------------------------------------------


def median(x, mask=None, axis=None, out=None):
    """
    Return median of array.

    Parameters
    ----------
    x : sequence or ndarray
        The input array. NaN values are discarded. Complex and floats of
        precision greater than 64 are not handled
    mask : ndarray, optional
        Boolean array mask whose True values indicate an element to be
        discarded.
    axis : {None, int}, optional
        Axis along which the medians are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.

    Returns
    -------
    median : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead).

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> median(a)
    3.0
    >>> median(a, axis=1)
    array([[ 7.],
           [ 2.]])

    """
    x = np.array(x, copy=False, order='c', subok=True)
    shape = x.shape
    dtype = np.find_common_type([np.float64, x.dtype], [])
    if dtype != np.float64:
        raise TypeError("Invalid input type '{0}'.".format(dtype))
    x = np.asanyarray(x, dtype)

    if mask is None and hasattr(x, 'mask'):
        mask = x.mask
        if mask is not None and mask.shape != x.shape:
            raise ValueError('Incompatible mask shape.')
    if mask is not None:
        mask = np.array(mask, dtype=bool, order='c', copy=False).view(np.int8)

    if axis is not None:
        slow = product(shape[:axis])
        fast = product(shape[axis+1:])
        x = x.reshape((slow,-1,fast))
        if mask is not None:
            mask = mask.reshape((slow,-1,fast)).view(np.int8)
        if out is not None:
            if out.nbytes != slow * fast * dtype.itemsize:
                raise ValueError('Incompatible output buffer length.')
            if out.shape != shape[:axis] + shape[axis+1:]:
                raise ValueError('Incompatible output shape.')
            out.dtype = dtype
        else:
            out = np.empty(shape[:axis] + shape[axis+1:], dtype)
        out_ = out.reshape((slow,fast))
    else:
        out = np.empty((), dtype)
        out_ = out

    if mask is axis is None:
        tmf.math.median(x.ravel(), out)
    elif axis is None:
        tmf.math.median_mask(x.ravel(), mask.ravel(), out)
    elif mask is None:
        tmf.math.median_axis(x.T, out_.T)
    else:
        tmf.math.median_mask_axis(x.T, mask.T, out_.T)

    if out.ndim == 0:
        out = out.flat[0]

    return out


#-------------------------------------------------------------------------------


def minmax(v):
    """Returns min and max values of an array, discarding NaN values."""
    v = np.asanyarray(v)
    if np.all(np.isnan(v)):
        return np.array([np.nan, np.nan])
    return np.array((np.nanmin(v), np.nanmax(v)))
   

#-------------------------------------------------------------------------------


def plot_tod(tod, mask=None, **kw):
    """Plot the signal timelines in a Tod and show masked samples.

    Plotting every detector timelines may be time consuming, so it is
    recommended to use this method on one or few detectors like this:
    >>> plot_tod(tod[idetector])
    """
    if mask is None:
        mask = getattr(tod, 'mask', None)

    ndetectors = product(tod.shape[0:-1])
    tod = tod.view().reshape((ndetectors, -1))
    if mask is not None:
        mask = mask.view().reshape((ndetectors, -1))
        if np.all(mask):
            print('There is no valid sample.')
            return

    for idetector in range(ndetectors):
        pyplot.plot(tod[idetector], **kw)
        if mask is not None:
            index=np.where(mask[idetector])
            pyplot.plot(index, tod[idetector,index],'ro')

    unit = getattr(tod, 'unit', '')
    if unit:
        pyplot.ylabel('Signal [' + unit + ']')
    else:
        pyplot.ylabel('Signal')
    pyplot.xlabel('Time sample')


#-------------------------------------------------------------------------------


def profile(input, origin=None, bin=1., nbins=None, histogram=False):
    """
    Returns axisymmetric profile of a 2d image.
    x, y[, n] = profile(image, [origin, bin, nbins, histogram])

    Parameters
    ----------
    input: array
        2d input array
    origin: (x0,y0)
        center of the profile. (Fits convention). Default is the image center
    bin: number
        width of the profile bins (in unit of pixels)
    nbins: integer
        number of profile bins
    histogram: boolean
        if set to True, return the histogram
    """
    input = np.ascontiguousarray(input, float)
    if origin is None:
        origin = (np.array(input.shape[::-1], float) + 1) / 2
    else:
        origin = np.ascontiguousarray(origin, float)
    
    if nbins is None:
        nbins = int(max(input.shape[0]-origin[1], origin[1],
                        input.shape[1]-origin[0], origin[0]) / bin)

    x, y, n = tmf.profile_axisymmetric_2d(input.T, origin, bin, nbins)
    if histogram:
        return x, y, n
    else:
        return x, y


#-------------------------------------------------------------------------------


def diff(input, output, axis=0, comm=None):
    """
    Inplace discrete difference
    """

    if not isinstance(input, np.ndarray):
        raise TypeError('Input array is not an ndarray.')

    if input.dtype != float:
        raise TypeError('The data type of the input array is not ' + \
                        str(float.type) + '.')

    if axis < 0:
        raise ValueError("Invalid negative axis '" + str(axis) + "'.")

    if comm is None:
        comm = MPI.COMM_WORLD

    ndim = input.ndim
    if ndim == 0:
        output.flat = 0
        return

    if axis >= ndim:
        raise ValueError("Invalid axis '" + str(axis) + "'. Expected value is" \
                         ' less than ' + str(ndim-1) + '.')

    inplace = input.__array_interface__['data'][0] == \
              output.__array_interface__['data'][0]

    if axis != 0 or comm.size == 1:
        if input.size == 0:
            return
        tmf.operators.diff(input.ravel(), output.ravel(), ndim-axis,
                           np.asarray(input.T.shape), inplace)
        return

    if product(input.shape[1:]) == 0:
        return
    with filter_comm(input.shape[0] > 0, comm) as fcomm:
        if fcomm is not None:
            status = tmf.operators_mpi.diff(input.ravel(), output.ravel(),
                ndim-axis, np.asarray(input.T.shape), inplace, fcomm.py2f())
            if status != 0: raise RuntimeError()


#-------------------------------------------------------------------------------


def diffT(input, output, axis=0, comm=None):
    """
    Inplace discrete difference transpose
    """

    if not isinstance(input, np.ndarray):
        raise TypeError('Input array is not an ndarray.')

    if input.dtype != float:
        raise TypeError('The data type of the input array is not ' + \
                        str(float.type) + '.')

    if axis < 0:
        raise ValueError("Invalid negative axis '" + str(axis) + "'.")

    if comm is None:
        comm = MPI.COMM_WORLD

    ndim = input.ndim
    if ndim == 0:
        output.flat = 0
        return

    if axis >= ndim:
        raise ValueError("Invalid axis '" + str(axis) + "'. Expected value is" \
                         ' less than ' + str(ndim-1) + '.')

    inplace = input.__array_interface__['data'][0] == \
              output.__array_interface__['data'][0]

    if axis != 0 or comm.size == 1:
        if input.size == 0:
            return
        tmf.operators.difft(input.ravel(), output.ravel(), ndim-axis,
                            np.asarray(input.T.shape), inplace)
        return

    if product(input.shape[1:]) == 0:
        return
    with filter_comm(input.shape[0] > 0, comm) as fcomm:
        if fcomm is not None:
            status = tmf.operators_mpi.difft(input.ravel(), output.ravel(),
                ndim-axis, np.asarray(input.T.shape), inplace, fcomm.py2f())
            if status != 0: raise RuntimeError()

    
#-------------------------------------------------------------------------------


def diffTdiff(input, output, axis=0, scalar=1., comm=None):
    """
    Inplace discrete difference transpose times discrete difference
    """

    if not isinstance(input, np.ndarray):
        raise TypeError('Input array is not an ndarray.')

    if input.dtype != float:
        raise TypeError('The data type of the input array is not ' + \
                        str(float.type) + '.')

    if axis < 0:
        raise ValueError("Invalid negative axis '" + str(axis) + "'.")

    if comm is None:
        comm = MPI.COMM_WORLD

    scalar = np.asarray(scalar, float)
    ndim = input.ndim
    if ndim == 0:
        output.flat = 0
        return
    
    if axis >= ndim:
        raise ValueError("Invalid axis '" + str(axis) + "'. Expected value is" \
                         ' less than ' + str(ndim) + '.')

    inplace = input.__array_interface__['data'][0] == \
              output.__array_interface__['data'][0]

    if axis != 0 or comm.size == 1:
        if input.size == 0:
            return
        tmf.operators.difftdiff(input.ravel(), output.ravel(), ndim-axis,
                                np.asarray(input.T.shape), scalar, inplace)
        return

    if product(input.shape[1:]) == 0:
        return
    with filter_comm(input.shape[0] > 0, comm) as fcomm:
        if fcomm is not None:
            status = tmf.operators_mpi.difftdiff(input.ravel(), output.ravel(),
                ndim-axis, np.asarray(input.T.shape), scalar, inplace,
                fcomm.py2f())
            if status != 0: raise RuntimeError()


#-------------------------------------------------------------------------------


def shift(input, output, n, axis=-1):
    """
    Shift array elements inplace along a given axis.

    Elements that are shifted beyond the last position are not re-introduced
    at the first.

    Parameters
    ----------
    array : float array
        Input array to be modified

    n : integer number or array
        The number of places by which elements are shifted. If it is an array,
        specific offsets are applied along the first dimensions.

    axis : int, optional
        The axis along which elements are shifted. By default, it is the first
        axis.

    Examples
    --------
    >>> a = ones(8)
    >>> shift(a, 3); a
    array([0., 0., 0., 1., 1., 1., 1., 1., 1.])

    >>> a = array([[1.,1.,1.,1.],[2.,2.,2.,2.]])
    >>> shift(a, [1,-1], axis=1); a
    array([[0., 1., 1., 1.],
           [2., 2., 2., 0.]])
    """
    if not isinstance(input, np.ndarray):
        raise TypeError('Input array is not an ndarray.')

    if input.dtype != float:
        raise TypeError('The data type of the input array is not ' + \
                        str(float.name) + '.')

    rank = input.ndim
    n = np.array(n, ndmin=1).ravel()
    
    if axis < 0:
        axis = rank + axis

    if axis == 0 and n.size > 1 or n.size != 1 and n.size not in \
       np.cumproduct(input.shape[0:axis]):
        raise ValueError('The offset size is incompatible with the first dime' \
                         'nsions of the array')
    if rank == 0:
        output[()] = 0
    else:
        tmf.shift(input.ravel(), output.ravel(), rank-axis,
                  np.asarray(input.T.shape), n)
