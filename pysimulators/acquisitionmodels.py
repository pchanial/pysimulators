from __future__ import division

import functools
import inspect
import numpy as np
import operator
import os
import pyoperators

from pyoperators import (Operator, BlockColumnOperator, BlockDiagonalOperator,
                         CompositionOperator, DiagonalOperator,
                         DiagonalNumexprOperator,
                         DistributionIdentityOperator, IdentityOperator,
                         MaskOperator, MultiplicationOperator, NumexprOperator,
                         ZeroOperator)
from pyoperators.config import LOCAL_PATH
from pyoperators.decorators import (linear, orthogonal, real, square, symmetric,
                                    universal, inplace)
try:
    from pyoperators.memory import empty
except ImportError:
    from pyoperators.memory import allocate as empty
from pyoperators.utils import (isscalar, openmp_num_threads,
                               operation_assignment, product, tointtuple)
from pyoperators.utils.mpi import MPI, distribute_shape, distribute_shapes
from tamasis import tmf

from . import _flib as flib
from .datatypes import Map, Tod
from .quantities import Quantity, _divide_unit, _multiply_unit
from .utils import diff, diffT, diffTdiff, shift
from .wcsutils import fitsheader2shape, str2fitsheader
from .mpiutils import gather_fitsheader_if_needed, scatter_fitsheader

try:
    import fftw3
    MAX_FFTW_NUM_THREADS = 1 if fftw3.planning.lib_threads is None \
        else openmp_num_threads()
    FFTW_WISDOM_FILE = os.path.join(LOCAL_PATH, 'fftw3.wisdom')
    FFTW_WISDOM_MIN_DELAY = 0.1
    try:
        fftw3.import_system_wisdom()
    except IOError:
        pass
    _is_fftw_wisdom_loaded = False
except:
    print('Warning: Library PyFFTW3 is not installed.')

__all__ = [
    'BlackBodyOperator',
    'CompressionAverageOperator',
    'DistributionLocalOperator',
    'DdTddOperator',
    'DiscreteDifferenceOperator',
    'DownSamplingOperator',
    'FftHalfComplexOperator',
    'InvNttOperator',
    'PackOperator',
    'PadOperator',
    'PowerLawOperator',
    'ProjectionOperator',
    'ConvolutionTruncatedExponentialOperator',
    'RollOperator',
    'ShiftOperator',
    'SqrtInvNttOperator',
    'UnpackOperator',
    'PointingMatrix',
]

def block_diagonal(*partition_args, **keywords):
    """
    Class decorator that partitions an Operator along a specified axis.
    It adds a 'partitionin' keyed argument to the class constructor.

    Subclasses can also benefit from a decorated class, as long as they do not
    define an __init__ method.
    
    Parameters
    ----------
    axisin : int
        Input partition axis for chunk partitioning.
    new_axisin : int
        Input partition axis for stack partitioning.

    *partition_args: string varargs
       Specify the class' arguments of the __init__ method that  can be a
       sequence to be dispatched to the partitioned operators.

    Example
    -------
    >>> @block_diagonal('value', axisin=0)
    >>> @linear
    >>> class MyOp(Operator):
    >>>     def __init__(self, value, shapein=None):
    >>>         self.value = value
    >>>         Operator.__init__(self, lambda i,o: np.multiply(i,value,o),
    >>>                           shapein=shapein)
    >>>
    >>> op = MyOp([1,2,3], shapein=3, partitionin=(1,1,1))
    >>> op.todense()
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])

    >>> [o.value for o in op.operands]
    [1, 2, 3]
        
    """
    # the following way to extract keywords is unnecessary in Python3 (PEP 3102)
    if 'axisin' not in keywords and 'new_axisin' not in keywords: 
        raise TypeError("Missing 'axisin' or 'new_axisin' keyword.")

    if 'axisin' in keywords:
        axisin = keywords['axisin']
        new_axisin = None
        del keywords['axisin']
    else:
        axisin = None
        new_axisin = keywords['new_axisin']
        del keywords['new_axisin']

    if len(keywords) > 0:
        raise TypeError('Invalid keyed argument.')

    def func(cls):

        @functools.wraps(cls.__init__)
        def partition_init(self, *args, **keywords):

            # get number of blocks through the partitionin keyword
            n1 = 0
            partitionin = None
            if 'partitionin' in keywords:
                if keywords['partitionin'] is not None:
                    partitionin = tointtuple(keywords['partitionin'])
                    n1 = len(partitionin)
                del keywords['partitionin']

            # get __init__ argument names, except self
            class_args, jk, jk, jk = inspect.getargspec(cls.__init_original__)
            class_args.pop(0)

            # get number of blocks through the arguments
            ns = [0 if isscalar(a) else len(a) for i,a in enumerate(args) \
                  if class_args[i] in partition_args] + \
                 [0 if isscalar(v) else len(v) for k,v in keywords.items() \
                  if k in partition_args]

            n2 = 0 if len(ns) == 0 else max(ns)
            if any(n not in (0, n2) for n in ns):
                raise ValueError('The partition variables do not have the same '
                                 'number of elements.')

            # bail if no partitioning is found
            n = max(n1, n2)
            if n == 0:
                cls.__init_original__(self, *args, **keywords)
                return

            # check the two methods are compatible
            if n1 != 0 and n2 != 0 and n1 != n2:
                raise ValueError('The specified partitioning is incompatible wi'
                    'th the number of elements in the partition variables.')

            # Implicit partition
            if partitionin is None:
                if axisin is not None:
                    partitionin = n * (None,)
                else:
                    partitionin = n * (1,)

            # dispatch arguments
            n = len(partitionin)
            argss = tuple(tuple(a[i] if class_args[j] in partition_args and \
                not isscalar(a) else a for j,a in enumerate(args)) \
                for i in range(n))
            keyss = tuple(dict((k, v[i]) if k in partition_args and \
                not isscalar(v) else (k,v) for k,v in keywords.items()) \
                for i in range(n))
            
            # the input shapein/out describe the BlockDiagonalOperator
            def _reshape(s, p, a, na):
                s = list(tointtuple(s))
                if a is not None:
                    s[a] = p
                else:
                    np.delete(s, na)
                return tuple(s)

            if 'shapein' in keywords:
                shapein = keywords['shapein']
                for keys, p in zip(keyss, partitionin):
                    keys['shapein'] = _reshape(shapein, p, axisin, new_axisin)
            if 'shapeout' in keywords:
                shapeout = keywords['shapeout']
                for keys, p in zip(keyss, partitionin):
                    keys['shapeout'] = _reshape(shapeout, p, axisin, new_axisin)

            # instantiate the partitioned operators
            ops = [cls.__new__(type(self), *a, **k) for a,k in zip(argss,keyss)]
            for o, a, k in zip(ops, argss, keyss):
                if isinstance(o, cls):
                    cls.__init_original__(o, *a, **k)

            self.__class__ = BlockDiagonalOperator
            self.__init__(ops, partitionin=partitionin, axisin=axisin,
                          new_axisin=new_axisin)

        cls.__init_original__ = cls.__init__
        cls.__init__ = partition_init
        return cls

    return func


class BlackBodyOperator(Operator):
    """
    Instanceless class whose __new__ method specialises in
    BlackBodyFixedTemperatureOperator and BlackBodyFreeTemperatureOperator.

    """
    def __new__(cls, wavelength=None, wavelength0=None, temperature=None,
                beta=0.):
        if temperature is not None:
            return BlackBodyFixedTemperatureOperator(temperature, wavelength,
                                                     wavelength0, beta)
        raise NotImplementedError()


@symmetric
class BlackBodyFixedTemperatureOperator(NumexprOperator):
    """
    Diagonal operator whose normalised values follow the Planck equation,
    optionnally modified by a power-law emissivity of given slope.

    """
    def __new__(cls, temperature=None, wavelength=None, wavelength0=None,
                beta=0.):
        if not isscalar(wavelength):
            return BlockColumnOperator([BlackBodyFixedTemperatureOperator(
                temperature, w, wavelength0, beta) for w in wavelength],
                new_axisout=0)
        return NumexprOperator.__new__(cls)

    def __init__(self, temperature, wavelength, wavelength0=None, beta=0.):
        """
        Parameters
        ----------
        temperature : float
            Black body temperature, in Kelvin.
        wavelength : float
            Wavelength, in meters, at which black body values will be
            computed.
        wavelength0 : float
            Reference wavelength, in meters, for which the operator returns 1.
        beta : float
            Slope of the emissivity (the spectrum is multiplied by nu**beta)
            The default value is 0 (non-modified black body).

        """
        self.temperature = np.array(temperature, float, copy=False)
        self.wavelength = float(wavelength)
        self.wavelength0 = float(wavelength0)
        self.beta = float(beta)
        c = 2.99792458e8
        h = 6.626068e-34
        k = 1.380658e-23
        if self.temperature.size == 1:
            coef = (self.wavelength0/self.wavelength)**(3+self.beta) * \
                   (np.exp(h*c/(self.wavelength0*k*self.temperature))-1) / \
                   (np.exp(h*c/(self.wavelength*k*self.temperature))-1)
            expr = 'coef * input'
            global_dict = {'coef':coef}
        else:
            coef1 = (self.wavelength0/self.wavelength)**(3+self.beta)
            coef2 = h*c/(self.wavelength0*k)
            coef3 = h*c/(self.wavelength*k)
            expr = 'coef1 * (exp(coef2/T) - 1) / (exp(coef3/T) - 1)'
            global_dict = {'coef1':coef1, 'coef2':coef2, 'coef3':coef3,
                           'T':self.temperature}
        NumexprOperator.__init__(self, expr, global_dict, dtype=float)


@real
class PowerLawOperator(DiagonalNumexprOperator):
    """
    Diagonal operator which extrapolates an input following the power law:
        output = (x/x0)**alpha * input

    This operator is lightweight, in the sense that it does not copy the
    potentially large argument 'alpha' and that the diagonal elements are
    computed on the fly.

    Parameters
    ----------
    alpha : array-like
        The broadcastable power-law slope.
    x : float
        Abscissa at which the extrapolation is performed.
    x0 : float
        The reference abscissa.

    Example
    -------
    >>> import scipy.constants
    >>> c = scipy.constants.c
    >>> nu0 = c/10.e-6
    >>> op = PowerLawOperator(-1, c/11.e-6, nu0)
    >>> fnu0 = 1e-3
    >>> op(fnu0)
    array(0.0010999999999999998)

    """
    def __init__(self, alpha, x, x0, scalar=1, **keywords):
        x = np.asarray(x)
        if x.ndim != 0:
            raise TypeError('The input is not a scalar.')
        x0 = np.asarray(x0)
        if x0.ndim != 0:
            raise TypeError('The reference input is not a scalar.')
        alpha = np.asarray(alpha)
        if alpha.ndim > 0:
            keywords['shapein'] = alpha.shape
        scalar = np.array(scalar)
        if scalar.ndim != 0:
            raise TypeError('The scalar coefficient is not a scalar.')
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        global_dict = {'x':x, 'x0':x0, 's':scalar}
        DiagonalNumexprOperator.__init__(self, alpha, 's * (x / x0) ** alpha',
                                         global_dict, var='alpha', **keywords)
        self.x = x
        self.x0 = x0
        self.scalar = scalar
        self.set_rule('{HomothetyOperator}.', lambda o, s: PowerLawOperator(
                      alpha, x, x0, o.data * s.scalar), CompositionOperator)
        self.set_rule('.{ConstantOperator}', lambda s, o: PowerLawOperator(
                      alpha, x, x0, o.data * s.scalar) if o.broadcast ==
                      'scalar' else None, MultiplicationOperator)

    @staticmethod
    def _rule_block(self, op, shape, partition, axis, new_axis, func_operation):
        return DiagonalOperator._rule_block(self, op, shape, partition, axis,
                   new_axis, func_operation, self.x, self.x0, scalar=
                   self.scalar)


@block_diagonal('factor', axisin=-1)
@real
@linear
@universal
class CompressionOperator(Operator):
    """
    Abstract class for compressing the input signal.
    Compression is operated on the fastest axis.

    """
    def __new__(cls, factor=None, shapein=None, **keywords):
        if factor == 1:
            op = IdentityOperator(shapein=shapein, dtype=float, **keywords)
            op.factor = 1
            return op
        return Operator.__new__(cls)

    def __init__(self, factor, shapein=None, shapeout=None, **keywords):
        self.factor = int(factor)
        Operator.__init__(self, shapein=shapein, shapeout=shapeout, **keywords)

    def reshapein(self, shapein):
        return shapein[:-1] + (shapein[-1] // self.factor,)

    def reshapeout(self, shapeout):
        return shapeout[:-1] + (shapeout[-1] * self.factor,)

    def validatein(self, shapein):
        if shapein[-1] % self.factor != 0:
            raise ValueError("The input timeline size '{0}' is incompatible wit"
                "h the compression factor '{1}'.".format(shapein[-1],
                self.factor))

    def __str__(self):
        return super(CompressionOperator, self).__str__() + ' (x{0})'.format(
               self.factor)


class CompressionAverageOperator(CompressionOperator):
    """
    Compress the input signal by averaging blocks of specified size.

    """
    def direct(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.compression_average_direct(input_, ishape[0], ishape[1], istride,
            output_, oshape[1], ostride, self.factor)

    def transpose(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.compression_average_transpose(input_, ishape[0], ishape[1],
            istride, output_, oshape[1], ostride, self.factor)


class DownSamplingOperator(CompressionOperator):
    """
    Downsample the input signal by picking up one sample out of a number
    specified by the compression factor

    """
    def direct(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.downsampling_direct(input_, ishape[0], ishape[1], istride, output_,
            oshape[1], ostride, self.factor)

    def transpose(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.downsampling_transpose(input_, ishape[0], ishape[1], istride,
            output_, oshape[1], ostride, self.factor)
      

@real
@symmetric
class InvNttOperator(Operator):

    def __new__(cls, obs=None, method='uncorrelated', **keywords):
        if obs is None:
            return Operator.__new__(cls)
        nsamples = obs.get_nsamples()
        comm_tod = obs.instrument.comm
        method = method.lower()
        if method not in ('uncorrelated', 'uncorrelated python'):
            raise ValueError("Invalid method '{0}'.".format(method))

        filter = obs.get_filter_uncorrelated(**keywords)
        if filter.ndim == 2:
            filter = filter[np.newaxis,...]
        nfilters = filter.shape[0]
        if nfilters != 1 and nfilters != len(nsamples):
            raise ValueError("Incompatible number of filters '{0}'. Expected nu"
                             "mber is '{1}'".format(nfilters, len(nsamples)))
        ncorrelations = filter.shape[-1] - 1
        filter_length = np.asarray(2**np.ceil(np.log2(np.array(nsamples) + \
                                   2 * ncorrelations)), int)
        fft_filters = []
        for i, n in enumerate(filter_length):
            i = min(i, nfilters-1)
            fft_filter, status = tmf.fft_filter_uncorrelated(filter[i].T, n)
            if status != 0: raise RuntimeError()
            np.maximum(fft_filter, 0, fft_filter)
            fft_filters.append(fft_filter.T)
        
        norm = comm_tod.allreduce(max(np.max(f) for f in fft_filters),
                                  op=MPI.MAX)
        for f in fft_filters:
            np.divide(f, norm, f)
            
        if method == 'uncorrelated':
            cls = InvNttUncorrelatedOperator
        else:
            cls = InvNttUncorrelatedPythonOperator
        #XXX should generate BlockDiagonalOperator with no duplicates...
        return BlockDiagonalOperator([cls(f, ncorrelations, n) \
                   for f, n in zip(fft_filters, nsamples)], axisin=-1)


@real
@symmetric
class SqrtInvNttOperator(InvNttOperator):
    def __init__(self, *args, **kw):
        invntt = InvNttOperator(*args, **kw)
        self._tocomposite(op.Composition, invntt.blocks)
        data = self.blocks[2].data
        np.sqrt(data, data)


@real
@symmetric
@inplace
class InvNttUncorrelatedOperator(Operator):

    def __init__(self, fft_filter, ncorrelations, nsamples,
                 fftw_flags=['measure', 'unaligned']):
        filter_length = fft_filter.shape[-1]
        array = np.empty(filter_length, dtype=float)
        self.fft_filter = fft_filter
        self.filter_length = filter_length
        self.left = ncorrelations
        self.right = filter_length - nsamples - ncorrelations
        self.fftw_flags = fftw_flags
        self.fplan = fftw3.Plan(array, direction='forward', flags=fftw_flags,
            realtypes=['halfcomplex r2c'], nthreads=1)
        self.bplan = fftw3.Plan(array, direction='backward', flags=fftw_flags,
            realtypes=['halfcomplex c2r'], nthreads=1)
        Operator.__init__(self, shapein=(fft_filter.shape[0], nsamples))
        
    def direct(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        if self.isalias(input, output):
            tmf.operators.invntt_uncorrelated_inplace(input_, ishape[1],
                istride, self.fft_filter.T, self.fplan._get_parameter(),
                self.bplan._get_parameter(), self.left, self.right)
            return
        output_, oshape, ostride = _ravel_strided(output)
        tmf.operators.invntt_uncorrelated_outplace(input_, ishape[1], istride,
            output_, ostride, self.fft_filter.T, self.fplan._get_parameter(),
            self.bplan._get_parameter(), self.left, self.right)


@real
@symmetric
class InvNttUncorrelatedPythonOperator(Operator):

    def __new__(cls, fft_filter=None, ncorrelations=None, nsamples=None):
        if fft_filter is None:
            return Operator.__new__(cls)
        filter_length = fft_filter.shape[-1]
        invntt = DiagonalOperator(fft_filter)
        fft = FftHalfComplexOperator(filter_length)
        padding = PadOperator(left=ncorrelations, right=filter_length - \
                              nsamples - ncorrelations)
        return padding.T * fft.T * invntt * fft * padding


@block_diagonal('left', 'right', axisin=-1)
@real
@linear
@universal
class PadOperator(Operator):
    """
    Pads before and after along the fast dimension of an ndarray.

    """
    def __init__(self, left=0, right=0., **keywords):
        left = int(left)
        right = int(right)
        if left < 0:
            raise ValueError('Left padding is not positive.')
        if right < 0:
            raise ValueError('Right padding is not positive.')
        self.left = left
        self.right = right
        Operator.__init__(self, **keywords)        
   
    def direct(self, input, output):
        right = -self.right if self.right != 0 else output.shape[-1]
        output[...,:self.left] = 0
        output[...,self.left:right] = input
        output[...,right:] = 0
   
    def transpose(self, input, output):
        right = -self.right if self.right != 0 else input.shape[-1]
        output[...] = input[...,self.left:right]

    def reshapein(self, shapein):
        return shapein[:-1] + (shapein[-1] + self.left + self.right,)
       
    def reshapeout(self, shapeout):
        return shapeout[:-1] + (shapeout[-1] - self.left - self.right,)


class PointingMatrix(np.ndarray):
    DTYPE = np.dtype([('value', np.float32), ('index', np.int32)])
    def __new__(cls, array, shape_input, info=None, copy=True, ndmin=0):
        result = np.array(array, copy=copy, ndmin=ndmin).view(cls)
        result.info = info
        result.shape_input = shape_input
        return result

    def __array_finalize__(self, obj):
        self.info = getattr(obj, 'info', None)
        self.shape_input = getattr(obj, 'shape_input', None)

    def __getattr__(self, name):
        if self.dtype.names is not None and name in self.dtype.names:
            return self[name].view(np.ndarray)
        return super(PointingMatrix, self).__getattribute__(name)

    @classmethod
    def empty(cls, shape, shape_input, info=None, verbose=True):
        buffer = empty(shape, cls.DTYPE, description='for the pointing matrix',
                       verbose=verbose)
        return PointingMatrix(buffer, shape_input, info=info, copy=False)

    @classmethod
    def zeros(cls, shape, shape_input, info=None, verbose=True):
        buffer = empty(shape, cls.DTYPE, description='for the pointing matrix',
                       verbose=verbose)
        buffer.value = 0
        buffer.index = -1
        return PointingMatrix(buffer, shape_input, info=info, copy=False)

    def check(self, npixels=None):
        """
        Return true if the indices in the pointing matrix are greater or equal
        to -1 or less than the number of pixels in the map.
        """
        if self.size == 0:
            return True
        if self.info is not None and 'header' in self.info:
            npixels = product(fitsheader2shape(self.info['header']))
        elif npixels is None:
            raise ValueError('The number of map pixels is not specified.')
        result = flib.pointingmatrix.check(self.ravel().view(np.int64),
                     self.shape[-1], product(self.shape[:-1]), npixels)
        return bool(result)

    def pack(self, mask):
        """
        Inplace renumbering of the pixel indices in the pointing matrix,
        discarding masked or unobserved pixels.

        """
        if self.size == 0:
            return
        tmf.pointing_matrix_pack(self.ravel().view(np.int64), mask.view(
            np.int8).T, self.shape[-1], self.size // self.shape[-1])
        self.shape_input = tointtuple(mask.size - np.sum(mask))

    def get_mask(self, out=None):
        if out is None:
            out = empty(self.shape_input, np.bool8, 'as new mask')
            out[...] = True
            out = Map(out, header=self.info.get('header', None), copy=False,
                      dtype=bool)
        elif out.dtype != bool:
            raise TypeError('The output mask argument has an invalid type.')
        elif out.shape != self.shape_input:
            raise ValueError('The output mask argument has an incompatible shap'
                             'e.')
        if self.size == 0:
            return out
        tmf.pointing_matrix_mask(self.ravel().view(np.int64), out.view(
            np.int8).T, self.shape[-1], self.size // self.shape[-1])
        return out


def ProjectionOperator(input, method=None, header=None, resolution=None,
                       npixels_per_sample=0, units=None, derived_units=None,
                       downsampling=False, packed=False, commin=MPI.COMM_WORLD,
                       commout=MPI.COMM_WORLD, **keywords):
    """
    Projection operator factory, to handle operations by one or more pointing
    matrices.

    It is assumed that each pointing matrix row has at most 'npixels_per_sample'
    non-null elements, or in other words, an output element can only intersect
    a fixed number of input elements.

    Given an input vector x and an output vector y, y = P(x) translates into:
        y[i] = sum(P.matrix[i,:].value * x[P.matrix[i,:].index])

    If the input is not MPI-distributed unlike the output, the projection
    operator is automatically multiplied by the operator DistributionIdentity-
    Operator, to enable MPI reductions.

    If the input is MPI-distributed, this operator is automatically packed (see
    below) and multiplied by the operator DistributionLocalOperator, which
    takes the local input as argument.

    Arguments
    ---------
    input : pointing matrix (or sequence of) or observation (deprecated)
    method : deprecated
    header : deprecated
    resolution : deprecated
    npixels_per_sample : deprecated
    downsampling : deprecated
    packed deprecated

    """
    # check if there is only one pointing matrix
    isobservation = hasattr(input, 'get_pointing_matrix')
    if isobservation:
        if hasattr(input, 'slice'):
            nmatrices = len(input.slice)
        else:
            nmatrices = 1
        commout = input.instrument.comm
    else:
        if isinstance(input, PointingMatrix):
            input = (input,)
        if any(not isinstance(i, PointingMatrix) for i in input):
            raise TypeError('The input is not a PointingMatrix, (nor a sequence'
                            ' of).')
        nmatrices = len(input)

    ismapdistributed = commin.size > 1
    istoddistributed = commout.size > 1

    # get the pointing matrix from the input observation
    if isobservation:
        if header is None:
            if not hasattr(input, 'get_map_header'):
                raise AttributeError("No map header has been specified and "
                    "the observation has no 'get_map_header' method.")
            header_global = input.get_map_header(resolution=resolution,
                                downsampling=downsampling)
        else:
            if isinstance(header, str):
                header = str2fitsheader(header)
            header_global = gather_fitsheader_if_needed(header, comm=commin)
        #XXX we should hand over the local header
        input = input.get_pointing_matrix(header_global, npixels_per_sample,
                    method=method, downsampling=downsampling, comm=commin)
        if isinstance(input, PointingMatrix):
            input = (input,)
    else:
        header_global = input[0].info['header']

    # check shapein
    shapeins = [i.shape_input for i in input]
    if any(s != shapeins[0] for s in shapeins):
        raise ValueError('The pointing matrices do not have the same input '
            "shape: {0}.".format(', '.join(str(shapeins))))
    shapein = shapeins[0]

    # the output is simply a ProjectionOperator instance
    if nmatrices == 1 and not ismapdistributed and not istoddistributed \
       and not packed:
        return ProjectionInMemoryOperator(input[0], units=units, derived_units=
            derived_units, commin=commin, commout=commout, **keywords)

    if packed or ismapdistributed:
        # compute the map mask before this information is lost while packing
        mask_global = Map.ones(shapein, dtype=np.bool8, header=header_global)
        for i in input:
            i.get_mask(out=mask_global)
        for i in input:
            i.pack(mask_global)
        if ismapdistributed:
            shapes = distribute_shapes(mask_global.shape, comm=commin)
            shape = shapes[commin.rank]
            mask = np.empty(shape, bool)
            commin.Reduce_scatter([mask_global, MPI.BYTE], [mask, MPI.BYTE],
                                  [product(s) for s in shapes], op=MPI.BAND)
        else:
            mask = mask_global

    operands = [ProjectionInMemoryOperator(i, units=units, derived_units=
                derived_units, commout=commout, **keywords)
                for i in input]
    result = BlockColumnOperator(operands, axisout=-1)

    if nmatrices > 1:
        def apply_mask(mask):
            mask = np.asarray(mask, np.bool8)
            dest = 0
            if mask.shape != result.shapeout:
                raise ValueError("The mask shape '{0}' is incompatible with tha"
                    "t of the projection operator '{1}'.".format(mask.shape,
                    result.shapeout))
            for p in result.operands:
                n = p.matrix.shape[1]
                p.apply_mask(mask[...,dest:dest+n])
                dest += n
        def get_mask(out=None):
            for p in result.operands:
                out = p.get_mask(out=out)
            return out
        def get_pTp(out=None):
            for p in result.operands:
                out = p.get_pTp(out=out)
            return out
        def get_pTx_pT1(x, out=None, mask=None):
            dest = 0
            for p in result.operands:
                n = p.matrix.shape[1]
                out = p.get_pTx_pT1(x[...,dest:dest+n], out=out, mask=mask)
                dest += n
            return out
        def intersects(out=None):
            raise NotImplementedError('email-me')

        result.apply_mask = apply_mask
        result.get_mask = get_mask
        result.get_pTp = get_pTp
        result.get_pTx_pT1 = get_pTx_pT1
        result.intersects = intersects

    if not istoddistributed and not ismapdistributed and not packed:
        return result

    def not_implemented(out=None):
        raise NotImplementedError('email-me')

    if packed or ismapdistributed:
        def get_mask(out=None):
            if out is not None:
                out &= mask
            else:
                out = mask
            return out
        if ismapdistributed:
            header = scatter_fitsheader(header_global, comm=commin)
            result *= DistributionLocalOperator(mask_global, commin=commin,
                                                attrin={'header':header})
        elif packed:
            result *= PackOperator(mask)
        result.get_mask = get_mask

    if istoddistributed and not ismapdistributed:
        get_mask_closure = result.get_mask
        def get_mask(out=None):
            out = get_mask_closure(out=out)
            commout.Allreduce(MPI.IN_PLACE, [out, MPI.BYTE], op=MPI.BAND)
            return out        
        result *= DistributionIdentityOperator(commout=commout)
        result.get_mask = get_mask

    result.apply_mask = not_implemented
    result.get_pTp = not_implemented
    result.get_pTx_pT1 = not_implemented
    result.intersects = not_implemented

    return result


@real
@linear
class ProjectionBaseOperator(Operator):
    """
    Abstract class for projection operators.

    """
    def direct(self, input, output):
        matrix = self.matrix
        if matrix.size == 0:
            return
        if not output.flags.contiguous:
            if pyoperators.memory.verbose:
                print 'Optimize me: Projection output is not contiguous.'
            output_ = np.empty_like(output)
        else:
            output_ = output
        tmf.pointing_matrix_direct(matrix.ravel().view(np.int64), input.T,
                                   output_.T, matrix.shape[-1])
        if not output.flags.contiguous:
            output[...] = output_

    def transpose(self, input, output, operation=operation_assignment):
        matrix = self.matrix
        if operation is operation_assignment:
            output[...] = 0
        elif operation is not operator.iadd:
            raise ValueError('Invalid reduction operation.')
        if matrix.size == 0:
            return
        if not input.flags.contiguous:
            if pyoperators.memory.verbose:
                print 'Optimize me: Projection.T input is not contiguous.'
            input_ = np.ascontiguousarray(input)
        else:
            input_ = input
        tmf.pointing_matrix_transpose(matrix.ravel().view(np.int64),
                                      input_.T, output.T, matrix.shape[-1])

    def set_attrin(self, attr):
        matrix = self.matrix
        header = matrix.info.get('header', None)
        if header is not None:
            attr['_header'] = header.copy()
        unitout = attr['_unit'] if '_unit' in attr else {}
        if unitout:
            attr['_unit'] = _divide_unit(unitout, self.unit)
        if self.duin is not None:
            attr['_derived_units'] = self.duin

    def set_attrout(self, attr):
        if '_header' in attr:
            del attr['_header']
        unitin = attr['_unit'] if '_unit' in attr else {}
        if unitin:
            attr['_unit'] = _multiply_unit(unitin, self.unit)
        if self.duout is not None:
            attr['_derived_units'] = self.duout

    def apply_mask(self, mask):
        """
        Set pointing matrix values to zero following an input mask of shape
        the operator output shape.

        """
        raise NotImplementedError()

    def get_mask(self, out=None):
        return self.matrix.get_mask(out=out)

    def get_pTp(self, out=None):
        matrix = self.matrix
        npixels = product(matrix.shape_input)
        if out is None:
            out = empty((npixels,npixels), np.bool8, 'for pTp array')
            out[...] = 0
        elif out.dtype != np.bool8:
            raise TypeError('The output ptp argument has an invalid type.')
        elif out.shape != (npixels, npixels):
            raise ValueError('The output ptp argument has an incompatible shape'
                             '.')
        if matrix.size == 0:
            return out
        tmf.pointing_matrix_ptp(matrix.ravel().view(np.int64), out.T,
            matrix.shape[-1], matrix.size // matrix.shape[-1], npixels)
        return out

    def get_pTx_pT1(self, x, out=None, mask=None):
        """
        Return a tuple of two arrays: the transpose of the projection
        applied over the input and over one. In other words, it returns the back
        projection of the input and its coverage.

        """
        matrix = self.matrix
        if out is None:
            shape = matrix.shape_input
            out = Map.zeros(shape), Map.zeros(shape)
            self.set_attrin(out[0].__dict__)
            if hasattr(x, 'unit'):
                out[0].unit = x.unit
        if matrix.size == 0:
            return out
        mask = mask or getattr(x, 'mask', None)
        if mask is not None:
            tmf.backprojection_weight_mask(matrix.ravel().view(np.int64),
                x.ravel(), x.mask.ravel().view(np.int8), out[0].ravel(),
                out[1].ravel(), matrix.shape[-1])
        else:
            tmf.backprojection_weight(matrix.ravel().view(np.int64),
                x.ravel(), out[0].ravel(), out[1].ravel(), matrix.shape[-1])
        return out

    def intersects(self, fitscoords, axis=None, f=None):
        """
        Return True if a map pixel is seen by the pointing matrix.

        """
        matrix = self.matrix
        axes = (1,) + matrix.shape_input[:0:-1]
        index = np.array(fitscoords) - 1
        index = int(np.sum(index * np.cumproduct(axes)))

        shape = matrix.shape
        matrix_ = matrix.ravel().view(np.int64)
        if f is not None:
            out = f(matrix_, index, shape[-1], shape[-2], shape[-3])
        elif axis is None:
            out = tmf.operators.pmatrix_intersects(matrix_, index, shape[-1],
                      shape[-2], shape[-3])
        elif axis == 0 or axis == -3:
            out = tmf.operators.pmatrix_intersects_axis3(matrix_, index,
                      shape[-1], shape[-2], shape[-3])
        elif axis == 1 or axis == -2:
            out = tmf.operators.pmatrix_intersects_axis2(matrix_, index,
                      shape[-1], shape[-2], shape[-3])
        else:
            raise ValueError('Invalid axis.')

        if isinstance(out, np.ndarray):
            out = out.view(bool)
        else:
            out = bool(out)
        return out


@real
@linear
class ProjectionInMemoryOperator(ProjectionBaseOperator):
    """
    Projection operator that stores the pointing matrix in memory.

    Parameters
    ----------
    matrix : PointingMatrix
        The pointing matrix.
    units : unit
        Unit of the pointing matrix. It is used to infer the output unit
        from that of the input.
    derived_units : tuple (derived_units_out, derived_units_in)
        Derived units to be added to the direct or transposed outputs.

    Attributes
    ----------
    header : pyfits.Header
        The map FITS header
    matrix : composite dtype ('value', 'f4'), ('index', 'i4')
        The pointing matrix.

    """
    def __init__(self, matrix, units=None, derived_units=None, **keywords):

        if 'shapein' in keywords:
            raise TypeError("The 'shapein' keyword should not be used.")
        shapein = matrix.shape_input

        if 'shapeout' in keywords:
            raise TypeError("The 'shapeout' keyword should not be used.")
        shapeout = matrix.shape[:-1]

        if units is None:
            units = matrix.info.get('units', None)
        if derived_units is None:
            derived_units = matrix.info.get('derived_units', None)

        if units is None:
            units = ('', '')
        if derived_units is None:
            derived_units = ({}, {})

        unit = _divide_unit(Quantity(1, units[0])._unit,
                            Quantity(1, units[1])._unit)

        Operator.__init__(self, shapein=shapein, shapeout=shapeout,
                          classin=Map, classout=Tod, dtype=float,
                          attrin=self.set_attrin, attrout=self.set_attrout,
                          **keywords)
        self.matrix = matrix
        self.unit = unit
        self.duout, self.duin = derived_units

    def apply_mask(self, mask):
        mask = np.asarray(mask, np.bool8)
        matrix = self.matrix
        if mask.shape != self.shapeout:
            raise ValueError("The mask shape '{0}' is incompatible with that of"
                " the pointing matrix '{1}'.".format(mask.shape, matrix.shape))
        matrix.value.T[...] *= 1 - mask.T


@real
@linear
@square
@inplace
@universal
class ConvolutionTruncatedExponentialOperator(Operator):
    """
    Apply a truncated exponential response to the signal

    Parameters
    ==========
    
    tau: number
        Time constant divided by the signal sampling period.

    """    
    def __init__(self, tau, **keywords):
        """
        """
        if hasattr(tau, 'SI'):
            tau = tau.SI
            if tau.unit != '':
                raise ValueError('The time constant must be dimensionless.')
        self.tau = np.array(tau, dtype=float, ndmin=1)
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.convolution_trexp_direct(input_, ishape[0], ishape[1], istride,
            output_, oshape[1], ostride, self.tau)

    def transpose(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        output_, oshape, ostride = _ravel_strided(output)
        tmf.convolution_trexp_transpose(input_, ishape[0], ishape[1], istride,
            output_, oshape[1], ostride, self.tau)


@real
@linear
@square
@inplace
class DiscreteDifferenceOperator(Operator):
    """
    Discrete difference operator.

    Calculate the nth order discrete difference along given axis.

    """
    def __init__(self, axis=-1, **keywords):
        Operator.__init__(self, dtype=float, **keywords)
        self.axis = axis
        self.set_rule('.T.', self._rule_ddtdd, CompositionOperator)

    def direct(self, input, output):
        diff(input, output, self.axis, comm=self.commin)

    def transpose(self, input, output):
        diffT(input, output, self.axis, comm=self.commin)

    @staticmethod
    def _rule_ddtdd(dT, self):
        return DdTddOperator(self.axis, commin=self.commin)


@real
@symmetric
@inplace
class DdTddOperator(Operator):
    """
    Calculate operator dX.T dX along a given axis.

    """
    def __new__(cls, axis=-1, scalar=1., **keywords):
        if scalar == 0:
            return ZeroOperator(flags='square', **keywords)
        return Operator.__new__(cls)

    def __init__(self, axis=-1, scalar=1., **keywords):
        Operator.__init__(self, dtype=float, **keywords)
        self.axis = axis
        self.scalar = scalar
        self.set_rule('{HomothetyOperator}.', lambda o,s: DdTddOperator(
                      self.axis, o.data * s.scalar), CompositionOperator)

    def direct(self, input, output):
        diffTdiff(input, output, self.axis, self.scalar, comm=self.commin)

    def __str__(self):
        s = (str(self.scalar) + ' ') if self.scalar != 1 else ''
        s += super(DdTddOperator, self).__str__()
        return s

@real
@linear
class DistributionLocalOperator(Operator):
    """
    Scatter a distributed map to different MPI processes under the control of a
    local non-distributed mask.

    """
    def __init__(self, mask, operator=MPI.SUM, commin=MPI.COMM_WORLD,
                 **keywords):
        commout = commin
        shapeout = mask.size - int(np.sum(mask))
        shapein = distribute_shape(mask.shape, comm=commin)
        Operator.__init__(self, shapein=shapein, shapeout=shapeout,
                          commin=commin, commout=commout, **keywords)
        self.mask = mask
        self.chunk_size = product(mask.shape[1:])
        self.operator = operator

    def direct(self, input, output):
        ninput = input.size
        noutput = output.size
        input = input.ravel() if ninput > 0 else np.array(0.)
        output = output.ravel() if noutput > 0 else np.array(0.)
        status = tmf.operators_mpi.allscatterlocal(input, self.mask.view(
            np.int8).T, output, self.chunk_size, self.commin.py2f(),
            ninput=ninput, noutput=noutput)
        if status == 0: return
        if status < 0:
            rank = self.commin.rank
            if status == -1:
                raise ValueError('Invalid mask on node {0}.'.format(rank))
            if status == -2:
                raise ValueError('Invalid input on node {0}.'.format(rank))
            if status == -3:
                raise RuntimeError('Transmission failure {0}.'.format(rank))
            assert False
        raise MPI.Exception(status)

    def transpose(self, input, output):
        output[...] = 0
        status = tmf.operators_mpi.allreducelocal(input.T, self.mask.view(
            np.int8).T, output.T, self.chunk_size, self.operator.py2f(),
            self.commin.py2f())
        if status == 0: return
        if status < 0:
            rank = self.commin.rank
            if status == -1:
                raise ValueError('Invalid mask on node {0}.'.format(rank))
            assert False
        raise MPI.Exception(status)


@real
@linear
@inplace
class PackOperator(Operator):
    """
    Convert an nd array in a 1d map, under the control of a mask.
    The elements for which the mask is true are discarded.

    """
    def __init__(self, mask, **keywords):
        mask = np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=mask.shape, shapeout=np.sum(mask == 0),
                          dtype=float, **keywords)
        self.mask = mask
        self.set_rule('.T', lambda s: UnpackOperator(s.mask))
        self.set_rule('.{UnpackOperator}', self._rule_left_unpack, CompositionOperator)
        self.set_rule('{UnpackOperator}.', self._rule_right_unpack, CompositionOperator)

    def direct(self, input, output):
        mask = self.mask.view(np.int8).ravel()
        if self.isalias(input, output):
            tmf.operators.pack_inplace(input.ravel(), mask, input.size)
        else:
            tmf.operators.pack_outplace(input.ravel(), mask, output)

    @staticmethod
    def _rule_left_unpack(self, op):
        if self.mask.shape != op.mask.shape:
            return
        if np.any(self.mask != op.mask):
            return
        return IdentityOperator()

    @staticmethod
    def _rule_right_unpack(op, self):
        if self.mask.shape != op.mask.shape:
            return
        if np.any(self.mask != op.mask):
            return
        return MaskOperator(self.mask)


@real
@linear
@inplace
class UnpackOperator(Operator):
    """
    Convert an nd array in a 1d map, under the control of a mask.
    The elements for which the mask is true are set to zero.

    """
    def __init__(self, mask, **keywords):
        mask = np.array(mask, np.bool8, copy=False)
        Operator.__init__(self, shapein=np.sum(mask == 0), shapeout=mask.shape,
                          dtype=float, **keywords)
        self.mask = mask
        self.set_rule('.T', lambda s: PackOperator(s.mask))

    def direct(self, input, output):
        mask = self.mask.view(np.int8).ravel()
        if self.isalias(input, output):
            tmf.operators.unpack_inplace(output.ravel(), mask, input.size)
        else:
            tmf.operators.unpack_outplace(input, mask, output.ravel())


@real
@linear
@square
@inplace
class ShiftOperator(Operator):

    def __init__(self, n, axis=None, **keywords):
        Operator.__init__(self, dtype=float, **keywords)
        if axis is None:
            axis = -1 if isscalar(n) else range(-len(n), 0)
        self.axis = (axis,) if isscalar(axis) else tuple(axis)
        self.n = (n,) * len(self.axis) if isscalar(n) else \
                 tuple([np.array(m,int) for m in n])
        if len(self.axis) != len(self.n):
            raise ValueError('There is a mismatch between the number of axes an'
                             'd offsets.')

    def direct(self, input, output):
        shift(input, output, self.n[0], self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            shift(output, output, n, axis)

    def transpose(self, input, output):
        shift(input, output, -self.n[0], self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            shift(output, output, -n, axis)


@orthogonal
@inplace
class RollOperator(Operator):
    
    def __init__(self, n, axis=None, **keywords):
        Operator.__init__(self, **keywords)
        if axis is None:
            axis = -1 if isscalar(n) else range(-len(n), 0)
        self.axis = (axis,) if isscalar(axis) else tuple(axis)
        self.n = (n,) * len(self.axis) if isscalar(n) else tuple(n)
        if len(self.axis) != len(self.n):
            raise ValueError('There is a mismatch between the number of axes an'
                             'd offsets.')

    def direct(self, input, output):
        output[...] = np.roll(input, self.n[0], axis=self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            output[...] = np.roll(output, n, axis=axis)

    def transpose(self, input, output):
        output[...] = np.roll(input, -self.n[0], axis=self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            output[...] = np.roll(output, -n, axis=axis)


@real
@orthogonal
@inplace
class FftHalfComplexOperator(Operator):
    """
    Performs real-to-half-complex fft

    """
    def __init__(self, size, fftw_flags=['measure', 'unaligned'], **keywords):
        size = int(size)
        array1 = np.empty(size, dtype=float)
        array2 = np.empty(size, dtype=float)
        fplan = fftw3.Plan(array1, array2, direction='forward', flags= \
            fftw_flags, realtypes=['halfcomplex r2c'], nthreads=1)
        bplan = fftw3.Plan(array1, array2, direction='backward', flags=\
            fftw_flags, realtypes=['halfcomplex c2r'], nthreads=1)
        ifplan = fftw3.Plan(array1, direction='forward', flags=fftw_flags,
            realtypes=['halfcomplex r2c'], nthreads=1)
        ibplan = fftw3.Plan(array1, direction='backward', flags=fftw_flags,
            realtypes=['halfcomplex c2r'], nthreads=1)
        self.size = size
        self.fftw_flags = fftw_flags
        self.fplan = fplan
        self.bplan = bplan
        self.ifplan = ifplan
        self.ibplan = ibplan
        Operator.__init__(self, **keywords)

    def direct(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        if self.isalias(input, output):
            tmf.fft_plan_inplace(input_, ishape[0], ishape[1], istride,
                                 self.ifplan._get_parameter())
            return
        output_, oshape, ostride = _ravel_strided(output)
        tmf.fft_plan_outplace(input_, ishape[0], ishape[1], istride, output_,
                     oshape[1], ostride, self.fplan._get_parameter())

    def transpose(self, input, output):
        input_, ishape, istride = _ravel_strided(input)
        if self.isalias(input, output):
            tmf.fft_plan_inplace(input_, ishape[0], ishape[1], istride,
                                 self.ibplan._get_parameter())
        else:
            output_, oshape, ostride = _ravel_strided(output)
            tmf.fft_plan_outplace(input_, ishape[0], ishape[1], istride,
                output_, oshape[1], ostride, self.bplan._get_parameter())
        output /= self.size

    def validatein(self, shapein):
        if shapein[-1] != self.size:
            raise ValueError("Invalid input dimension '{0}'. Expected dimension"
                             " is '{1}'".format(shapein[-1], self.size))


def _ravel_strided(array):
    # array_ = array.reshape((-1,array.shape[-1]))
    #XXX bug in numpy 1.5.1: reshape with -1 leads to bogus strides
    array_ = array.reshape(reduce(lambda x,y:x*y, array.shape[0:-1], 1),
                           array.shape[-1])
    n2, n1 = array_.shape
    if array_.flags.c_contiguous:
        return array_.ravel(), array_.shape, n1
    if not array_[0,:].flags.c_contiguous:
        raise RuntimeError()
    s2, s1 = array_.strides
    s2 //= s1
    base = array
    while not base.flags.c_contiguous:
        base = base.base
    start = (array.__array_interface__['data'][0] - \
             base.__array_interface__['data'][0]) // s1
    stop = start + (n2 - 1) * s2 + n1
    flat = base.ravel()[start:stop]
    return flat, array_.shape, s2
