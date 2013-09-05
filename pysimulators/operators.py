from __future__ import division

import functools
import inspect
import numpy as np
import operator
import pyoperators
import scipy.constants

from pyoperators import (Operator, BlockDiagonalOperator, CompositionOperator,
                         DiagonalOperator, DiagonalNumexprOperator,
                         MultiplicationOperator)
from pyoperators.decorators import linear, orthogonal, real, inplace
from pyoperators.memory import empty
from pyoperators.utils import (isscalar, operation_assignment, product,
                               tointtuple)

from . import _flib as flib
from .datatypes import FitsArray, Map
from .quantities import Quantity, _divide_unit, _multiply_unit
from .wcsutils import create_fitsheader

__all__ = [
    'BlackBodyOperator',
    'PointingMatrix',
    'PowerLawOperator',
    'ProjectionInMemoryOperator',
    'ProjectionOnFlyOperator',
    'RollOperator',
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
    >>> op = MyOp([1, 2, 3], shapein=3, partitionin=(1, 1, 1))
    >>> op.todense()
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])

    >>> [o.value for o in op.operands]
    [1, 2, 3]

    """
    # the following way to extract keywords is unnecessary in Python3 (PEP3102)
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
            ns = [0 if isscalar(a) else len(a) for i, a in enumerate(args)
                  if class_args[i] in partition_args] + \
                 [0 if isscalar(v) else len(v) for k, v in keywords.items()
                  if k in partition_args]

            n2 = 0 if len(ns) == 0 else max(ns)
            if any(n not in (0, n2) for n in ns):
                raise ValueError('The partition variables do not have the same'
                                 ' number of elements.')

            # bail if no partitioning is found
            n = max(n1, n2)
            if n == 0:
                cls.__init_original__(self, *args, **keywords)
                return

            # check the two methods are compatible
            if n1 != 0 and n2 != 0 and n1 != n2:
                raise ValueError('The specified partitioning is incompatible w'
                                 'ith the number of elements in the partition '
                                 'variables.')

            # Implicit partition
            if partitionin is None:
                if axisin is not None:
                    partitionin = n * (None,)
                else:
                    partitionin = n * (1,)

            # dispatch arguments
            n = len(partitionin)
            argss = tuple(tuple(a[i] if class_args[j] in partition_args and
                          not isscalar(a) else a for j, a in enumerate(args))
                          for i in range(n))
            keyss = tuple(dict((k, v[i]) if k in partition_args and not
                          isscalar(v) else (k, v) for k, v in keywords.items())
                          for i in range(n))

            # the input shapein/out describe the BlockDiagonalOperator
            def reshape(s, p, a, na):
                s = list(tointtuple(s))
                if a is not None:
                    s[a] = p
                else:
                    np.delete(s, na)
                return tuple(s)

            if 'shapein' in keywords:
                shapein = keywords['shapein']
                for keys, p in zip(keyss, partitionin):
                    keys['shapein'] = reshape(shapein, p, axisin, new_axisin)
            if 'shapeout' in keywords:
                shapeout = keywords['shapeout']
                for keys, p in zip(keyss, partitionin):
                    keys['shapeout'] = reshape(shapeout, p, axisin, new_axisin)

            # instantiate the partitioned operators
            ops = [cls.__new__(type(self), *a, **k) for a, k in
                   zip(argss, keyss)]
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


@real
class BlackBodyOperator(DiagonalOperator):
    """
    Diagonal operator whose normalised diagonal values are given by the Planck
    equation, optionally modified by a power-law emissivity of given slope.

    In other words, assuming that the set of spectra fnu[i](nu) is proportional
    to the black body law of temperature T[i], and that the flux density of the
    spectra at the reference frequency nu0 is given by fnu_nu0[i], then the
    flux densities fnu_nu[i] at a frequency nu can be determined by using the
    BlackBodyOperator:
        bb = BlackBodyOperator(T, frequency=nu, frequency0=nu0)
        fnu_nu = bb(fnu_nu0)

    Example
    -------
    >>> from pysimulators import gaussian
    >>> T = gaussian((128, 128), sigma=32)  # Temperature map
    >>> ws = np.arange(90, 111) * 1e-6
    >>> bb = [BlackBodyOperator(T, wavelength=w, wavelength0=100e-6)
              for w in ws]

    """

    def __init__(self, temperature, frequency=None, frequency0=None,
                 wavelength=None, wavelength0=None, beta=0, **keywords):
        """
        Parameters
        ----------
        temperature : array-like or scalar of float
            Black body temperature, in Kelvin.
        frequency : scalar float
            Frequency, in Hertz, at which black body values will be computed.
        frequency0 : scalar float
            Reference frequency, in Hertz. If the frequency keyword is equal
            to frequency0, the operator is the identity operator.
        wavelength : scalar float
            Wavelength, in meters, at which black body values will be computed.
            (alternative to the frequency keyword.)
        wavelength0 : scalar float
            Reference wavelength, in meters. If the wavelength keyword is equal
            to wavelength0, the operator is the identity operator.
            (alternative to the frequency0 keyword.)
        beta : array-like or scalar of float
            Slope of the emissivity (the spectrum is multiplied by nu**beta)
            The default value is 0 (non-modified black body).

        """
        temperature = np.asarray(temperature, float)
        if frequency is None and wavelength is None:
            raise ValueError('The operating frequency or wavelength is not spe'
                             'cified.')
        if frequency0 is None and wavelength0 is None:
            raise ValueError('The reference frequency or wavelength is not spe'
                             'cified.')
        if frequency is not None and wavelength is not None:
            raise ValueError('Ambiguous operating frequency / wavelength.')
        if frequency0 is not None and wavelength0 is not None:
            raise ValueError('Ambiguous reference frequency / wavelength.')

        c = scipy.constants.c
        h = scipy.constants.h
        k = scipy.constants.k

        if frequency is not None:
            nu = np.asarray(frequency, float)
        else:
            nu = c / np.asarray(wavelength, float)
        if nu.ndim != 0:
            raise TypeError('The operating frequency or wavelength is not a sc'
                            'alar.')
        if frequency0 is not None:
            nu0 = np.asarray(frequency0, float)
        else:
            nu0 = c / np.asarray(wavelength0, float)
        if nu.ndim != 0:
            raise TypeError('The operating frequency or wavelength is not a sc'
                            'alar.')
        beta = float(beta)
        data = (nu / nu0)**(3 + beta) * \
            np.expm1(h * nu0 / (k * temperature)) / \
            np.expm1(h * nu / (k * temperature))
        DiagonalOperator.__init__(self, data, **keywords)
        self.temperature = temperature
        self.beta = beta
        self.frequency = frequency
        self.frequency0 = frequency0
        self.wavelength = wavelength
        self.wavelength0 = wavelength0


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
        x = np.asarray(x, float)
        if x.ndim != 0:
            raise TypeError('The input is not a scalar.')
        x0 = np.asarray(x0, float)
        if x0.ndim != 0:
            raise TypeError('The reference input is not a scalar.')
        alpha = np.asarray(alpha, float)
        if alpha.ndim > 0:
            keywords['shapein'] = alpha.shape
        scalar = np.asarray(scalar, float)
        if scalar.ndim != 0:
            raise TypeError('The scalar coefficient is not a scalar.')
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        global_dict = {'x': x, 'x0': x0, 's': scalar}
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
    def _rule_block(self, op, shape, partition, axis, new_axis,
                    func_operation):
        return DiagonalOperator._rule_block(
            self, op, shape, partition, axis, new_axis, func_operation,
            self.x, self.x0, scalar=self.scalar)


class PointingMatrix(FitsArray):
    default_dtype = np.dtype([('index', np.int32), ('value', np.float32)])

    def __new__(cls, array, shape_input, copy=True, ndmin=0):
        result = FitsArray.__new__(cls, array, copy=copy, ndmin=ndmin).view(
            cls.default_dtype, cls)
        result.header = create_fitsheader(result.shape[::-1])
        result.shape_input = shape_input
        return result

    def __array_finalize__(self, array):
        FitsArray.__array_finalize__(self, array)
        self.shape_input = getattr(array, 'shape_input', None)

    @classmethod
    def empty(cls, shape, shape_input, verbose=True):
        buffer = empty(shape, cls.default_dtype, verbose=verbose,
                       description='for the pointing matrix')
        return PointingMatrix(buffer, shape_input, copy=False)

    @classmethod
    def ones(cls, shape, shape_input, verbose=True):
        buffer = empty(shape, cls.default_dtype, verbose=verbose,
                       description='for the pointing matrix')
        buffer['index'] = -1
        buffer['value'] = 1
        return PointingMatrix(buffer, shape_input, copy=False)

    @classmethod
    def zeros(cls, shape, shape_input, verbose=True):
        buffer = empty(shape, cls.default_dtype, verbose=verbose,
                       description='for the pointing matrix')
        buffer['index'] = -1
        buffer['value'] = 0
        return PointingMatrix(buffer, shape_input, copy=False)

    def isvalid(self):
        """
        Return true if the indices in the pointing matrix are greater or equal
        to -1 and lesser than the number of pixels in the map.

        """
        if self.size == 0:
            return True
        npixels = product(self.shape_input)
        result = flib.pointingmatrix.isvalid(
            self.ravel().view(np.int64), self.shape[-1],
            product(self.shape[:-1]), npixels)
        return bool(result)

    def pack(self, mask):
        """
        Inplace renumbering of the pixel indices in the pointing matrix,
        discarding masked or unobserved pixels.

        """
        if self.size == 0:
            return
        flib.pointingmatrix.pack(self.ravel().view(np.int64), mask.view(
            np.int8).T, self.shape[-1], self.size // self.shape[-1])
        self.shape_input = tointtuple(mask.size - np.sum(mask))

    def get_mask(self, out=None):
        if out is None:
            out = empty(self.shape_input, np.bool8, description='as new mask')
            out[...] = True
        elif out.dtype != bool:
            raise TypeError('The output mask argument has an invalid type.')
        elif out.shape != self.shape_input:
            raise ValueError('The output mask argument has an incompatible sha'
                             'pe.')
        if self.size == 0:
            return out
        flib.pointingmatrix.mask(self.ravel().view(np.int64), out.view(
            np.int8).T, self.shape[-1], self.size // self.shape[-1])
        return out


@real
@linear
class ProjectionBaseOperator(Operator):
    """
    Abstract class for projection operators.

    """
    def __init__(self, units=None, derived_units=None, attrin={}, attrout={},
                 **keywords):

        if units is None:
            units = ('', '')
        if derived_units is None:
            derived_units = ({}, {})

        unit = _divide_unit(Quantity(1, units[0])._unit,
                            Quantity(1, units[1])._unit)

        Operator.__init__(self, dtype=float, attrin=self.set_attrin,
                          attrout=self.set_attrout, **keywords)
        self._attrin = attrin
        self._attrout = attrout
        self.unit = unit
        self.duout, self.duin = derived_units

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
        flib.pointingmatrix.direct(matrix.ravel().view(np.int64), input.T,
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
        flib.pointingmatrix.transpose(matrix.ravel().view(np.int64),
                                      input_.T, output.T, matrix.shape[-1])

    def set_attrin(self, attr):
        if '_header' in attr:
            del attr['_header']
        unitout = attr['_unit'] if '_unit' in attr else {}
        if unitout:
            attr['_unit'] = _divide_unit(unitout, self.unit)
        if self.duin is not None:
            attr['_derived_units'] = self.duin
        attr.update(self._attrin)

    def set_attrout(self, attr):
        if '_header' in attr:
            del attr['_header']
        unitin = attr['_unit'] if '_unit' in attr else {}
        if unitin:
            attr['_unit'] = _multiply_unit(unitin, self.unit)
        if self.duout is not None:
            attr['_derived_units'] = self.duout
        attr.update(self._attrout)

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
            out = empty((npixels, npixels), bool, description='for pTp array')
            out[...] = 0
        elif out.dtype != np.bool8:
            raise TypeError('The output ptp argument has an invalid type.')
        elif out.shape != (npixels, npixels):
            raise ValueError('The output ptp argument has an incompatible shap'
                             'e.')
        if matrix.size == 0:
            return out
        flib.pointingmatrix.ptp(matrix.ravel().view(np.int64), out.T,
                                matrix.shape[-1],
                                matrix.size // matrix.shape[-1], npixels)
        return out

    def get_pTx_pT1(self, x, out=None, mask=None):
        """
        Return a tuple of two arrays: the transpose of the projection
        applied over the input and over one. In other words, it returns
        the back projection of the input and its coverage.

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
            flib.pointingmatrix.backprojection_weight_mask(matrix.ravel().view(
                np.int64), x.ravel(), x.mask.ravel().view(np.int8),
                out[0].ravel(), out[1].ravel(), matrix.shape[-1])
        else:
            flib.pointingmatrix.backprojection_weight(matrix.ravel().view(
                np.int64), x.ravel(), out[0].ravel(), out[1].ravel(),
                matrix.shape[-1])
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
            out = flib.pointingmatrix.intersects(
                matrix_, index, shape[-1], shape[-2], shape[-3])
        elif axis == 0 or axis == -3:
            out = flib.pointingmatrix.intersects_axis3(
                matrix_, index, shape[-1], shape[-2], shape[-3])
        elif axis == 1 or axis == -2:
            out = flib.pointingmatrix.intersects_axis2(
                matrix_, index, shape[-1], shape[-2], shape[-3])
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

    Attributes
    ----------
    matrix : composite dtype ('value', 'f4'), ('index', 'i4')
        The pointing matrix.

    """
    def __init__(self, matrix, units=None, derived_units=None, **keywords):
        """
        Parameters
        ----------
        matrix : PointingMatrix
            The pointing matrix.
        units : unit
            Unit of the pointing matrix. It is used to infer the output unit
            from that of the input.
        derived_units : tuple (derived_units_out, derived_units_in)
            Derived units to be added to the direct or transposed outputs.

        """
        if 'shapein' in keywords:
            raise TypeError("The 'shapein' keyword should not be used.")
        shapein = matrix.shape_input

        if 'shapeout' in keywords:
            raise TypeError("The 'shapeout' keyword should not be used.")
        shapeout = matrix.shape[:-1]

        ProjectionBaseOperator.__init__(self, shapein=shapein,
                                        shapeout=shapeout, units=units,
                                        derived_units=derived_units,
                                        **keywords)
        self.matrix = matrix
        self.set_rule('.T.', self._rule_ptp, CompositionOperator)
        self.set_rule('{DiagonalOperator}.', self._rule_diagonal,
                      CompositionOperator)

    def apply_mask(self, mask):
        mask = np.asarray(mask, np.bool8)
        matrix = self.matrix
        if mask.shape != self.shapeout:
            raise ValueError("The mask shape '{0}' is incompatible with that o"
                             "f the pointing matrix '{1}'.".format(mask.shape,
                             matrix.shape))
        matrix.value.T[...] *= 1 - mask.T

    @staticmethod
    def _rule_ptp(pt, p):
        if p.matrix.shape[-1] != 1:
            return
        cov = np.zeros(p.shapein)
        flib.pointingmatrix.ptp_one(p.matrix.ravel().view(np.int64),
                                    cov.ravel(), p.matrix.size, cov.size)
        return DiagonalOperator(cov)

    @staticmethod
    def _rule_diagonal(d, self):
        # rightward broadcasting can fail with structured arrays (np 1.7)
        if self.matrix.shape[-1] != 1 or d.broadcast == 'rightward':
            return
        matrix = self.matrix.copy()
        result = ProjectionInMemoryOperator(matrix)
        matrix.value *= d.get_data()
        return result


@real
@linear
class ProjectionOnFlyOperator(ProjectionBaseOperator):
    """
    Projection operator that recomputes the pointing matrix on the fly.

    ProjectionOnFlyOperator instances are grouped (usually organised as a block
    column operator) and they share a dictionary, which holds one pointing
    matrix and a hashable to identify with which element of the group this
    pointing matrix is associated. If another element requests the
    computation of its pointing matrix, the common dict is updated accordingly,
    and the pointing matrix is substituted.

    """
    def __init__(self, place_holder, id, func, units=None, derived_units=None,
                 **keywords):
        """
        Parameters
        ----------
        place_holder : dict
            dict whose keys 'matrix' contains the pointing matrix and 'id' the
            hashable to identify with which operator it is associated.
        id : hashable
            The on-fly projection operator identifier.
        func : callable
            Function used to compute the pointing matrix on the fly:
                 matrix = func(id)
        units : unit
            Unit of the pointing matrix. It is used to infer the output unit
            from that of the input.
        derived_units : tuple (derived_units_out, derived_units_in)
            Derived units to be added to the direct or transposed outputs.

        """
        if 'matrix' not in place_holder:
            place_holder['matrix'] = None
            place_holder['id'] = None
        self.place_holder = place_holder
        self.id = id
        self.func = func
        ProjectionBaseOperator.__init__(self, units=units,
                                        derived_units=derived_units,
                                        **keywords)

    @property
    def matrix(self):
        if self.place_holder['id'] is self.id:
            return self.place_holder['matrix']
        self.place_holder['matrix'] = None
        matrix = self.func(self.id)
        self.place_holder['id'], self.place_holder['matrix'] = self.id, matrix
        return matrix


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
            raise ValueError('There is a mismatch between the number of axes a'
                             'nd offsets.')

    def direct(self, input, output):
        output[...] = np.roll(input, self.n[0], axis=self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            output[...] = np.roll(output, n, axis=axis)

    def transpose(self, input, output):
        output[...] = np.roll(input, -self.n[0], axis=self.axis[0])
        for n, axis in zip(self.n, self.axis)[1:]:
            output[...] = np.roll(output, -n, axis=axis)
