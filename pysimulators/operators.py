#encoding: utf-8
from __future__ import division

import functools
import inspect
import numpy as np
import operator
import pyoperators
import scipy.constants

from pyoperators import (
    Operator, BlockDiagonalOperator, Cartesian2SphericalOperator,
    CompositionOperator, ConstantOperator, DenseOperator,
    DenseBlockDiagonalOperator, DiagonalOperator, DiagonalNumexprOperator,
    HomothetyOperator, MultiplicationOperator, Spherical2CartesianOperator)
from pyoperators.decorators import linear, orthogonal, real, inplace
from pyoperators.memory import empty, ones, zeros
from pyoperators.utils import (
    float_dtype, isscalarlike, operation_assignment, product, strenum,
    tointtuple)

from . import _flib as flib
from .datatypes import FitsArray, Map
from .quantities import Quantity, _divide_unit, _multiply_unit
from .sparse import FSRMatrix, FSRRotation3dMatrix, SparseOperator
from .wcsutils import create_fitsheader

__all__ = [
    'BlackBodyOperator',
    'CartesianEquatorial2GalacticOperator',
    'CartesianGalactic2EquatorialOperator',
    'CartesianEquatorial2HorizontalOperator',
    'CartesianHorizontal2EquatorialOperator',
    'PointingMatrix',
    'PowerLawOperator',
    'ProjectionOperator',
    'ProjectionInMemoryOperator',
    'ProjectionOnFlyOperator',
    'RollOperator',
    'SphericalEquatorial2GalacticOperator',
    'SphericalGalactic2EquatorialOperator',
    'SphericalEquatorial2HorizontalOperator',
    'SphericalHorizontal2EquatorialOperator',
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
            ns = [0 if isscalarlike(a) else len(a) for i, a in enumerate(args)
                  if class_args[i] in partition_args] + \
                 [0 if isscalarlike(v) else len(v) for k, v in keywords.items()
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
                                not isscalarlike(a) else a
                                for j, a in enumerate(args))
                          for i in range(n))
            keyss = tuple(dict((k, v[i]) if k in partition_args and
                               not isscalarlike(v) else (k, v)
                               for k, v in keywords.items())
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
        alpha = np.asarray(alpha, float)
        if alpha.ndim > 0:
            keywords['shapein'] = alpha.shape
        x = np.asarray(x, float)
        if x.ndim != 0:
            raise TypeError('The input is not a scalar.')
        x0 = np.asarray(x0, float)
        if x0.ndim != 0:
            raise TypeError('The reference input is not a scalar.')
        scalar = np.asarray(scalar, float)
        if scalar.ndim != 0:
            raise TypeError('The scalar coefficient is not a scalar.')
        if 'dtype' not in keywords:
            keywords['dtype'] = float
        global_dict = {'x': x, 'x0': x0, 's': scalar}
        DiagonalNumexprOperator.__init__(self, alpha, 's * (x / x0) ** alpha',
                                         global_dict, var='alpha', **keywords)
        self.alpha = alpha
        self.x = x
        self.x0 = x0
        self.scalar = scalar
        self.set_rule((HomothetyOperator, '.'),
                      lambda o, s: PowerLawOperator(
                          s.alpha, s.x, s.x0, o.data * s.scalar),
                      CompositionOperator)

    @staticmethod
    def _rule_block(self, op, shape, partition, axis, new_axis,
                    func_operation):
        return DiagonalOperator._rule_block(
            self, op, shape, partition, axis, new_axis, func_operation,
            self.x, self.x0, scalar=self.scalar)

    def __str__(self):
        return u'powerlaw(..., \u03B1={0}, x/x0={1})'\
            .encode('utf-8') \
            .format(self.alpha, self.x / self.x0)


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
        self.set_rule('T,.', self._rule_ptp, CompositionOperator)
        self.set_rule((DiagonalOperator, '.'), self._rule_diagonal,
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


class ProjectionOperator(SparseOperator):
    """
    Projection operator. It is a SparseOperator with convenience methods.

    Attributes
    ----------
    matrix : FSRMatrix or FSRRotation3dMatrix
        The projection sparse matrix.

    """
    def __init__(self, arg, **keywords):
        if not isinstance(arg, (FSRMatrix, FSRRotation3dMatrix)):
            raise TypeError('The input sparse matrix type is invalid.')
        SparseOperator.__init__(self, arg, **keywords)

    def canonical_basis_in_kernel(self, out=None,
                                  operation=operation_assignment):
        """
        Return a boolean array whose values are True if the component of
        the canonical basis is in the operator's kernel.

        """
        data = self.matrix.data
        shape = self.shapein
        if isinstance(self.matrix, FSRRotation3dMatrix):
            shape = shape[:-1]
        if out is None:
            if operation is not operation_assignment:
                raise ValueError(
                    'The array for inplace operation is not specified')
            out = empty(shape, bool)
        elif out.dtype != bool:
            raise TypeError("The keyword 'out' has an invalid type.")
        elif out.shape != shape:
            raise ValueError("The keyword 'out' has an incompatible shape.")
        if data.size == 0:
            operation(out, True)
            return out

        i, m = [str(_.dtype.itemsize) for _ in (data.index, self.matrix)]
        f = 'fsr{0}_kernel_i{1}_m{2}'.format(
            '' if isinstance(self.matrix, FSRMatrix) else '_rot3d', i, m)
        n = product(shape)
        if hasattr(flib.operators, f):
            if operation in (operation_assignment, operator.iand,
                             operator.imul):
                if operation is operation_assignment:
                    out[...] = True
                kernel = out
            else:
                kernel = np.ones_like(out)

            func = getattr(flib.operators, f)
            func(data.view(np.int8).ravel(), kernel.view(np.int8).ravel(),
                 self.matrix.ncolmax, data.shape[0])
            if operation in (operation_assignment, operator.iand,
                             operator.imul):
                return out
        else:
            if isinstance(self.matrix, FSRMatrix):
                indices = data.index[data.value != 0]
            else:
                indices = data.index[data.r11 != 0]
            kernel = np.histogram(indices, n, range=(0, n))[0] == 0
        operation(out, kernel)
        return out

    def pT1(self, out=None, operation=operation_assignment):
        """
        Return the transpose of the projection over one. In other words,
        it returns the projection coverage.

        """
        if out is None:
            shape = self.shapein
            if isinstance(self.matrix, FSRRotation3dMatrix):
                shape = shape[:-1]
            out = zeros(shape, self.dtype)
        elif not isinstance(out, np.ndarray):
            raise TypeError('The output keyword is not an ndarray.')
        elif out.dtype != self.dtype:
            raise TypeError('The output array has a dtype incompatible with th'
                            'at of the input timeline.')
        elif operation is operation_assignment:
            out[...] = 0
        if operation not in (operation_assignment, operator.iadd):
            raise ValueError('Invalid reduction operation.')

        i, m, v = [str(_.dtype.itemsize) for _ in (self.matrix.data.index,
                                                   self.matrix, self)]
        f = 'fsr{0}_pt1_i{1}_m{2}_v{3}'.format(
            '' if isinstance(self.matrix, FSRMatrix) else '_rot3d', i, m, v)
        if not hasattr(flib.operators, f):
            if isinstance(self.matrix, FSRMatrix):
                self.T(ones(self.shapeout, self.dtype), out=out,
                       operation=operation)
            else:
                x_ = ones(self.shapeout, self.dtype)
                if operation is operator.iadd:
                    out_ = zeros(self.shapein, self.dtype)
                    out_[..., 0] = out
                else:
                    out_ = empty(self.shapein, self.dtype)
                self.T(x_, out=out_, operation=operation)
                out[...] = out_[..., 0]
        else:
            func = getattr(flib.operators, f)
            func(self.matrix.data.view(np.int8).ravel(), out.ravel(),
                 self.matrix.ncolmax, self.matrix.data.shape[0])
        return out

    def pTx_pT1(self, x, out=None, operation=operation_assignment):
        """
        Return a tuple of two arrays: the transpose of the projection
        applied over the input and over one. In other words, it returns
        the back projection of the input and its coverage.

        """
        x = np.asarray(x)
        shapein = self.shapeout
        shapeout = self.shapein
        if isinstance(self.matrix, FSRRotation3dMatrix):
            shapein = shapein[:-1]
            shapeout = shapeout[:-1]
        if x.shape != shapein:
            raise ValueError("Invalid input shape '{0}'. Expected shape is '{1"
                             "}'.".format(shapein))
        if out is None:
            pTx = zeros(shapeout, self.dtype)
            pT1 = zeros(shapeout, self.dtype)
        elif not isinstance(out, (list, tuple)) or len(out) != 2:
            raise TypeError('The out keyword is not a 2-tuple.')
        elif out[0].dtype != x.dtype or out[1].dtype != x.dtype:
            raise TypeError('The output arrays have a dtype incompatible with '
                            'that of the input timeline.')
        else:
            pTx, pT1 = out
            if operation is operation_assignment:
                pTx[...] = 0
                pT1[...] = 0
        if operation not in (operation_assignment, operator.iadd):
            raise ValueError('Invalid reduction operation.')

        i, m, v = [str(_.dtype.itemsize) for _ in (self.matrix.data.index,
                                                   self.matrix, x)]
        f = 'fsr{0}_ptx_pt1_i{1}_m{2}_v{3}'.format(
            '' if isinstance(self.matrix, FSRMatrix) else '_rot3d', i, m, v)
        if not hasattr(flib.operators, f):
            if isinstance(self.matrix, FSRMatrix):
                self.T(x, out=pTx, operation=operator.iadd)
                self.T(ones(x.shape, self.dtype), out=pT1, operation=operation)
            else:
                x_ = ones(self.shapeout, x.dtype)
                if operation is operator.iadd:
                    out_ = zeros(self.shapein, self.dtype)
                    out_[..., 0] = pT1
                else:
                    out_ = empty(self.shapein, self.dtype)
                self.T(x_, out=out_, operation=operation)
                pT1[...] = out_[..., 0]
                if operation is operator.iadd:
                    out_[..., 0] = pTx
                x_[..., 0] = x                    
                self.T(x_, out=out_, operation=operation)
                pTx[...] = out_[..., 0]
        else:
            func = getattr(flib.operators, f)
            func(self.matrix.data.view(np.int8).ravel(), x.ravel(),
                 pTx.ravel(), pT1.ravel(), self.matrix.ncolmax)
        return pTx, pT1

    def restrict(self, mask, n=None):
        """
        Restrict the projection to a subspace defined by a mask
        (True means that the element is kept). Indices are renumbered in-place.

        """
        idtype = self.matrix.data.index.dtype
        mask = np.asarray(mask)
        expected = self.shapein[:-1] \
                   if isinstance(self.matrix, FSRRotation3dMatrix) \
                   else self.shapein
        if mask.shape != expected:
            raise ValueError("Invalid shape '{}'. Expected value is '{}'.".
                             format(mask.shape, expected))
        if mask.dtype == bool:
            if n is None:
                n = np.sum(mask)
            new_index = empty(mask.shape, idtype)
            new_index[...] = -1
            new_index[mask] = np.arange(n, dtype=idtype)
        elif n is None:
            raise ValueError('The size of the restriction is not specified.')
        if isinstance(self.matrix, FSRRotation3dMatrix):
            shapein = (n, 3)
        else:
            shapein = n
        undef = self.matrix.data.index < 0
        self.matrix.data.index = new_index[self.matrix.data.index]
        self.matrix.data.index[undef] = -1
        self.matrix.shape = (self.matrix.shape[0], product(shapein))
        self._reset(shapein=shapein)


@orthogonal
@inplace
class RollOperator(Operator):
    def __init__(self, n, axis=None, **keywords):
        Operator.__init__(self, **keywords)
        if axis is None:
            axis = -1 if isscalarlike(n) else range(-len(n), 0)
        self.axis = (axis,) if isscalarlike(axis) else tuple(axis)
        self.n = (n,) * len(self.axis) if isscalarlike(n) else tuple(n)
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


@orthogonal
class _CartesianEquatorialGalactic(DenseOperator):
    _g2e = np.array([[-0.0548755604,  0.4941094279, -0.8676661490],
                     [-0.8734370902, -0.4448296300, -0.1980763734],
                     [-0.4838350155,  0.7469822445,  0.4559837762]])


class CartesianEquatorial2GalacticOperator(_CartesianEquatorialGalactic):
    """
    ICRS-to-Galactic cartesian coordinate transform.

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0)
        - the Earth North pole of coordinates (0, 0, 1)

    The galactic direct referential is defined by:
        - the Sun center as the origin
        - the galactic center of coordinates (1, 0, 0)
        - the galactic pole of coordinates (0, 0, 1)

    Note that the equatorial-to-galactic conversion is considered to be
    a rotation, so we neglect the Earth-to-Sun origin translation.

    The galactic pole and center in the ICRS frame are defined by
    (Hipparcos 1997):
        αp = 192.85948°
        δp = 27.12825°
        αc = 266.40510°
        δc = -28.936175°

    The last dimension of the inputs/outputs must be 3.

    Example
    -------
    >>> op = CartesianEquatorial2GalacticOperator()
    >>> op([0, 0, 1])
    array([-0.48383502,  0.74698224,  0.45598378])

    """
    def __init__(self, **keywords):
        _CartesianEquatorialGalactic.__init__(self, self._g2e.T, **keywords)
        self.set_rule('T', lambda s: CartesianGalactic2EquatorialOperator())
        self.set_rule(('.', CartesianGalactic2EquatorialOperator), '1',
                      CompositionOperator)


class CartesianGalactic2EquatorialOperator(_CartesianEquatorialGalactic):
    """
    Galactic-to-ICRS cartesian coordinate transform.

    The galactic direct referential is defined by:
        - the Sun center as the origin
        - the galactic center of coordinates (1, 0, 0)
        - the galactic pole of coordinates (0, 0, 1)

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0)
        - the Earth North pole of coordinates (0, 0, 1)

    Note that the galactic-to-equatorial conversion is considered to be
    a rotation, so we neglect the Sun-to-Earth origin translation.

    The galactic pole and center in the ICRS frame are defined by
    (Hipparcos 1997):
        αp = 192.85948°
        δp = 27.12825°
        αc = 266.40510°
        δc = -28.936175°

    The last dimension of the inputs/outputs must be 3.

    Example
    -------
    >>> op = CartesianGalactic2EquatorialOperator()
    >>> op([0, 0, 1])
    array([-0.86766615, -0.19807637,  0.45598378])

    """
    def __init__(self, **keywords):
        _CartesianEquatorialGalactic.__init__(self, self._g2e, **keywords)
        self.set_rule('T', lambda s: CartesianEquatorial2GalacticOperator())
        self.set_rule(('.', CartesianEquatorial2GalacticOperator), '1',
                      CompositionOperator)


class _CartesianEquatorialHorizontal(DenseBlockDiagonalOperator):
    def __init__(self, convention, time, latitude, longitude, transpose,
                 dtype=None, **keywords):
        conventions = ('NE',)  # 'SW', 'SE')
        if convention not in conventions:
            raise ValueError(
                "Invalid azimuth angle convention '{0}'. Expected values are {"
                "1}.".format(convention, strenum(conventions)))
        lst = self._gst2lst(self._jd2gst(time.jd), longitude)
        lst = np.radians(lst * 15)
        latitude = np.radians(latitude)
        if dtype is None:
            dtype = np.find_common_type([float_dtype(lst.dtype),
                                         float_dtype(latitude.dtype)], [])
        slat = np.sin(latitude)
        clat = np.cos(latitude)
        slst = np.sin(lst)
        clst = np.cos(lst)
        m = np.empty(np.broadcast(latitude, lst).shape + (3, 3), dtype)
        m[..., 0, 0] = -slat * clst
        m[..., 0, 1] = -slat * slst
        m[..., 0, 2] = clat
        m[..., 1, 0] = -slst
        m[..., 1, 1] = clst
        m[..., 1, 2] = 0
        m[..., 2, 0] = clat * clst
        m[..., 2, 1] = clat * slst
        m[..., 2, 2] = slat
        if transpose:
            m = m.swapaxes(-1, -2)
        keywords['flags'] = self.validate_flags(keywords.get('flags', {}),
                                                orthogonal=True)
        DenseBlockDiagonalOperator.__init__(self, m, **keywords)

    @staticmethod
    def _jd2gst(jd):
        """
        Convert Julian Dates into Greenwich Sidereal Time.

        From Duffett-Smith, P. : 1988, Practical astronomy with your
        calculator, Cambridge University Press, third edition

        """
        jd0 = np.floor(jd - 0.5) + 0.5
        T = (jd0 - 2451545) / 36525
        T0 = 6.697374558 + 2400.051336 * T + 0.000025862 * T**2
        T0 %= 24
        ut = (jd - jd0) * 24
        T0 += ut * 1.002737909
        T0 %= 24
        return T0

    @staticmethod
    def _gst2lst(gst, longitude):
        """
        Convert Greenwich Sidereal Time into Local Sidereal Time.
        longitude : Geographic longitude EAST in degrees.
        """
        return (gst + longitude / 15) % 24


class CartesianEquatorial2HorizontalOperator(_CartesianEquatorialHorizontal):
    """
    Conversion between equatorial-to-horizontal cartesian coordinates.

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0)
        - the Earth North pole of coordinates (0, 0, 1)

    The horizontal referential is defined by:
        - the observer geographic position as the origin
        - the azimuth reference (North or South) of coordinates (1, 0, 0)
        - whether the azimuth is measured towards the East or West
        - the zenith of coordinates (0, 0, 1)

    Example
    -------
    >>> from astropy.time import Time, TimeDelta
    >>> t0 = Time(['2000-01-01 00:00:00.0'], scale='utc')
    >>> dt = TimeDelta(np.arange(1000)/10, format='sec')
    >>> lat, lon = 48.853291, 2.348751
    >>> op = CartesianEquatorial2HorizontalOperator('NE', t0 + dt, lat, lon)

    """
    def __init__(self, convention, time, latitude, longitude, **keywords):
        """
        convention : 'NE', 'SW', 'SE'
            The azimuth angle convention:
                - 'NE' : measured from the North towards the East (indirect)
                - 'SW' : from the South towards the West (indirect)
                - 'SE' : from the South towards the East (direct)
            But so far, only the 'NE' convention is implemented.
        time : astropy.time.Time
            The observer's time.
        latitude : array-like
            The observer's latitude, in degrees.
        longitude : array-like
            The observation's longitude counted positively eastward,
            in degrees.

        """
        _CartesianEquatorialHorizontal.__init__(
            self, convention, time, latitude, longitude, False, **keywords)


class CartesianHorizontal2EquatorialOperator(_CartesianEquatorialHorizontal):
    """
    Conversion between horizontal-to-equatorial cartesian coordinates.

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0)
        - the Earth North pole of coordinates (0, 0, 1)

    The horizontal referential is defined by:
        - the observer geographic position as the origin
        - the azimuth reference (North or South) of coordinates (1, 0, 0)
        - whether the azimuth is measured towards the East or West
        - the zenith of coordinates (0, 0, 1)

    Example
    -------
    >>> from astropy.time import Time, TimeDelta
    >>> t0 = Time(['2000-01-01 00:00:00.0'], scale='utc')
    >>> dt = TimeDelta(np.arange(1000)/10, format='sec')
    >>> lat, lon = 48.853291, 2.348751
    >>> op = CartesianHorizontal2EquatorialOperator('NE', t0 + dt, lat, lon)

    """
    def __init__(self, convention, time, latitude, longitude, **keywords):
        """
        convention : 'NE', 'SW', 'SE'
            The azimuth angle convention:
                - 'NE' : measured from the North towards the East (indirect)
                - 'SW' : from the South towards the West (indirect)
                - 'SE' : from the South towards the East (direct)
            But so far, only the 'NE' convention is implemented.
        time : astropy.time.Time
            The observer's time.
        latitude : array-like
            The observer's latitude, in degrees.
        longitude : array-like
            The observation's longitude counted positively eastward,
            in degrees.

        """
        _CartesianEquatorialHorizontal.__init__(
            self, convention, time, latitude, longitude, True, **keywords)


class SphericalEquatorial2GalacticOperator(CompositionOperator):
    """
    ICRS-to-Galactic cartesian coordinate transform.

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the Earth North pole of coordinates (0, 0, 1) (zenith=0)

    The galactic direct referential is defined by:
        - the Sun center as the origin
        - the galactic center of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the galactic pole of coordinates (0, 0, 1) (zenith=0)

    Note that the equatorial-to-galactic conversion is considered to be
    a rotation, so we neglect the Earth-to-Sun origin translation.

    The galactic pole and center in the ICRS frame are defined by
    (Hipparcos 1997):
        αp = 192.85948°
        δp = 27.12825°
        αc = 266.40510°
        δc = -28.936175°

    The last dimension of the inputs/outputs must be 2.

    Example
    -------
    >>> op = SphericalEquatorial2GalacticOperator(degrees=True)
    >>> op([266.40510, -28.936175])
    array([  4.70832664e-05,  -7.91258821e-05])

    """
    def __init__(self, conventionin='azimuth,elevation',
                 conventionout='azimuth,elevation', degrees=False,
                 **keywords):
        """
        Parameters
        ----------
        conventionin : str
            One of the following spherical coordinate conventions for
            the input equatorial angles: 'zenith,azimuth', 'azimuth,zenith',
            'elevation,azimuth' and 'azimuth,elevation'.
        conventionout : str
            The spherical coordinate convention for the output galactic angles.
        degrees : boolean, optional
            If true, the angle units are degrees (radians otherwise).

        """
        operands = [
            Cartesian2SphericalOperator(conventionout, degrees=degrees),
            CartesianEquatorial2GalacticOperator(),
            Spherical2CartesianOperator(conventionin, degrees=degrees)]
        CompositionOperator.__init__(self, operands, **keywords)


class SphericalGalactic2EquatorialOperator(CompositionOperator):
    """
    Galactic-to-ICRS cartesian coordinate transform.

    The galactic direct referential is defined by:
        - the Sun center as the origin
        - the galactic center of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the galactic pole of coordinates (0, 0, 1) (zenith=0)

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the Earth North pole of coordinates (0, 0, 1) (zenith=0)

    Note that the galactic-to-equatorial conversion is considered to be
    a rotation, so we neglect the Sun-to-Earth origin translation.

    The galactic pole and center in the ICRS frame are defined by
    (Hipparcos 1997):
        αp = 192.85948°
        δp = 27.12825°
        αc = 266.40510°
        δc = -28.936175°

    The last dimension of the inputs/outputs must be 2.

    Example
    -------
    >>> op = SphericalGalactic2EquatorialOperator(degrees=True)
    >>> op([0, 0])
    array([266.4049948, -28.93617396])

    """
    def __init__(self, conventionin='azimuth,elevation',
                 conventionout='azimuth,elevation', degrees=False,
                 **keywords):
        """
        Parameters
        ----------
        conventionin : str
            One of the following spherical coordinate conventions for
            the input galactic angles: 'zenith,azimuth', 'azimuth,zenith',
            'elevation,azimuth' and 'azimuth,elevation'.
        conventionout : str
            The spherical coordinate convention for the output equatorial
            angles.
        degrees : boolean, optional
            If true, the angle units are degrees (radians otherwise).

        """
        operands = [
            Cartesian2SphericalOperator(conventionout, degrees=degrees),
            CartesianGalactic2EquatorialOperator(),
            Spherical2CartesianOperator(conventionin, degrees=degrees)]
        CompositionOperator.__init__(self, operands, **keywords)


class SphericalEquatorial2HorizontalOperator(CompositionOperator):
    """
    Conversion between equatorial-to-horizontal cartesian coordinates.

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the Earth North pole of coordinates (0, 0, 1) (zenith=0)

    The horizontal referential is defined by:
        - the observer geographic position as the origin
        - the azimuth reference (North or South) (1, 0, 0) (zenith=pi/2, az=0)
        - whether the azimuth is measured towards the East or West
        - the zenith of coordinates (0, 0, 1) (zenith=0)

    The last dimension of the inputs/outputs must be 2.

    Example
    -------
    >>> from astropy.time import Time, TimeDelta
    >>> t0 = Time(['2000-01-01 00:00:00.0'], scale='utc')
    >>> dt = TimeDelta(np.arange(1000)/10, format='sec')
    >>> lat, lon = 48.853291, 2.348751
    >>> op = SphericalEquatorial2HorizontalOperator('NE', t0 + dt, lat, lon)

    """
    def __init__(self, convention_horizontal, time, latitude, longitude,
                 conventionin='azimuth,elevation',
                 conventionout='azimuth,elevation', degrees=False, **keywords):
        """
        convention_horizontal : 'NE', 'SW', 'SE'
            The azimuth angle convention:
                - 'NE' : measured from the North towards the East (indirect)
                - 'SW' : from the South towards the West (indirect)
                - 'SE' : from the South towards the East (direct)
            But so far, only the 'NE' convention is implemented.
        time : astropy.time.Time
            The observer's time.
        latitude : array-like
            The observer's latitude, in degrees.
        longitude : array-like
            The observation's longitude counted positively eastward,
            in degrees.
        conventionin : str
            One of the following spherical coordinate conventions for
            the input equatorial angles: 'zenith,azimuth', 'azimuth,zenith',
            'elevation,azimuth' and 'azimuth,elevation'.
        conventionout : str
            The spherical coordinate convention for the output horizontal
            angles.
        degrees : boolean, optional
            If true, the angle units are degrees (radians otherwise).

        """
        operands = [
            Cartesian2SphericalOperator(conventionout, degrees=degrees),
            CartesianEquatorial2HorizontalOperator(
                convention_horizontal, time, latitude, longitude),
            Spherical2CartesianOperator(conventionin, degrees=degrees)]
        CompositionOperator.__init__(self, operands, **keywords)


class SphericalHorizontal2EquatorialOperator(CompositionOperator):
    """
    Conversion between horizontal-to-equatorial cartesian coordinates.

    The horizontal referential is defined by:
        - the observer geographic position as the origin
        - the azimuth reference (North or South) (1, 0, 0) (zenith=pi/2, az=0)
        - whether the azimuth is measured towards the East or West
        - the zenith of coordinates (0, 0, 1) (zenith=0)

    The ICRS equatorial direct referential is defined by:
        - the Earth center as the origin
        - the vernal equinox of coordinates (1, 0, 0) (zenith=pi/2, az=0)
        - the Earth North pole of coordinates (0, 0, 1) (zenith=0)

    The last dimension of the inputs/outputs must be 2.

    Example
    -------
    >>> from astropy.time import Time, TimeDelta
    >>> t0 = Time(['2000-01-01 00:00:00.0'], scale='utc')
    >>> dt = TimeDelta(np.arange(1000)/10, format='sec')
    >>> lat, lon = 48.853291, 2.348751
    >>> op = CartesianHorizontal2EquatorialOperator('NE', t0 + dt, lat, lon)

    """
    def __init__(self, convention_horizontal, time, latitude, longitude,
                 conventionin='azimuth,elevation',
                 conventionout='azimuth,elevation', degrees=False, **keywords):
        """
        convention_horizontal : 'NE', 'SW', 'SE'
            The azimuth angle convention:
                - 'NE' : measured from the North towards the East (indirect)
                - 'SW' : from the South towards the West (indirect)
                - 'SE' : from the South towards the East (direct)
            But so far, only the 'NE' convention is implemented.
        time : astropy.time.Time
            The observer's time.
        latitude : array-like
            The observer's latitude, in degrees.
        longitude : array-like
            The observation's longitude counted positively eastward,
            in degrees.
        conventionin : str
            One of the following spherical coordinate conventions for
            the input horizontal angles: 'zenith,azimuth', 'azimuth,zenith',
            'elevation,azimuth' and 'azimuth,elevation'.
        conventionout : str
            The spherical coordinate convention for the output equatorial
            angles.
        degrees : boolean, optional
            If true, the angle units are degrees (radians otherwise).

        """
        operands = [
            Cartesian2SphericalOperator(conventionout, degrees=degrees),
            CartesianHorizontal2EquatorialOperator(
                convention_horizontal, time, latitude, longitude),
            Spherical2CartesianOperator(conventionin, degrees=degrees)]
        CompositionOperator.__init__(self, operands, **keywords)
