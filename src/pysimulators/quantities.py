# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#

import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from numpy.lib import NumpyVersion

from pyoperators.memory import empty

from . import _flib as flib

__all__ = ['ConversionInput', 'ConversionType', 'Quantity', 'UnitError', 'units']


numpy_version = NumpyVersion(np.__version__)
if numpy_version < '2.2.0':
    np.matvec = None
    np.vecmat = None


_RE_UNIT = re.compile(r' *([/*])? *([a-zA-Z_"\']+|\?+)(\^-?[0-9]+(\.[0-9]*)?)? *')


class UnitError(Exception):
    pass


UnitType = dict[str, float]


class ConversionInput(Enum):
    FIRST = auto()
    SECOND = auto()
    SEQUENCE = auto()
    SEQUENCE_EXCEPT_FIRST = auto()
    FIRST_SEQUENCE = auto()
    FIRST_NESTED = auto()
    SECOND_SEQUENCE = auto()
    FIRST_AND_SECOND = auto()
    FIRST_AND_THIRD = auto()
    SECOND_AND_THIRD = auto()


class ConversionOutput(Enum):
    DEFAULT = auto()  # The function simply returns an array
    SEQUENCE = auto()  # The function returns a sequence of arrays
    FIRST_IF_SEQUENCE = (
        auto()
    )  # If the output is a sequence, only convert the first element
    FIRST_IN_SEQUENCE = auto()  # Only convert the first element of the output sequence
    NONE = auto()  # Function returns None


class ConversionAction(Enum):
    SAME = auto()  # homogenize and propagate the input units to the output
    EACH = (
        auto()
    )  # when the input is *args and output is a sequence, each quantity is propagated
    SQUARE = auto()  # square the input units
    PROD = auto()  # output units is the product of the inputs
    UNITLESS = auto()  # the output is an ndarray


@dataclass
class ConversionType:
    input: ConversionInput = ConversionInput.FIRST
    output: ConversionOutput = ConversionOutput.DEFAULT
    action: ConversionAction = ConversionAction.SAME


DEFAULT_CONVERSION = ConversionType()

FUNCTIONS = {
    np.amax: DEFAULT_CONVERSION,
    np.amin: DEFAULT_CONVERSION,
    np.append: ConversionType(ConversionInput.FIRST_AND_SECOND),
    np.around: DEFAULT_CONVERSION,
    np.array_split: ConversionType(output=ConversionOutput.SEQUENCE),
    np.atleast_1d: DEFAULT_CONVERSION,
    np.atleast_2d: DEFAULT_CONVERSION,
    np.atleast_3d: DEFAULT_CONVERSION,
    np.average: DEFAULT_CONVERSION,
    # np.block: ConversionType(ConversionInput.FIRST_NESTED),
    np.broadcast_arrays: ConversionType(
        ConversionInput.SEQUENCE, ConversionOutput.SEQUENCE, ConversionAction.EACH
    ),
    np.broadcast_to: DEFAULT_CONVERSION,
    np.choose: ConversionType(ConversionInput.SECOND_SEQUENCE),
    np.clip: DEFAULT_CONVERSION,
    np.column_stack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.compress: ConversionType(ConversionInput.SECOND),
    np.concat: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.concatenate: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.convolve: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.copy: DEFAULT_CONVERSION,
    # np.corrcoef: NO_CONVERSION,
    np.correlate: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.cov: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.cross: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.cumsum: DEFAULT_CONVERSION,
    getattr(np, 'cumulative_sum', None): DEFAULT_CONVERSION,  # Numpy 2.1
    np.delete: DEFAULT_CONVERSION,
    np.diag: DEFAULT_CONVERSION,
    np.diagflat: DEFAULT_CONVERSION,
    np.diagonal: DEFAULT_CONVERSION,
    np.diff: DEFAULT_CONVERSION,
    np.dot: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.dsplit: ConversionType(output=ConversionOutput.SEQUENCE),
    np.dstack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.einsum: ConversionType(
        ConversionInput.SEQUENCE_EXCEPT_FIRST, action=ConversionAction.PROD
    ),
    np.expand_dims: DEFAULT_CONVERSION,
    np.extract: ConversionType(ConversionInput.SECOND),
    np.fill_diagonal: ConversionType(
        ConversionInput.FIRST_AND_SECOND, output=ConversionOutput.NONE
    ),
    np.fix: DEFAULT_CONVERSION,
    np.flip: DEFAULT_CONVERSION,
    np.fliplr: DEFAULT_CONVERSION,
    np.flipud: DEFAULT_CONVERSION,
    np.hsplit: ConversionType(output=ConversionOutput.SEQUENCE),
    np.hstack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.imag: DEFAULT_CONVERSION,
    np.inner: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.insert: ConversionType(ConversionInput.FIRST_AND_THIRD),
    np.interp: DEFAULT_CONVERSION,
    np.intersect1d: ConversionType(ConversionInput.FIRST_AND_SECOND),
    np.kron: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.max: DEFAULT_CONVERSION,
    np.mean: DEFAULT_CONVERSION,
    np.median: DEFAULT_CONVERSION,
    np.min: DEFAULT_CONVERSION,
    np.moveaxis: DEFAULT_CONVERSION,
    np.nan_to_num: DEFAULT_CONVERSION,
    np.nancumsum: DEFAULT_CONVERSION,
    np.nanmax: DEFAULT_CONVERSION,
    np.nanmean: DEFAULT_CONVERSION,
    np.nanmedian: DEFAULT_CONVERSION,
    np.nanmin: DEFAULT_CONVERSION,
    np.nanstd: DEFAULT_CONVERSION,
    np.nansum: DEFAULT_CONVERSION,
    np.nanvar: ConversionType(action=ConversionAction.SQUARE),
    np.outer: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.pad: DEFAULT_CONVERSION,
    np.partition: DEFAULT_CONVERSION,
    np.permute_dims: DEFAULT_CONVERSION,
    np.ptp: DEFAULT_CONVERSION,
    np.ravel: DEFAULT_CONVERSION,
    np.real: DEFAULT_CONVERSION,
    np.real_if_close: DEFAULT_CONVERSION,
    np.repeat: DEFAULT_CONVERSION,
    np.reshape: DEFAULT_CONVERSION,
    np.resize: DEFAULT_CONVERSION,
    np.roll: DEFAULT_CONVERSION,
    np.rollaxis: DEFAULT_CONVERSION,
    np.rot90: DEFAULT_CONVERSION,
    np.round: DEFAULT_CONVERSION,
    np.row_stack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.select: ConversionType(ConversionInput.SECOND_SEQUENCE),
    np.setdiff1d: ConversionType(ConversionInput.FIRST_AND_SECOND),
    np.setxor1d: ConversionType(ConversionInput.FIRST_AND_SECOND),
    np.sort: DEFAULT_CONVERSION,
    np.sort_complex: DEFAULT_CONVERSION,
    np.split: ConversionType(output=ConversionOutput.SEQUENCE),
    np.squeeze: DEFAULT_CONVERSION,
    np.stack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.std: DEFAULT_CONVERSION,
    np.sum: DEFAULT_CONVERSION,
    np.swapaxes: DEFAULT_CONVERSION,
    np.take: DEFAULT_CONVERSION,
    np.take_along_axis: DEFAULT_CONVERSION,
    np.tensordot: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.tile: DEFAULT_CONVERSION,
    np.trace: DEFAULT_CONVERSION,
    np.transpose: DEFAULT_CONVERSION,
    np.trim_zeros: DEFAULT_CONVERSION,
    np.union1d: ConversionType(ConversionInput.FIRST_AND_SECOND),
    np.unique: ConversionType(output=ConversionOutput.FIRST_IF_SEQUENCE),
    np.unique_all: ConversionType(output=ConversionOutput.FIRST_IN_SEQUENCE),
    # np.unique_counts: NO_CONVERSION,
    # np.unique_inverse: NO_CONVERSION,
    np.unique_values: DEFAULT_CONVERSION,
    getattr(np, 'unstack', None): ConversionType(
        output=ConversionOutput.SEQUENCE
    ),  # Numpy 2.1
    np.unwrap: DEFAULT_CONVERSION,
    np.var: ConversionType(action=ConversionAction.SQUARE),
    np.vdot: ConversionType(
        ConversionInput.FIRST_AND_SECOND, action=ConversionAction.PROD
    ),
    np.vsplit: ConversionType(output=ConversionOutput.SEQUENCE),
    np.vstack: ConversionType(ConversionInput.FIRST_SEQUENCE),
    np.where: ConversionType(ConversionInput.SECOND_AND_THIRD),
}
FUNCTIONS.pop(None, None)


def _extract_unit(string: str | UnitType) -> UnitType:
    """
    Convert the input string into a unit as a dictionary.

    """
    if string is None:
        return {}

    if isinstance(string, dict):
        return string

    if not isinstance(string, str):
        raise TypeError(
            "Invalid unit type '"
            + type(string).__name__
            + "'. Expected types are string or dictionary."
        )

    string = string.strip()
    result = {}
    start = 0
    while start < len(string):
        match = _RE_UNIT.match(string, start)
        if match is None:
            raise ValueError("Unit '" + string[start:] + "' cannot be understood.")
        op = match.group(1)
        u = match.group(2)
        exp = match.group(3)
        if exp is None:
            exp = 1.0
        else:
            exp = float(exp[1:])
        if op == '/':
            exp = -exp
        result = _multiply_unit_inplace(result, u, exp)
        start = start + len(match.group(0))
    return result


def _multiply_unit_inplace(unit, key, val):
    """
    Multiply a unit as a dictionary by a non-composite one
    Unlike _divide_unit, _multiply_unit and _power_unit,
    the operation is done in place, to speed up main
    caller _extract_unit.

    """
    if key in unit:
        if unit[key] == -val:
            del unit[key]
        else:
            unit[key] += val
    else:
        unit[key] = val
    return unit


def _power_unit(unit, power):
    """
    Raise to power a unit as a dictionary

    """
    if len(unit) == 0 or power == 0:
        return {}
    result = unit.copy()
    for key in unit:
        result[key] *= power
    return result


def _multiply_unit(unit1, unit2):
    """
    Multiplication of units as dictionary.

    """
    unit = unit1.copy()
    for key, val in unit2.items():
        if key in unit:
            if unit[key] == -val:
                del unit[key]
            else:
                unit[key] += val
        else:
            unit[key] = val
    return unit


def _divide_unit(unit1, unit2):
    """
    Division of units as dictionary.

    """
    unit = unit1.copy()
    for key, val in unit2.items():
        if key in unit:
            if unit[key] == val:
                del unit[key]
            else:
                unit[key] -= val
        else:
            unit[key] = -val
    return unit


def _strunit(unit):
    """
    Convert a unit as dictionary into a string.

    """
    if len(unit) == 0:
        return ''
    result = ''
    has_pos = False

    for key, val in unit.items():
        if val >= 0:
            has_pos = True
            break

    for key, val in sorted(unit.items()):
        if val < 0 and has_pos:
            continue
        result += ' ' + key
        if val != 1:
            ival = int(val)
            if abs(val - ival) <= 1e-7 * abs(val):
                val = ival
            result += '^' + str(val)

    if not has_pos:
        return result[1:]

    for key, val in sorted(unit.items()):
        if val >= 0:
            continue
        val = -val
        result += ' / ' + key
        if val != 1:
            ival = int(val)
            if abs(val - ival) <= 1e-7 * abs(val):
                val = ival
            result += '^' + str(val)

    return result[1:]


def _grab_doc(doc, func):
    # ex: zeros.__doc__ = _grab_doc(np.zeros.__doc__, 'zeros')
    doc = doc.replace(func + '(shape, ', func + '(shape, unit=None, ')
    doc = doc.replace(
        '\n    dtype : ',
        '\n    unit : string\n        Unit of '
        'the new array, e.g. ``W``. Default is None for scalar.'
        '\n    dtype : ',
    )
    doc = doc.replace('np.' + func, 'unit.' + func)
    doc = doc.replace('\n    out : ndarray', '\n    out : Quantity')
    return doc


class Quantity(np.ndarray):
    """
    Represent a quantity, i.e. a scalar or array associated with a unit.
    if dtype is not specified, the quantity is upcasted to float64 if its dtype
    is not real nor complex

    Examples
    --------

    A Quantity can be converted to a different unit:
    >>> a = Quantity(18, 'm')
    >>> a.inunit('km')
    >>> a
    Quantity(0.018, 'km')

    A more useful conversion:
    >>> sky = Quantity(4 * np.pi, 'sr')
    >>> print(sky.tounit('deg^2'))
    41252.9612494 deg^2

    Quantities can be compared:
    >>> Quantity(0.018, 'km') > Quantity(10., 'm')
    True
    >>> np.minimum(Quantity(1, 'm'), Quantity(0.1, 'km'))
    Quantity(1.0, 'm')

    Quantities can be operated on:
    >>> time = Quantity(0.2, 's')
    >>> a / time
    Quantity(0.09, 'km / s')

    Units do not have to be standard and ? can be used as a non-standard one:
    >>> value = Quantity(1., '?/detector')
    >>> value *= Quantity(100, 'detector')
    >>> value
    Quantity(100.0, '?')

    A derived unit can be converted to a SI one:
    >>> print(Quantity(2, 'Jy').SI)
    2e-26 kg / s^2

    It is also possible to add a new derived unit:
    >>> unit['kloug'] = Quantity(3, 'kg')
    >>> print(Quantity(2, 'kloug').SI)
    6.0 kg

    In fact, a non-standard unit is not handled differently than one of
    the 7 SI units, as long as it is not in the unit table:
    >>> unit['krouf'] = Quantity(0.5, 'broug^2')
    >>> print(Quantity(1, 'krouf').SI)
    0.5 broug^2

    """

    default_dtype = np.float64
    _unit = None
    _derived_units = None

    def __new__(
        cls,
        data,
        unit=None,
        derived_units=None,
        dtype=None,
        copy=True,
        order='C',
        subok=False,
        ndmin=0,
    ):

        data = np.asanyarray(data)
        if dtype is None and data.dtype.kind == 'i':
            dtype = float

        # get a new Quantity instance (or a subclass if subok is True)
        result = np.array(
            data, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin
        )
        if not subok and type(result) is not cls or not isinstance(result, cls):
            result = result.view(cls)

        # set unit attribute
        if unit is not None:
            result.unit = unit

        # set derived_units attribute
        if derived_units is not None:
            result.derived_units = derived_units.copy()

        return result

    def __array_finalize__(self, obj):
        # for some numpy methods (append): the result doesn't go through
        # __new__ and obj is None. We have to set the instance attributes
        self._unit = getattr(obj, '_unit', {})
        self._derived_units = getattr(obj, '_derived_units', {})

    @property
    def __array_priority__(self):
        return 1.0 if len(self._unit) == 0 else 1.5

    def __array_function__(self, func, types, args, kwargs):
        """Handle NumPy functions on Quantity objects.

        This method intercepts NumPy function calls to provide custom behavior,
        ensuring that functions use the Quantity methods instead of going through
        ufuncs when appropriate.
        """
        conversion = FUNCTIONS.get(func)
        if conversion is None:
            return super().__array_function__(func, types, args, kwargs)

        input_units = self._get_function_input_units(args, conversion)
        output_units = self._get_function_output_units(input_units, conversion)

        converted_inputs, kwargs = self._convert_function_inputs(
            args, kwargs, output_units, conversion
        )
        converted_inputs = self._view_function_inputs(converted_inputs, conversion)
        outputs = func(*converted_inputs, **kwargs)
        if conversion.output is ConversionOutput.NONE:
            return None

        converted_outputs = self._convert_function_outputs(
            outputs, output_units, conversion
        )
        return converted_outputs

    @classmethod
    def _get_function_input_units(
        cls, arrays: Iterable[Any], conversion: ConversionType
    ) -> tuple[UnitType, ...]:
        match conversion.input:
            case ConversionInput.FIRST:
                return (cls._get_units(arrays[0]),)
            case ConversionInput.SECOND:
                return (cls._get_units(arrays[1]),)
            case ConversionInput.SEQUENCE:
                return cls._get_sequence_units(arrays)
            case ConversionInput.SEQUENCE_EXCEPT_FIRST:
                return cls._get_sequence_units(arrays[1:])
            case ConversionInput.FIRST_AND_SECOND:
                return cls._get_sequence_units([arrays[0], arrays[1]])
            case ConversionInput.SECOND_AND_THIRD:
                return cls._get_sequence_units([arrays[1], arrays[2]])
            case ConversionInput.FIRST_AND_THIRD:
                return cls._get_sequence_units([arrays[0], arrays[2]])
            case ConversionInput.FIRST_SEQUENCE:
                return cls._get_sequence_units(arrays[0])
            case ConversionInput.SECOND_SEQUENCE:
                return cls._get_sequence_units(arrays[1])
            case _:
                assert False, 'unreachable'

    @classmethod
    def _get_function_output_units(
        cls, input_units: tuple[UnitType, ...], conversion: ConversionType
    ) -> UnitType | tuple[UnitType, ...]:
        match conversion.action:
            case ConversionAction.SAME:
                return next((_ for _ in input_units if _), {})
            case ConversionAction.EACH:
                return input_units
            case ConversionAction.SQUARE:
                return _power_unit(input_units[0], 2)
            case ConversionAction.PROD:
                output_unit = input_units[0]
                for input_unit in input_units[1:]:
                    output_unit = _multiply_unit(output_unit, input_unit)
                return output_unit
            case _:
                assert False, 'unreachable'

    @classmethod
    def _convert_function_inputs(
        cls,
        arrays: Any,
        keywords: dict[str, Any],
        output_units: UnitType | Sequence[UnitType],
        conversion: ConversionType,
    ) -> tuple[Any, ...]:
        out = keywords.get('out')
        if out is not None:
            keywords['out'] = cls._view_function_input(out)

        if conversion.action is not ConversionAction.SAME or conversion.input in (
            ConversionInput.FIRST,
            ConversionInput.SECOND,
        ):
            return arrays, keywords
        assert isinstance(output_units, dict)

        common_unit = output_units
        if not common_unit:
            return arrays, keywords

        homogenized_args = []
        for iarray, array in enumerate(arrays):
            if (
                iarray == 0
                and conversion.input is ConversionInput.FIRST_SEQUENCE
                or iarray == 1
                and conversion.input is ConversionInput.SECOND_SEQUENCE
            ):
                array = tuple(
                    (
                        input.tounit(common_unit)
                        if isinstance(input, Quantity) and unit and unit != common_unit
                        else input
                    )
                    for input, unit in zip(array, cls._get_sequence_units(array))
                )
            elif (
                iarray == 0
                and conversion.input
                in (
                    ConversionInput.FIRST,
                    ConversionInput.FIRST_AND_SECOND,
                    ConversionInput.FIRST_AND_THIRD,
                )
                or iarray == 1
                and conversion.input
                in (
                    ConversionInput.SECOND,
                    ConversionInput.FIRST_AND_SECOND,
                    ConversionInput.SECOND_AND_THIRD,
                )
                or iarray == 2
                and conversion.input
                in (ConversionInput.FIRST_AND_THIRD, ConversionInput.SECOND_AND_THIRD)
                or iarray > 0
                and conversion.input is ConversionInput.SEQUENCE_EXCEPT_FIRST
                or conversion.input is ConversionInput.SEQUENCE
            ):
                if (
                    isinstance(array, Quantity)
                    and array._unit
                    and array._unit != common_unit
                ):
                    array = array.tounit(common_unit)
            homogenized_args.append(array)
        return tuple(homogenized_args), keywords

    @classmethod
    def _view_function_inputs(
        cls, arrays: Any, conversion: ConversionType
    ) -> tuple[Any, ...]:
        viewed_arrays = []
        for iarray, array in enumerate(arrays):
            if (
                iarray == 0
                and conversion.input is ConversionInput.FIRST_SEQUENCE
                or iarray == 1
                and conversion.input is ConversionInput.SECOND_SEQUENCE
            ):
                array = cls._view_function_input_iterable(array)
            else:
                array = cls._view_function_input(array)
            viewed_arrays.append(array)
        return tuple(viewed_arrays)

    @classmethod
    def _view_function_input(cls, array):
        if isinstance(array, np.ndarray) and type(array) is not np.ndarray:
            return array.view(np.ndarray)
        return array

    @classmethod
    def _view_function_input_iterable(cls, arrays: Iterable[Any]) -> list[Any]:
        return [cls._view_function_input(array) for array in arrays]

    def _convert_function_outputs(
        self,
        arrays: Any,
        output_units: UnitType | Sequence[UnitType],
        conversion: ConversionType,
    ) -> Any | tuple[Any, ...]:
        if np.isscalar(arrays) or isinstance(arrays, np.ndarray):
            # covers the case DEFAULT and FIRST_IF_SEQUENCE when then the output is not
            # a sequence
            if np.isscalar(arrays):
                output = type(self)(arrays)
            else:
                output = arrays.view(type(self))
            output.__array_finalize__(self)
            output._unit = output_units
            return output

        if isinstance(output_units, dict):
            output_units = len(arrays) * [output_units]

        # Check if arrays is a namedtuple
        is_namedtuple = (
            isinstance(arrays, tuple)
            and hasattr(arrays, '_fields')
            and hasattr(arrays, '_make')
        )

        outputs = []
        for iarray, (array, units) in enumerate(zip(arrays, output_units)):
            if (
                iarray == 0
                and conversion.output
                in (
                    ConversionOutput.FIRST_IF_SEQUENCE,
                    ConversionOutput.FIRST_IN_SEQUENCE,
                )
                or conversion.output is ConversionOutput.SEQUENCE
            ):
                array = array.view(type(self))
                array.__array_finalize__(self)
                array._unit = units
            outputs.append(array)

        # If input was a namedtuple, reconstruct it with the same type
        if is_namedtuple:
            return type(arrays)(*outputs)

        return outputs

    @classmethod
    def _get_sequence_units(cls, arrays) -> tuple[UnitType, ...]:
        return tuple(cls._get_units(_) for _ in arrays)

    @classmethod
    def _get_units(cls, array) -> UnitType:
        if not isinstance(array, Quantity):
            return {}
        return array._unit

    @classmethod
    def _get_inputs_with_common_units(
        cls,
        arrays: Sequence[Any, ...],
    ) -> tuple[Any, UnitType]:
        input_units = cls._get_sequence_units(arrays)
        common_unit = next((_ for _ in input_units if _), {})
        inputs = tuple(
            input.tounit(common_unit) if unit and unit != common_unit else input
            for input, unit in zip(arrays, input_units)
        )
        return inputs, common_unit

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle ufunc operations on Quantity objects."""
        # Get the output if provided
        out = kwargs.get('out', None)

        input_units = self._get_sequence_units(inputs)

        if ufunc in (
            np.add,
            np.clip,
            np.fmod,
            np.hypot,
            np.maximum,
            np.minimum,
            np.subtract,
        ):
            inputs, output_unit = self._get_inputs_with_common_units(inputs)

        elif ufunc in (
            np.equal,
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.not_equal,
        ):
            inputs, _ = self._get_inputs_with_common_units(inputs)
            output_unit = None

        elif ufunc in (
            np.absolute,
            np.ceil,
            np.conjugate,
            np.copysign,
            np.fabs,
            np.float_power,
            np.floor,
            np.fmax,
            np.fmin,
            np.hypot,
            np.negative,
            np.rint,
            np.trunc,
        ):
            output_unit = self._unit

        elif ufunc is np.reciprocal:
            output_unit = _power_unit(input_units[0], -1)

        elif ufunc is np.cbrt:
            output_unit = _power_unit(input_units[0], 1 / 3)

        elif ufunc is np.sqrt:
            output_unit = _power_unit(input_units[0], 0.5)

        elif ufunc is np.square:
            output_unit = _power_unit(input_units[0], 2)

        elif ufunc is np.power:
            power = (
                inputs[1].view(np.ndarray)
                if isinstance(inputs[1], Quantity)
                else inputs[1]
            )
            output_unit = _power_unit(input_units[0], power)

        elif ufunc in (np.matmul, np.matvec, np.multiply, np.vecdot, np.vecmat):
            output_unit = _multiply_unit(input_units[0], input_units[1])

        elif ufunc in (np.floor_divide, np.true_divide, np.divide):
            output_unit = _divide_unit(input_units[0], input_units[1])

        elif ufunc in (np.exp, np.expm1, np.exp2, np.log, np.log10, np.log2, np.log1p):
            # Transcendental functions return dimensionless results
            output_unit = {}

        else:
            output_unit = None

        # Process outputs if provided
        # TODO: check me
        if out is not None:
            kwargs['out'] = tuple(
                o.view(np.ndarray) if isinstance(o, Quantity) else o
                for o in (out if isinstance(out, tuple) else (out,))
            )

        # Call the ufunc
        result = super().__array_ufunc__(
            ufunc,
            method,
            *[
                input.view(np.ndarray) if isinstance(input, Quantity) else input
                for input in inputs
            ],
            **kwargs,
        )
        if result is None or result is NotImplemented or output_unit is None:
            return result

        # Handle tuple results (for ufuncs with multiple outputs)
        if isinstance(result, tuple):
            return tuple(self._process_ufunc_result(r, output_unit) for r in result)

        return self._process_ufunc_result(result, output_unit)

    def _process_ufunc_result(self, result, output_unit):
        """
        Process the result of a ufunc call, setting the appropriate unit.
        """
        if np.isscalar(result):
            result = type(self)(result)
        else:
            result = result.view(type(self))
        result.__array_finalize__(self)
        result._unit = output_unit
        return result

    #    def __getattr__(self, name):
    #        if self.dtype.names is None or name not in self.dtype.names:
    #            raise AttributeError(
    #                "'" + self.__class__.__name__ + "' object has"
    #                " no attribute '" + name + "'"
    #            )
    #        return self[name]

    def __getitem__(self, key):
        """
        x.__getitem__(y) <==> x[y]

        Return the item described by the key.

        """
        if key is Ellipsis:
            return self

        item = np.ndarray.__getitem__(self, key)
        if not isinstance(item, np.ndarray):
            return item
        if self.dtype.kind == 'V' and isinstance(key, str):
            return item.view(Quantity)

        if isinstance(key, list):
            key = tuple(key)

        if not isinstance(key, tuple):
            key = (key,)

        # Replace Ellipsis with ':'
        try:
            pos = key.index(Ellipsis)
            key = key[:pos] + (self.ndim - len(key)) * (slice(None),) + key[pos + 1 :]
        except ValueError:
            pass

        key += (self.ndim - len(key)) * (slice(None),)

        # update the broadcastable derived units
        du = None
        for d, v in item._derived_units.items():
            try:
                pos = d.index('[')
            except ValueError:
                continue

            # the derived unit is broadcastable, let's update it
            broadcast = d[pos + 1 : -1]
            if broadcast == 'leftward':
                v = v[key[-v.ndim :]]
            else:
                v = v[key[: v.ndim]]

            if du is None:
                du = item._derived_units.copy()

            # if the derived unit has become a scalar, remove the brackets
            if v.ndim == 0:
                del du[d]
                du[d[:pos]] = v
            else:
                du[d] = v

        if du is not None:
            item._derived_units = du

        return item

    def __getslice__(self, i, j):
        """
        x.__getslice__(i, j) <==> x[i:j]

        Return the slice described by (i, j).  The use of negative
        indices is not supported.

        """
        return self.__getitem__(slice(i, j))

    @property
    def magnitude(self):
        """
        x.magnitude <==> x.view(np.ndarray)

        Return the magnitude of the quantity
        """
        return self.view(np.ndarray)

    @magnitude.setter
    def magnitude(self, value):
        """
        Set the magnitude of the quantity
        """
        if self.ndim == 0:
            self.shape = (1,)
            try:
                self.view(np.ndarray)[:] = value
            except ValueError as error:
                raise error
            finally:
                self.shape = ()
        else:
            self.view(np.ndarray)[:] = value

    @property
    def unit(self):
        """
        Return the Quantity unit as a string

        Example
        -------
        >>> Quantity(32., 'm/s').unit
        'm / s'

        """
        return _strunit(self._unit)

    @unit.setter
    def unit(self, unit):
        self._unit = _extract_unit(unit)

    @property
    def derived_units(self):
        """
        Return the derived units associated with the quantity

        """
        return self._derived_units

    @derived_units.setter
    def derived_units(self, derived_units):
        if derived_units is not None:
            if not isinstance(derived_units, dict):
                raise TypeError('Input derived units are not a dict.')
            for key in derived_units:
                if not isinstance(derived_units[key], Quantity) and not hasattr(
                    derived_units[key], '__call__'
                ):
                    raise UnitError(
                        "The user derived unit '" + key + "' is not a Quantity."
                    )
                try:
                    pos = key.index('[')
                except ValueError:
                    continue
                if key[pos:] not in ('[leftward]', '[rightward]'):
                    raise UnitError(
                        f'Invalid broadcast : {key[pos:]!r}. Valid values a'
                        f"re '[leftward]' or '[rightward]'."
                    )
        self._derived_units = derived_units

    def tounit(self, unit: str | UnitType) -> Self:
        """
        Convert a Quantity into a new unit

        Parameters
        ----------
        unit : string
             A string representing the unit into which the Quantity
             should be converted

        Returns
        -------
        res : Quantity
            A shallow copy of the Quantity in the new unit

        Example
        -------
        >>> q = Quantity(1., 'km')
        >>> p = q.tounit('m')
        >>> print(p)
        1000.0 m
        """
        result = self.copy()
        result.inunit(unit)
        return result

    def inunit(self, unit: str | UnitType) -> None:
        """
        In-place conversion of a Quantity into a new unit

        Parameters
        ----------
        unit : string
             A string representing the unit into which the Quantity
             should be converted

        Example
        -------
        >>> q = Quantity(1., 'km')
        >>> q.inunit('m')
        >>> print(q)
        1000.0 m
        """

        newunit = _extract_unit(unit)

        if len(self._unit) == 0 or len(newunit) == 0:
            self._unit = newunit
            return

        # the header may contain information used to do unit conversions
        if hasattr(self, '_header'):
            q1 = self.__class__(
                1,
                header=self._header,
                unit=self._unit,
                derived_units=self.derived_units,
            ).SI
            q2 = self.__class__(
                1, header=self._header, unit=newunit, derived_units=self.derived_units
            ).SI
        else:
            q1 = Quantity(1.0, self._unit, self.derived_units).SI
            q2 = Quantity(1.0, newunit, self.derived_units).SI

        if q1._unit != q2._unit:
            raise UnitError(
                "Units '"
                + self.unit
                + "' and '"
                + _strunit(newunit)
                + "' are incompatible."
            )
        factor = q1.magnitude / q2.magnitude
        if self.ndim == 0:
            self.magnitude *= factor
        else:
            self.magnitude.T[:] *= factor.T
        self._unit = newunit

    @property
    def SI(self):
        """
        Return the quantity in SI unit. If the quantity has
        no units, the quantity itself is returned, otherwise, a
        shallow copy is returned.

        Example
        -------
        >>> print(Quantity(1., 'km').SI)
        1000.0 m
        """
        if len(self._unit) == 0:
            return self

        ffast = Quantity(1.0, '')
        fslow = Quantity(1.0, '')
        for key, val in self._unit.items():

            # check if the unit is a local derived unit
            newfactor, broadcast = _check_du(self, key, val, self.derived_units)

            # check if the unit is a global derived unit
            if newfactor is None:
                newfactor, broadcast = _check_du(self, key, val, units)

            # if the unit is not derived, we add it to the dictionary
            if newfactor is None:
                _multiply_unit_inplace(fslow._unit, key, val)
                continue

            # factor may be broadcast
            if broadcast == 'leftward':
                fslow = fslow * newfactor
            else:
                ffast = (ffast.T * newfactor.T).T

        result = self * fslow.magnitude
        result = (result.T * ffast.magnitude.T).T
        result._unit = _multiply_unit(fslow._unit, ffast._unit)

        return result

    def __reduce__(self):
        state = list(np.ndarray.__reduce__(self))
        try:
            subclass_state = self.__dict__.copy()
        except AttributeError:
            subclass_state = {}
        for cls in self.__class__.__mro__:
            for slot in cls.__dict__.get('__slots__', ()):
                try:
                    subclass_state[slot] = getattr(self, slot)
                except AttributeError:
                    pass
        state[2] = (state[2], subclass_state)
        return tuple(state)

    def __setstate__(self, state):
        ndarray_state, subclass_state = state
        np.ndarray.__setstate__(self, ndarray_state)
        for k, v in subclass_state.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            type(self).__name__
            + '('
            + str(np.asarray(self))
            + ", '"
            + _strunit(self._unit)
            + "')"
        )

    def __str__(self):
        result = str(np.asarray(self))
        if len(self._unit) == 0:
            return result
        return result + ' ' + _strunit(self._unit)

    @classmethod
    def empty(
        cls, shape, unit=None, derived_units=None, dtype=None, order=None, **keywords
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            empty(shape, dtype, order),
            dtype=dtype,
            unit=unit,
            derived_units=derived_units,
            copy=False,
            **keywords,
        )

    @classmethod
    def ones(
        cls, shape, unit=None, derived_units=None, dtype=None, order=None, **keywords
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.ones(shape, dtype, order),
            dtype=dtype,
            unit=unit,
            derived_units=derived_units,
            copy=False,
            **keywords,
        )

    @classmethod
    def zeros(
        cls, shape, unit=None, derived_units=None, dtype=None, order=None, **keywords
    ):
        if dtype is None:
            dtype = cls.default_dtype
        return cls(
            np.zeros(shape, dtype, order),
            dtype=dtype,
            unit=unit,
            derived_units=derived_units,
            copy=False,
            **keywords,
        )


def _check_du(input, key, val, derived_units):
    if len(derived_units) == 0:
        return None, None
    if (key, val) in derived_units:
        du, broadcast = _get_du(input, (key, val), derived_units)
        return (None, None) if du is None else (du.SI, broadcast)
    if (key, -val) in derived_units:
        du, broadcast = _get_du(input, (key, -val), derived_units)
        return (None, None) if du is None else ((1 / du).SI, broadcast)
    du, broadcast = _get_du(input, key, derived_units)
    if du is None:
        return None, None
    if val == 1.0:
        return du.SI, broadcast
    if val == -1.0:
        return (1 / du).SI, broadcast
    return (du**val).SI, broadcast


def _get_du(input, key, derived_units):
    try:
        du, broadcast = _check_in_du(key, derived_units)
    except ValueError:
        return None, None
    if isinstance(du, Callable):
        du = du(input)
    if du is None:
        return None, None
    if hasattr(input, '_header'):
        du = du.view(type(input))
        du._header = input._header
    du._derived_units = input._derived_units
    return du, broadcast


def _check_in_du(key, derived_units):
    for du, value in derived_units.items():
        try:
            pos = du.index('[')
            if key == du[:pos]:
                return value, du[pos + 1 : -1]
        except ValueError:
            if key == du:
                return value, 'leftward'
    raise ValueError


def pixel_to_pixel_reference(input):
    """
    Returns the pixel area in units of reference pixel.
    """
    if not hasattr(input, 'header'):
        return Quantity(1, 'pixel_reference')
    required = 'CRPIX,CRVAL,CTYPE'.split(',')
    keywords = np.concatenate(
        [
            (lambda i: [r + str(i + 1) for r in required])(i)
            for i in range(input.header['NAXIS'])
        ]
    )
    if not all([k in input.header for k in keywords]):
        return Quantity(1, 'pixel_reference')

    scale, status = flib.wcsutils.projection_scale(
        str(input.header).replace('\n', ''),
        input.header['NAXIS1'],
        input.header['NAXIS2'],
    )
    if status != 0:
        raise RuntimeError()

    return Quantity(scale.T, 'pixel_reference', copy=False)


def pixel_reference_to_solid_angle(input):
    """
    Returns the reference pixel area in the units defined by the FITSheader
    """
    if not hasattr(input, 'header'):
        return None

    header = input.header
    if all([key in header for key in ('CD1_1', 'CD2_1', 'CD1_2', 'CD2_2')]):
        cd = np.array(
            [[header['cd1_1'], header['cd1_2']], [header['cd2_1'], header['cd2_2']]]
        )
        area = abs(np.linalg.det(cd))
    elif 'CDELT1' in header and 'CDELT2' in header:
        area = abs(header['CDELT1'] * header['CDELT2'])
    else:
        return None

    cunit1 = header['CUNIT1'] if 'CUNIT1' in header else 'deg'
    cunit2 = header['CUNIT2'] if 'CUNIT2' in header else 'deg'
    return area * Quantity(1, cunit1) * Quantity(1, cunit2)


units_table = {
    # SI
    'A': None,
    'cd': None,
    'K': None,
    'kg': None,
    'm': None,
    'mol': None,
    'rad': None,
    's': None,
    'sr': None,
    # angle
    "'": Quantity(1, 'arcmin'),
    '"': Quantity(1, 'arcsec'),
    'arcmin': Quantity(1 / 60.0, 'deg'),
    'arcsec': Quantity(1 / 3600.0, 'deg'),
    'deg': Quantity(np.pi / 180.0, 'rad'),
    # flux_densities
    'uJy': Quantity(1.0e-6, 'Jy'),
    'mJy': Quantity(1.0e-3, 'Jy'),
    'Jy': Quantity(1.0e-26, 'W/Hz/m^2'),
    'MJy': Quantity(1.0e6, 'Jy'),
    # force
    'N': Quantity(1.0, 'kg m / s^2'),
    # frequency
    'Hz': Quantity(1.0, 's^-1'),
    # energy
    'J': Quantity(1.0, 'kg m^2 / s^2'),
    # length
    'AU': Quantity(149597870700.0, 'm'),
    'km': Quantity(1000.0, 'm'),
    'mm': Quantity(1e-3, 'm'),
    'um': Quantity(1e-6, 'm'),
    'pc': Quantity(30.857e15, 'm'),
    # power
    'W': Quantity(1.0, 'kg m^2 / s^3'),
    # pressure
    'atm': Quantity(101325.0, 'Pa'),
    'bar': Quantity(1e5, 'Pa'),
    'mmHg': Quantity(1 / 760, 'atm'),
    'cmHg': Quantity(10, 'mmHg'),
    'Pa': Quantity(1, 'kg / m / s^2'),
    'Torr': Quantity(1, 'mmHg'),
    # resistivity
    'Ohm': Quantity(1.0, 'V/A'),
    # solid angle
    ('rad', 2): Quantity(1.0, 'sr'),
    'pixel': pixel_to_pixel_reference,
    'pixel_reference': pixel_reference_to_solid_angle,
    # time
    'ms': Quantity(1e-3, 's'),
    'us': Quantity(1e-6, 's'),
    # misc
    'C': Quantity(1.0, 'A s'),
    'V': Quantity(1.0, 'kg m^2 / A / s^3'),
}


class Unit(dict):
    def __init__(self):
        for k, v in units_table.items():
            self[k] = v
            if isinstance(k, str):
                setattr(self, k, Quantity(1, k))


units = Unit()
