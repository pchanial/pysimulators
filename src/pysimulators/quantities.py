# Copyrights 2010-2011 Pierre Chanial
# All rights reserved
#

import re
from collections.abc import Callable

import numpy as np

from pyoperators.memory import empty

from . import _flib as flib

__all__ = ['Quantity', 'UnitError', 'units']

_RE_UNIT = re.compile(r' *([/*])? *([a-zA-Z_"\']+|\?+)(\^-?[0-9]+(\.[0-9]*)?)? *')


class UnitError(Exception):
    pass


def _extract_unit(string):
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


def _get_units(units):
    return [getattr(q, '_unit', {}) for q in units]


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

    def __array_prepare__(self, array, context=None):
        """
        Homogenise ufunc's argument units and cast array to the class of the
        argument of highest __array_priority__
        """

        if self is not array:
            if not isinstance(array, type(self)):
                array = array.view(type(self))
            # copy over attributes
            array.__dict__.update(self.__dict__)

        if context is None or len(self._unit) == 0:
            return array

        ufunc = context[0]
        if ufunc in (
            np.add,
            np.subtract,
            np.maximum,
            np.minimum,
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.equal,
            np.not_equal,
        ):
            for arg in context[1]:
                u = getattr(arg, '_unit', {})
                if len(u) == 0 or u == self._unit:
                    continue

                print(
                    "Warning: applying function '" + str(ufunc) + "' to Quan"
                    'tities of different units may have changed operands to '
                    "common unit '" + _strunit(self._unit) + "'."
                )
                arg.inunit(self._unit)

        return array

    def __array_wrap__(self, array, context=None):
        """
        Set unit of the result of a ufunc.

        Since different quantities can share their _unit attribute, a
        a change in unit of the ufunc result must be preceded by a copy
        of the argument unit.
        Not all ufuncs are currently handled. For an exhaustive list of
        ufuncs, see http://docs.scipy.org/doc/numpy/reference/ufuncs.html
        """

        if np.__version__ < '1.4.1' and self is not array:
            array = array.view(type(self))

        if context is None:
            return array

        ufunc = context[0]
        args = context[1]
        if ufunc in (np.add, np.subtract, np.maximum, np.minimum):
            if self is not array:
                # self has highest __array_priority__
                array._unit = self._unit
            else:
                # inplace operation
                if len(self._unit) == 0:
                    self._unit = getattr(args[1], '_unit', {})

        elif ufunc is np.reciprocal:
            array._unit = _power_unit(args[0]._unit, -1)

        elif ufunc is np.sqrt:
            array._unit = _power_unit(args[0]._unit, 0.5)

        elif ufunc in (np.square, np.var):
            array._unit = _power_unit(args[0]._unit, 2)

        elif ufunc is np.power:
            array._unit = _power_unit(args[0]._unit, args[1])

        elif ufunc in (np.multiply, np.vdot):
            units = _get_units(args)
            array._unit = _multiply_unit(units[0], units[1])

        elif ufunc in (np.floor_divide, np.true_divide):
            units = _get_units(args)
            array._unit = _divide_unit(units[0], units[1])

        elif ufunc is np.divide:
            units = _get_units(args)
            array._unit = _divide_unit(units[0], units[1])

        elif ufunc in (
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.equal,
            np.not_equal,
            np.iscomplex,
            np.isfinite,
            np.isinf,
            np.isnan,
            np.isreal,
            np.bitwise_or,
            np.invert,
            np.logical_and,
            np.logical_not,
            np.logical_or,
            np.logical_xor,
        ):
            if array.ndim == 0:
                return bool(array)
            if hasattr(array, 'coverage'):
                array.coverage = None
            if hasattr(array, '_mask'):
                array._mask = None
            array._unit = {}

        elif ufunc in (
            np.arccos,
            np.arccosh,
            np.arcsin,
            np.arcsinh,
            np.arctan,
            np.arctanh,
            np.arctan2,
            np.cos,
            np.cosh,
            np.exp,
            np.exp2,
            np.log,
            np.log2,
            np.log10,
            np.sin,
            np.sinh,
            np.tan,
            np.tanh,
        ):
            array._unit = {}

        elif ufunc in (np.abs, np.negative):
            array._unit = self._unit
        else:
            array._unit = {}

        return array

    def __getattr__(self, name):
        if self.dtype.names is None or name not in self.dtype.names:
            raise AttributeError(
                "'" + self.__class__.__name__ + "' object has"
                " no attribute '" + name + "'"
            )
        return self[name]

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

    def tounit(self, unit):
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

    def inunit(self, unit):
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

    def min(self, *args, **kw):
        return self._wrap_func(np.min, self.unit, *args, **kw)

    min.__doc__ = np.ndarray.min.__doc__

    def max(self, *args, **kw):
        return self._wrap_func(np.max, self.unit, *args, **kw)

    max.__doc__ = np.ndarray.max.__doc__

    def sum(self, *args, **kw):
        return self._wrap_func(np.sum, self.unit, *args, **kw)

    sum.__doc__ = np.ndarray.sum.__doc__

    def mean(self, *args, **kw):
        return self._wrap_func(np.mean, self.unit, *args, **kw)

    mean.__doc__ = np.ndarray.mean.__doc__

    def median(self, *args, **kw):
        return self._wrap_func(np.median, self.unit, *args, **kw)

    median.__doc__ = np.median.__doc__

    def ptp(self, *args, **kw):
        return self._wrap_func(np.ptp, self.unit, *args, **kw)

    ptp.__doc__ = np.ndarray.ptp.__doc__

    def round(self, *args, **kw):
        return self._wrap_func(np.round, self.unit, *args, **kw)

    round.__doc__ = np.ndarray.round.__doc__

    def std(self, *args, **kw):
        return self._wrap_func(np.std, self.unit, *args, **kw)

    std.__doc__ = np.ndarray.std.__doc__

    def var(self, *args, **kw):
        return self._wrap_func(np.var, _power_unit(self._unit, 2), *args, **kw)

    var.__doc__ = np.ndarray.var.__doc__

    def _wrap_func(self, func, unit, *args, **kw):
        result = func(self.magnitude, *args, **kw).view(type(self))
        if not isinstance(result, np.ndarray):
            return result
        result.__array_finalize__(self)
        if unit is not None:
            result.unit = unit
        return result


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
    raise ValueError()


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
