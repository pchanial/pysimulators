import numpy as np
import glob
import os
import pickle

from numpy.testing import assert_array_equal
from pyoperators.utils.testing import assert_is_none
from pysimulators import Quantity, FitsArray, Map, Tod, create_fitsheader
from pysimulators.utils import all_eq, get_attributes
from uuid import uuid1

filename = 'pysimulatorstest-'+str(uuid1())
default_type = float
types = [Quantity, FitsArray, Map, Tod]

def teardown():
    for file in glob.glob(filename+'*'):
        os.remove(file)

a = np.ones((4,3))
a[1,2] = 4
q = Quantity(a, unit='myunit', derived_units={'myunit': Quantity(2., 'Jy')})
f = FitsArray(q, header=create_fitsheader(fromdata=q, cdelt=0.5, crval=(4.,8.)))
m = Map(f, origin='upper', error=a*2, coverage=a*3)
mask = np.zeros((4,3), np.bool8)
mask[0,2] = True
t = Tod(f, mask=mask)
del mask

def test_copy_false_subok_true():
    def func(obj1, t):
        obj2 = t(obj1, copy=False, subok=True)
        if isinstance(obj, t):
            assert obj1 is obj2
        else:
            assert obj1 is not obj2
            assert all_eq(obj1, obj2)
    for obj in [q, f, m, t]:
        for ty in types:
            yield func, obj, ty

def test_copy_false_subok_false():
    def func(obj1, t):
        obj2 = t(obj1, copy=False, subok=False)
        if type(obj1) is t:
            assert obj1 is obj2
        else:
            assert obj1 is not obj2
            assert all_eq(obj1, obj2)
    for obj in [q, f, m, t]:
        for ty in types:
            yield func, obj, ty

def test_copy_true_subok_true():
    def func(obj1, ty):
        obj2 = ty(obj1, copy=True, subok=True)
        assert obj1 is not obj2
        assert all_eq(obj1, obj2)
        if isinstance(obj1, ty):
            assert type(obj2) is type(obj1)
        else:
            assert type(obj2) is ty
    for obj1 in [q, f, m, t]:
        for ty in types:
            yield func, obj1, ty

def test_copy_true_subok_false():
    def func(obj1, ty):
        obj2 = ty(obj1, copy=True, subok=False)
        assert obj1 is not obj2
        assert all_eq(obj1, obj2)
        assert type(obj2) is ty
    for obj1 in [q, f, m, t]:
        for ty in types:
            yield func, obj1, ty

def test_input1():
    def func(t, i):
        d = t(i)
        assert d.shape == (1,)
        assert d[0] == 2
    for t in types:
        for i in [(2,), [2], np.array([2])]:
            yield func, t, i

def test_input2():
    def func(t, i):
        d = t(i)
        assert d.shape == ()
        assert d[...] == 2
    for t in types:
        for i in [2, np.array(2)]:
            yield func, t, i

def test_input3():
    def func(t):
        d = t([])
        assert d.shape == (0,)
    for t in types:
        yield func, t

def test_view():
    array = np.ones((10,32))
    aq = ['unit', 'derived_units']
    af = aq + ['header']
    attrs = [aq, af, af + ['coverage', 'error', 'origin'], af + ['mask']]
    def func(t, attr):
        d = array.view(t)
        for a in attr:
            assert hasattr(d, a)
    for t, attr in zip(types, attrs):
        yield func, t, attr

def test_operation():
    def func(t):
        a = t.ones(10)
        a2 = a + 1
        assert type(a2) is t
    for t in types[1:]:
        yield func, t

def test_default_dtype():
    def func(t):
        a = t([10,20])
        assert a.dtype == default_type
    for t in types:
        yield func, t

def test_dtype():
    def func(t, dtype):
        d = t(a, dtype=dtype, copy=False)
        assert d.dtype == dtype
    for t in types:
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            yield func, t, dtype

def test_tod_save():
    m = np.ndarray((10,2,10), dtype='int8')
    m.flat = np.random.random(m.size)*2
    a = Tod(np.random.random_sample((10,2,10)), mask=m, unit='Jy')
    a.save(filename+'_tod.fits')
    b = Tod(filename+'_tod.fits')
    assert_array_equal(a, b)

def test_map():
    class MAP(Map):
        def __init__(self, data):
            self.info = 'info'
    m = MAP(np.ones(3))
    assert get_attributes(m) == ['_derived_units', '_header', '_unit',
                                 'coverage', 'error', 'info', 'origin']

def test_pickling():
    objs = (q,f,m,t)
    def func1(v, o):
        o2 = pickle.loads(pickle.dumps(o,v))
        assert all_eq(o, o2)
    for v in range(pickle.HIGHEST_PROTOCOL):
        for o in objs:
            yield func1, v, o
    def func2(o):
        o.save(filename+'_obj.fits')
        o2 = type(o)(filename+'_obj.fits')
        if all_eq(o, o2):
            return
        print 'o', repr(o)
        print 'o2', repr(o2)
        assert False
    for o in objs[1:]:
        yield func2, o

def test_ndarray_funcs():
    data = [[1,2,3,4],[5,6,7,8]]
    mask = [[True,False,False,False],[False,True,True,False]]
    funcs = np.min, np.max, np.sum, np.mean, np.ptp, np.round, np.std, np.var
    axes = (None, 0, 1)

    def f(cls, func, axis, m):
        keywords_array = {'unit':'u'}
        if cls is Tod:
            keywords_array['mask'] = m
        array = cls(data, **keywords_array)
        keywords_func = {'axis':axis} if func is not np.round else {}
        result = func(array, **keywords_func)
        if cls is Tod:
            ref = func(np.ma.MaskedArray(array.magnitude,
                       mask=m), **keywords_func)
            if not isinstance(ref, np.ndarray):
                ref = np.ma.MaskedArray(ref)
        else:
            ref = func(array.magnitude, **keywords_func)
            if not isinstance(ref, np.ndarray):
                ref = np.array(ref)
        assert all_eq(result, ref)
        if func is np.var:
            assert result._unit == {'u':2}
        else:
            assert result._unit == {'u':1}
        if cls is Map:
            assert_is_none(result.coverage)
            assert_is_none(result.error)
        elif cls is Tod:
            assert all_eq(result.mask, ref.mask)

    for cls in (Quantity, FitsArray, Map, Tod):
        if cls is Tod:
            masks = None, mask
        else:
            masks = None,
        for func in funcs:
            for axis in (None,) if func is np.round else axes:
                for m in masks:
                    yield f, cls, func, axis, m

def test_astype():
    dtypes = np.int8, np.int32, np.int64, np.float32, np.float64
    def func(d, e):
        m = Map(np.array([1,2,3], dtype=d), coverage=np.array([0,1,0], dtype=d),
                error=np.array([2,2,2], dtype=d))
        m2 = m.astype(e)
        assert m2.dtype == e
        assert m2.coverage.dtype == e
        assert m2.error.dtype == e
    for d in dtypes:
        for e in dtypes:
            yield func, d, e
