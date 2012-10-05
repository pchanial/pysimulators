import itertools
import numpy as np

from pyoperators.utils import product
from pyoperators.utils.testing import skiptest
from pysimulators.utils import (all_eq, any_neq, assert_all_eq, diff,
                                integrated_profile, median, minmax, profile,
                                shift)
from pysimulators.datautils import distance

def test_any_neq1():
    assert all_eq(1, 1+1.e-15)
    assert all_eq([1], [1+1.e-15])
    assert any_neq(1, 1+1.e-13)
    assert any_neq([1], [1+1.e-13])
    assert all_eq(1, np.asarray(1+1.e-8, dtype='float32'))
    assert all_eq([1], [np.asarray(1+1.e-8, dtype='float32')])
    assert any_neq(1, np.asarray(1+1.e-6, dtype='float32'))
    assert any_neq([1], [np.asarray(1+1.e-6, dtype='float32')])

def test_any_neq2():
    NaN = np.nan
    assert all_eq(NaN, NaN)
    assert all_eq([NaN], [NaN])
    assert all_eq([NaN,1], [NaN,1])
    assert any_neq([NaN,1,NaN], [NaN,1,3])
    assert all_eq(minmax([NaN, 1., 4., NaN, 10.]), [1., 10.])

class A(np.ndarray):
    __slots__ = ('__dict__', 'info1')
    def __new__(cls, data, dtype=None, info1=None, info2=None, info3=None):
        data = np.array(data, dtype=dtype).view(cls)
        data.info1 = info1
        data.info2 = info2
        data.info3 = info3
        return data
    def __array_finalize__(self, obj):
        from copy import deepcopy
        if not hasattr(self, 'info1'):
            self.info1 = deepcopy(getattr(obj, 'info1', None))
        if not hasattr(self, 'info2'):
            self.info2 = deepcopy(getattr(obj, 'info2', None))
        if not hasattr(self, 'info3'):
            self.info3 = deepcopy(getattr(obj, 'info3', None))
    def copy(self):
        import copy
        return copy.deepcopy(self)
    def __repr__(self):
        desc="""\
array(data=%s,
      info1=%s,
      info2=%s,
      info3=%s"""
        return desc % (np.ndarray.__repr__(self), repr(self.info1), repr(self.info2), repr(self.info3))

dtypes=(np.bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128)

def test_any_neq3():
    for dtype in dtypes:
        arr = np.ones((2,3), dtype=dtype)
        a = A(arr, info1='1', info2=True)
        b1 = A(arr, info1='1', info2=True)
        b1[0,1] = 0
        b2 = A(arr, info1='2', info2=True)
        b3 = A(arr, info1='1', info2=False)
        def func1(a, b):
            assert any_neq(a, b)
        for b in (b1, b2, b3):
            yield func1, a, b
        b = a.copy()
        assert all_eq(a,b)
        a.info3 = b
        b1 = a.copy(); b1.info3[0,1] = 0
        b2 = a.copy(); b2.info3.info1 = '2'
        b3 = a.copy(); b3.info3.info2 = False
        def func2(a, b):
            assert any_neq(a, b)
        for b in (b1, b2, b3):
            yield func2, a, b
        b = a.copy()
        assert all_eq(a,b)

def test_profile():
    def profile_slow(input, origin=None, bin=1.):
        d = distance(input.shape, origin=origin)
        d /= bin
        d = d.astype(int)
        m = np.max(d)
        p = np.ndarray(int(m+1))
        n = np.zeros(m+1, int)
        for i in range(m+1):
            p[i] = np.mean(input[d == i])
            n[i] = np.sum(d==i)
        return p, n
    d = distance((10,20))
    x, y, n = profile(d, origin=(4,5), bin=2., histogram=True)
    y2, n2 = profile_slow(d, origin=(4,5), bin=2.)
    assert all_eq(y, y2[0:y.size])
    assert all_eq(n, n2[0:n.size])

def test_integrated_profile():
    def integrated_profile_slow(input, origin=None, bin=1.):
        d = distance(input.shape, origin=origin)
        m = np.max(d)
        x = np.ndarray(int(m+1))
        y = np.ndarray(int(m+1))
        for i in range(m+1):
            x[i] = (i + 1) * bin
            y[i] = np.sum(input[d < x[i]])
        return x, y
    d = distance((10,20))
    x, y = integrated_profile(d, origin=(4,5), bin=2.)
    x2, y2 = integrated_profile_slow(d, origin=(4,5), bin=2.)
    assert all_eq(x, x2[0:y.size])
    assert all_eq(y, y2[0:y.size])

@skiptest
def test_diff():
    def func(naxis, axis):
        shape = [i+8 for i in range(naxis)]
        a=np.random.random_integers(1, 10, size=shape).astype(float)
        ref = -np.diff(a, axis=axis)
        diff(a, a, axis=axis)
        s = tuple([slice(0,s) for s in ref.shape])
        print s, a.shape, ref.shape
        a[s] -= ref
        assert_all_eq(a, 0)
    for naxis in range(5):
        for axis in range(naxis):
            yield func, naxis, axis

def _median(x, mask=None, axis=None):
    x = np.asarray(x)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        x = x.copy()
        x[mask] = np.nan
    if axis is None:
        return np.median(x[np.isfinite(x)])
    if np.all(np.isfinite(x)):
        return np.median(x, axis=axis)
    slow = product(x.shape[:axis])
    fast = product(x.shape[axis+1:])
    out = np.empty(x.shape[:axis] + x.shape[axis+1:])
    out_ = out.reshape((slow,fast))
    x_ = x.reshape((slow,-1,fast))
    for i in range(slow):
        for j in range(fast):
            b = x_[i,:,j]
            out_[i,j] = np.median(b[np.isfinite(b)])
    return out

@skiptest
def test_median():
    def func(x, m, a):
        assert all_eq(median(x, mask=m, axis=a), _median(x, m, a))

    x = [0.2, 0.91, 0.9, 0.4, 0.5]
    m = len(x) * [False]

    iterator = itertools.chain(itertools.combinations(range(len(x)), 2),
                               iter([()]), iter([(0,1,2,3,4)]))
    for i in iterator:
        x_ = list(x)
        m_ = list(m)
        for i_ in i:
            x_[i_] = np.nan
            m_[i_] = True
        yield func, x_, None, None
        yield func, x, m_, None

    x = np.array([[2, 1, 3, 7, 4],
                  [8, 3, 2, 4, 2],
                  [3, 4, 1, 4, 8.]])
    m = np.zeros_like(x, dtype=bool)

    for a in range(2):
        iterator = itertools.chain(itertools.combinations(range(x.shape[a]), 2),
                                   iter([()]), iter([(0,1,2)]))
        for i in iterator:
            if len(i) == 3 and a == 1:
                # don't remove 3 elements from axis=1, since medians differ
                # for an odd number of elements
                continue
            for j in range(x.shape[1-a]):
                x_ = x.copy()
                m_ = m.copy()
                for i_ in i:
                    index = (i_,j) if a == 0 else (j,i_)
                    x_[index] = np.nan
                    m_[index] = True
                yield func, x_, None, a
                yield func, x, m_, a

@skiptest
def test_shift1():
    def func(a, s):
        b = np.empty_like(a)
        shift(b, b, s, axis=-1)
        assert_all_eq(b, 0)
    for a in (np.ones(10),np.ones((12,10))):
        for s in (10,11,100,-10,-11,-100):
            yield func, a, s

@skiptest
def test_shift2():
    a = np.array([[1.,1.,1.,1.],[2.,2.,2.,2.]])
    shift(a, a, [1,-1], axis=1)
    assert_all_eq(a, [[0,1,1,1],[2,2,2,0]])

@skiptest
def test_shift3():
    a = np.array([[0.,0,0],[0,1,0],[0,0,0]])
    b = np.empty_like(a)
    shift(a, b, 1, axis=0)
    assert_all_eq(b, np.roll(a,1,axis=0))
    shift(b, b, -2, axis=0)
    assert_all_eq(b, np.roll(a,-1,axis=0))

@skiptest
def test_shift4():
    a = np.array([[0.,0,0],[0,1,0],[0,0,0]])
    b = np.empty_like(a)
    shift(a, b, 1, axis=1)
    assert_all_eq(b, np.roll(a,1,axis=1))
    shift(b, b, -2, axis=1)
    assert_all_eq(b, np.roll(a,-1,axis=1))
