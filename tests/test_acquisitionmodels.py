import numpy as np
import scipy
import scipy.signal

from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from pyoperators import Operator, AdditionOperator, CompositionOperator, BlockDiagonalOperator, asoperator, decorators
from pyoperators.utils import isscalar
from pyoperators.utils.testing import assert_is_instance, skiptest
from pysimulators.acquisitionmodels import BlackBodyOperator, ConvolutionOperator, CompressionAverageOperator, DdTddOperator, DiscreteDifferenceOperator, DownSamplingOperator, FftHalfComplexOperator, IdentityOperator, InvNttUncorrelatedOperator, InvNttUncorrelatedPythonOperator,  MaskOperator, PackOperator, PadOperator, ConvolutionTruncatedExponentialOperator, RollOperator, ShiftOperator, UnpackOperator, block_diagonal
from pysimulators.utils import all_eq

def test_partitioning():

    @block_diagonal('value', 'mykey', axisin=0)
    @decorators.square
    class MyOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey
        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    @block_diagonal('value', 'mykey', axisin=0)
    @decorators.square
    class MySupOp(Operator):
        def __init__(self, arg1, value, arg3, mykey=None, **keywords):
            Operator.__init__(self, **keywords)
            self.arg1 = arg1
            self.value = value
            self.arg3 = arg3
            self.mykey = mykey
    class MySubOp(MySupOp):
        def direct(self, input, output):
            output[...] = self.value * input
        __str__ = Operator.__repr__

    arg1 = [1, 2, 3, 4, 5]
    arg3 = ['a', 'b', 'c', 'd']

    def func(cls, n, v, k):
        n1 = 1 if isscalar(v) else len(v)
        n2 = 1 if isscalar(k) else len(k)
        nn = max(n1,n2) if n is None else 1 if isscalar(n) else len(n)
        if not isscalar(v) and not isscalar(k) and n1 != n2:
            # the partitioned arguments do not have the same length
            assert_raises(ValueError, lambda: cls(arg1, v, arg3, mykey=k,
                                                  partitionin=n))
            return
        if nn != max(n1, n2) and (not isscalar(v) or not isscalar(k)):
            # the partition is incompatible with the partitioned arguments
            return # test assert_raises(ValueError)

        op = cls(arg1, v, arg3, mykey=k, partitionin=n)
        if nn == 1:
            v = v if isscalar(v) else v[0]
            k = k if isscalar(k) else k[0]
            func2(cls, op, v, k)
        else:
            assert op.__class__ is BlockDiagonalOperator
            assert len(op.operands) == nn
            if n is None:
                assert op.partitionin == nn * (None,)
                assert op.partitionout == nn * (None,)
                return
            v = nn * [v] if isscalar(v) else v
            k = nn * [k] if isscalar(k) else k
            for op_, v_, k_ in zip(op.operands, v, k):
                func2(cls, op_, v_, k_)
            expected = np.hstack(n_*[v_] for n_, v_ in zip(n,v))
            input = np.ones(np.sum(n))
            output = op(input)
            assert_equal(output, expected)

    def func2(cls, op, v, k):
        assert op.__class__ is cls
        assert op.arg1 is arg1
        assert op.value is v
        assert hasattr(op, 'arg3')
        assert op.arg3 is arg3
        assert op.mykey is k
        input = np.ones(1)
        output = op(input)
        assert_equal(output, v)

    for c in (MyOp, MySubOp):
        for n in (None, 2, (2,), (4,2)):
            for v in (2., (2.,), (2., 3)):
                for k in (0., (0.,), (0., 1.)):
                    yield func, c, n, v, k

def test_blackbody():
    def bb(w,T):
        c = 2.99792458e8
        h = 6.626068e-34
        k = 1.380658e-23
        nu = c/w
        return 2*h*nu**3/c**2 / (np.exp(h*nu/(k*T))-1)

    w = np.arange(90.,111) * 1e-6
    T = 15.
    flux = bb(w, T) / bb(w[10], T)
    ops = [BlackBodyOperator(wave, 100e-6, T) for wave in w]
    flux2 = [op(1.) for op in ops]
    assert all_eq(flux, flux2)

    w, T = np.ogrid[90:111,15:20]
    w = w * 1.e-6
    flux = bb(w, T) / bb(w[10], T)
    ops = [BlackBodyOperator(wave, 100e-6, T.squeeze()) for wave in w]
    flux2 = np.array([op(np.ones(T.size)) for op in ops])
    assert all_eq(flux, flux2)
    
@skiptest
def test_compression_average1():
    data = np.array([1., 2., 2., 3.])
    compression = CompressionAverageOperator(2)
    compressed = compression(data)
    assert_equal(compressed, [1.5, 2.5])
    assert_equal(compression.T(compressed), [0.75, 0.75, 1.25, 1.25])

@skiptest
def test_compression_average2():
    partition = (10,5)
    tod = np.empty((2,15), float)
    tod[0,:] = [1,1,1,1,1,3,3,3,3,3,4,4,4,4,4]
    tod[1,:] = [1,2,1.,0.5,0.5,5,0,0,0,0,1,2,1,2,1.5]
    compression = CompressionAverageOperator(5, partitionin=partition)
    tod2 = compression(tod)
    assert tod2.shape == (2,3)
    assert_equal(tod2, [[1.,3.,4.],[1.,1.,1.5]])

    tod3 = compression.T(tod2)
    assert tod3.shape == (2,15)
    assert_almost_equal(tod3[0,:], (0.2,0.2,0.2,0.2,0.2,0.6,0.6,0.6,0.6,0.6,0.8,0.8,0.8,0.8,0.8))
    
    tod = np.array([1,2,2,3,3,3,4,4,4,4])
    compression = CompressionAverageOperator([1,2,3,4], partitionin=[1,2,3,4])
    tod2 = compression(tod)
    assert_almost_equal(tod2, [1,2,3,4])
    tod3 = compression.T(tod2)
    assert_almost_equal(tod3, 10*[1])

@skiptest
def test_compression_average3():
    a = CompressionAverageOperator(3)
    assert_almost_equal(a.todense(9).T, a.T.todense(3))

@skiptest
def test_downsampling1():
    partition = (1,2,3,4)
    tod = np.array([1,2,1,3,1,1,4,1,1,1])
    compression=DownSamplingOperator([1,2,3,4], partitionin=partition)
    tod2 = compression(tod)
    assert_equal(tod2, [1,2,3,4])
    tod3 = compression.T(tod2)
    assert_equal(tod3, [1,2,0,3,0,0,4,0,0,0])

@skiptest
def test_downsampling2():
    a = CompressionAverageOperator(3)
    assert_almost_equal(a.todense(9).T, a.T.todense(3))

def test_padding1():
    padding = PadOperator(left=1,right=20)
    a = np.arange(10*15).reshape((10,15))
    b = padding(a)
    assert b.shape == (10,36)
    assert_equal(b[:,0:1], 0)
    assert_equal(b[:,1:16], a)
    assert_equal(b[:,16:], 0)
    shapein = (10,15)
    shapeout = (10,15+1+20)
    assert_equal(padding.T.todense(shapeout), padding.todense(shapein).T)

def test_padding2():
    padding = PadOperator(left=1,right=(4,20), partitionin=(12,3))
    a = np.arange(10*15).reshape((10,15))
    b = padding(a)
    assert b.shape == (10,41)
    assert_equal(b[:,0:1], 0)
    assert_equal(b[:,1:13], a[:,0:12])
    assert_equal(b[:,13:17], 0)
    assert_equal(b[:,17:18], 0)
    assert_equal(b[:,18:21], a[:,12:])
    assert_equal(b[:,21:], 0)
    shapein = (10,15)
    shapeout = (10,(12+1+4)+(3+1+20))
    assert_equal(padding.T.todense(shapeout), padding.todense(shapein).T)

@skiptest
def test_convolution_truncated_exponential():
    r = ConvolutionTruncatedExponentialOperator(1., shapein=(1,10))
    a = np.ones((1,10))
    b = r(a)
    assert np.allclose(a, b)

    a[0,1:]=0
    b = r(a)
    assert_almost_equal(b[0,:], [np.exp(-t/1.) for t in range(0,10)])
    assert_almost_equal(r.T.todense(), r.todense().T)

@skiptest
def test_ffthalfcomplex1():
    n = 100
    fft = FftHalfComplexOperator(n)
    a = np.random.random(n)+1
    b = fft.T(fft(a))
    assert np.allclose(a, b)
    a = np.random.random((3,n))+1
    b = fft.T(fft(a))
    assert np.allclose(a, b)

@skiptest
def test_ffthalfcomplex2():
    nsamples = 1000
    fft = FftHalfComplexOperator(nsamples)
    a = np.random.random((10,nsamples))+1
    b = fft.T(fft(a))
    assert np.allclose(a, b)

@skiptest
def test_ffthalfcomplex3():
    partition = (100,300,5,1000-100-300-5)
    ffts = [FftHalfComplexOperator(p) for p in partition]
    fft = BlockDiagonalOperator(ffts, partitionin=partition, axisin=-1)
    a = np.random.random((10,np.sum(partition)))+1
    b = fft(a)
    b_ = np.hstack([ffts[0](a[:,:100]), ffts[1](a[:,100:400]), ffts[2](a[:,400:405]), ffts[3](a[:,405:])])
    assert np.allclose(b, b_)
    b = fft.T(fft(a))
    assert np.allclose(a, b)

@skiptest
def test_invntt_uncorrelated():
    filter = np.array([0., 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0]).reshape((1,-1))
    ncorrelations = 2
    invntt = InvNttUncorrelatedOperator(filter, ncorrelations, 3)
    invntt_todense = invntt.todense()
    assert all_eq(invntt_todense, invntt.todense(inplace=True))
    invntt2 = InvNttUncorrelatedPythonOperator(filter, ncorrelations, 3)
    invntt2_todense = invntt2.todense()
    assert all_eq(invntt2_todense, invntt2.todense(inplace=True))
    assert all_eq(invntt_todense, invntt2_todense)

@skiptest
def test_addition():
    def func(nops):
        ops = [ DiscreteDifferenceOperator(axis=axis,shapein=(2,3,4,5)) \
                for axis in range(nops) ]
        model = AdditionOperator(ops)
        v = np.arange(2*3*4*5.).reshape(2,3,4,5)
        a = model(v)
        b = sum([op(v) for op in ops])
        c = model(v,v)
        assert np.all(a == b)
        assert np.all(b == c)
    for nops in range(1, 5):
        yield func, nops

@skiptest
def test_additionT():
    def func(nops):
        ops = [ DiscreteDifferenceOperator(axis=axis,shapein=(2,3,4,5)) \
                for axis in range(nops) ]
        model = AdditionOperator(ops)
        assert model.T.T is model
        v = np.arange(2*3*4*5.).reshape(2,3,4,5)
        a = model.T(v)
        b = sum([op.T(v) for op in ops])
        c = model.T(v,v)
        assert np.all(a == b)
        assert np.all(b == c)
    for nops in range(1, 5):
        yield func, nops

@skiptest
def test_composition():
    def func(nops):
        ops = [ DiscreteDifferenceOperator(axis=axis,shapein=(2,3,4,5)) \
                for axis in range(nops) ]
        model = CompositionOperator(ops)
        v = np.arange(2*3*4*5.).reshape(2,3,4,5)
        a = model(v)
        b = v.copy()
        for m in reversed(ops):
            b = m(b)
        c = model(v,v)
        assert np.all(a == b) and np.all(b == c)
    for nops in range(1, 5):
        yield func, nops

@skiptest
def test_compositionT():
    def func(nops):
        ops = [ DiscreteDifferenceOperator(axis=axis,shapein=(2,3,4,5)) \
                for axis in range(nops) ]
        model = CompositionOperator(ops)
        assert model.T.T is model
        v = np.arange(2*3*4*5.).reshape(2,3,4,5)
        a = model.T(v)
        b = v.copy()
        for m in ops:
            b = m.T(b)
        c = model.T(v,v)
        assert np.all(a == b) and np.all(b == c)
    for nops in range(1, 5):
        yield func, nops


@skiptest
def test_packing():

    p = PackOperator([False, True, True, False])
    assert all_eq(p([1,2,3,4]), [1,4])
    assert all_eq(p.T([1,4]), [1,0,0,4])

    u = UnpackOperator([False, True, True, False])
    assert all_eq(u([1,4]), [1,0,0,4])
    assert all_eq(u.T([1,2,3,4]), [1,4])

    pdense = p.todense()
    udense = u.todense()
    assert all_eq(pdense, p.todense(inplace=True))
    assert all_eq(udense, u.todense(inplace=True))
    assert all_eq(pdense, udense.T)

    assert_is_instance(p*u, IdentityOperator)
    assert_is_instance(u*p, MaskOperator)
    m = u * p
    assert all_eq(np.dot(udense, pdense), m.todense())
    

@skiptest
def test_convolution():
    imashape = (7, 7)
    kershape = (3, 3)
    kerorig = (np.array(kershape) - 1) // 2
    kernel = np.zeros(kershape)
    kernel[kerorig[0]-1:kerorig[0]+2,kerorig[1]-1:kerorig[1]+2] = 0.5 ** 4
    kernel[kerorig[0], kerorig[1]] = 0.5
    kernel[kerorig[0]-1,kerorig[1]-1] *= 2
    kernel[kerorig[0]+1,kerorig[1]+1] = 0

    image = np.zeros(imashape)
    image[3,3] = 1.
    ref = scipy.signal.convolve(image, kernel, mode='same')
    convol=ConvolutionOperator(image.shape, kernel)
    con = convol(image)
    assert np.allclose(ref, con, atol=1.e-15)

    image = np.array([0,1,0,0,0,0,0])
    kernel = [1,1,0.5]
    convol = ConvolutionOperator(image.shape, [1,1,1])
    con = convol(image)
    ref = scipy.signal.convolve(image, kernel, mode='same')

    for kx in range(1,4,2):
        kshape = (kx,)
        kernel = np.ones(kshape)
        kernel.flat[-1] = 0.5
        for ix in range(kx*2, kx*2+3):
          ishape = (ix,)
          image = np.zeros(ishape)
          image.flat[image.size//2] = 1.
          convol = ConvolutionOperator(image.shape, kernel)
          con = convol(image)
          ref = scipy.signal.convolve(image, kernel, mode='same')
          assert np.allclose(con, ref, atol=1.e-15)
          assert np.allclose(convol.todense().T, convol.T.todense(),
                             atol=1.e-15)

    for kx in range(1,4,2):
      for ky in range(1,4,2):
        kshape = (kx,ky)
        kernel = np.ones(kshape)
        kernel.flat[-1] = 0.5
        for ix in range(kx*2+1, kx*2+3):
          for iy in range(ky*2+1, ky*2+3):
            ishape = (ix,iy)
            image = np.zeros(ishape)
            image[tuple([s//2 for s in image.shape])] = 1.
            convol = ConvolutionOperator(image.shape, kernel)
            con = convol(image)
            ref = scipy.signal.convolve(image, kernel, mode='same')
            assert np.allclose(con, ref, atol=1.e-15)
            assert np.allclose(convol.todense().T, convol.T.todense(),
                               atol=1.e-15)

    for kx in range(1,4,2):
      for ky in range(1,4,2):
        for kz in range(1,4,2):
          kshape = (kx,ky,kz)
          kernel = np.ones(kshape)
          kernel.flat[-1] = 0.5
          for ix in range(kx*2+1, kx*2+3):
            for iy in range(ky*2+1, ky*2+3):
              for iz in range(kz*2+1, kz*2+3):
                ishape = (ix,iy,iz)
                image = np.zeros(ishape)
                image[tuple([s//2 for s in image.shape])] = 1.
                convol = ConvolutionOperator(image.shape, kernel)
                con = convol(image)
                ref = scipy.signal.convolve(image, kernel, mode='same')
                assert np.allclose(con, ref, atol=1.e-15)
                assert np.allclose(convol.todense().T, convol.T.todense(),
                                   atol=1.e-15)

def test_scipy_linear_operator():
    diagonal = np.arange(10.)
    M = scipy.sparse.dia_matrix((diagonal, 0), shape=2*diagonal.shape)
    model = asoperator(M)
    vec = np.ones(10)
    assert np.all(model(vec)   == diagonal)
    assert np.all(model.T(vec) == diagonal)
    assert np.all(model.matvec(vec)  == diagonal)
    assert np.all(model.rmatvec(vec) == diagonal)

@skiptest
def test_diff():
    def func(shape, axis):
        dX = DiscreteDifferenceOperator(axis=axis, shapein=shape)
        dTd = DdTddOperator(axis=axis, shapein=shape)
        dX_dense = dX.todense()
        assert_equal(dX_dense, dX.todense(inplace=True))

        dXT_dense = dX.T.todense()
        assert_equal(dXT_dense, dX.T.todense(inplace=True))
        assert_equal(dX_dense.T, dXT_dense)

        dtd_dense = dTd.todense()
        assert_equal(dtd_dense, dTd.todense(inplace=True))
        assert_equal(np.matrix(dXT_dense) * np.matrix(dX_dense), dtd_dense)

    for shape in ((3,), (3,4), (3,4,5), (3,4,5,6)):
        for axis in range(len(shape)):
            yield func, shape, axis

@skiptest
def test_shift1():
    for axis in range(4):
        shift = ShiftOperator(1, axis=axis, shapein=(3,4,5,6))
        yield assert_equal, shift.todense().T, shift.T.todense()

@skiptest
def test_shift2():
    for axis in range(1,4):
        shift = ShiftOperator(((1,2,3),), axis=axis, shapein=(3,4,5,6))
        yield assert_equal, shift.todense().T, shift.T.todense()

@skiptest
def test_shift3():
    for offset in ( (3,), (3,4), (3,4,5) ):
        for axis in range(len(offset),4):
            s = np.random.random_integers(-2,2,offset)
            shift = ShiftOperator([s], axis=axis, shapein=(3,4,5,6))
            yield assert_equal, shift.todense().T, shift.T.todense()

def test_roll():
    shape = np.arange(2,6)
    v = np.arange(2*3*4*5).reshape(shape)
    for n in range(4):
        for axis in ((0,),(1,),(2,),(3,),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),
                     (0,1,2),(0,1,3),(0,2,3),(1,2,3),(1,2,3),(0,1,2,3)):
            expected = v.copy()
            for a in axis:
                expected = np.roll(expected, n, a)
            result = RollOperator(axis=axis, n=n)(v)
            yield assert_equal, result, expected
