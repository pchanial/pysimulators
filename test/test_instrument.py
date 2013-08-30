from pyoperators import I
from pyoperators.utils.testing import assert_raises, assert_same
from pysimulators import Instrument, Imager, Layout


def test_instrument():

    def func(cls, keywords):
        instrument = cls(name, Layout(shape), **keywords)
        assert instrument.name == name
        assert instrument.detector.shape == shape
        assert len(instrument.detector) == len(instrument.detector.packed)
        assert instrument

    name = 'instrument'
    shape = (3, 2)
    yield func, Instrument, {}
    yield func, Imager, {'object2image': I}
    yield func, Imager, {'image2object': I}


def test_error():
    assert_raises(ValueError, Imager, 'instrument', (3, 2))


def test_pack_unpack():
    layout = Layout(4, removed=[False, True, False, False])
    instrument = Instrument('instrument', layout)
    v = [1, 2, 3, 4]
    assert_same(instrument.pack(v), [1, 3, 4])
    assert_same(instrument.unpack([1, 3, 4]), [1, -1, 3, 4])
