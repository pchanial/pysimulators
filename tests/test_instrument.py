import pytest
from numpy.testing import assert_equal

from pyoperators import I
from pysimulators import Imager, Instrument, PackedTable


@pytest.mark.parametrize(
    'cls, keywords',
    [
        (Instrument, {}),
        (Imager, {'object2image': I}),
        (Imager, {'image2object': I}),
    ],
)
def test_instrument(cls, keywords):
    name = 'instrument'
    shape = (3, 2)
    instrument = cls(name, PackedTable(shape), **keywords)
    assert instrument.name == name
    assert instrument.detector.shape == shape
    assert len(instrument.detector) == len(instrument.detector.all)
    assert instrument


def test_error():
    with pytest.raises(ValueError):
        Imager('instrument', (3, 2))


def test_pack_unpack():
    layout = PackedTable(4, selection=[True, False, True, True])
    instrument = Instrument('instrument', layout)
    v = [1, 2, 3, 4]
    assert_equal(instrument.pack(v), [1, 3, 4])
    assert_equal(instrument.unpack([1, 3, 4]), [1, -1, 3, 4])
