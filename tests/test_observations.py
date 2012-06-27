import numpy as np

from numpy.testing import assert_raises
from pysimulators.observations import MaskPolicy
from pysimulators.utils import all_eq

flags = ['bad', 'u1', 'u2']


def test_mask_policy1():
    good_policy = ['kEep', 'removE', 'MASK']
    mask_policy = MaskPolicy(flags, good_policy)
    assert all_eq(np.array(mask_policy), (0, 2, 1))
    assert (
        mask_policy.bad == 'keep'
        and mask_policy.u1 == 'remove'
        and mask_policy.u2 == 'mask'
    )


def test_mask_policy2():
    bad_policy = ['remove', 'KKeep']
    assert_raises(ValueError, MaskPolicy, flags, bad_policy)
    assert_raises(KeyError, MaskPolicy, flags[0:2], bad_policy)
