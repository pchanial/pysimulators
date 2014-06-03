from __future__ import absolute_import, division, print_function
import warnings
from warnings import warn


class PySimulatorsWarning(UserWarning):
    pass


class PySimulatorsDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter('always', category=PySimulatorsWarning)
warnings.simplefilter('module', category=PySimulatorsDeprecationWarning)
del warnings
