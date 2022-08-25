from __future__ import absolute_import, division, print_function
from warnings import warn
import warnings


class PySimulatorsWarning(UserWarning):
    pass


class PySimulatorsDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter('always', category=PySimulatorsWarning)
warnings.simplefilter('module', category=PySimulatorsDeprecationWarning)
del warnings
