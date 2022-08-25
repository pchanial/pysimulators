import warnings


class PySimulatorsWarning(UserWarning):
    pass


class PySimulatorsDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter('always', category=PySimulatorsWarning)
warnings.simplefilter('module', category=PySimulatorsDeprecationWarning)
del warnings
