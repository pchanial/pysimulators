from pyoperators.utils.mpi import MPI

# force gfortran's read statement to always use the dot sign as fraction
# separator (PR47007)
import locale

locale.setlocale(locale.LC_NUMERIC, 'POSIX')
del locale
from importlib.metadata import version as _version

from . import _flib
from . import geometry
from .beams import *
from .operators import *
from .datatypes import *
from .packedtables import *
from .instruments import *

# from .mpiutils import *
from .acquisitions import *

# from .processing import *
from .quantities import *
from .datautils import *
from .sparse import *
from .wcsutils import *

__version__ = _version('pysimulators')