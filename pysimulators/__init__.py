from pyoperators.utils.mpi import MPI

# force gfortran's read statement to always use the dot sign as fraction
# separator (PR47007)
import locale

locale.setlocale(locale.LC_NUMERIC, 'POSIX')
del locale

from . import _flib
from . import geometry
from .version import VERSION as __version__

del version
from .acquisitionmodels import *
from .datatypes import *
from .instruments import *

# from .mpiutils import *
from .configurations import *
from .pointings import *

# from .processing import *
from .quantities import *
from .datautils import *
from .wcsutils import *
