from pyoperators.utils.mpi import MPI

# force gfortran's read statement to always use the dot sign as fraction
# separator (PR47007)
import locale
locale.setlocale(locale.LC_NUMERIC, 'POSIX')
del locale

from . import _flib
from . import geometry
from .operators import *
from .datatypes import *
from .layouts import *
from .discretesurfaces import *
from .instruments import *
#from .mpiutils import *
from .acquisitions import *
from .pointings import *
#from .processing import *
from .quantities import *
from .datautils import *
from .wcsutils import *

import pkg_resources  # part of setuptools
__version__ = pkg_resources.require("pysimulators")[0].version
