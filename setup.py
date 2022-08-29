#!/usr/bin/env python
import sys
from distutils.util import get_platform

import hooks
import numpy as np
import setuptools
from hooks import cmdclass
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

hooks.F2PY_TABLE = {
    'integer': {
        'int8': 'char',
        'int16': 'short',
        'int32': 'int',
        'int64': 'long_long',
    },
    'real': {
        'sp': 'float',
        'dp': 'double',
        'p': 'double',
        'real32': 'float',
        'real64': 'double',
    },
    'complex': {
        'sp': 'complex_float',
        'dp': 'complex_double',
        'p': 'complex_double',
        'real32': 'complex_float',
        'real64': 'complex_double',
    },
}
hooks.F90_COMPILE_ARGS_GFORTRAN += ['-fpack-derived']
hooks.F90_COMPILE_ARGS_IFORT += ['-align norecords']
if sys.platform == 'darwin':
    hooks.F90_COMPILE_OPT_GFORTRAN = ['-O2']

define_macros = [('GFORTRAN', None), ('PRECISION_REAL', 8)]
setuptools_version = tuple(map(int, setuptools.__version__.split('.')))
if setuptools_version >= (62, 1):
    mod_dir = f'build/temp.{get_platform()}-{sys.implementation.cache_tag}'
else:
    major, minor = sys.version_info[:2]
    mod_dir = f'build/temp.{get_platform()}-{major}.{minor}'

flib = (
    'fmod',
    {
        'sources': [
            'src/flib/module_precision.f90',
            'src/flib/module_tamasis.f90',
            'src/flib/module_string.f90',
            'src/flib/module_fitstools.f90',
            'src/flib/module_geometry.f90.src',
            'src/flib/module_math.f90.src',
            'src/flib/module_math_old.f90',
            'src/flib/module_pointingmatrix.f90',
            'src/flib/module_operators.f90.src',
            'src/flib/module_sort.f90',
            'src/flib/module_wcs.f90',
        ],
        'depends': [],
        'macros': define_macros,
        'include_dirs': [np.get_include()],
    },
)

ext_modules = [
    Extension(
        'pysimulators._flib',
        sources=[
            'src/flib/datautils.f90.src',
            'src/flib/geometry.f90',
            'src/flib/operators.f90.src',
            'src/flib/pointingmatrix_old.f90',
            'src/flib/projection.f90.src',
            'src/flib/sparse.f90.src',
            'src/flib/wcsutils.f90',
        ],
        define_macros=define_macros,
        include_dirs=[np.get_include(), mod_dir],
        libraries=[flib],
    )
]


setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
