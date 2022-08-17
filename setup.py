#!/usr/bin/env python
import numpy as np
import sys
from distutils.util import get_platform

import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

import hooks
from hooks import cmdclass

hooks.F2PY_TABLE = {
    'integer': {'int8': 'char', 'int16': 'short', 'int32': 'int', 'int64': 'long_long'},
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
            'src/module_precision.f90',
            'src/module_tamasis.f90',
            'src/module_string.f90',
            'src/module_fitstools.f90',
            'src/module_geometry.f90.src',
            'src/module_math.f90.src',
            'src/module_math_old.f90',
            'src/module_pointingmatrix.f90',
            'src/module_operators.f90.src',
            'src/module_sort.f90',
            'src/module_wcs.f90',
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
            'src/datautils.f90.src',
            'src/geometry.f90',
            'src/operators.f90.src',
            'src/pointingmatrix_old.f90',
            'src/projection.f90.src',
            'src/sparse.f90.src',
            'src/wcsutils.f90',
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
