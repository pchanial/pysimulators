#!/usr/bin/env python
import numpy as np
import sys
from distutils.util import get_platform

import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

import hooks
from hooks import cmdclass, get_version

VERSION = '1.1'

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

name = 'pysimulators'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'
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
    name=name,
    version=get_version(name, VERSION),
    description='Tools to build an instrument model.',
    long_description=long_description,
    url='http://pchanial.github.com/pysimulators',
    author='Pierre Chanial',
    author_email='pierre.chanial@gmail.com',
    maintainer='Pierre Chanial',
    maintainer_email='pierre.chanial@gmail.com',
    install_requires=['pyoperators>=0.12.15', 'astropy>=0.3.2'],
    packages=[
        'pysimulators',
        'pysimulators/interfaces',
        'pysimulators/interfaces/healpy',
        'pysimulators/interfaces/madmap1',
        'pysimulators/packedtables',
    ],
    platforms=platforms.split(','),
    keywords=keywords.split(','),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    license='CeCILL-B',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Fortran',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
