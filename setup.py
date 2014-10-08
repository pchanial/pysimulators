#!/usr/bin/env python
import numpy as np
import hooks
import sys

from distutils.util import get_platform
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from numpy.distutils.misc_util import Configuration
from hooks import get_cmdclass, get_version

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

VERSION = '1.1'

name = 'pysimulators'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'
define_macros = [('GFORTRAN', None), ('PRECISION_REAL', 8)]
extra_f90_compile_args = [
    '-Ofast -funroll-loops -march=native -cpp',
    '-fopenmp -fpack-derived -Wall -g',
]
# debugging options: -fcheck=all -fopt-info-vec-missed
# -fprofile-use -fprofile-correction
# ifort: -xHost -vec-report=1
mod_dir = 'build/temp.' + get_platform() + '-%s.%s' % sys.version_info[:2]

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
            'src/module_sort.f90',
            'src/module_wcs.f90',
            'src/module_pointingmatrix.f90',
        ],
        'depends': [],
        'macros': define_macros,
        'extra_f90_compile_args': extra_f90_compile_args,
        'include_dirs': [np.get_include()],
    },
)

ext_modules = [
    Extension(
        'pysimulators._flib',
        sources=[
            'src/datautils.f90',
            'src/geometry.f90',
            'src/operators.f90.src',
            'src/pointingmatrix_old.f90',
            'src/projection.f90.src',
            'src/sparse.f90.src',
            'src/wcsutils.f90',
        ],
        define_macros=define_macros,
        extra_compile_args=['-Wno-format'],
        extra_f90_compile_args=extra_f90_compile_args,
        include_dirs=[np.get_include(), mod_dir],
        libraries=['gomp', flib],
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
    cmdclass=get_cmdclass(),
    ext_modules=ext_modules,
    license='CeCILL-B',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2 :: Only',
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
