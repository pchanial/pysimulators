#!/usr/bin/env python
import numpy as np
import os
import sys

from distutils.util import get_platform
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from hooks import get_cmdclass, get_version

VERSION = '1.0'

name = 'pysimulators'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'
define_macros = [('GFORTRAN', None), ('PRECISION_REAL', 8)]
extra_f90_compile_args = ['-g -cpp -fopenmp -fpack-derived']

if any(c in sys.argv for c in ('build', 'build_ext')):
    # write f2py's type mapping file
    root = os.path.dirname(__file__)
    with open(os.path.join(root, '.f2py_f2cmap'), 'w') as f:
        f.write("{'real': {'sp': 'float', 'dp': 'double', 'p': 'double'},"
                " 'complex': {'sp': 'complex', 'dp': 'complex_double', 'p': 'complex_double'}}\n")


def configuration(parent_package='', top_path=None):
    config = Configuration('', parent_package, top_path)
    config.add_library('fmod',
                       sources=['src/module_precision.f90',
                                'src/module_tamasis.f90',
                                'src/module_string.f90',
                                'src/module_fitstools.f90',
                                'src/module_geometry.f90.src',
                                'src/module_math.f90',
                                'src/module_math2.f90.src',
                                'src/module_sort.f90',
                                'src/module_projection.f90',
                                'src/module_wcs.f90',
                                'src/module_pointingmatrix.f90'],
                       macros=define_macros,
                       extra_f90_compile_args=extra_f90_compile_args)

    # how to get the build base ?
    temp_dir = 'build/temp.' + get_platform() + '-%s.%s' % sys.version_info[:2]
    config.add_extension('pysimulators._flib',
                         sources=['pysimulators/module_datautils.f90',
                                  'pysimulators/module_geometry.f90',
                                  'src/operators.f90.src',
                                  'pysimulators/module_pointingmatrix.f90',
                                  'src/projection.f90.src',
                                  'src/sparse.f90.src',
                                  'pysimulators/module_wcsutils.f90'],
                         define_macros=define_macros,
                         extra_f90_compile_args=extra_f90_compile_args,
                         include_dirs=['.', np.get_include(), temp_dir],
                         libraries=['fmod', 'gomp'])
    return config


setup(configuration=configuration,
      name=name,
      version=get_version(name, VERSION),
      description='Tools to build an instrument model.',
      long_description=long_description,
      url='http://pchanial.github.com/pysimulators',
      author='Pierre Chanial',
      author_email='pierre.chanial@gmail.com',
      maintainer='Pierre Chanial',
      maintainer_email='pierre.chanial@gmail.com',
      install_requires=['pyoperators>=0.12.4', 'astropy>=0.3.2'],
      packages=['pysimulators',
                'pysimulators/interfaces',
                'pysimulators/interfaces/healpy',
                'pysimulators/interfaces/madmap1',
                'pysimulators/packedtables'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass=get_cmdclass(),
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
          'Topic :: Scientific/Engineering :: Physics'])
