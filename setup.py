#!/usr/bin/env python
import numpy as np
import os

#from distutils.extension import Extension
from distutils.util import get_platform
from numpy.distutils.core import setup, Extension
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import Configuration
from glob import glob

def version():
    import os, re
    f = open(os.path.join('pysimulators', 'config.py')).read()
    m = re.search(r"VERSION = '(.*)'", f)
    return m.groups()[0]

version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

import sys
sys.argv += ['config_fc', "--f90flags='-cpp -DGFORTRAN -DPRECISION_REAL=8 -fopenmp'"]

ext_modules = [Extension('pysimulators._flib',
                         sources=glob('pysimulators/module_*f90'),
                         include_dirs=['.', np.get_include()],
                         f2py_options=['skip:', 'pmatrix_direct',
                                       'pmatrix_direct_one_pixel_per_sample',
                                       'pmatrix_transpose',
                                       'pmatrix_transpose_one_pixel_per_sample',
                                       'pmatrix_ptp',
                                       'pmatrix_mask',
                                       'pmatrix_pack',
                                       'backprojection_weight__inner',
                                       'intersection_polygon_unity_square',
                                       'intersection_segment_unity_square',
                                       'angle_lonlat__inner',
                                       ':',
                                      ])]

# write f2py's type mapping file
with open(os.path.join(os.path.dirname(__file__), '.f2py_f2cmap'), 'w') as f:
    f.write("{'real':{'p':'double'}, 'complex':{'p':'complex_double'}}\n")

def configuration(parent_package='',top_path=None):
    config = Configuration('', parent_package,top_path)
    config.add_library('fmod',
                       sources=['src/module_precision.f90',
                                'src/module_tamasis.f90',
                                'src/module_string.f90',
                                'src/module_fitstools.f90',
                                'src/module_math.f90',
                                'src/module_sort.f90',
                                'src/module_stack.f90',
                                'src/module_projection.f90',
                                'src/module_wcs.f90',
                                'src/module_pointingmatrix.f90'])

    # how to get the build base ?
    temp_dir = 'build/temp.' + get_platform() + '-%s.%s' % sys.version_info[:2]
    config.add_extension('pysimulators._flib',
                         sources=glob('pysimulators/module_*f90'),
                         include_dirs=['.', np.get_include(), temp_dir],
                         libraries=['fmod'],
                         f2py_options=['skip:', 'pmatrix_direct',
                                       'pmatrix_direct_one_pixel_per_sample',
                                       'pmatrix_transpose',
                                       'pmatrix_transpose_one_pixel_per_sample',
                                       'pmatrix_ptp',
                                       'pmatrix_mask',
                                       'pmatrix_pack',
                                       'backprojection_weight__inner',
                                       'intersection_polygon_unity_square',
                                       'intersection_segment_unity_square',
                                       ':',
                                      ],
                        )
    return config

setup(configuration=configuration,
      name='pysimulators',
      version=version,
      description='Tools to build an instrument model.',
      long_description=long_description,
      url='http://pchanial.github.com/pysimulators',
      author='Pierre Chanial',
      author_email='pierre.chanial@gmail.com',
      maintainer='Pierre Chanial',
      maintainer_email='pierre.chanial@gmail.com',
      install_requires=['pyoperators>=0.6.1', 'pyfits>=3'],
      packages=['pysimulators'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass = {'build_ext': build_ext},
#      ext_modules = ext_modules,
      license='CeCILL-B',
      classifiers = [
          'Programming Language :: Python',
          'Programming Language :: Python :: 2 :: Only',
          'Programming Language :: Fortran',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Physics',
          ])
