#!/usr/bin/env python
import numpy as np
import os
import re
import sys
import subprocess

from distutils.util import get_platform
from glob import glob
from numpy.distutils.core import setup, Command
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.misc_util import Configuration
from subprocess import Popen, PIPE

VERSION = '0.6.4'


def version_sdist():
    p = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE,
              stderr=PIPE)
    stdout, stderr = p.communicate()
    if stderr:
        return VERSION
    branch = stdout[:-1]
    if re.search('^v[0-9]', branch) is not None:
        branch = branch[1:]
    if branch != 'master':
        return VERSION
    p = Popen(['git', 'rev-parse', '--verify', '--short', 'HEAD'], stdout=PIPE,
              stderr=PIPE)
    stdout, stderr = p.communicate()
    version = VERSION
    if not stderr:
        version += '-' + stdout[:-1]
    return version

version = version_sdist()
if 'install' in sys.argv[1:]:
    if '-' in version:
        version = VERSION + '-dev'

if any(c in sys.argv[1:] for c in ('install', 'sdist')):
    init = open('pysimulators/__init__.py.in').readlines()
    init += ['\n', "__version__ = '" + version + "'\n"]
    open('pysimulators/__init__.py', 'w').writelines(init)

long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

if any(c in sys.argv for c in ('build', 'build_ext', 'config', 'install')):
    sys.argv += ['config_fc',
                 "--f90flags='-cpp -DGFORTRAN -DPRECISION_REAL=8 -fopenmp'"]

# write f2py's type mapping file
with open(os.path.join(os.path.dirname(__file__), '.f2py_f2cmap'), 'w') as f:
    f.write("{'real':{'p':'double'}, 'complex':{'p':'complex_double'}}\n")


class NewCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

class CoverageCommand(NewCommand):
    description = "run the package coverage"

    def run(self):
        subprocess.call(['nosetests', '--with-coverage', '--cover-package',
                         'pysimulators'])
        subprocess.call(['coverage', 'html'])


class TestCommand(NewCommand):
    description = "run the test suite"

    def run(self):
        subprocess.call(['nosetests', 'test'])


def configuration(parent_package='', top_path=None):
    config = Configuration('', parent_package, top_path)
    config.add_library('fmod',
                       sources=['src/module_precision.f90',
                                'src/module_tamasis.f90',
                                'src/module_string.f90',
                                'src/module_fitstools.f90',
                                'src/module_math.f90',
                                'src/module_sort.f90',
                                'src/module_projection.f90',
                                'src/module_wcs.f90',
                                'src/module_pointingmatrix.f90'])

    # how to get the build base ?
    temp_dir = 'build/temp.' + get_platform() + '-%s.%s' % sys.version_info[:2]
    config.add_extension('pysimulators._flib',
                         sources=glob('pysimulators/module_*f90'),
                         include_dirs=['.', np.get_include(), temp_dir],
                         libraries=['fmod', 'gomp'])
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
      install_requires=['pyoperators>=0.8', 'astropy>=0.2'],
      packages=['pysimulators', 'pysimulators/interfaces'],
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass={'build_ext': build_ext,
                'coverage': CoverageCommand,
                'test': TestCommand},
      license='CeCILL-B',
      classifiers=[
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
          'Topic :: Scientific/Engineering :: Physics'])
