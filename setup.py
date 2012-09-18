#!/usr/bin/env python
import numpy as np

# from distutils.extension import Extension
from numpy.distutils.core import setup, Extension
from numpy.distutils.command.build_ext import build_ext


def version():
    import os, re

    f = open(os.path.join('pysimulators', 'config.py')).read()
    m = re.search(r"VERSION = '(.*)'", f)
    return m.groups()[0]


version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

ext_modules = [
    Extension(
        'pysimulators._wcsutils',
        sources=['pysimulators/module_wcsutils.f90'],
        include_dirs=['.', np.get_include()],
        f2py_options=[],
    )
]

setup(
    name='pysimulators',
    version=version,
    description='Tools to build an instrument model.',
    long_description=long_description,
    url='http://pchanial.github.com/pysimulators',
    author='Pierre Chanial',
    author_email='pierre.chanial@gmail.com',
    maintainer='Pierre Chanial',
    maintainer_email='pierre.chanial@gmail.com',
    requires=['numpy (>1.6)', 'scipy (>0.9)', 'pyoperators'],
    packages=['pysimulators'],
    platforms=platforms.split(','),
    keywords=keywords.split(','),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    license='CeCILL-B',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Astrophysics',
    ],
)
