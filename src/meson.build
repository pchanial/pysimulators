# Python source files
python_sources = [
  'pysimulators/__init__.py',
  'pysimulators/acquisitions.py',
  'pysimulators/beams.py',
  'pysimulators/datatypes.py',
  'pysimulators/datautils.py',
  'pysimulators/geometry.py',
  'pysimulators/instruments.py',
  'pysimulators/mpiutils.py',
  'pysimulators/noises.py',
  'pysimulators/operators.py',
  'pysimulators/quantities.py',
  'pysimulators/sparse.py',
  'pysimulators/warnings.py',
  'pysimulators/wcsutils.py',
]

# Install Python sources
py.install_sources(
  python_sources,
  preserve_path: true,
)

# Install interfaces subpackage
py.install_sources(
  [
    'pysimulators/interfaces/__init__.py',
    'pysimulators/interfaces/madmap1.py',
  ],
  preserve_path: true,
)

# Install interfaces/healpy subpackage
py.install_sources(
  [
    'pysimulators/interfaces/healpy/__init__.py',
    'pysimulators/interfaces/healpy/operators.py',
    'pysimulators/interfaces/healpy/scenes.py',
  ],
  preserve_path: true,
)

# Install packedtables subpackage
py.install_sources(
  [
    'pysimulators/packedtables/__init__.py',
    'pysimulators/packedtables/core.py',
    'pysimulators/packedtables/layouts.py',
    'pysimulators/packedtables/samplings.py',
    'pysimulators/packedtables/scenes.py',
  ],
  preserve_path: true,
)

# Build Fortran extensions
subdir('flib')
