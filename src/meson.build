# Automatically discover all Python source files recursively
python_sources = run_command(
  'find', 'pysimulators', '-name', '*.py', '-type', 'f',
  check: true,
  capture: true
).stdout().strip().split('\n')

# Install Python sources
py.install_sources(
  python_sources,
  preserve_path: true,
)

# Build Fortran extensions
subdir('flib')
