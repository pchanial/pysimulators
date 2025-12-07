# Fortran library and f2py extension for pysimulators
# ======================================================

# Generate Fortran source files from Jinja2 templates
# ----------------------------------------------------

generate_fortran = find_program(meson.project_source_root() / 'tools' / 'generate_fortran.py')

# Template files that need to be processed
template_files = [
  'module_geometry.f90.j2',
  'module_math.f90.j2',
  'module_operators.f90.j2',
  'datautils.f90.j2',
  'operators.f90.j2',
  'projection.f90.j2',
  'sparse.f90.j2',
]

# Generate .f90 files from templates
generated_fortran_sources = []
foreach template : template_files
  output_file = template.replace('.j2', '')
  gen_src = custom_target(
    output_file,
    input: template,
    output: output_file,
    command: [
      generate_fortran,
      '@INPUT@',
      '-o', meson.current_build_dir(),
      '--precision-real', precision_real.to_string(),
    ],
    install: false,
  )
  generated_fortran_sources += gen_src
endforeach

# Static Fortran library (fmod)
# ------------------------------

# Fortran module sources (in dependency order)
fmod_sources = [
  'module_precision.f90',
  'module_tamasis.f90',
  'module_string.f90',
  'module_fitstools.f90',
  'module_pointingmatrix.f90',
  'module_sort.f90',
  'module_wcs.f90',
  'module_math_old.f90',
]

# Add generated sources for the static library
fmod_generated = [
  'module_geometry.f90',
  'module_math.f90',
  'module_operators.f90',
]

# Get the generated source objects
fmod_gen_objs = []
foreach gen_file : fmod_generated
  foreach gen_src : generated_fortran_sources
    if gen_src.full_path().endswith(gen_file)
      fmod_gen_objs += gen_src
      break
    endif
  endforeach
endforeach

# Build static library
fmod_lib = static_library(
  'fmod',
  fmod_sources + fmod_gen_objs,
  fortran_args: fortran_args,
  dependencies: [openmp_dep],
  install: false,
)

# f2py extension module (_flib)
# ------------------------------
#
# We use Meson's native f2py integration:
# 1. custom_target generates the f2py wrappers (_flibmodule.c, _flib-f2pywrappers.f)
# 2. py.extension_module compiles everything with proper dependencies (OpenMP, etc.)

# Fortran sources for the Python extension
flib_sources = [
  'geometry.f90',
  'pointingmatrix_old.f90',
  'wcsutils.f90',
]

# Generated sources for f2py (excluding files with type(c_ptr) that f2py can't handle)
flib_generated = [
  'datautils.f90',
  'operators.f90',
  'projection.f90',
  'sparse.f90',
]

# Get the generated source files
# Note: We use endswith with '/' prefix to avoid matching 'module_operators.f90' when looking for 'operators.f90'
flib_gen_files = []
foreach gen_file : flib_generated
  foreach gen_src : generated_fortran_sources
    if gen_src.full_path().endswith('/' + gen_file)
      flib_gen_files += gen_src
      break
    endif
  endforeach
endforeach

# f2py include directory (for fortranobject.c and fortranobject.h)
incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

# Create .f2py_f2cmap file to fix type mappings
# f2py maps real64 to float by default, we need double
# This file must be in the build directory where f2py runs
f2cmap_gen = custom_target(
  'f2py_f2cmap',
  output: '.f2py_f2cmap',
  command: [
    py, '-c',
    '''import sys; open(sys.argv[1], "w").write("""{
    'integer': {'int8': 'char', 'int16': 'short', 'int32': 'int', 'int64': 'long_long'},
    'real': {'real32': 'float', 'real64': 'double', 'p': 'double'},
    'complex': {'real32': 'complex_float', 'real64': 'complex_double'},
}
""")''',
    '@OUTPUT@',
  ],
)

# Generate f2py wrappers
# Note: f2py generates _flibmodule.c and _flib-f2pywrappers.f in the current directory
# We use a helper script to run f2py in the correct output directory
generate_f2py_wrappers = find_program(meson.project_source_root() / 'tools' / 'generate_f2py_wrappers.py')

# f2py generates different wrapper filenames depending on source format:
# - .f (fixed form): _flib-f2pywrappers.f
# - .f90 (free form): _flib-f2pywrappers2.f90
_flib_wrappers = custom_target(
  '_flib_wrappers',
  input: flib_sources + flib_gen_files,
  output: ['_flibmodule.c', '_flib-f2pywrappers2.f90'],
  command: [
    generate_f2py_wrappers,
    '--output-dir', '@OUTDIR@',
    '--module-name', '_flib',
    '@INPUT@',
  ],
  depends: [f2cmap_gen],
)

# Build the extension module using Meson's py.extension_module
# This properly handles OpenMP and links with libfmod.a
# Note: On macOS, use GCC (not Apple Clang) so that -fopenmp works
py.extension_module(
  '_flib',
  [flib_sources, flib_gen_files, _flib_wrappers, incdir_f2py / 'fortranobject.c'],
  include_directories: [incdir_numpy, incdir_f2py],
  dependencies: [py_dep, openmp_dep],
  link_with: fmod_lib,
  fortran_args: fortran_args,
  install: true,
  subdir: 'pysimulators',
)
