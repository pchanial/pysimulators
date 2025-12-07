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
    if gen_src.full_path().contains(gen_file)
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

# For now, we use a Python script to build the f2py extension
# This is simpler than trying to integrate f2py directly into Meson

build_f2py_script = find_program(meson.project_source_root() / 'tools' / 'build_flib.py')

# Fortran sources for the Python extension (uses the static library)
flib_sources = [
  'geometry.f90',
  'pointingmatrix_old.f90',
  'wcsutils.f90',
]

# Add generated sources for f2py
# Note: We must exclude any files that use type(c_ptr) as f2py cannot handle them
flib_generated = [
  'datautils.f90',
  'operators.f90',
  'projection.f90',
  'sparse.f90',
]

# Exclude module_operators.f90 as it contains type(c_ptr)
flib_generated_exclude = [
  'module_operators.f90',
]

# Get the generated source files
flib_gen_files = []
foreach gen_file : flib_generated
  foreach gen_src : generated_fortran_sources
    src_path = gen_src.full_path()
    # Skip files that contain c_ptr types
    skip = false
    foreach exclude : flib_generated_exclude
      if src_path.contains(exclude)
        skip = true
        break
      endif
    endforeach
    if not skip and src_path.contains(gen_file)
      flib_gen_files += gen_src
      break
    endif
  endforeach
endforeach

# Determine the extension suffix for the current platform
py_suffix = run_command(py, ['-c',
  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'],
  check: true).stdout().strip()

# Build the extension using our custom Python script
_flib_target = custom_target(
  '_flib',
  input: flib_sources + flib_gen_files,
  output: '_flib@0@'.format(py_suffix),
  command: [
    py,
    build_f2py_script,
    '--build-dir', meson.current_build_dir(),
    '--source-dir', meson.current_source_dir(),
    '--static-lib', fmod_lib.full_path(),
    '--output', '@OUTPUT@',
    '--fortran-args', ' '.join(fortran_args),
    '--sources', '@INPUT@',
  ],
  depends: [fmod_lib] + flib_gen_files,
  install: true,
  install_dir: py.get_install_dir() / 'pysimulators',
  build_by_default: true,
)
