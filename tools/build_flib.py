#!/usr/bin/env python3
"""
Script to build the _flib extension using f2py.

This script is called by Meson to build the Fortran extension.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Build _flib extension with f2py')
    parser.add_argument('--build-dir', required=True, help='Build directory')
    parser.add_argument('--source-dir', required=True, help='Source directory')
    parser.add_argument(
        '--static-lib', required=True, help='Path to static library (libfmod.a)'
    )
    parser.add_argument('--output', required=True, help='Output .so file')
    parser.add_argument('--fortran-args', default='', help='Fortran compiler arguments')
    parser.add_argument(
        '--sources', required=True, nargs='+', help='Fortran source files'
    )

    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    source_dir = Path(args.source_dir)
    output_file = Path(args.output)

    # Change to build directory
    os.chdir(build_dir)

    # Create .f2py_f2cmap file to fix type mappings
    # f2py by default maps real64 to float (4 bytes) instead of double (8 bytes)
    # Also map 'p' (precision parameter from module_tamasis) to double
    f2cmap_content = """dict(real=dict(real32='float', real64='double', p='double'),
     integer=dict(int8='signed char', int16='short', int32='int', int64='long long'))
"""
    with open('.f2py_f2cmap', 'w') as f:
        f.write(f2cmap_content)

    # Prepare source files list
    # Files can be absolute paths or relative to source/build dir
    source_files = []
    for src in args.sources:
        src_path = Path(src)

        # Try different possible locations
        candidates = [
            src_path if src_path.is_absolute() else None,
            source_dir / src_path,
            build_dir / src_path,
            build_dir / src_path.name,  # Just filename in build dir (generated files)
            source_dir / src_path.name,  # Just filename in source dir
        ]

        found = False
        for candidate in candidates:
            if candidate and candidate.exists():
                source_files.append(str(candidate.resolve()))
                found = True
                break

        if not found:
            print(f"Error: Cannot find source file {src}")
            print(f"  Looked in:")
            for candidate in candidates:
                if candidate:
                    print(f"    {candidate} (exists: {candidate.exists()})")
            sys.exit(1)

    # Build f2py command
    # NumPy 2.0 with Python >= 3.12 uses meson backend
    # which has different command-line options
    cmd = [
        sys.executable,
        '-m',
        'numpy.f2py',
        '-c',  # Compile
    ]

    # For Meson backend, we need to set environment variables instead of flags
    env = os.environ.copy()

    # Combine all Fortran flags
    all_fflags = ['-cpp', '-fopenmp']  # Always enable preprocessor and OpenMP
    if args.fortran_args:
        all_fflags.extend([arg for arg in args.fortran_args.split() if arg])

    # Add module directory from static library location
    # The .mod files are in the same directory as libfmod.a
    static_lib_path = Path(args.static_lib)
    module_dir = static_lib_path.parent / 'libfmod.a.p'
    if module_dir.exists():
        all_fflags.append(f'-I{module_dir}')  # For reading module files

    # Set environment variables for Meson backend
    env['FFLAGS'] = ' '.join(all_fflags)
    # Add static library to linker flags
    env['LDFLAGS'] = f'-fopenmp {args.static_lib}'  # OpenMP linking + static lib

    # Note: Not adding static library as a separate argument since
    # f2py's Meson backend doesn't handle it properly - using LDFLAGS instead

    # Add source files
    cmd.extend(source_files)

    # Module name
    cmd.extend(['-m', '_flib'])

    # Run f2py
    print(f"Running f2py command in {build_dir}:")
    print(' '.join(cmd))
    print()

    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True, env=env)

    # Always print stdout and stderr
    if result.stdout:
        print('f2py stdout:')
        print(result.stdout)
    if result.stderr:
        print('f2py stderr:')
        print(result.stderr)

    if result.returncode != 0:
        print(f"f2py failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Find the generated .so file
    # f2py creates a file like _flib.cpython-312-x86_64-linux-gnu.so
    so_files = list(build_dir.glob('_flib*.so'))

    if not so_files:
        print('Error: f2py did not generate expected .so file')
        print(f"Contents of {build_dir}:")
        for f in build_dir.iterdir():
            print(f"  {f}")
        sys.exit(1)

    generated_so = so_files[0]

    # Move/copy the file to the expected output location
    output_path = build_dir / output_file.name
    if generated_so != output_path:
        shutil.move(str(generated_so), str(output_path))

    print(f"Successfully built {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
