"""
Setup script for QVCache Python bindings
Uses CMake to build the extension module
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension
from pybind11 import get_cmake_dir
import pybind11

# Get the directory containing this setup.py
SETUP_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SETUP_DIR.parent


class CMakeExtension(Extension):
    """Wrapper for CMake-based extensions"""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext that uses CMake"""
    
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
            print(f"Found CMake: {out.decode('utf-8').split()[2]}")
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DBUILD_PYTHON_BINDINGS=ON',
            f'-DCMAKE_BUILD_TYPE=Release',
        ]
        
        # Add platform-specific flags
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        if sys.platform.startswith('darwin'):
            cmake_args += ['-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64']
        
        # Use Ninja if available, otherwise make
        if shutil.which('ninja'):
            cmake_args += ['-GNinja']
        
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        
        # Configure CMake
        subprocess.check_call(['cmake', str(PROJECT_ROOT)] + cmake_args, cwd=build_temp)
        
        # Build
        build_args = []
        if shutil.which('ninja'):
            build_args = ['--build', '.', '--target', 'qvcache_python_module']
        else:
            build_args = ['--build', '.', '--target', 'qvcache_python_module', '--', '-j']
        
        subprocess.check_call(['cmake'] + build_args, cwd=build_temp)
        
        # Copy the built module to the extension directory
        built_module = None
        for suffix in ['.so', '.pyd', '.dylib']:
            candidate = os.path.join(build_temp, f'qvcache{suffix}')
            if os.path.exists(candidate):
                built_module = candidate
                break
        
        if not built_module:
            # Try python directory (CMake might output there)
            for suffix in ['.so', '.pyd', '.dylib']:
                pattern = f'qvcache*{suffix}'
                import glob
                candidates = glob.glob(os.path.join(PROJECT_ROOT, 'python', pattern))
                if candidates:
                    built_module = candidates[0]
                    break
        
        if built_module and os.path.exists(built_module):
            dest = self.get_ext_fullpath(ext.name)
            dest_dir = os.path.dirname(dest)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy(built_module, dest)
            print(f"Copied {built_module} to {dest}")
        else:
            raise RuntimeError(f"Could not find built qvcache module. Searched in {build_temp}")


# Read README if it exists
readme_file = SETUP_DIR / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()


# Collect Python files for the package
package_files = ['bruteforce_backend.py']

setup(
    name="qvcache",
    version="0.1.0",
    author="Anıl Eren Göçer",
    author_email="agoecer@ethz.ch",
    license="MIT",
    description="Python bindings for QVCache - A vector cache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension('qvcache')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    packages=find_packages(where=".") if find_packages(where=".") else [],
    py_modules=['bruteforce_backend'],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
