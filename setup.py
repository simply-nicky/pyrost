import os
import sys
from setuptools import setup, find_packages
from distutils.core import Extension
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'
extension_args = {'language': 'c',
                  'extra_compile_args': ['-fopenmp', '-std=c99'],
                  'extra_link_args': ['-lgomp', '-Wl,-rpath,/usr/local/lib'],
                  'libraries': ['gsl', 'gslcblas', 'fftw3', 'fftw3_omp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(os.path.dirname(__file__), 'pyrost/include')]}

src_files = ["pyrost/include/pocket_fft.c", "pyrost/include/fft_functions.c",
             "pyrost/include/array.c", "pyrost/include/routines.c", "pyrost/include/median.c"]
extensions = [Extension(name='pyrost.bin.simulation',
                        sources=['pyrost/bin/simulation' + ext,] + src_files, **extension_args),
              Extension(name='pyrost.bin.pyrost',
                        sources=['pyrost/bin/pyrost' + ext,], **extension_args),
              Extension(name='pyrost.bin.pyfftw',
                        sources=['pyrost/bin/pyfftw' + ext,], **extension_args)]

if USE_CYTHON:
    extensions = cythonize(extensions, annotate=True, language_level="3",
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True,
                                                'embedsignature': True})

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='pyrost',
      version='0.5.2',
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/pyrost",
      packages=find_packages(),
      include_package_data=True,
      package_data={'pyrost': ['config/*.ini', 'ini_templates/*.ini', 'data/*.npz']},
      install_requires=['h5py', 'numpy', 'scipy'],
      extras_require={'interactive': ['matplotlib', 'jupyter', 'pyximport', 'ipykernel', 'ipywidgets']},
      ext_modules=extensions,
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.6')
