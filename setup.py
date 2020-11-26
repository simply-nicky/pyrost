from setuptools import setup
from distutils.core import Extension
import numpy
import sys
import os
import cython_gsl

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON:
    extensions = [Extension(name='beam_calc',
                            sources=["pyrost/bin/beam_calc.pyx"],
                            language="c",
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-lomp'],
                            libraries=cython_gsl.get_libraries(),
                            library_dirs=[cython_gsl.get_library_dir(), '/usr/local/lib',
                                          os.path.join(sys.prefix, 'lib')],
                            include_dirs=[numpy.get_include(),
                                          cython_gsl.get_cython_include_dir(),
                                          os.path.join(sys.prefix, 'include')],
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
                  Extension(name='st_utils',
                            sources=["pyrost/bin/st_utils.pyx"],
                            language="c",
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-lomp'],
                            libraries=cython_gsl.get_libraries(),
                            library_dirs=[cython_gsl.get_library_dir(), '/usr/local/lib',
                                          os.path.join(sys.prefix, 'lib')],
                            include_dirs=[numpy.get_include(),
                                          cython_gsl.get_cython_include_dir(),
                                          os.path.join(sys.prefix, 'include')],
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]
    extensions = cythonize(extensions, annotate=False, language_level="3",
                           build_dir="build",
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True,
                                                'embedsignature': True})
else:
    extensions = [Extension(name="pyrost/bin/*",
                            sources=["pyrost/bin/*.c"],
                            include_dirs=[numpy.get_include()])]

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='pyrost',
      version='0.1.0',
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/rst",
      install_requires=[
          'Cython',
          'CythonGSL',
          'h5py',
          'numpy',
          'scipy',
      ],
      extras_require={'interactive': ['matplotlib', 'jupyter', 'pyximport']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      ext_modules=extensions,
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.7',
      options={'build': {'build_lib': 'pyrost/bin'}})
