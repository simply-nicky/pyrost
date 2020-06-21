from __future__ import absolute_import

from distutils.core import setup, Extension
import numpy
import cython_gsl

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON:
    extensions = [Extension(name='beam_calc', sources=["beam_calc.pyx"], language="c",
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-lomp'],
                            libraries=cython_gsl.get_libraries(),
                            library_dirs=[cython_gsl.get_library_dir(), '/usr/local/lib'],
                            include_dirs=[numpy.get_include(), cython_gsl.get_cython_include_dir()])]
    extensions = cythonize(extensions, annotate=False, language_level="3",
                           compiler_directives={'cdivision': True,
                                                'boundscheck': False,
                                                'wraparound': False,
                                                'binding': True})
else:
    extensions = [Extension(name="*", sources=["*.c"], include_dirs=[numpy.get_include()])]

setup(ext_modules=extensions)