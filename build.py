import os
import sys
import numpy
from distutils.core import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"
extension_args = {
    "language": "c",
    "extra_compile_args": ["-fopenmp", "-std=c99"],
    "extra_link_args": ["-lgomp", "-Wl,-rpath,/usr/local/lib"],
    "libraries": ["gsl", "gslcblas", "fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp"],
    "library_dirs": ["/usr/local/lib", os.path.join(sys.prefix, "lib")],
    "include_dirs": [
        numpy.get_include(),
        os.path.join(sys.prefix, "include"),
        os.path.join(os.path.dirname(__file__), "pyrost/include"),
        os.path.join(os.path.dirname(__file__), "pyrost/bin"),
    ],
}

src_files = [
    "pyrost/include/pocket_fft.c",
    "pyrost/include/fft_functions.c",
    "pyrost/include/array.c",
    "pyrost/include/routines.c",
    "pyrost/include/median.c",
]
extensions = [
    Extension(
        name="pyrost.bin.simulation",
        sources=[
            "pyrost/bin/simulation" + ext,
        ]
        + src_files,
        **extension_args
    ),
    Extension(
        name="pyrost.bin.pyfftw",
        sources=[
            "pyrost/bin/pyfftw" + ext,
        ],
        **extension_args
    ),
    Extension(
        name="pyrost.bin.pyrost",
        sources=[
            "pyrost/bin/pyrost" + ext,
        ],
        **extension_args
    ),
]

if USE_CYTHON:
    extensions = cythonize(
        extensions,
        annotate=True,
        language_level="3",
        compiler_directives={
            "cdivision": True,
            "boundscheck": False,
            "wraparound": False,
            "binding": True,
            "embedsignature": False,
        },
    )


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": extensions})
