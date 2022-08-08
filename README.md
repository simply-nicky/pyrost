## Current build status
[![PyPI](https://img.shields.io/pypi/v/pyrost?color=brightgreen)](https://pypi.org/project/pyrost/)
[![Documentation Status](https://readthedocs.org/projects/robust-speckle-tracking/badge/?version=latest)](https://robust-speckle-tracking.readthedocs.io/en/latest/?badge=latest)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pyrost?color=brightgreen)](https://anaconda.org/conda-forge/pyrost)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6574364.svg)](https://doi.org/10.5281/zenodo.6574364)

# pyrost
Python Robust Speckle Tracking (**pyrost**) is a library for wavefront metrology
and sample imaging based on ptychographic speckle tracking algorithm. This
project takes over Andrew Morgan's [speckle_tracking](https://github.com/andyofmelbourne/speckle-tracking)
project as an improved version aiming to add robustness to the optimisation
algorithm in the case of the high noise present in the measured data.

The documentation can be found on [Read the Docs](https://robust-speckle-tracking.readthedocs.io/en/latest/).

Read the open access scientific article about the technical details of the developed technique published in [Optica Express](https://opg.optica.org/oe/abstract.cfm?URI=oe-30-14-25450).

## Dependencies

- [Python](https://www.python.org/) 3.6 or later (Python 2.x is **not** supported).
- [GNU Scientific Library](https://www.gnu.org/software/gsl/) 2.4 or later.
- [LLVM's OpenMP library](http://openmp.llvm.org) 10.0.0 or later.
- [FFTW](http://www.fftw.org) 3.3.8 or later.
- [h5py](https://www.h5py.org) 2.10.0 or later.
- [NumPy](https://numpy.org) 1.19.0 or later.
- [SciPy](https://scipy.org) 1.5.2 or later.
- [tqdm](https://github.com/tqdm/tqdm) 4.56.0 or later.

## Installation
We recommend **not** building from source, but install the release via the
[conda](https://anaconda.org/conda-forge/pyrost) manager:

    conda install -c conda-forge pyrost

The package is available in [conda-forge](https://anaconda.org/conda-forge/pyrost) on OS X and Linux.

Also you can install the release from [pypi](https://pypi.org/project/pyrost/)
with the pip package installer:

    pip install pyrost

The source distribution is available in [pypi](https://pypi.org/project/pyrost/) as for now.

## Installation from source
In order to build the package from source simply execute the following command:

    python setup.py install

or:

    pip install -r requirements.txt -e . -v

That cythonizes the Cython extensions and builds them into ``/pyrost/bin``.