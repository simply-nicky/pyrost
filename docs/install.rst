Installation
============

Dependencies
------------
pyrost has the following **mandatory** runtime dependencies:

* `Python <https://www.python.org/>`_ 3.6 or later (Python 2.x is
  **not** supported).
* `FFTW library <http://www.fftw.org/#documentation>`_ 3.3.8 or later,
  which is used for fast Fourier transform computations.
* `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ 2.4
  or later, which is used for pseudo-random number generation.
* `LLVM's OpenMP library <http://openmp.llvm.org>`_ 10.0.0 or later, which
  is used for parallelization.
* `h5py <https://www.h5py.org>`_ 2.10.0 or later, which is used to work with
  CXI files.
* `NumPy <https://numpy.org>`_ 1.19.0 or later.
* `SciPy <https://scipy.org>`_ 1.5.2 or later.
* `tqdm <https://github.com/tqdm/tqdm>`_ 4.56.0 or later.

Packages
--------
pyrost packages are available through `pypi <https://pypi.org/project/pyrost/>`_ and
`conda <https://anaconda.org/conda-forge/pyrost>`_ on OS X and Linux platforms.

conda
^^^^^
pyrost binary distribution is available via the `conda <https://anaconda.org/conda-forge/pyrost>`_
package manager for Linux and OSX (Windows is **not** supported) in `conda-forge <https://conda-forge.org/>`_
channel. In order to install pyrost via conda, you just need to add `conda-forge`
to the channels, and install as follows:

.. code-block:: console

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install pyrost

The conda packages for pyrost are regularly updated when a new version is released.

pip
^^^
pyrost source distribution is available via the `pip <https://pip.pypa.io/en/stable/>`_
package installer. Installation is pretty straightforward:

.. code-block:: console

    $ pip install pyrost

If you want to install pyrost for a single user instead of
system-wide, you can do:

.. code-block:: console

    $ pip install --user pyrost

Installation from source
------------------------
In order to install pyrost from the source you will need:

* a C++ compiler (gcc or clang will do).
* `Python <https://www.python.org/>`_ 3.6 or later (Python 2.x is
  **not** supported).
* `FFTW library <http://www.fftw.org/#documentation>`_ 3.3.8 or later,
  which is used for fast Fourier transform computations.
* `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ 2.4
  or later, which is used for pseudo-random number generation.
* `LLVM's OpenMP library <http://openmp.llvm.org>`_ 10.0.0 or later, which
  is used for parallelization.
* `Cython <https://cython.org>`_ 0.29 or later.
* `NumPy <https://numpy.org>`_ 1.19.0 or later.
* `SciPy <https://scipy.org>`_ 1.5.2 or later.
* `h5py <https://www.h5py.org>`_ 2.10.0 or later, which is used to work with
  CXI files.
* `tqdm <https://github.com/tqdm/tqdm>`_ 4.56.0 or later.

After installing the dependencies, you can download the source code from
the `GitHub pyrost repository page <https://github.com/simply-nicky/pyrost>`_.
Or you can download the last version of pyrost repository with ``git``:

.. code-block:: console

    $ git clone https://github.com/simply-nicky/pyrost.git

After downloading the pyrost's source code, ``cd`` into the repository root folder
and build the C++ libraries using ``setuputils``:

.. code-block:: console

    $ cd pyrost
    $ python setup.py install

OR

.. code-block:: console

    $ pip install -r requirements.txt -e . -v

Getting help
------------
If you run into troubles when installing pyrost, please do not hesitate
to contact me either through `my mail <nikolay.ivanov@desy.de>`_
or by opening an issue report on `github <https://github.com/simply-nicky/pyrost/issues>`_.