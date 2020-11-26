Installation
============

Dependencies
------------
pyrst has the following **mandatory** runtime dependencies:

* `Python <https://www.python.org/>`_ 3.7 or later (Python 2.x is
  **not** supported).
* `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ 2.4
  or later, which is used for numerical integration and pseudo-random
  number generation.
* `LLVM's OpenMP library <http://openmp.llvm.org>`_ 10.0.0 or later, which
  is used for parallelization.
* `h5py <https://www.h5py.org>`_ 2.10.0 or later, which is used to work with
  CXI files.
* `NumPy <https://numpy.org>`_ 1.19.0 or later.
* `SciPy <https://scipy.org>`_ 1.5.2 or later.
* `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_ 0.12.0 or later, which is used
  for the Fourier Transform wavefront reconstruction (:func:`pyrst.bin.ct_integrate`).

Packages
--------
pyrst packages are available through `pypi <https://test.pypi.org/project/rst/>`_ on
OS X as for now.

pip
^^^
pyrst is available on OS X via the `pip <https://pip.pypa.io/en/stable/>`_
package installer. Installation is pretty straightforward:

.. code-block:: console

    $ pip install pyrst

If you want to install pyrst for a single user instead of
system-wide, you can do:

.. code-block:: console

    $ pip install --user pyrst

Installation from source
------------------------
In order to install pyrst from source you will need:

* a C++ compiler (gcc or clang will do).
* `Python <https://www.python.org/>`_ 3.7 or later (Python 2.x is
  **not** supported).
* `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ 2.4
  or later, which is used for numerical integration and pseudo-random
  number generation.
* `LLVM's OpenMP library <http://openmp.llvm.org>`_ 10.0.0 or later, which
  is used for parallelization.
* `Cython <https://cython.org>`_ 0.29 or later.
* `CythonGSL <https://github.com/twiecki/CythonGSL>`_ 0.2.2 or later.
* `NumPy <https://numpy.org>`_ 1.19.0 or later.

After installing the dependencies, you can download the the source code from
the `GitHub pyrst repository page <https://github.com/simply-nicky/pyrst>`_.
Or you can download the last version of pyrst repository with ``git``:

.. code-block:: console

    $ git clone https://github.com/simply-nicky/pyrst.git

After downloading the pyrst's source code, ``cd`` into the repository root folder
and build the C++ libraries using ``setuputils``:

.. code-block:: console

    $ cd pyrst
    $ python setup.py install

OR

.. code-block:: console

    $ pip install -r requirements.txt -e . -v

Getting help
------------
If you run into troubles installing pyrst, please do not hesitate
to contact me either through `my mail <nikolay.ivanov@desy.de>`_
or by opening an issue report on `github <https://github.com/simply-nicky/pyrst/issues>`_.