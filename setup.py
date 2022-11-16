# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pyrost', 'pyrost.bin', 'pyrost.multislice', 'pyrost.simulation']

package_data = \
{'': ['*'], 'pyrost': ['config/*', 'data/*', 'include/*']}

install_requires = \
['h5py>=2.10.0,<3.0.0',
 'numpy>=1.19.0,<2.0.0',
 'scipy>=1.5.2,<2.0.0',
 'tqdm>=4.56.0,<5.0.0']

setup_kwargs = {
    'name': 'pyrost',
    'version': '0.7.6',
    'description': 'Python Robust Speckle Tracking (pyrost) is a library for wavefront metrology and sample imaging based on ptychographic speckle tracking algorithm.',
    'long_description': "## Current build status\n[![PyPI](https://img.shields.io/pypi/v/pyrost?color=brightgreen)](https://pypi.org/project/pyrost/)\n[![Documentation Status](https://readthedocs.org/projects/robust-speckle-tracking/badge/?version=latest)](https://robust-speckle-tracking.readthedocs.io/en/latest/?badge=latest)\n[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pyrost?color=brightgreen)](https://anaconda.org/conda-forge/pyrost)\n[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6574364.svg)](https://doi.org/10.5281/zenodo.6574364)\n\n# pyrost\nPython Robust Speckle Tracking (**pyrost**) is a library for wavefront metrology\nand sample imaging based on ptychographic speckle tracking algorithm. This\nproject takes over Andrew Morgan's [speckle_tracking](https://github.com/andyofmelbourne/speckle-tracking)\nproject as an improved version aiming to add robustness to the optimisation\nalgorithm in the case of the high noise present in the measured data.\n\nThe documentation can be found on [Read the Docs](https://robust-speckle-tracking.readthedocs.io/en/latest/).\n\nRead the open access scientific article about the technical details of the developed technique published in [Optica Express](https://opg.optica.org/oe/abstract.cfm?URI=oe-30-14-25450).\n\n## Dependencies\n\n- [Python](https://www.python.org/) 3.6 or later (Python 2.x is **not** supported).\n- [GNU Scientific Library](https://www.gnu.org/software/gsl/) 2.4 or later.\n- [LLVM's OpenMP library](http://openmp.llvm.org) 10.0.0 or later.\n- [FFTW](http://www.fftw.org) 3.3.8 or later.\n- [h5py](https://www.h5py.org) 2.10.0 or later.\n- [NumPy](https://numpy.org) 1.19.0 or later.\n- [SciPy](https://scipy.org) 1.5.2 or later.\n- [tqdm](https://github.com/tqdm/tqdm) 4.56.0 or later.\n\n## Installation\nWe recommend **not** building from source, but install the release via the\n[conda](https://anaconda.org/conda-forge/pyrost) manager:\n\n    conda install -c conda-forge pyrost\n\nThe package is available in [conda-forge](https://anaconda.org/conda-forge/pyrost) on OS X and Linux.\n\nAlso you can install the release from [pypi](https://pypi.org/project/pyrost/)\nwith the pip package installer:\n\n    pip install pyrost\n\nThe source distribution is available in [pypi](https://pypi.org/project/pyrost/) as for now.\n\n## Installation from source\nIn order to build the package from source simply execute the following command:\n\n    python setup.py install\n\nor:\n\n    pip install -r requirements.txt -e . -v\n\nThat cythonizes the Cython extensions and builds them into ``/pyrost/bin``.",
    'author': 'Nikolay Ivanov',
    'author_email': 'nikolay.ivanov@desy.de',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
