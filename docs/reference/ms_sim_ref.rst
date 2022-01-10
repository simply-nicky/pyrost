Core classes
============

The following classes are central to pyrost.multislice module:

* :class:`pyrost.multislice.MLL` is the mutlilayer Laue lens class, that generates a
  transmission profile of a MLL for the given pair of materials and set of experimental
  parameters.

* :class:`pyrost.multislice.MSPropagator` accepts a set of simulation parameters from
  :class:`pyrost.multislice.MSParams` and a MLL object from :class:`pyrost.multislice.MLL`
  and performs the multislice beam propagation using the FFT algorithm.

.. automodule:: pyrost.multislice.mslice

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/mll
    classes/ms_propagator