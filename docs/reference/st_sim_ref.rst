Core Classes
============

The following classes are central to pyrost.simulation module:

* :class:`pyrost.simulation.STSim` accepts a set of simulation parameters from
  :class:`pyrost.simulation.STParams` and performs the wavefront propagation of
  an X-ray beam from the lens plane to the detector to generate a speckle tracking
  dataset.

* :class:`pyrost.simulation.STConverter` takes the generated data from
  :class:`pyrost.simulation.STSim` and a CXI protocol :class:`pyrost.CXIProtocol`
  to transfer the data to a data container :class:`pyrost.STData` or to save it
  to a CXI file.

.. automodule:: pyrost.simulation.st_sim

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/st_sim
    classes/st_converter