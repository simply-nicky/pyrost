.. _ms-parameters:

Simulation parameters
=====================

.. raw:: html
    :file:  ../figures/ms_sim_exp_geom.svg

|

Simulation parameters class(:class:`pyrost.simulation.MSParams`) for the
multislice beam propagation stores all the experimental parameters
and provides additional methods necessary to perform the propagation. 

List of simulation parameters:

* **Experimental geometry parameters**:

  * `x_min`, `x_max` : Wavefront span along the x-axis [um].
  * `x_step` : Beam sampling interval along the x-axis [um].
  * `z_step` : Distance between the slices [um].
  * `wl` : Beam's wavelength [um].

The ms_sim library can generate the transmission profile of the 
Multilayer Laue (MLL) lens. The required parameters to yield a transmission
profile are as follows: 

* **MLL materials**:

  * `material1` : the first material in the MLL's bilayers.
    * `formula` : Chemical formula of the material.
    * `density` : Atomic density of the material [g / cm^3].
  * `material2` : the second material in the MLL's bilayers.
    * `formula` : Chemical formula of the material.
    * `density` : Atomic density of the material [g / cm^3].

* **MLL parameters**:

  * `n_min`, `n_max` : zone number of the first and the last layer.
  * `focus` : MLL's focal distance [um].
  * `mll_sigma` : Bilayer's interdiffusion length [um].
  * `mll_depth` : MLL's thickness [um].
  * `mll_wl` : Wavelength of the MLL [um].

.. note::

    You can save parameters to an INI file with :func:`pyrost.multislice.MSParams.to_ini`
    and import parameters from an INI file with :func:`pyrost.multislice.MSParams.import_ini`.

The default parameters are accessed with :func:`pyrost.multislice.MSParams.import_default`.
The parameters are given by:

.. code-block:: ini

    [multislice]
    x_max = 30.0
    x_min = 0.0
    x_step = 1e-4
    z_step = 5e-3
    wl = 7.293187822128047e-5

    [material1]
    formula = W
    density = 18.0

    [material2]
    formula = SiC
    density = 2.8

    [mll]
    n_min = 100
    n_max = 8000
    focus = 1500.0
    mll_sigma = 1e-4
    mll_depth = 5.0
    mll_wl = 7.293187822128047e-5

.. automodule:: pyrost.multislice.ms_parameters

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/material
    classes/ms_params