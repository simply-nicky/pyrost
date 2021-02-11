st_sim Parameters
=================

.. raw:: html
    :file: ../figures/st_sim_exp_geom.svg

|

Experimental parameters class(:class:`pyrost.simulation.STParams`) for the
one-dimenional Speckle Tracking scan stores all the parameters and provides
additional methods necessary to perform the simulation. 

List of experimental parameters:

* **Experimental geometry parameters**:

    * `defocus` : Lens' defocus distance [um].
    * `det_dist` : Distance between the barcode and the
      detector [um].
    * `step_size` : Scan step size [um].
    * `n_frames` : Number of frames.

* **Detector parameters**:

    * `fs_size` : Detector's size along the fast axis in
      pixels.
    * `ss_size` : Detector's size along the slow axis in
      pixels.
    * `pix_size` : Detector's pixel size [um].

* **Source parameters**:

    * `p0` : Source beam flux [cnt / s].
    * `wl` : Incoming beam's wavelength [um].
    * `th_s` : Source rocking curve width [rad].

* **Lens parameters**:

    * `ap_x` : Lens' aperture size along the x axis [um].
    * `ap_y` : Lens' aperture size along the y axis [um].
    * `focus` : Focal distance [um].
    * `alpha` : Third order abberations ceofficient [rad/mrad^3].
    * `ab_cnt` : Lens' abberations center point [0.0 - 1.0].

* **Barcode sample parameters**:

    * `bar_size` : Average bar's size [um].
    * `bar_sigma` : Bar bluriness width [um].
    * `bar_atn` : Bar's attenuation coefficient [0.0 - 1.0].
    * `bulk_atn` : Barcode's bulk attenuation coefficient [0.0 - 1.0].
    * `rnd_dev` : Bar's coordinates random deviation [0.0 - 1.0].
    * `offset` : Barcode's offset at the beginning and at the end
      of the scan from the detector's bounds [um].

.. note::

    You can save protocol to an INI file with :func:`pyrost.Protocol.export_ini`
    and import protocol from INI file with :func:`pyrost.Protocol.import_ini`.

The default parameters are accessed width :func:`pyrost.simulation.parameters`.
The parameters are given by:

.. code-block:: ini

    [exp_geom]
    defocus = 4e2
    det_dist = 2e6
    step_size = 0.1
    n_frames = 300

    [detector]
    fs_size = 2000
    ss_size = 1000
    pix_size = 55

    [source]
    p0 = 2e5
    wl = 7.29e-5
    th_s = 2e-4

    [lens]
    ap_x = 40
    ap_y = 2
    focus = 1.5e3
    alpha = -0.05
    ab_cnt = 0.5

    [barcode]
    bar_size = 0.1
    bar_sigma = 0.01
    bar_atn = 0.3
    bulk_atn = 0.0
    rnd_dev = 0.6
    offset = 0

    [system]
    verbose = False

.. automodule:: pyrost.simulation.st_sim_param

.. toctree::
    :maxdepth: 1
    :caption: Contents

    classes/st_params
    classes/parameters