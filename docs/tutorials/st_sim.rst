Generating a speckle tracking dataset
=====================================

You can simulate an one-dimensional speckle tracking scan either using
Python interface or Terminal.

Python interface
----------------

Experimental parameters
^^^^^^^^^^^^^^^^^^^^^^^

Before performing the simulation, you need to choose the experimental
parameters. You can do it with :class:`pyrost.simulation.STParams`. The
st_sim library has built-in default parameters, which can be accessed
with :func:`pyrost.simulation.STParams.import_default`.

.. note:: Full list of experimental parameters is written in
    :doc:`../reference/st_params_ref`. All the spatial parameters are
    assumed to be in microns.

.. doctest::

    >>> import pyrost.simulation as st_sim
    >>> params = st_sim.STParams.import_default(bar_size=0.7, bar_sigma=0.12, bar_atn=0.18,
    >>>                                         bulk_atn=0.2, p0=5e4, th_s=8e-5, n_frames=100,
    >>>                                         offset=2.0, step_size=0.1, defocus=150, alpha=0.05,
    >>>                                         ab_cnt=0.7, bar_rnd=0.8)

Performing the simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

Now you're able to generate the simulated data. It takes time to calculate the
wavefronts, :class:`pyrost.simulation.STSim` will post it's status during the process. You can
either generate a stack of frames or a ptychograph. :class:`pyrost.simulation.STConverter`
generates all the data attributes necessary for the speckle tracking reconstruction. Also
it provides an interface to save the generated attributes to a CXI file with
:func:`pyrost.simulation.STConverter.save` method.

.. testsetup:: [st_simulation]

    import pyrost.simulation as st_sim
    params = st_sim.STParams.import_default()

.. doctest:: [st_simulation]

    >>> sim_obj = st_sim.STSim(params)
    >>> ptych = sim_obj.ptychograph()
    >>> st_conv = st_sim.STConverter(sim_obj, ptych)
    >>> st_conv.save('sim.cxi', mode='overwrite') # doctest: +SKIP

    >>> fig, ax = plt.subplots(figsize=(14, 6)) # doctest: +SKIP
    >>> ax.imshow(ptych[:, 0, 500:1480]) # doctest: +SKIP
    >>> ax.set_title('Ptychograph', fontsize=20) # doctest: +SKIP
    >>> ax.tick_params(labelsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/sim_ptychograph.png
    :width: 100 %
    :alt: Ptychograph.

Or you can save the simulated data and generate an :class:`pyrost.STData` data container with 
:func:`pyrost.simulation.STConverter` method.

.. note:: :func:`pyrost.simulation.STConverter` returns a :class:`pyrost.STData` container
    without any attributes already loaded, use :func:`pyrost.STData.load` to load the data
    from the file.

.. doctest:: [st_simulation]

    >>> sim_obj = st_sim.STSim(params)
    >>> ptych = sim_obj.ptychograph()
    >>> st_conv = st_sim.STConverter(sim_obj, ptych)
    >>> data = st_conv.export_data('sim.cxi')
    >>> data = data.load()

Command-line interface
----------------------

You can perform the whole simulation procedure with one command :code:`python -m pyrost.simulation`. To see all available arguments
just type :code:`python -m pyrost.simulation --help`.

.. code-block:: console

    $ python -m pyrost.simulation --help
    usage: __main__.py [-h] [-f INI_FILE] [--defocus DEFOCUS]
                       [--det_dist DET_DIST] [--step_size STEP_SIZE]
                       [--step_rnd STEP_RND] [--n_frames N_FRAMES]
                       [--detx_size DETX_SIZE] [--dety_size DETY_SIZE] [--p0 P0]
                       [--wl WL] [--th_s TH_S] [--ap_x AP_X] [--ap_y AP_Y]
                       [--focus FOCUS] [--alpha ALPHA] [--ab_cnt AB_CNT]
                       [--bar_size BAR_SIZE] [--bar_sigma BAR_SIGMA]
                       [--bar_atn BAR_ATN] [--bulk_atn BULK_ATN]
                       [--bar_rnd BAR_RND] [--offset OFFSET] [-p]
                       out_path

    Run Speckle Tracking simulation

    positional arguments:
      out_path              Output folder path

    optional arguments:
      -h, --help            show this help message and exit
      -f INI_FILE, --ini_file INI_FILE
                            Path to an INI file to fetch all of the simulation
                            parameters (default: None)
      --defocus DEFOCUS     Lens defocus distance, [um] (default: 100.0)
      --det_dist DET_DIST   Distance between the barcode and the detector [um]
                            (default: 2000000.0)
      --step_size STEP_SIZE
                            Scan step size [um] (default: 0.1)
      --step_rnd STEP_RND   Random deviation of sample translations [0.0 - 1.0]
                            (default: 0.2)
      --n_frames N_FRAMES   Number of frames (default: 300)
      --detx_size DETX_SIZE
                            horizontal axis frames size in pixels (default: 2000)
      --dety_size DETY_SIZE
                            vertical axis frames size in pixels (default: 1000)
      --p0 P0               Source beam flux [cnt / s] (default: 200000.0)
      --wl WL               Wavelength [um] (default: 7.29e-05)
      --th_s TH_S           Source rocking curve width [rad] (default: 0.0002)
      --ap_x AP_X           Lens size along the x axis [um] (default: 40.0)
      --ap_y AP_Y           Lens size along the y axis [um] (default: 2.0)
      --focus FOCUS         Focal distance [um] (default: 1500.0)
      --alpha ALPHA         Third order aberrations [rad/mrad^3] (default: -0.05)
      --ab_cnt AB_CNT       Lens' aberrations center point [0.0 - 1.0] (default:
                            0.5)
      --bar_size BAR_SIZE   Average bar size [um] (default: 0.5)
      --bar_sigma BAR_SIGMA
                            Bar haziness width [um] (default: 0.12)
      --bar_atn BAR_ATN     Bar attenuation (default: 0.15)
      --bulk_atn BULK_ATN   Bulk attenuation (default: 0.15)
      --bar_rnd BAR_RND     Bar random deviation (default: 0.9)
      --offset OFFSET       Sample's offset at the beginning and the end of the
                            scan [um] (default: 0.0)
      -p, --ptych           Generate ptychograph data (default: False)

    $ python -m pyrost.simulation sim.cxi --bar_size 0.7 --bar_sigma 0.12 \
    --bar_atn 0.18 --bulk_atn 0.2 --p0 5e4 --th_s 8e-5 --n_frames 200 --offset 2 \
    --step_size 0.1 --defocus 150 --alpha 0.05 --ab_cnt 0.7 --bar_rnd 0.8 -p
    The simulation results have been saved to sim.cxi

As you can see below, the simulated Speckle Tracking scan was saved to a CXI file.

.. code-block:: console

    $ h5ls -r sim.cxi
    /                        Group
    /entry                   Group
    /entry/data              Group
    /entry/data/data         Dataset {200/Inf, 1, 985}
    /entry/instrument        Group
    /entry/instrument/detector Group
    /entry/instrument/detector/distance Dataset {SCALAR}
    /entry/instrument/detector/x_pixel_size Dataset {SCALAR}
    /entry/instrument/detector/y_pixel_size Dataset {SCALAR}
    /entry/instrument/source Group
    /entry/instrument/source/wavelength Dataset {SCALAR}
    /speckle_tracking        Group
    /speckle_tracking/basis_vectors Dataset {200/Inf, 2, 3}
    /speckle_tracking/defocus_x Dataset {SCALAR}
    /speckle_tracking/defocus_y Dataset {SCALAR}
    /speckle_tracking/mask   Dataset {200/Inf, 1, 985}
    /speckle_tracking/pixel_translations Dataset {200/Inf, 2}
    /speckle_tracking/translations Dataset {200/Inf, 3}
    /speckle_tracking/whitefield Dataset {1, 985}