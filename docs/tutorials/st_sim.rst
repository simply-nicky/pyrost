Speckle Tracking Simulation
===========================

You can simulate an one-dimensional Speckle Tracking scan either using
Python interface or Terminal.

Python interface
----------------

Experimental parameters
^^^^^^^^^^^^^^^^^^^^^^^

Before performing the simulation, you need to choose experimental
parameters. You can do it with :class:`pyrost.simulation.STParams` or
:func:`pyrost.simulation.parameters`.

.. note:: Full list of experimental parameters is written in
    :doc:`../reference/st_params_ref`. All the spatial parameters are
    assumed to be in microns.

.. doctest::

    >>> import pyrost.simulation as st_sim
    >>> params = st_sim.parameters(bar_size=0.7, bar_sigma=0.12, bar_atn=0.18,
    >>>                            bulk_atn=0.2, p0 = 5e4, th_s=8e-5, n_frames=100,
    >>>                            offset=2.0, step_size=0.1, defocus=150, alpha=0.05,
    >>>                            x0=0.7, rnd_dev=0.8)

Performing the simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

Now you're able to generate the simulated data. It takes time to calculate the
wavefronts, :class:`pyrost.simulation.STSim` will post it's status during the process. You can
either generate a stack of frames or a ptychograph. :class:`pyrost.simulation.STConverter`
saves the results to a CXI file using the provided CXI protocol.

.. note:: :class:`pyrost.simulation.STSim` logs the simulation procedure in `[package_root_folder]/logs`.

.. doctest::

    >>> with st_sim.STSim(params) as sim_obj:
    >>>     ptych = sim_obj.ptychograph()
    >>>     st_conv = st_sim.STConverter()
    >>>     st_conv.save_sim(ptych, sim_obj, 'results/sim_results')
    2020-11-25 15:50:34,548 - STSim - INFO - Initializing
    2020-11-25 15:50:34,549 - STSim - INFO - Current parameters
    2020-11-25 15:50:34,550 - STSim - INFO - Initializing coordinate arrays at the sample's plane
    2020-11-25 15:50:34,551 - STSim - INFO - Number of points in x axis: 4828
    2020-11-25 15:50:34,552 - STSim - INFO - Number of points in y axis: 905
    2020-11-25 15:50:34,554 - STSim - INFO - Generating wavefields at the sample's plane
    2020-11-25 15:50:44,708 - STSim - INFO - The wavefields have been generated
    2020-11-25 15:50:44,709 - STSim - INFO - Generating barcode's transmission coefficients
    2020-11-25 15:50:44,770 - STSim - INFO - The coefficients have been generated
    2020-11-25 15:50:44,771 - STSim - INFO - Generating wavefields at the detector's plane
    2020-11-25 15:50:52,562 - STSim - INFO - The wavefields have been generated
    2020-11-25 15:50:52,564 - STSim - INFO - Making ptychograph...
    2020-11-25 15:50:52,565 - STSim - INFO - Source blur size: 160.000000 um
    2020-11-25 15:50:52,581 - STSim - INFO - The ptychograph is generated, data shape: (200, 1, 2000)
    2020-11-25 15:50:52,583 - STSim - INFO - Saving data in the directory: results/sim_results
    2020-11-25 15:50:52,583 - STSim - INFO - Making ini files...
    2020-11-25 15:50:52,595 - STSim - INFO - results/sim_results/update_pixel_map.ini saved
    2020-11-25 15:50:52,596 - STSim - INFO - results/sim_results/zernike.ini saved
    2020-11-25 15:50:52,597 - STSim - INFO - results/sim_results/calculate_phase.ini saved
    2020-11-25 15:50:52,599 - STSim - INFO - results/sim_results/make_reference.ini saved
    2020-11-25 15:50:52,600 - STSim - INFO - results/sim_results/calc_error.ini saved
    2020-11-25 15:50:52,601 - STSim - INFO - results/sim_results/speckle_gui.ini saved
    2020-11-25 15:50:52,603 - STSim - INFO - results/sim_results/generate_pixel_map.ini saved
    2020-11-25 15:50:52,604 - STSim - INFO - results/sim_results/update_translations.ini saved
    2020-11-25 15:50:52,605 - STSim - INFO - results/sim_results/parameters.ini saved
    2020-11-25 15:50:52,607 - STSim - INFO - results/sim_results/protocol.ini saved
    2020-11-25 15:50:52,607 - STSim - INFO - Making a cxi file...
    2020-11-25 15:50:52,608 - STSim - INFO - Using the following cxi protocol:
    2020-11-25 15:50:52,609 - STSim - INFO - basis_vectors [float]: '/entry_1/instrument_1/detector_1/basis_vectors' 
    2020-11-25 15:50:52,609 - STSim - INFO - data [float]: '/entry_1/data_1/data' 
    2020-11-25 15:50:52,610 - STSim - INFO - defocus [float]: '/speckle_tracking/defocus' 
    2020-11-25 15:50:52,611 - STSim - INFO - defocus_fs [float]: '/speckle_tracking/dfs' 
    2020-11-25 15:50:52,612 - STSim - INFO - defocus_ss [float]: '/speckle_tracking/dss' 
    2020-11-25 15:50:52,612 - STSim - INFO - distance [float]: '/entry_1/instrument_1/detector_1/distance' 
    2020-11-25 15:50:52,613 - STSim - INFO - energy [float]: '/entry_1/instrument_1/source_1/energy' 
    2020-11-25 15:50:52,614 - STSim - INFO - good_frames [int]: '/frame_selector/good_frames' 
    2020-11-25 15:50:52,614 - STSim - INFO - m0 [int]: '/speckle_tracking/m0' 
    2020-11-25 15:50:52,615 - STSim - INFO - mask [bool]: '/speckle_tracking/mask' 
    2020-11-25 15:50:52,616 - STSim - INFO - n0 [int]: '/speckle_tracking/n0' 
    2020-11-25 15:50:52,617 - STSim - INFO - phase [float]: '/speckle_tracking/phase' 
    2020-11-25 15:50:52,617 - STSim - INFO - pixel_map [float]: '/speckle_tracking/pixel_map' 
    2020-11-25 15:50:52,618 - STSim - INFO - pixel_abberations [float]: '/speckle_tracking/pixel_abberations' 
    2020-11-25 15:50:52,619 - STSim - INFO - pixel_translations [float]: '/speckle_tracking/pixel_translations' 
    2020-11-25 15:50:52,619 - STSim - INFO - reference_image [float]: '/speckle_tracking/reference_image' 
    2020-11-25 15:50:52,620 - STSim - INFO - roi [int]: '/speckle_tracking/roi' 
    2020-11-25 15:50:52,621 - STSim - INFO - translations [float]: '/entry_1/sample_1/geometry/translations' 
    2020-11-25 15:50:52,621 - STSim - INFO - wavelength [float]: '/entry_1/instrument_1/source_1/wavelength' 
    2020-11-25 15:50:52,622 - STSim - INFO - whitefield [float]: '/speckle_tracking/whitefield' 
    2020-11-25 15:50:52,623 - STSim - INFO - x_pixel_size [float]: '/entry_1/instrument_1/detector_1/x_pixel_size' 
    2020-11-25 15:50:52,624 - STSim - INFO - y_pixel_size [float]: '/entry_1/instrument_1/detector_1/y_pixel_size' 
    2020-11-25 15:50:52,646 - STSim - INFO - results/sim_results/data.cxi saved

    >>> fig, ax = plt.subplots(figsize=(14, 6)) # doctest: +SKIP
    >>> ax.imshow(ptych[:, 0, 500:1480]) # doctest: +SKIP
    >>> ax.set_title('Ptychograph', fontsize=20) # doctest: +SKIP
    >>> ax.tick_params(labelsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/ptychograph.png
    :width: 100 %
    :alt: Ptychograph.

Or you can directly generate an :class:`pyrost.STData` data container to perform the Speckle Tracking algorithm.

.. doctest::

    >>> with st_sim.STSim(params) as sim_obj:
    >>>     ptych = sim_obj.ptychograph()
    >>>     st_conv = st_sim.STConverter()
    >>>     st_data = st_conv.export_data(ptych, sim_obj)
    2020-11-25 15:50:34,548 - STSim - INFO - Initializing
    2020-11-25 15:50:34,549 - STSim - INFO - Current parameters
    2020-11-25 15:50:34,550 - STSim - INFO - Initializing coordinate arrays at the sample's plane
    2020-11-25 15:50:34,551 - STSim - INFO - Number of points in x axis: 4828
    2020-11-25 15:50:34,552 - STSim - INFO - Number of points in y axis: 905
    2020-11-25 15:50:34,554 - STSim - INFO - Generating wavefields at the sample's plane
    2020-11-25 15:50:44,708 - STSim - INFO - The wavefields have been generated
    2020-11-25 15:50:44,709 - STSim - INFO - Generating barcode's transmission coefficients
    2020-11-25 15:50:44,770 - STSim - INFO - The coefficients have been generated
    2020-11-25 15:50:44,771 - STSim - INFO - Generating wavefields at the detector's plane
    2020-11-25 15:50:52,562 - STSim - INFO - The wavefields have been generated
    2020-11-25 15:50:52,564 - STSim - INFO - Making ptychograph...
    2020-11-25 15:50:52,565 - STSim - INFO - Source blur size: 160.000000 um
    2020-11-25 15:50:52,581 - STSim - INFO - The ptychograph is generated, data shape: (200, 1, 2000)


Command-line interface
----------------------

You can perform the whole simulation procedure with one command :code:`python -m pyrost.simulation`. To see all available arguments
just type :code:`python -m pyrost.simulation --help`.

.. code-block:: console

    $ python -m pyrost.simulation --help      
    usage: __main__.py [-h] [-f INI_FILE] [--defocus DEFOCUS]
                       [--det_dist DET_DIST] [--step_size STEP_SIZE]
                       [--n_frames N_FRAMES] [--fs_size FS_SIZE]
                       [--ss_size SS_SIZE] [--p0 P0] [--wl WL] [--th_s TH_S]
                       [--ap_x AP_X] [--ap_y AP_Y] [--focus FOCUS] [--alpha ALPHA]
                       [--x0 X0] [--bar_size BAR_SIZE] [--bar_sigma BAR_SIGMA]
                       [--bar_atn BAR_ATN] [--bulk_atn BULK_ATN]
                       [--rnd_dev RND_DEV] [--offset OFFSET] [-v] [-p]
                       out_path

    Run Speckle Tracking simulation

    positional arguments:
      out_path              Output folder path

    optional arguments:
      -h, --help            show this help message and exit
      -f INI_FILE, --ini_file INI_FILE
                            Path to an INI file to fetch all of the simulation
                            parameters (default: None)
      --defocus DEFOCUS     Lens defocus distance, [um] (default: 400.0)
      --det_dist DET_DIST   Distance between the barcode and the detector [um]
                            (default: 2000000.0)
      --step_size STEP_SIZE
                            Scan step size [um] (default: 0.1)
      --n_frames N_FRAMES   Number of frames (default: 300)
      --fs_size FS_SIZE     Fast axis frames size in pixels (default: 2000)
      --ss_size SS_SIZE     Slow axis frames size in pixels (default: 1000)
      --p0 P0               Source beam flux [cnt / s] (default: 200000.0)
      --wl WL               Wavelength [um] (default: 7.29e-05)
      --th_s TH_S           Source rocking curve width [rad] (default: 0.0002)
      --ap_x AP_X           Lens size along the x axis [um] (default: 40.0)
      --ap_y AP_Y           Lens size along the y axis [um] (default: 2.0)
      --focus FOCUS         Focal distance [um] (default: 1500.0)
      --alpha ALPHA         Third order abberations [rad/mrad^3] (default: -0.05)
      --x0 X0               Lens' abberations center point [0.0 - 1.0] (default:
                            0.5)
      --bar_size BAR_SIZE   Average bar size [um] (default: 0.1)
      --bar_sigma BAR_SIGMA
                            Bar haziness width [um] (default: 0.01)
      --bar_atn BAR_ATN     Bar attenuation (default: 0.3)
      --bulk_atn BULK_ATN   Bulk attenuation (default: 0.0)
      --rnd_dev RND_DEV     Bar random deviation (default: 0.6)
      --offset OFFSET       sample's offset at the beginning and the end of the
                            scan [um] (default: 0.0)
      -v, --verbose         Turn on verbosity (default: True)
      -p, --ptych           Generate ptychograph data (default: False)

    $ python -m pyrost.simulation results/sim_results --bar_size 0.7 --bar_sigma 0.12 \
    --bar_atn 0.18 --bulk_atn 0.2 --p0 5e4 --th_s 8e-5 --n_frames 200 --offset 2 \
    --step_size 0.1 --defocus 150 --alpha 0.05 --x0 0.7 --rnd_dev 0.8 -p -v
    2020-11-25 17:29:35,570 - STSim - INFO - Initializing
    2020-11-25 17:29:35,570 - STSim - INFO - Current parameters
    2020-11-25 17:29:35,571 - STSim - INFO - Initializing coordinate arrays at the sample's plane
    2020-11-25 17:29:35,571 - STSim - INFO - Number of points in x axis: 4828
    2020-11-25 17:29:35,571 - STSim - INFO - Number of points in y axis: 905
    2020-11-25 17:29:35,571 - STSim - INFO - Generating wavefields at the sample's plane
    2020-11-25 17:29:46,155 - STSim - INFO - The wavefields have been generated
    2020-11-25 17:29:46,155 - STSim - INFO - Generating barcode's transmission coefficients
    2020-11-25 17:29:46,193 - STSim - INFO - The coefficients have been generated
    2020-11-25 17:29:46,193 - STSim - INFO - Generating wavefields at the detector's plane
    2020-11-25 17:29:53,171 - STSim - INFO - The wavefields have been generated
    2020-11-25 17:29:53,171 - STSim - INFO - Making ptychograph...
    2020-11-25 17:29:53,171 - STSim - INFO - Source blur size: 160.000000 um
    2020-11-25 17:29:53,186 - STSim - INFO - The ptychograph is generated, data shape: (200, 1, 2000)
    2020-11-25 17:29:53,186 - STSim - INFO - Saving data in the directory: results/sim_results
    2020-11-25 17:29:53,186 - STSim - INFO - Making ini files...
    2020-11-25 17:29:53,194 - STSim - INFO - results/sim_results/update_pixel_map.ini saved
    2020-11-25 17:29:53,194 - STSim - INFO - results/sim_results/zernike.ini saved
    2020-11-25 17:29:53,195 - STSim - INFO - results/sim_results/calculate_phase.ini saved
    2020-11-25 17:29:53,195 - STSim - INFO - results/sim_results/make_reference.ini saved
    2020-11-25 17:29:53,196 - STSim - INFO - results/sim_results/calc_error.ini saved
    2020-11-25 17:29:53,196 - STSim - INFO - results/sim_results/speckle_gui.ini saved
    2020-11-25 17:29:53,196 - STSim - INFO - results/sim_results/generate_pixel_map.ini saved
    2020-11-25 17:29:53,197 - STSim - INFO - results/sim_results/update_translations.ini saved
    2020-11-25 17:29:53,197 - STSim - INFO - results/sim_results/parameters.ini saved
    2020-11-25 17:29:53,197 - STSim - INFO - results/sim_results/protocol.ini saved
    2020-11-25 17:29:53,197 - STSim - INFO - Making a cxi file...
    2020-11-25 17:29:53,198 - STSim - INFO - Using the following cxi protocol:
    2020-11-25 17:29:53,198 - STSim - INFO - basis_vectors [float]: '/entry_1/instrument_1/detector_1/basis_vectors' 
    2020-11-25 17:29:53,198 - STSim - INFO - data [float]: '/entry_1/data_1/data' 
    2020-11-25 17:29:53,198 - STSim - INFO - defocus [float]: '/speckle_tracking/defocus' 
    2020-11-25 17:29:53,198 - STSim - INFO - defocus_fs [float]: '/speckle_tracking/dfs' 
    2020-11-25 17:29:53,198 - STSim - INFO - defocus_ss [float]: '/speckle_tracking/dss' 
    2020-11-25 17:29:53,204 - STSim - INFO - distance [float]: '/entry_1/instrument_1/detector_1/distance' 
    2020-11-25 17:29:53,204 - STSim - INFO - energy [float]: '/entry_1/instrument_1/source_1/energy' 
    2020-11-25 17:29:53,204 - STSim - INFO - good_frames [int]: '/frame_selector/good_frames' 
    2020-11-25 17:29:53,204 - STSim - INFO - m0 [int]: '/speckle_tracking/m0' 
    2020-11-25 17:29:53,204 - STSim - INFO - mask [bool]: '/speckle_tracking/mask' 
    2020-11-25 17:29:53,204 - STSim - INFO - n0 [int]: '/speckle_tracking/n0' 
    2020-11-25 17:29:53,204 - STSim - INFO - phase [float]: '/speckle_tracking/phase' 
    2020-11-25 17:29:53,204 - STSim - INFO - pixel_map [float]: '/speckle_tracking/pixel_map' 
    2020-11-25 17:29:53,204 - STSim - INFO - pixel_abberations [float]: '/speckle_tracking/pixel_abberations' 
    2020-11-25 17:29:53,204 - STSim - INFO - pixel_translations [float]: '/speckle_tracking/pixel_translations' 
    2020-11-25 17:29:53,204 - STSim - INFO - reference_image [float]: '/speckle_tracking/reference_image' 
    2020-11-25 17:29:53,204 - STSim - INFO - roi [int]: '/speckle_tracking/roi' 
    2020-11-25 17:29:53,204 - STSim - INFO - translations [float]: '/entry_1/sample_1/geometry/translations' 
    2020-11-25 17:29:53,205 - STSim - INFO - wavelength [float]: '/entry_1/instrument_1/source_1/wavelength' 
    2020-11-25 17:29:53,205 - STSim - INFO - whitefield [float]: '/speckle_tracking/whitefield' 
    2020-11-25 17:29:53,205 - STSim - INFO - x_pixel_size [float]: '/entry_1/instrument_1/detector_1/x_pixel_size' 
    2020-11-25 17:29:53,205 - STSim - INFO - y_pixel_size [float]: '/entry_1/instrument_1/detector_1/y_pixel_size' 
    2020-11-25 17:29:53,221 - STSim - INFO - results/sim_results/data.cxi saved

As you can see below, the simulated Speckle Tracking scan was saved to a CXI file.

.. code-block:: console

    $ h5ls -r results/sim_results/data.cxi
    /                        Group
    /entry_1                 Group
    /entry_1/data_1          Group
    /entry_1/data_1/data     Dataset {200, 1, 2000}
    /entry_1/instrument_1    Group
    /entry_1/instrument_1/detector_1 Group
    /entry_1/instrument_1/detector_1/basis_vectors Dataset {200, 2, 3}
    /entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/energy Dataset {SCALAR}
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_1        Group
    /entry_1/sample_1/geometry Group
    /entry_1/sample_1/geometry/translations Dataset {200, 3}
    /frame_selector          Group
    /frame_selector/good_frames Dataset {200}
    /speckle_tracking        Group
    /speckle_tracking/defocus Dataset {SCALAR}
    /speckle_tracking/mask   Dataset {1, 2000}
    /speckle_tracking/roi    Dataset {4}
    /speckle_tracking/whitefield Dataset {1, 2000}