1D Speckle Tracking Scan
========================

Speckle Tracking update procedure of one-dimensional scan doesn't differ much
from the case of two-dimensional scan. See :doc:`diatom` for an example.

1D Scan CXI File
----------------
In order to obtain the file generate it using :doc:`st_sim`. The file has
the following structure:

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

The file contains a ptychograph (set of frames summed over one of detector axes)
of 200 frames.

Loading the file
----------------
Load the file with :class:`pyrost.STLoader`. In the case of simulated data you can
import the protocol file, which is located in the same folder with `data.cxi`.

.. doctest::

    >>> import pyrost as rst
    >>> protocol = rst.Protocol.import_ini('results/sim_results/protocol.ini')
    >>> loader = rst.loader(protocol=protocol)
    >>> data = loader.load('results/sim_results/data.cxi')

Speckle Tracking update
-----------------------
You can perform the Speckle Tracking procedure with :class:`pyrost.SpeckleTracking`.

.. note:: You should pay outmost attention to choose the right length scales of reference
    image and pixel mapping (`ls_ri`, `ls_pm`). Essentually they stand for high frequency
    cut-off of the measured data, it helps to supress Poisson noise. If the values are too
    high you'll lose useful information. If the values are too low in presence of high noise,
    you won't get accurate results.

.. doctest::

    >>> st_obj = data.get_st()
    >>> st_res, errors = st_obj.iter_update(sw_fs=10, ls_pm=2.5, ls_ri=5, verbose=True, n_iter=10)
    Initial MSE = 0.267591
    Iteration No. 1: Total MSE = 0.229052
    Iteration No. 2: Total MSE = 0.188193
    Iteration No. 3: Total MSE = 0.157726
    Iteration No. 4: Total MSE = 0.137449
    Iteration No. 5: Total MSE = 0.125132
    Iteration No. 6: Total MSE = 0.118468
    Iteration No. 7: Total MSE = 0.114671
    Iteration No. 8: Total MSE = 0.112955
    Iteration No. 9: Total MSE = 0.112231
    Iteration No. 10: Total MSE = 0.111924

**OR** you can perform iterative update, where the reference image length scale is updated
based on gradeint descent with momentum algorithm, which in general gives lower final error.

.. doctest::

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.iter_update_gd(sw_fs=8, ls_pm=2.5, ls_ri=50., verbose=True, n_iter=20)
    Initial MSE = 0.179852, Initial ls_ri = 50.00
    Iteration No. 1: Total MSE = 0.144939, ls_ri = 51.46
    Iteration No. 2: Total MSE = 0.113126, ls_ri = 52.37
    Iteration No. 3: Total MSE = 0.088769, ls_ri = 52.70
    Iteration No. 4: Total MSE = 0.070811, ls_ri = 51.99
    Iteration No. 5: Total MSE = 0.058375, ls_ri = 50.74
    Iteration No. 6: Total MSE = 0.050156, ls_ri = 48.79
    Iteration No. 7: Total MSE = 0.044550, ls_ri = 46.61
    Iteration No. 8: Total MSE = 0.040678, ls_ri = 44.36
    Iteration No. 9: Total MSE = 0.038191, ls_ri = 42.28
    Iteration No. 10: Total MSE = 0.036637, ls_ri = 40.21
    Iteration No. 11: Total MSE = 0.035661, ls_ri = 38.12
    Iteration No. 12: Total MSE = 0.034942, ls_ri = 36.07
    Iteration No. 13: Total MSE = 0.034417, ls_ri = 34.13
    Iteration No. 14: Total MSE = 0.034110, ls_ri = 32.35
    Iteration No. 15: Total MSE = 0.034038, ls_ri = 30.79
    Iteration No. 16: Total MSE = 0.034014, ls_ri = 29.45
    Iteration No. 17: Total MSE = 0.034177, ls_ri = 28.35
    Iteration No. 18: Total MSE = 0.034302, ls_ri = 27.75
    Iteration No. 19: Total MSE = 0.034382, ls_ri = 27.54
    Iteration No. 20: Total MSE = 0.034349, ls_ri = 27.64

    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # doctest: +SKIP
    >>> axes[0].plot(np.arange(st_res.reference_image.shape[1]) - st_res.m0, # doctest: +SKIP
    >>>              st_res.reference_image[0]) # doctest: +SKIP
    >>> axes[0].set_title('Reference image', fontsize=20) # doctest: +SKIP
    >>> axes[1].plot((st_res.pixel_map - st_obj.pixel_map)[1, 0]) # doctest: +SKIP
    >>> axes[1].set_title('Pixel mapping', fontsize=20) # doctest: +SKIP
    >>> for ax in axes: # doctest: +SKIP
    >>>     ax.tick_params(labelsize=15) # doctest: +SKIP
    >>>     ax.set_xlabel('Fast axis, pixels', fontsize=20) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/1d_sim_res.png
    :width: 100 %
    :alt: Speckle Tracking update results

Phase reconstruction
--------------------
After we got the pixel map we're able to reconstruct the phase profile and fit it with
polynomial function.

.. doctest::

    >>> data.update_phase(st_res)
    >>> fit = data.fit_phase(axis=1, max_order=2)
    >>> fit['alpha'] # alpha in the simulation
    -0.05065824525080925

    >>> fit_obj = data.get_fit(axis=1) # doctest: +SKIP
    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # doctest: +SKIP
    >>> axes[0].plot(fit_obj.pixels, fit_obj.pixel_aberrations) # doctest: +SKIP
    >>> axes[0].plot(fit_obj.pixels, fit_obj.model(fit['fit'])) # doctest: +SKIP
    >>> axes[0].set_title('Pixel aberrations', fontsize=20) # doctest: +SKIP
    >>> axes[1].plot(fit_obj.pixels, fit_obj.phase) # doctest: +SKIP
    >>> axes[1].plot(fit_obj.pixels, fit_obj.model(fit['ph_fit']), # doctest: +SKIP
    >>>              label=r'$\alpha$ = {:.5f} rad/mrad^3'.format(fit['alpha'])) # doctest: +SKIP
    >>> axes[1].set_title('Phase', fontsize=20) # doctest: +SKIP
    >>> axes[1].legend(fontsize=15) # doctest: +SKIP
    >>> for ax in axes: # doctest: +SKIP
    >>>     ax.tick_params(axis='both', which='major', labelsize=15) # doctest: +SKIP
    >>>     ax.set_xlabel('fast axis', fontsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/1d_sim_fits.png
    :width: 100 %
    :alt: Phase polynomial fit.

Saving the results
------------------
In the end you can save the results to a CXI file.

.. doctest::

    >>> with h5py.File('results/sim_results/data_proc.cxi', 'w') as cxi_file:
    >>>     data.write_cxi(cxi_file)

.. code-block:: console

    $   h5ls -r results/sim_results/data_proc.cxi
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
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_1        Group
    /entry_1/sample_1/geometry Group
    /entry_1/sample_1/geometry/translations Dataset {200, 3}
    /frame_selector          Group
    /frame_selector/good_frames Dataset {200}
    /speckle_tracking        Group
    /speckle_tracking/error_frame Dataset {1, 2000}
    /speckle_tracking/dfs    Dataset {SCALAR}
    /speckle_tracking/dss    Dataset {SCALAR}
    /speckle_tracking/mask   Dataset {1, 2000}
    /speckle_tracking/phase  Dataset {1, 2000}
    /speckle_tracking/pixel_aberrations Dataset {2, 1, 2000}
    /speckle_tracking/pixel_map Dataset {2, 1, 2000}
    /speckle_tracking/pixel_translations Dataset {200, 2}
    /speckle_tracking/reference_image Dataset {1, 5754}
    /speckle_tracking/roi    Dataset {4}
    /speckle_tracking/whitefield Dataset {1, 2000}