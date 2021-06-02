Speckle Tracking of a 1D scan
=============================

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
Load the file with :class:`pyrost.CXILoader`. In the case of simulated data you can
import the protocol file, which is located in the same folder with `data.cxi`.

.. doctest::

    >>> import pyrost as rst
    >>> protocol = rst.CXIProtocol.import_ini('results/sim_results/protocol.ini')
    >>> loader = rst.CXILoader(protocol=protocol)
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

**OR** you can perform iterative update, where the reference image length scale is updated
based on gradeint descent with momentum algorithm, which in general gives lower final error.

.. doctest::

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.iter_update_gd(sw_fs=8, ls_pm=2.5, ls_ri=50., verbose=True, n_iter=20)

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
    >>> fit['c_3'] # third order fit coefficient
    -0.05065824525080925

    >>> fit_obj = data.get_fit(axis=1) # doctest: +SKIP
    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # doctest: +SKIP
    >>> axes[0].plot(fit_obj.pixels, fit_obj.pixel_aberrations) # doctest: +SKIP
    >>> axes[0].plot(fit_obj.pixels, fit_obj.model(fit['fit'])) # doctest: +SKIP
    >>> axes[0].set_title('Pixel aberrations', fontsize=20) # doctest: +SKIP
    >>> axes[1].plot(fit_obj.pixels, fit_obj.phase) # doctest: +SKIP
    >>> axes[1].plot(fit_obj.pixels, fit_obj.model(fit['ph_fit']), # doctest: +SKIP
    >>>              label=r'$\alpha$ = {:.5f} rad/mrad^3'.format(fit['c_3'])) # doctest: +SKIP
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