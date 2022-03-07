Speckle tracking reconstruction of a simulated dataset
======================================================

Speckle Tracking update procedure of one-dimensional scan doesn't differ much
from the case of two-dimensional scan. See :doc:`diatom` for an example.

1D scan CXI file
----------------
In order to obtain the file generate it using :doc:`st_sim`. The file has
the following structure:

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

The file contains a ptychograph (set of frames summed over one of detector axes)
of 200 frames.

Loading the file
----------------
The procedure for loading the data from a file is the same as in :doc:`diatom`:

* Create a :class:`pyrost.CXIProtocol` protocol.
* Open the file with :class:`pyrost.CXIStore` file handler.
* Create a :class:`pyrost.STData` R-PXST data container.
* Load all the data from the file with :func:`pyrost.STData.load`.

.. code-block:: python

    >>> import pyrost as rst
    >>> protocol = rst.CXIProtocol.import_default()
    >>> files = rst.CXIStore(input_files='sim.cxi', output_file='sim.cxi',
    >>>                      protocol=protocol)
    >>> data = rst.STData(files=files)
    >>> data = data.load()

The file already contains all the necessary attributes to perform the speckle tracking
reconstruction:

.. code-block:: python

    >>> data.contents()
    ['whitefield', 'x_pixel_size', 'files', 'y_pixel_size', 'data', 'num_threads',
     'distance', 'defocus_x', 'good_frames', 'defocus_y', 'basis_vectors', 'translations',
     'mask', 'frames', 'wavelength', 'pixel_translations']

Speckle tracking update
-----------------------
The steps to perform the speckle tracking update are also the same as in :doc:`diatom`:

* Create a :class:`pyrost.SpeckleTracking` object.
* Find an optimal kernel bandwidth with :func:`pyrost.SpeckleTracking.find_hopt`.
* Perform the iterative R-PXST update  with :func:`pyrost.SpeckleTracking.train`
  or :func:`pyrost.SpeckleTracking.train_adapt`.

.. code-block:: python

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.train_adapt(search_window=(0.0, 10.0, 0.1), blur=8., verbose=True, n_iter=20)

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    >>> axes[0].plot(np.arange(st_res.reference_image.shape[1]) - st_res.ref_orig[1],
    >>>              st_res.reference_image[0])
    >>> axes[0].set_title('Reference image', fontsize=20)
    >>> axes[1].plot((st_res.pixel_map - st_obj.pixel_map)[1, 0])
    >>> axes[1].set_title('Pixel mapping', fontsize=20)
    >>> for ax in axes:
    >>>     ax.tick_params(labelsize=15)
    >>>     ax.set_xlabel('Fast axis, pixels', fontsize=20)
    >>> plt.show()

.. image:: ../figures/1d_sim_res.png
    :width: 100 %
    :alt: Speckle tracking update results.

Phase reconstruction
--------------------
After we got the pixel map we're able to reconstruct the phase profile and fit it with
polynomial function.

.. code-block:: python

    >>> data.import_st(st_res)
    >>> fit = data.fit_phase(axis=1, max_order=2)
    >>> fit['c_3'] # third order fit coefficient
    -0.04798021776009187

    >>> fit_obj = data.get_fit(axis=1)
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    >>> axes[0].plot(fit_obj.pixels, fit_obj.pixel_aberrations)
    >>> axes[0].plot(fit_obj.pixels, fit_obj.model(fit['fit']))
    >>> axes[0].set_title('Pixel aberrations', fontsize=20)
    >>> axes[1].plot(fit_obj.pixels, fit_obj.phase)
    >>> axes[1].plot(fit_obj.pixels, fit_obj.model(fit['ph_fit']),
    >>>              label=r'$\alpha$ = {:.5f} rad/mrad^3'.format(fit['c_3']))
    >>> axes[1].set_title('Phase', fontsize=20)
    >>> axes[1].legend(fontsize=15)
    >>> for ax in axes:
    >>>     ax.tick_params(axis='both', which='major', labelsize=15)
    >>>     ax.set_xlabel('horizontal axis', fontsize=15)
    >>> plt.show()

.. image:: ../figures/1d_sim_fits.png
    :width: 100 %
    :alt: Phase polynomial fit.

Saving the results
------------------
In the end you can save the results to a CXI file.

.. code-block:: python

    >>> data.save()

.. code-block:: console

    $   h5ls -r results/sim_results/data_proc.cxi
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
    /speckle_tracking/phase  Dataset {1, 985}
    /speckle_tracking/pixel_aberrations Dataset {2, 1, 985}
    /speckle_tracking/pixel_translations Dataset {200/Inf, 2}
    /speckle_tracking/reference_image Dataset {1, 5924}
    /speckle_tracking/scale_map Dataset {1, 985}
    /speckle_tracking/translations Dataset {200/Inf, 3}
    /speckle_tracking/whitefield Dataset {1, 985}