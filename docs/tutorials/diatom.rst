Speckle tracking reconstruction of a 2d dataset
===============================================

Diatom dataset CXI file
-----------------------
First download the `diatom.cxi <https://www.cxidb.org/data/134/diatom.cxi>`_
file from the `CXIDB <https://www.cxidb.org/>`_. The file has the following
structure:

.. code-block:: console

    $ h5ls -r diatom.cxi
    /                        Group
    /entry_1                 Group
    /entry_1/data_1          Group
    /entry_1/data_1/data     Dataset {121, 516, 1556}
    /entry_1/data_1/experiment_identifier Dataset {121}
    /entry_1/end_time        Dataset {SCALAR}
    /entry_1/experiment_identifier Dataset, same as /entry_1/data_1/experiment_identifier
    /entry_1/instrument_1    Group
    /entry_1/instrument_1/detector_1 Group
    /entry_1/instrument_1/detector_1/basis_vectors Dataset {121, 2, 3}
    /entry_1/instrument_1/detector_1/corner_positions Dataset {121, 3}
    /entry_1/instrument_1/detector_1/count_time Dataset {121, 1}
    /entry_1/instrument_1/detector_1/data Dataset, same as /entry_1/data_1/data
    /entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/experiment_identifier Dataset, same as /entry_1/data_1/experiment_identifier
    /entry_1/instrument_1/detector_1/mask Dataset {516, 1556}
    /entry_1/instrument_1/detector_1/name Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/name Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/energy Dataset {SCALAR}
    /entry_1/instrument_1/source_1/name Dataset {SCALAR}
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_1        Group
    /entry_1/sample_1/geometry Group
    /entry_1/sample_1/geometry/orientation Dataset {121, 6}
    /entry_1/sample_1/geometry/translation Dataset {121, 3}
    /entry_1/sample_1/name   Dataset {SCALAR}
    /entry_1/start_time      Dataset {SCALAR}

As we can see in :code:`entry_1/data_1/data` the file contains a two-dimensional 11x11 scan,
where each frame is an image of 516x1556 pixels.

Loading the file
----------------
Load the CXI file into a data container :class:`pyrost.STData` with :class:`pyrost.CXILoader`.
:func:`pyrost.CXILoader.import_default` returns the default loader with the default CXI protocol
(:func:`pyrost.CXIProtocol.import_default`).

.. note:: :class:`pyrost.CXILoader` will raise an :class:`AttributeError` while loading the data
    from the CXI file if some of the necessary attributes for Speckle Tracking algorithm
    are not provided. You can see full list of the necessary attributes in
    :class:`pyrost.STData`. Adding the missing attributes to :func:`pyrost.CXILoader.load`
    solves the problem.

.. doctest::

    >>> import pyrost as rst
    >>> loader = rst.CXILoader.import_default()
    >>> data = loader.load('diatom.cxi') # doctest: +SKIP

Moreover, you can crop the data with the provided region of interest at the detector plane,
or mask bad frames and bad pixels (See :func:`pyrost.STData.crop_data`,
:func:`pyrost.STData.mask_frames`, :func:`pyrost.STData.update_mask`).

.. code-block:: python

    >>> data = loader.load('diatom.cxi', roi=(75, 420, 55, 455), good_frames=np.arange(1, 121))
    >>> data = data.update_mask(method='perc-bad')

OR

.. code-block:: python

    >>> data = data.crop_data(roi=(75, 420, 55, 455))
    >>> data = data.mask_frames(good_frames=np.arange(1, 121))
    >>> data = data.update_mask(method='perc-bad')

It worked! But still we can not perform the Speckle Tracking update procedure without the
estimates of the defocus distance. You can estimate it with :func:`pyrost.STData.defocus_sweep`.
It generates sample profiles for a set of defocus distances and yields average values
of the gradient magnitude squared (:math:`\left< R[i, j] \right>`, see
:func:`pyrost.STData.defocus_sweep`), which characterizes reference image's shaprness
(the higher is the value the sharper is the reference profile).

.. code-block:: python

    >>> defoci = np.linspace(2e-3, 3e-3, 50)
    >>> sweep_scan = data.defocus_sweep(defoci, size=5, extra_args={'hval': 1.5})
    >>> defocus = defoci[np.argmax(sweep_scan)]
    >>> print(defocus)
    0.002204081632653061

    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> ax.plot(defoci * 1e3, sweep_scan)
    >>> ax.set_xlabel('Defocus distance, [mm]', fontsize=20)
    >>> ax.set_title('Average gradient magnitude squared', fontsize=20)
    >>> ax.tick_params(labelsize=15)
    >>> plt.show()

.. image:: ../figures/sweep_scan.png
    :width: 100 %
    :alt: Defocus sweep scan.

Let's update the data container with the defocus distance we got. 

.. code-block:: python

    >>> data = data.update_defocus(defocus)

.. _diatom-st-update:

Speckle tracking update
-----------------------
Now we're ready to generate a :class:`pyrost.SpeckleTracking` object, which does the heavy
lifting of calculating the pixel mapping between reference plane and detector plane (`pixel_map`),
and generating the unabberated profile of the sample (`reference_image`) following the ptychographic
speckle tracking algorithm [ST]_.

For the speckle tracking update you've got two options to choose from:

    * :func:`pyrost.SpeckleTracking.train` : performs the iterative reference image
      and pixel mapping updates with the constant kernel bandwidths for the reference image
      estimator (`h0`).

    * :func:`pyrost.SpeckleTracking.train_adapt` : does ditto, but updates the bandwidth
      value for the reference image estimator at each iteration by the help of the BFGS method
      to attain the minimal error value.

.. note:: You should pay outmost attention to choosing the right kernel bandwidth of the
    reference image estimator (`h0` in :func:`pyrost.SpeckleTracking.update_reference`). Essentially it
    stands for the high frequency cut-off imposed during the reference profile update, so it helps to
    supress the noise. If the value is too high you'll lose useful information in the reference
    profile. If the value is too low and the data is noisy, you won't get an acurate reconstruction.
    An optimal kernel bandwidth can be found with :func:`pyrost.SpeckleTracking.find_hopt` method.
    
.. note:: Next important parameter is `blur` in :func:`pyrost.SpeckleTracking.update_pixel_map`.
    It helps to prevent the noise propagation to the next iteration by the means of kernel
    smoothing of the updated pixel mapping. **As a rule of thumb, `blur` should be several times
    larger than `h0`**.

.. note:: Apart from pixel mapping update you may try to perform the sample shifts update if you've
    got a low precision or credibilily of sample shifts measurements. You can do it by setting
    the `update_translations` parameter to True.

.. code-block:: python

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.train(sw_x=15, sw_y=15, h0=1.2, blur=8.0,
                              verbose=True, n_iter=5)

    >>> fig, ax = plt.subplots(figsize=(10, 10))
    >>> ax.imshow(st_res.reference_image[700:1200, 100:700], vmin=0.7, vmax=1.3,
    >>>           extent=[100, 700, 1200, 700])
    >>> ax.set_title('Reference image', fontsize=20)
    >>> ax.set_xlabel('horizontal axis', fontsize=15)
    >>> ax.set_ylabel('vertical axis', fontsize=15)
    >>> ax.tick_params(labelsize=15)
    >>> plt.show()

.. image:: ../figures/diatom_image.png
    :width: 100 %
    :alt: Diatom close-up view.

Phase reconstruction
--------------------
We got the pixel mapping between from the detector plane to the reference plane, which can
be easily translated to the angular diplacement profile of the lens. Following the Hartmann sensor
principle (look [ST]_ page 762 for more information), we reconstruct the lens' phase
profile with :func:`pyrost.STData.update_phase` method. Besides, you can fit the phase
profile with polynomial function using :class:`pyrost.AberrationsFit` fitter object,
which can be obtained with :func:`pyrost.STData.get_fit` method.

.. code-block:: python

    >>> data.update_phase(st_res)
    >>> fit_obj_ss = data.get_fit(axis=0)
    >>> fit_ss = fit_obj_ss.fit(max_order=3)
    >>> fit_obj_fs = data.get_fit(axis=1)
    >>> fit_fs = fit_obj_fs.fit(max_order=3)

    >>> fig, ax = plt.subplots(figsize=(10, 10))
    >>> ax.imshow(data.get('phase'))
    >>> ax.set_title('Phase', fontsize=20)
    >>> ax.set_xlabel('horizontal axis', fontsize=15)
    >>> ax.set_ylabel('vertical axis', fontsize=15)
    >>> ax.tick_params(labelsize=15)
    >>> plt.show()

.. image:: ../figures/diatom_phase.png
    :width: 100 %
    :alt: Phase profile.

.. code-block:: python

    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    >>> axes[0].plot(fit_obj_fs.pixels, fit_obj_fs.phase, label='Reconstructed profile')
    >>> axes[0].plot(fit_obj_fs.pixels, fit_obj_fs.model(fit_fs['ph_fit']),
                     label='Polynomial fit')
    >>> axes[0].set_xlabel('horizontal axis', fontsize=15)
    >>> axes[1].plot(fit_obj_ss.pixels, fit_obj_ss.phase, label='Reconstructed profile')
    >>> axes[1].plot(fit_obj_ss.pixels, fit_obj_ss.model(fit_ss['ph_fit']),
    >>>              label='Polynomial fit')
    >>> axes[1].set_xlabel('vertical axis')
    >>> for ax in axes:
    >>>     ax.set_title('Phase', fontsize=20)
    >>>     ax.tick_params(labelsize=15)
    >>>     ax.legend(fontsize=15)
    >>> plt.show()

.. image:: ../figures/phase_fit.png
    :width: 100 %
    :alt: Phase fit.

.. _diatom-saving:

Saving the results
------------------
In the end you can save the results to a CXI file.

.. code-block:: python

    >>> with h5py.File('results/diatom_proc.cxi', 'w') as cxi_file:
    >>>     data.write_cxi(cxi_file)

.. code-block:: console

    $   h5ls -r diatom_proc.cxi
    /                        Group
    /entry_1                 Group
    /entry_1/data_1          Group
    /entry_1/data_1/data     Dataset {121, 516, 1556}
    /entry_1/instrument_1    Group
    /entry_1/instrument_1/detector_1 Group
    /entry_1/instrument_1/detector_1/basis_vectors Dataset {121, 2, 3}
    /entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_1        Group
    /entry_1/sample_1/geometry Group
    /entry_1/sample_1/geometry/translations Dataset {121, 3}
    /frame_selector          Group
    /frame_selector/good_frames Dataset {120}
    /speckle_tracking        Group
    /speckle_tracking/error_frame Dataset {516, 1556}
    /speckle_tracking/defocus_ss    Dataset {SCALAR}
    /speckle_tracking/defocus_fs    Dataset {SCALAR}
    /speckle_tracking/mask   Dataset {516, 1556}
    /speckle_tracking/phase  Dataset {516, 1556}
    /speckle_tracking/pixel_aberrations Dataset {2, 516, 1556}
    /speckle_tracking/pixel_map Dataset {2, 516, 1556}
    /speckle_tracking/pixel_translations Dataset {121, 2}
    /speckle_tracking/reference_image Dataset {1455, 1498}
    /speckle_tracking/roi    Dataset {4}
    /speckle_tracking/whitefield Dataset {516, 1556}

As you can see all the results have been saved using the same CXI protocol.

References
----------

.. [ST] `"Ptychographic X-ray speckle tracking", Morgan, A. J., Quiney, H. M., Bajt,
        S. & Chapman, H. N. (2020). J. Appl. Cryst. 53, 760-780. <https://doi.org/10.1107/S1600576720005567>`_