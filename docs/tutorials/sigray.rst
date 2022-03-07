Processing a wavefront metrology experiment
===========================================

Converting raw data at the Sigray lab
-------------------------------------

**For the experiments at the Sigray lab only**

All we need to convert the raw experimental data are a scan number, X-ray target
used during the measurements (Molybdenum, Cupper or Rhodium), and the distance
between a MLL lens and the detector in meters. We parse it to
:func:`pyrost.cxi_converter_sigray` function as follows:

.. code-block:: python

    >>> import pyrost as rst
    >>> data = rst.cxi_converter_sigray(out_path='sigray.cxi', scan_num=2989, target='Mo')

The function reads the log files and a detector bad pixels mask to initiate `basis_vectors`,
`distance`, `mask`, `translations`, `x_pixel_size`, `y_pixel_size`, and `wavelength`:

.. code-block:: python

    >>> data.contents()
    ['good_frames', 'wavelength', 'translations', 'basis_vectors', 'data', 'whitefield', 'mask',
    'frames', 'x_pixel_size', 'distance', 'y_pixel_size', 'files', 'num_threads']

.. note::
    We may save the data container to a CXI file at any time with :func:`pyrost.STData.save`
    method, see the section :ref:`diatom-saving` in the Diatom dataset tutorial.

Working with the data
---------------------
The function returns a :class:`pyrost.STData` data container, which has a set of utility routines
(see :class:`pyrost.STData` for the full list of methods). Usually the pre-processing of a Sigray
dataset consists of:

* Defining a region of interest (:class:`pyrost.Crop`, :func:`pyrost.STData.update_transform`).
* Mirroring the data around the vertical detector axis if needed (:class:`pyrost.Mirror`,
  :func:`pyrost.STData.update_transform`).
* Masking bad pixels (:func:`pyrost.STData.update_mask`).
* Integrating the stack of frames along the vertical detector axis (:func:`pyrost.STData.integrate_data`).

.. code-block:: python

    >>> data = data.update_mask(pmax=99.999, update='multiply')
    >>> data = data.integrate_data(axis=0)
    >>> data = data.crop_data(roi=(0, 1, 200, 1200))
    >>> data = data.mirror_data(axis=1)

    >>> fig, ax = plt.subplots(figsize=(12, 2))
    >>> ax.imshow(data.get('data')[:, 0])
    >>> ax.set_title('Ptychograph', fontsize=20)
    >>> ax.set_xlabel('horizontal axis', fontsize=15)
    >>> ax.set_ylabel('frames', fontsize=15)
    >>> ax.tick_params(labelsize=15)
    >>> plt.show()

.. image:: ../figures/sigray_ptychograph.png
    :width: 100 %
    :alt: Ptychograph

Also, prior to conducting the speckle tracking update one needs to know the
defocus distance. You can estimate it with :func:`pyrost.STData.defocus_sweep`.
It generates sample profiles for a set of defocus distances and yields average values
of the local variance (:math:`\left< R[i, j] \right>`, see
:func:`pyrost.STData.defocus_sweep`), which characterizes the reference image's
contrast (the higher is the value the sharper is the reference profile).

.. code-block:: python

    >>> defoci = np.linspace(5e-5, 3e-4, 50)
    >>> sweep_scan = data.defocus_sweep(defoci, size=50, extra_args={'hval': 30})
    >>> defocus = defoci[np.argmax(sweep_scan)]
    >>> print(defocus)
    0.00015204081632653058

    >>> fig, ax = plt.subplots(figsize=(12, 6))
    >>> ax.plot(defoci * 1e3, sweep_scan)
    >>> ax.set_xlabel('Defocus distance, [mm]', fontsize=20)
    >>> ax.set_title('Average gradient magnitude squared', fontsize=20)
    >>> ax.tick_params(labelsize=15)
    >>> ax.grid(True)
    >>> plt.show()

.. image:: ../figures/sweep_scan_sigray.png
    :width: 100 %
    :alt: Defocus sweep scan.

Let's update the data container with the defocus distance we got. 

.. code-block:: python

    >>> data = data.update_defocus(defocus)

Speckle tracking update
-----------------------
Now we're ready to generate a :class:`pyrost.SpeckleTracking` object, which is able to
perform the speckle tracking reconstruction with :func:`pyrost.SpeckleTracking.train_adapt`
method. For more information about the parameters see the section :ref:`diatom-st-update` in the
2d dataset tutorial.

.. code-block:: python

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.train_adapt(h0=15., blur=12., sw_x=5)
    >>> data.update_phase(st_res)

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    >>> axes[0].plot(np.arange(st_res.reference_image.shape[1]) - st_res.m0,
    >>>              st_res.reference_image[0])
    >>> axes[0].set_title('Reference image', fontsize=20)
    >>> axes[1].plot((st_res.pixel_map - st_obj.pixel_map)[1, 0])
    >>> axes[1].set_title('Pixel mapping', fontsize=20)
    >>> for ax in axes:
    >>>     ax.tick_params(labelsize=15)
    >>>     ax.set_xlabel('Fast axis, pixels', fontsize=15)
    >>>     ax.grid(True)
    >>> plt.show()

.. image:: ../figures/sigray_res.png
    :width: 100 %
    :alt: Speckle tracking update results.

Phase fitting
-------------
In the end we want to look at a angular displacement profile of the X-ray beam and
find the fit to the profile with a polynomial. All of it could be done with 
:class:`pyrost.AberrationsFit` fitter object, which can be obtained with
:func:`pyrost.STData.get_fit` method. We may parse the direct beam coordinate
in pixels to center the scattering angles aroung the direction of the direct beam:

.. code-block:: python

    >>> fit_obj = data.get_fit(axis=1, center=20)
    
Moreover we would like to remove the first order polynomial term from the displacement
profile with the :func:`pyrost.AberrationsFit.remove_linear_term`, since it
characterizes the beam's defocus and is of no interest to us. After that, you
can obtain the best fit to the displacement profile with :func:`pyrost.AberrationsFit.fit`
and to the phase profile with :func:`pyrost.AberrationsFit.fit_phase`:

.. code-block:: python

    >>> fit_obj = fit_obj.remove_linear_term()
    >>> fit = fit_obj.fit(max_order=3)

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    >>> axes[0].plot(fit_obj.thetas, fit_obj.theta_ab * 1e9, 'b')
    >>> axes[0].plot(fit_obj.thetas, fit_obj.model(fit['fit']) * fit_obj.ref_ap * 1e9,
    >>>              'b--', label=fr"R-PXST $c_4$ = {fit['c_4']:.4f} rad/mrad^4")
    >>> axes[0].set_title('Angular displacements, nrad', fontsize=20)
    >>> 
    >>> axes[1].plot(fit_obj.thetas, fit_obj.phase, 'b')
    >>> axes[1].plot(fit_obj.thetas, fit_obj.model(fit['ph_fit']), 'b--',
    >>>              label=fr"R-PXST $c_4$ ={fit['c_4']:.4f} rad/mrad^4")
    >>> axes[1].set_title('Phase, rad', fontsize=20)
    >>> for ax in axes:
    >>>     ax.legend(fontsize=15)
    >>>     ax.tick_params(labelsize=15)
    >>>     ax.set_xlabel('Scattering angles, rad', fontsize=15)
    >>>     ax.grid(True)
    >>> plt.show() 

.. image:: ../figures/sigray_fits.png
    :width: 100 %
    :alt: Phase polynomial fit.