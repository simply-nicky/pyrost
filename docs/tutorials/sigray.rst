Speckle Tracking at the Sigray Lab
==================================

Converting raw data
-------------------
All we need to convert the raw experimental data are a scan
number, X-ray target used during the measurements (Molybdenum,
Cupper or Rhodium), and the distance between a MLL lens and
the detector in meters. We parse it to
:func:`pyrost.cxi_converter_sigray` function as follows:

.. doctest::

    >>> import pyrost as rst
    >>> data = rst.cxi_converter_sigray(scan_num=2989, target='Mo', distance=2.)

.. note::
    We may save the data container to a CXI file at any time with
    :func:`pyrost.STData.write_cxi` function, see the section
    :ref:`diatom-saving` in the Diatom dataset tutorial.

Working with the data
---------------------
The function returns a :class:`pyrost.STData` data container,
which has a set of utility routines (see :class:`pyrost.STData`). For
instance, usually we work with one dimensional scans, so we can mask the bad
pixels, integrate the measured frames along the slow axis, mirror the data,
and crop it using a region of interest as follows:

.. doctest::

    >>> data = data.update_mask(pmax=99.999, update='multiply')
    >>> data = data.integrate_data(axis=0)
    >>> data = data.crop_data(roi=(0, 1, 200, 1240))
    >>> data = data.mirror_data(axis=0)

    >>> fig, ax = plt.subplots(figsize=(14, 6)) # doctest: +SKIP
    >>> ax.imshow(data.get('data')[:, 0]) # doctest: +SKIP
    >>> ax.set_title('Ptychograph', fontsize=20) # doctest: +SKIP
    >>> ax.set_xlabel('fast axis', fontsize=15) # doctest: +SKIP
    >>> ax.set_ylabel('frames', fontsize=15) # doctest: +SKIP
    >>> ax.tick_params(labelsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/sigray_ptychograph.png
    :width: 100 %
    :alt: Ptychograph

Also, prior to conducting the speckle tracking update one needs to know the
defocus distance. You can estimate it with :func:`pyrost.STData.defocus_sweep`.
It generates sample profiles for a set of defocus distances and yields average values
of the local variance (:math:`\left< R[i, j] \right>`, see
:func:`pyrost.STData.defocus_sweep`), which characterizes the reference image's
contrast (the higher is the value the sharper is the reference profile).

.. doctest::

    >>> defoci = np.linspace(5e-5, 3e-4, 50) # doctest: +SKIP
    >>> sweep_scan = data.defocus_sweep(defoci)
    >>> defocus = defoci[np.argmax(sweep_scan)] # doctest: +SKIP
    >>> print(defocus) # doctest: +SKIP
    0.00015204081632653058

    >>> fig, ax = plt.subplots(figsize=(12, 6)) # doctest: +SKIP
    >>> ax.plot(defoci * 1e3, sweep_scan) # doctest: +SKIP
    >>> ax.set_xlabel('Defocus distance, [mm]', fontsize=20) # doctest: +SKIP
    >>> ax.set_title('Average gradient magnitude squared', fontsize=20) # doctest: +SKIP
    >>> ax.tick_params(labelsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. image:: ../figures/sweep_scan_sigray.png
    :width: 100 %
    :alt: Defocus sweep scan.

Let's update the data container with the defocus distance we got. 

    .. doctest::
    
        >>> data = data.update_defocus(defocus)

Speckle Tracking update
-----------------------
Now we’re ready to generate a pyrost.SpeckleTracking object, which is able to perform the
speckle tracking procedure with :func:`pyrost.SpeckleTracking.iter_update_gd` method.
For more information about the parameters see the section :ref:`diatom-st-update` in the
Diatom dataset tutorial.

.. doctest::

    >>> st_obj = data.get_st()
    >>> st_res = st_obj.iter_update_gd(ls_ri=8., ls_pm=1.5, blur=12., sw_fs=5,
    >>>                                n_iter=150, learning_rate=5e0)
    >>> data = data.update_phase(st_res)

    >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # doctest: +SKIP
    >>> axes[0].plot(np.arange(st_res.reference_image.shape[1]) - st_res.m0, # doctest: +SKIP
    >>>              st_res.reference_image[0]) # doctest: +SKIP
    >>> axes[0].set_title('Reference image', fontsize=20) # doctest: +SKIP
    >>> axes[1].plot((st_res.pixel_map - st_obj.pixel_map)[1, 0]) # doctest: +SKIP
    >>> axes[1].set_title('Pixel mapping', fontsize=20) # doctest: +SKIP
    >>> for ax in axes: # doctest: +SKIP
    >>>     ax.tick_params(labelsize=15) # doctest: +SKIP
    >>>     ax.set_xlabel('Fast axis, pixels', fontsize=15) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

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

.. doctest::

    >>> fit_obj = data.get_fit(axis=1, center=20)
    
Moreover we would like to remove the first order polynomial term from the displacement
profile with the :func:`pyrost.AberrationsFit.remove_linear_term`, since it
characterizes the beam's defocus and is of no interest to us:

.. doctest::

    >>> fit_obj = fit_obj.remove_linear_term()

    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # doctest: +SKIP
    >>> axes[0].plot(fit_obj.thetas, fit_obj.theta_aberrations * 1e9, 'b') # doctest: +SKIP
    >>> axes[0].plot(fit_obj.thetas, fit_obj.model(fcf_rst['fit']) * fit_obj.ref_ap * 1e9, # doctest: +SKIP
    >>>              'b--', label=fr"RST $c_4 = {fcf_rst['c_4']:.4f} rad/mrad^4$") # doctest: +SKIP
    >>> axes[0].set_title('Angular displacements, nrad', fontsize=20) # doctest: +SKIP
    >>>  # doctest: +SKIP
    >>> axes[1].plot(fit_obj.thetas, fit_obj.phase, 'b') # doctest: +SKIP
    >>> axes[1].plot(fit_obj.thetas, fit_obj.model(fcf_rst['ph_fit']), 'b--', # doctest: +SKIP
    >>>              label=fr"RST $c_4={fcf_rst['c_4']:.4f} rad/mrad^4$") # doctest: +SKIP
    >>> axes[1].set_title('Phase, rad', fontsize=20) # doctest: +SKIP
    >>> for ax in axes: # doctest: +SKIP
    >>>     ax.legend(fontsize=15) # doctest: +SKIP
    >>>     ax.tick_params(labelsize=15) # doctest: +SKIP
    >>>     ax.set_xlabel('Scattering angles, rad', fontsize=15) # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

.. image:: ../figures/sigray_fits.png
    :width: 100 %
    :alt: Phase polynomial fit.