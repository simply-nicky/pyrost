Speckle Tracking at the Sigray Lab
==================================

Converting raw data to a CXI file
---------------------------------
All you need to convert the raw data are a scan number,
X-ray target used (Molybdenum, Cupper or Rhodium), and
the distance between a MLL lens and the detector in meters.
You parse it to the :func:`pyrost.cxi_converter_sigray` function:

.. doctest::

    >>> import pyrost as rst
    >>> data = rst.cxi_converter_sigray(scan_num=2989, target='Mo', distance=2.)

.. note::
    You amy save a data container to a CXI file at any time with
    :func:`pyrost.STData.write_cxi` function

Working with the data
---------------------
The function returns a :class:`pyrost.STData` data container,
which has a set of utility routines (see :class:`pyrost.STData`). For
instance, usually we work with one dimensional scans, so we can integrate
the measured frames along the slow axis, mirror the data, and crop
it using a regio of interest as follows:

.. doctest::

    >>> data = data.integrate_data(axis=0)
    >>> data = data.mirror_data(axis=0)
    >>> data = data.crop_data(roi=(0, 1, 200, 1240))

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
of the gradient magnitude squared (:math:`\left| \nabla I_{ref} \right|^2`), which characterizes
reference image's contrast (the higher the value the better the estimate of defocus distance
is). Also, it returns the set of sample profiles if `return_sweep` argument is True.

.. doctest::

    >>> defoci = np.linspace(5e-5, 3e-4, 50) # doctest: +SKIP
    >>> sweep_scan = data.defocus_sweep(defoci, return_sweep=True)
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

Speckle Tracking update
-----------------------