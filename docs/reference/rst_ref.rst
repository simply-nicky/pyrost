Core classes
============

.. raw:: html
    :file: ../figures/pyrost_flowchart.svg

|

The following classes are central to pyrost module:

* :class:`pyrost.STData` is the main data container, which provides an interface for the preprocessing of a XST dataset and can output an object, that performs the main loop of RST update.
* :class:`pyrost.SpeckleTracking` performs the reference image and mapping updates and allows to perform the iterative RST update.

:class:`STData <pyrost.STData>`
-------------------------------

:class:`pyrost.STData` contains various functions that are necessary to conduct an initial
data processing and create several maps and quantities that are needed prior to the
main speckle tracking update algorithm, such as the sample defocus or the
whitefield image:

* :func:`pyrost.STData.update_mask` generates a pixel mask that excludes bad and hot pixels
  from the following analysis.
* :func:`pyrost.STData.defocus_sweep` can help to estimate the focus-to-sample distance by
  generating a reference image together with calculating a mean local variance of the image
  for a set of distances. The mean local variance serves as a figure of merit of how sharp
  or blurry the reference image is. In the end, the defocus distance which yields the sharpest
  reference image is chosen.
* :func:`pyrost.STData.update_whitefield` generates a whitefield image by taking a median through
  a stack of measured frames.
* If the flat-fields are dynamically varying from a frame to a frame, :func:`pyrost.STData.update_flatfields`
  method can generate a flat-field for each frame separately based on Principal Component Analysis
  approach [PCA]_ or by applying median filtering through a stack of frames.

:class:`SpeckleTracking <pyrost.SpeckleTracking>`
-------------------------------------------------

:class:`pyrost.SpeckleTracking` provides an interface to perform the reference image
and lens wavefront reconstruction and offers two methods (:func:`pyrost.SpeckleTracking.iter_update`,
:func:`pyrost.SpeckleTracking.iter_update_gd`) to perform the iterative RST update until the error metric
converges to a minimum. The typical reconstruction cycle consists of:

* Estimating an optimal kernel bandwidth for the reference image estimate (:func:`pyrost.SpeckleTracking.find_hopt`,
  in :func:`pyrost.SpeckleTracking.iter_update_gd` only).
* Generating the reference image (:func:`pyrost.SpeckleTracking.update_reference`).
* Updating the discrete (pixel) mapping between a stack of frames and the generated reference image
  (:func:`pyrost.SpeckleTracking.update_pixel_map`).
* Updating the sample translations vectors (:func:`pyrost.SpeckleTracking.update_translations`).
* Calculating figures of merit (:func:`pyrost.SpeckleTracking.ref_total_error`,
  :func:`pyrost.SpeckleTracking.error_profile`).

Reference image update
++++++++++++++++++++++

:func:`pyrost.SpeckleTracking.update_reference` method supports Kernel regression estimator and Local
Weighted Linear Regression (LOWESS) estimator of the reference image, which can be chosen by the user
with `method` argument.

Pixel mapping update
++++++++++++++++++++

:func:`pyrost.SpeckleTracking.update_pixel_map` updates the pixel mapping at each pixel separately by
minimizing the error metric as a function of the map function. The minimization procedure may be performed
by *grid search*, *random search* or *differential evolution* algorithms. The algorithm is selected with
`method` argument.

The updated pixel mapping usually requires a further regularisation by the help of weighted
kernel smoothing. the kernel bandwidth is defined by `blur` argument.

Since the geometric mapping between the reference plane and the detector plane is defined in terms of the
gradient of a scalar function, the curl of the mapping must be zero. Such vector field is called ‘irrotational’.
However, the pixel mapping is updated at each point separately without any examination of the irrotationality.
In order to ensure that the mapping field is irrotational, one can first integrate the mapping function and
then numerically calculate the gradient to obtain a `curl-free` version. One can enable this procedure with 
`integrate` argument.

Contents
--------

.. toctree::
    :maxdepth: 1

    classes/st_data
    classes/speckle_tracking