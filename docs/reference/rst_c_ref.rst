Core functions
==============

Core functions of the robust speckle tracking reconstruction pipeline.
All the functions are written in `Cython`_ or C.

.. _Cython: https://cython.org

.. toctree::
    :maxdepth: 1

    funcs/gaussian_filter
    funcs/gaussian_gradient_magnitude
    funcs/KR_reference
    funcs/LOWESS_reference
    funcs/pm_gsearch
    funcs/pm_rsearch
    funcs/pm_devolution
    funcs/tr_gsearch
    funcs/pm_errors
    funcs/pm_total_error
    funcs/ref_errors
    funcs/ref_total_error
    funcs/ct_integrate