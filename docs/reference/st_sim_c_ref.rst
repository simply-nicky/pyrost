st_sim Core Functions
=====================

Core functions of the st_sim Speckle Tracking scan simulation.
Contains core tools to calculate wavefront propagation based on
Fresnel diffraction theory and to generate simulated data from
the wavefronts. All the functions are written in `Cython`_.

.. _Cython: https://cython.org

.. toctree::
    :maxdepth: 1
    :caption: Contents

    c_funcs/lens_wp
    c_funcs/aperture_wp
    c_funcs/fraunhofer_1d
    c_funcs/fraunhofer_1d_scan
    c_funcs/barcode_profile
    c_funcs/make_frames
    c_funcs/make_whitefield