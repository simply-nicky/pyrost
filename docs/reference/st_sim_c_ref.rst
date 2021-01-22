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

    c_funcs/rsc_wp
    c_funcs/fhf_wp
    c_funcs/fhf_wp_scan
    c_funcs/bar_positions
    c_funcs/barcode_profile
    c_funcs/fft_convolve
    c_funcs/fft_convolve_scan
    c_funcs/make_frames
    c_funcs/make_whitefield