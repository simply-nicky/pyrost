#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True, linetrace=False, profile=False
import numpy as np
import cython
from speckle_tracking import make_object_map, calc_error, update_pixel_map, update_translations
from cython.parallel import prange, parallel
from pyrost.bin import pyfftw
from typing import Tuple, Optional

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray layers, complex t0,
                                   complex t1, double sigma, unsigned num_threads):
    x_arr = check_array(x_arr, np.NPY_FLOAT64)
    layers = check_array(layers, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef int ndim = x_arr.ndim
    cdef np.npy_intp *dims = x_arr.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)

    cdef np.npy_intp isize = np.PyArray_SIZE(x_arr)
    cdef np.npy_intp lsize = np.PyArray_SIZE(layers)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef double *_x = <double *>np.PyArray_DATA(x_arr)
    cdef double *_lyrs = <double *>np.PyArray_DATA(layers)
    with nogil:
        fail = ml_profile(_out, _x, isize, _lyrs, lsize, t0, t1, sigma, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def barcode_profile(np.ndarray x_arr not None, np.ndarray bars not None, double bulk_atn,
                    double bar_atn, double bar_sigma, unsigned num_threads=1):
    r"""Return an array of barcode's transmission profile calculated
    at `x_arr` coordinates.

    Args:
        x_arr (numpy.ndarray) : Array of the coordinates, where the transmission
            coefficients are calculated [um].    
        bars (numpy.ndarray) : Coordinates of barcode's bar positions [um].
        bulk_atn (float) : Barcode's bulk attenuation coefficient (0.0 - 1.0).
        bar_atn (float) : Barcode's bar attenuation coefficient (0.0 - 1.0).
        bar_sigma (float) : Inter-diffusion length [um].
        num_threads (int) : Number of threads used in the calculations.
    
    Returns:
        numpy.ndarray : Array of barcode's transmission profiles.

    Notes:
        Barcode's transmission profile is given by:
        
        .. math::
            t_b(x) = t_{air} + \frac{t_1 - t_{air}}{2}
            \left( \tanh\left(\frac{x - x^b_1}{\sigma_b} \right) +
            \tanh\left(\frac{x^b_N - x}{\sigma_b}\right) \right)\\
            - \frac{t_2 - t_1}{2}\sum_{n = 1}^{(N - 1) / 2} 
            \left( \tanh\left(\frac{x - x^b_{2 n}}{\sigma_b}\right) + 
            \tanh\left(\frac{x^b_{2 n + 1} - x}{\sigma_b}\right) \right)
        
        where :math:`t_1 = \sqrt{1 - A_{bulk}}`, :math:`t_2 = \sqrt{1 - A_{bulk} - A_{bar}}`
        are the transmission coefficients for the bulk and the bars of the sample,
        :math:`x^b_n` is a set of bars coordinates, and :math:`\sigma_b` is the
        inter-diffusion length.
    """
    cdef complex t0 = sqrt(1.0 - bulk_atn)
    cdef complex t1 = sqrt(1.0 - bulk_atn - bar_atn)
    return ml_profile_wrapper(x_arr, bars, t0, t1, bar_sigma, num_threads)

def st_update(I_n: np.ndarray, W: np.ndarray, M: np.ndarray, dij_pix: np.ndarray, u: np.ndarray,
              search_window: Tuple[int, int], n_iter: int=5, filter: Optional[float]=None,
              update_translations: bool=False, verbose: bool=False):
    """
    Andrew's speckle tracking update algorithm
    
    Args:
        I_n : Measured data.
    W - whitefield
    basis - detector plane basis vectors
    x_ps, y_ps - x and y pixel sizes
    z - distance between the sample and the detector
    df - defocus distance
    wl - wavelength
    sw_max - pixel mapping search window size
    n_iter - number of iterations
    """
    I0, n0, m0 = make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=verbose)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                             search_window=search_window, subpixel=True,
                             fill_bad_pix=False, integrate=False,
                             quadratic_refinement=False, verbose=verbose,
                             filter=filter)[0]

        # make reference image
        I0, n0, m0 = make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

        # update translations
        if update_translations:
            dij_pix = update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]

    return {'u': u, 'I0': I0, 'errors': es, 'n0': n0, 'm0': m0}
