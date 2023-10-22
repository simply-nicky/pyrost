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

def robust_mean(np.ndarray inp not None, int axis=0, double r0=0.0, double r1=0.5,
                int n_iter=12, double lm=9.0, unsigned num_threads=1):
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)

    cdef int ndim = inp.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1

    cdef unsigned long *_dims = <unsigned long *>inp.shape

    cdef np.npy_intp *odims = <np.npy_intp *>malloc((ndim - 1) * sizeof(np.npy_intp))
    if odims is NULL:
        raise MemoryError('not enough memory')
    cdef int i
    for i in range(axis):
        odims[i] = inp.shape[i]
    for i in range(axis + 1, ndim):
        odims[i - 1] = inp.shape[i]

    cdef int type_num = np.PyArray_TYPE(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim - 1, odims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = robust_mean_c(_out, _inp, ndim, _dims, 8, axis, compare_double,
                                 get_double, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = robust_mean_c(_out, _inp, ndim, _dims, 4, axis, compare_float,
                                 get_float, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_INT32:
            fail = robust_mean_c(_out, _inp, ndim, _dims, 4, axis, compare_int,
                                 get_int, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = robust_mean_c(_out, _inp, ndim, _dims, 4, axis, compare_uint,
                                 get_uint, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = robust_mean_c(_out, _inp, ndim, _dims, 8, axis, compare_ulong,
                                 get_ulong, r0, r1, n_iter, lm, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    free(odims)
    return out

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
