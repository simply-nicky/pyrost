#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from libc.math cimport log
from libc.math cimport sqrt, exp, pi, floor, ceil
from libc.string cimport memcmp
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp
from scipy.ndimage import gaussian_gradient_magnitude as ggm, gaussian_filter as gf
import speckle_tracking as st

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.uint64_t uint_t
ctypedef np.complex128_t complex_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF NO_VAR = -1.0

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def init_fftw():
    fftw_init_threads()
    
    if Py_AtExit(fftw_cleanup_threads) != 0:
        raise ImportError('Failed to register the fftw library shutdown callback')

init_fftw()

cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    r"""If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

def make_whitefield(data: np.ndarray, mask: np.ndarray, axis: cython.int=0,
                    num_threads: cython.uint=1) -> np.ndarray:
    data = np.PyArray_GETCONTIGUOUS(data)
    mask = np.PyArray_GETCONTIGUOUS(mask)

    if not np.PyArray_ISBOOL(mask):
        raise TypeError('mask array must be of boolean type')
    cdef int ndim = data.ndim
    if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
        raise ValueError('mask and data arrays must have identical shapes')
    axis = axis if axis >= 0 else ndim + axis
    cdef np.npy_intp isize = np.PyArray_SIZE(data)
    cdef np.npy_intp *dims = <np.npy_intp *>malloc((ndim - 1) * sizeof(np.npy_intp))
    if dims is NULL:
        abort()
    cdef int i
    for i in range(axis):
        dims[i] = data.shape[i]
    cdef np.npy_intp npts = data.shape[axis]
    for i in range(axis + 1, ndim):
        dims[i - 1] = data.shape[i]
    cdef np.npy_intp istride = np.PyArray_STRIDE(data, axis) / np.PyArray_ITEMSIZE(data)
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim - 1, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    print(isize, npts, istride)
    with nogil:
        if type_num == np.NPY_FLOAT64:
                whitefield(_out, _data, _mask, isize, npts, istride, 8, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
                whitefield(_out, _data, _mask, isize, npts, istride, 4, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
                whitefield(_out, _data, _mask, isize, npts, istride, 4, compare_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    free(dims)
    return out

def st_update(I_n, dij, basis, x_ps, y_ps, z, df, search_window, n_iter=5,
              filter=None, update_translations=False, verbose=False):
    """
    Andrew's speckle tracking update algorithm
    
    I_n - measured data
    W - whitefield
    basis - detector plane basis vectors
    x_ps, y_ps - x and y pixel sizes
    z - distance between the sample and the detector
    df - defocus distance
    sw_max - pixel mapping search window size
    n_iter - number of iterations
    """
    M = np.ones((I_n.shape[1], I_n.shape[2]), dtype=bool)
    W = st.make_whitefield(I_n, M, verbose=verbose)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis,
                                            x_ps, y_ps, z,
                                            df, verbose=verbose)
    I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=verbose)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                                search_window=search_window, subpixel=True,
                                fill_bad_pix=False, integrate=False,
                                quadratic_refinement=False, verbose=verbose,
                                filter=filter)[0]

        # make reference image
        I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

        # update translations
        if update_translations:
            dij_pix = st.update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]
    return {'u':u, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

def pixel_translations(basis, dij, df, z):
    dij_pix = (basis * dij[:, None]).sum(axis=-1)
    dij_pix /= (basis**2).sum(axis=-1) * df / z
    dij_pix -= dij_pix.mean(axis=0)
    return np.ascontiguousarray(dij_pix[:, 0]), np.ascontiguousarray(dij_pix[:, 1])

# def str_update(I_n, W, dij, basis, x_ps, y_ps, z, df, sw_max=100, n_iter=5, l_scale=2.5):
#     """
#     Robust version of Andrew's speckle tracking update algorithm
    
#     I_n - measured data
#     W - whitefield
#     basis - detector plane basis vectors
#     x_ps, y_ps - x and y pixel sizes
#     z - distance between the sample and the detector
#     df - defocus distance
#     sw_max - pixel mapping search window size
#     n_iter - number of iterations
#     """
#     I_n = I_n.astype(np.float64)
#     W = W.astype(np.float64)
#     u0 = np.indices(W.shape, dtype=np.float64)
#     di, dj = pixel_translations(basis, dij, df, z)
#     I0, n0, m0 = make_reference(I_n=I_n, W=W, u=u0, di=di, dj=dj, ls=l_scale, sw_fs=0, sw_ss=0)

#     es = []
#     for i in range(n_iter):

#         # calculate errors
#         es.append(mse_total(I_n=I_n, W=W, I0=I0, u=u0, di=di - n0, dj=dj - m0, ls=l_scale))

#         # update pixel map
#         u = update_pixel_map_gs(I_n=I_n, W=W, I0=I0, u0=u0, di=di - n0, dj=dj - m0,
#                                 sw_ss=0, sw_fs=sw_max, ls=l_scale)
#         sw_max = int(np.max(np.abs(u - u0)))
#         u0 = u0 + gaussian_filter(u - u0, (0, 0, l_scale))

#         # make reference image
#         I0, n0, m0 = make_reference(I_n=I_n, W=W, u=u0, di=di, dj=dj, ls=l_scale, sw_ss=0, sw_fs=0)
#         I0 = gaussian_filter(I0, (0, l_scale))
#     return {'u':u0, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

# def phase_fit(pixel_ab, x_ps, z, df, wl, max_order=2, pixels=None):
#     def errors(fit, x, y):
#         return np.polyval(fit[:max_order + 1], x - fit[max_order + 1]) - y

#     # Apply ROI
#     if pixels is None:
#         pixels = np.arange(pixel_ab.shape[0])
#     else:
#         pixel_ab = pixel_ab[pixels]
#     x_arr = pixels * x_ps / z * 1e3

#     # Calculate initial argument
#     x0 = np.zeros(max_order + 2)
#     u0 = gaussian_filter(pixel_ab, pixel_ab.shape[0] / 10)
#     if np.median(np.gradient(np.gradient(u0))) > 0:
#         idx = np.argmin(u0)
#     else:
#         idx = np.argmax(u0)
#     x0[max_order + 1] = x_arr[idx]
#     lb = -np.inf * np.ones(max_order + 2)
#     ub = np.inf * np.ones(max_order + 2)
#     lb[max_order + 1] = x_arr.min()
#     ub[max_order + 1] = x_arr.max()
        
#     # Perform least squares fitting
#     fit = least_squares(errors, x0, bounds=(lb, ub), loss='cauchy', jac='3-point',
#                         args=(x_arr, pixel_ab), xtol=1e-14, ftol=1e-14)
#     if np.linalg.det(fit.jac.T.dot(fit.jac)):
#         cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
#         err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
#     else:
#         err = np.zeros(fit.x.size)

#     # Convert the fit
#     ang_fit = fit.x * x_ps / z
#     ang_fit[:max_order + 1] /= np.geomspace((x_ps / z)**max_order, 1, max_order + 1)
#     ph_fit = np.zeros(max_order + 3)
#     ph_fit[:max_order + 1] = ang_fit[:max_order + 1] * 2 * np.pi / wl * df / np.linspace(max_order + 1, 1, max_order + 1)
#     ph_fit[max_order + 2] = ang_fit[max_order + 1]
#     ph_fit[max_order + 1] = -np.polyval(ph_fit[:max_order + 2], pixels * x_ps / z - ph_fit[max_order + 2]).mean()

#     # evaluating errors
#     r_sq = 1 - np.sum(errors(fit.x, pixels, pixel_ab)**2) / np.sum((pixel_ab.mean() - pixel_ab)**2)
#     return {'pixels': pixels, 'pix_fit': fit.x, 'ang_fit': ang_fit,
#             'pix_err': err, 'ph_fit': ph_fit, 'r_sq': r_sq, 'fit': fit}
