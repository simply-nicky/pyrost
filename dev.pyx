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
from libc.math cimport sqrt, erf, sin, cos, exp, fabs
import speckle_tracking as st

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.uint64_t uint_t
ctypedef np.complex128_t complex_t
ctypedef np.float64_t double_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF NO_VAR = -1.0

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

cdef int binary_search_float(double *values, int l, int r, double x) nogil:
    cdef int m = l + (r - l) // 2
    if l <= r:
        if x == values[m]:
            return m
        elif x > values[m] and x <= values[m + 1]:
            return m + 1
        elif x < values[m]:
            return binary_search_float(values, l, m, x)
        else:
            return binary_search_float(values, m + 1, r, x)

cdef int searchsorted(double *values, double x, int r) nogil:
    if x < values[0]:
        return 0
    elif x > values[r - 1]:
        return r
    else:
        return binary_search_float(values, 0, r, x)

cdef complex mll_c(double *layers, int a, complex mt1, complex mt2, double x, double sgm) nogil:
    cdef:
        int b = 2 * (a // 2), j0
        double x0, x1
        complex tr
    j0 = searchsorted(layers, x, a) # even '-', odd '+'
    tr = 0
    if j0 > 0 and j0 < b:
        x0 = (x - layers[j0 - 1]) / sqrt(2) / sgm
        x1 = (x - layers[j0]) / sqrt(2) / sgm
        tr += (mt1 - mt2) / 2 * (j0 % 2 - 0.5) * (erf(x0) - erf(x1))
        tr -= (mt1 - mt2) / 4 * erf((x - layers[0]) / sqrt(2) / sgm)
        tr += (mt1 - mt2) / 4 * erf((x - layers[b - 1]) / sqrt(2) / sgm)
    tr += mt1 / 2 * erf((x - layers[0]) / sqrt(2) / sgm)
    tr -= mt1 / 2 * erf((x - layers[b - 1]) / sqrt(2) / sgm)
    return tr

def make_mll_slice(np.ndarray[double_t, ndim=1] x_arr, np.ndarray[double_t, ndim=1] layers, complex_t mt1, complex_t mt2, double sgm, double kdz):
    cdef:
        np.npy_intp a = np.PyArray_DIM(x_arr, 0), b = np.PyArray_DIM(layers, 0), i
        complex_t rf
        np.ndarray[complex_t, ndim=1] slc = np.empty(a, dtype=np.complex128)
        double_t *_layers = <double_t *>np.PyArray_DATA(layers)
    for i in prange(a, schedule='guided', nogil=True):
        rf = mll_c(_layers, b, mt1, mt2, x_arr[i], sgm)
        slc[i] = (cos(kdz * rf.real) + 1j * sin(kdz * rf.real)) * exp(-kdz * rf.imag)
    return slc

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

# def pixel_translations(basis, dij, df, z):
#     dij_pix = (basis * dij[:, None]).sum(axis=-1)
#     dij_pix /= (basis**2).sum(axis=-1) * df / z
#     dij_pix -= dij_pix.mean(axis=0)
#     return np.ascontiguousarray(dij_pix[:, 0]), np.ascontiguousarray(dij_pix[:, 1])

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
