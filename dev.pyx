#cython: language_level=3, boundscheck=True, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True, linetrace=True, profile=True
cimport numpy as np
import numpy as np
import cython
# import speckle_tracking as st
from libc.math cimport sqrt, exp, pi, floor, ceil, fabs
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from pyrost.bin cimport pyfftw
from pyrost.bin import pyfftw
from pyrost.bin cimport simulation as sim
cimport openmp

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef fused uint_t:
    np.uint64_t
    np.uint32_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF M_1_SQRT2PI = 0.3989422804014327

ctypedef double (*loss_func)(double a) nogil

cdef double Huber_loss(double a) nogil:
    cdef double aa = fabs(a)
    if aa < 1.345:
        return 0.5 * a * a
    elif 1.345 <= aa < 3.0:
        return 1.345 * (aa - 0.6725)
    else:
        return 3.1304875

cdef double Epsilon_loss(double a) nogil:
    cdef double aa = fabs(a)
    if aa < 0.25:
        return 0.0
    elif 0.25 <= aa < 3.0:
        return aa - 0.25
    else:
        return 2.75

cdef double l2_loss(double a) nogil:
    if -3.0 < a < 3.0:
        return a * a
    else:
        return 9.0

cdef double l1_loss(double a) nogil:
    if -3.0 < a < 3.0:
        return fabs(a)
    else:
        return 3.0

cdef loss_func choose_loss(str loss):
    cdef loss_func f
    if loss == 'Epsilon':
        f = Epsilon_loss
    elif loss == 'Huber':
        f = Huber_loss
    elif loss == 'L2':
        f = l2_loss
    elif loss == 'L1':
        f = l1_loss
    else:
        raise ValueError('loss keyword is invalid')
    return f

cdef float_t min_float(float_t* array, int a) nogil:
    cdef:
        int i
        float_t mv = array[0]
    for i in range(a):
        if array[i] < mv:
            mv = array[i]
    return mv

cdef float_t max_float(float_t* array, int a) nogil:
    cdef:
        int i
        float_t mv = array[0]
    for i in range(a):
        if array[i] > mv:
            mv = array[i]
    return mv

cdef double rbf(double dsq, double h) nogil:
    return exp(-0.5 * dsq / (h * h)) * M_1_SQRT2PI

cdef double FVU_interp(uint_t[:, :, ::1] I_n, float_t W, float_t[:, ::1] I0, float_t[::1] di, float_t[::1] dj, int j, int k,
                       float_t ux, float_t uy, double ds_y, double ds_x, double sigma, loss_func f) nogil:
    """Return fraction of variance unexplained between the validation set I and trained
    profile I0. Find the predicted values at the points (y, x) with bilinear interpolation.
    """
    cdef int N = I_n.shape[0], Y0 = I0.shape[0], X0 = I0.shape[1]
    cdef int i, y0, y1, x0, x1
    cdef double y, x, dy, dx, I0_bi, err = 0.0

    for i in range(N):
        y = (ux - di[i]) / ds_y
        x = (uy - dj[i]) / ds_x

        if y <= 0.0:
            dy = 0.0; y0 = 0; y1 = 0
        elif y >= Y0 - 1.0:
            dy = 0.0; y0 = Y0 - 1; y1 = Y0 - 1
        else:
            dy = y - floor(y)
            y0 = <int>floor(y); y1 = y0 + 1

        if x <= 0.0:
            dx = 0.0; x0 = 0; x1 = 0
        elif x >= X0 - 1.0:
            dx = 0.0; x0 = X0 - 1; x1 = X0 - 1
        else:
            dx = x - floor(x)
            x0 = <int>floor(x); x1 = x0 + 1

        I0_bi = (1.0 - dy) * (1.0 - dx) * I0[y0, x0] + \
                (1.0 - dy) * dx * I0[y0, x1] + \
                dy * (1.0 - dx) * I0[y1, x0] + \
                dy * dx * I0[y1, x1]
        err += f((<double>I_n[i, j, k] - W * I0_bi) / sigma)
    
    return err / N

cdef void pm_gsearcher(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, float_t[:, :, ::1] u,
                       float_t[:, ::1] derrs, float_t[::1] di, float_t[::1] dj, int j, int k, double sw_y, double sw_x,
                       unsigned wsize, double ds_y, double ds_x, double sigma, loss_func f) nogil:
    cdef double err, err0, uy_min = 0.0, ux_min = 0.0, err_min=FLOAT_MAX, ux, uy 
    cdef double dsw_y = 2.0 * sw_y / (wsize - 1), dsw_x = 2.0 * sw_x / (wsize - 1)
    cdef int ii, jj

    err0 = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k],
                      u[1, j, k], ds_y, ds_x, sigma, f)

    for ii in range(<int>wsize if dsw_y > 0.0 else 1):
        uy = dsw_y * (ii - 0.5 * (wsize - 1))
        for jj in range(<int>wsize if dsw_x > 0.0 else 1):
            ux = dsw_x * (jj - 0.5 * (wsize - 1))
            err = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k] + uy,
                             u[1, j, k] + ux, ds_y, ds_x, sigma, f)

            if err < err_min:
                uy_min = uy; ux_min = ux; err_min = err

    u[0, j, k] += uy_min; u[1, j, k] += ux_min
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_gsearch(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, ::1] I0 not None,
               float_t[:, :, ::1] u0 not None, float_t[::1] di not None, float_t[::1] dj not None,
               double sw_y, double sw_x, unsigned grid_size, double ds_y, double ds_x, double sigma,
               str loss='Huber', unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error
    (MSE). Perform a grid search within the search window of `sw_y`,
    `sw_x` size along the vertical and fast axes accordingly in order to
    minimize the MSE at each point of the detector grid separately.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        I0 (numpy.ndarray) : Reference image of the sample.
        u (numpy.ndarray) : The discrete geometrical mapping of the detector
            plane to the reference image.
        di (numpy.ndarray) : Initial sample's translations along the vertical
            detector axis in pixels.
        dj (numpy.ndarray) : Initial sample's translations along the horizontal
            detector axis in pixels.
        sw_y (float) : Search window size in pixels along the vertical detector
            axis.
        sw_x (float) : Search window size in pixels along the horizontal detector
            axis.
        grid_size (int) :  Grid size along one of the detector axes. The grid
            shape is then (grid_size, grid_size).
        ds_y (float) : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x (float) : Sampling interval of reference image in pixels along the
            horizontal axis.
        sigma (float) : The standard deviation of `I_n`.
        loss (str) : Choose between the following loss functions:

            * 'Epsilon': Epsilon loss function (epsilon = 0.5)
            * 'Huber' : Huber loss function (k = 1.345)
            * 'L1' : L1 norm loss function.
            * 'L2' : L2 norm loss function.

        num_threads (int) : Number of threads.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray] : A tuple of two elements ('u', 'derr').
        The elements are the following:

        * 'u' : Updated pixel mapping array.
        * 'derr' : Error decrease for each pixel in the detector grid.

    Notes:
        The error metric as a function of pixel mapping displacements
        is given by:

        .. math::

            \varepsilon_{pm}[i, j, i^{\prime}, j^{\prime}] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] + i^{\prime} - di[n],
            u[1, i, j] + j^{\prime} - dj[n]]}{\sigma} \right)

        where :math:`f(x)` is L1 norm, L2 norm or Huber loss function.
    """
    if ds_y <= 0.0 or ds_x <= 0.0:
        raise ValueError('Sampling intervals must be positive')

    cdef loss_func f = choose_loss(loss)

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k

    cdef np.npy_intp *u_shape = [2, Y, X]
    cdef np.ndarray u = np.PyArray_SimpleNew(3, u_shape, type_num)
    cdef np.ndarray derr = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef float_t[:, :, ::1] _u = u
    cdef float_t[:, ::1] _derr = derr

    for k in prange(X, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(Y):
            _u[0, j, k] = u0[0, j, k]; _u[1, j, k] = u0[1, j, k]
            if W[j, k] > 0.0:
                pm_gsearcher(I_n, W, I0, _u, _derr, di, dj, j, k, sw_y, sw_x,
                             grid_size, ds_y, ds_x, sigma, f)

    return u, derr

# def st_update(I_n, dij, basis, x_ps, y_ps, z, df, search_window, n_iter=5,
#               filter=None, update_translations=False, verbose=False):
#     """
#     Andrew's speckle tracking update algorithm
    
#     I_n - measured data
#     W - whitefield
#     basis - detector plane basis vectors
#     x_ps, y_ps - x and y pixel sizes
#     z - distance between the sample and the detector
#     df - defocus distance
#     wl - wavelength
#     sw_max - pixel mapping search window size
#     n_iter - number of iterations
#     """
#     M = np.ones((I_n.shape[1], I_n.shape[2]), dtype=bool)
#     W = st.make_whitefield(I_n, M, verbose=verbose)
#     u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis,
#                                             x_ps, y_ps, z,
#                                             df, verbose=verbose)
#     I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

#     es = []
#     for i in range(n_iter):

#         # calculate errors
#         error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=verbose)[0]

#         # store total error
#         es.append(error_total)

#         # update pixel map
#         u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
#                                 search_window=search_window, subpixel=True,
#                                 fill_bad_pix=False, integrate=False,
#                                 quadratic_refinement=False, verbose=verbose,
#                                 filter=filter)[0]

#         # make reference image
#         I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

#         # update translations
#         if update_translations:
#             dij_pix = st.update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]

#     return {'u':u, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}
