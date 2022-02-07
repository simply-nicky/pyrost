#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True, linetrace=False, profile=False
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

cdef double Huber_loss(double a) nogil:
    cdef double aa = fabs(a)
    if aa < 1.345:
        return a * a
    else:
        return 2.69 * aa - 1.809025

cdef double Berthe_loss(double a) nogil:
    cdef double aa = fabs(a)
    if aa < 1.345:
        return aa
    else:
        return 0.3717472118959108 * (aa * aa + 1.809025)

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

cdef void KR_frame_1d(float_t[:, ::1] I0, float_t[:, ::1] w0, uint_t[:, ::1] I_n,
                      float_t[:, ::1] W, float_t[:, :, ::1] u, float_t dj,
                      double ds_x, double h) nogil:
    cdef int X = I_n.shape[1], X0 = I0.shape[1], k, kk, k0, kk0, kk1
    cdef int dn = <int>ceil((4.0 * h) / ds_x)
    cdef double x, r

    for k in range(X):
        if W[0, k]:
            x = u[1, 0, k] - dj
            k0 = <int>(x / ds_x) + 1

            kk0 = k0 - dn if k0 - dn > 0 else 0
            kk1 = k0 + dn if k0 + dn < X0 else X0

            for kk in range(kk0, kk1):
                if W[0, kk]:
                    r = rbf((ds_x * kk - x) * (ds_x * kk - x), h)
                    I0[0, kk] += I_n[0, k] * W[0, k] * r
                    w0[0, kk] += W[0, k] * W[0, k] * r

cdef void KR_frame_2d(float_t[:, ::1] I0, float_t[:, ::1] w0, uint_t[:, ::1] I_n,
                      float_t[:, ::1] W, float_t[:, :, ::1] u, float_t di, float_t dj,
                      double ds_y, double ds_x, double h) nogil:
    cdef int Y = I_n.shape[0], X = I_n.shape[1], Y0 = I0.shape[0], X0 = I0.shape[1]
    cdef int j, k, jj, kk, j0, k0, jj0, jj1, kk0, kk1
    cdef int dn_y = <int>ceil((4.0 * h) / ds_y), dn_x = <int>ceil((4.0 * h) / ds_x)
    cdef double y, x, r

    for j in range(Y):
        for k in range(X):
            if W[j, k]:
                y = u[0, j, k] - di
                x = u[1, j, k] - dj
                j0 = <int>(y / ds_y) + 1
                k0 = <int>(x / ds_x) + 1

                jj0 = j0 - dn_y if j0 - dn_y > 0 else 0
                jj1 = j0 + dn_y if j0 + dn_y < Y0 else Y0
                kk0 = k0 - dn_x if k0 - dn_x > 0 else 0
                kk1 = k0 + dn_x if k0 + dn_x < X0 else X0

                for jj in range(jj0, jj1):
                    for kk in range(kk0, kk1):
                        if W[jj, kk]:
                            r = rbf((ds_y * jj - y) * (ds_y * jj - y) + (ds_x * kk - x) * (ds_x * kk - x), h)
                            I0[jj, kk] += I_n[j, k] * W[j, k] * r
                            w0[jj, kk] += W[j, k] * W[j, k] * r

def KR_reference(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, :, ::1] u not None,
                 float_t[::1] di not None, float_t[::1] dj not None, double ds_y, double ds_x, double h,
                 bint return_nm0=True, unsigned num_threads=1):
    r"""Generate an unabberated reference image of the sample based on the pixel
    mapping `u` and the measured data `I_n` using the Kernel regression.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        u (numpy.ndarray) : The discrete geometrical mapping of the detector
            plane to the reference image.
        di (numpy.ndarray) : Initial sample's translations along the vertical
            detector axis in pixels.
        dj (numpy.ndarray) : Initial sample's translations along the horizontal
            detector axis in pixels.
        ds_y (float) : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x (float) : Sampling interval of reference image in pixels along the
            horizontal axis.
        h (float) : Gaussian kernel bandwidth in pixels.
        return_nm0 (bool) : If True, also returns the lower bounds (`n0`, `m0`)
            of the reference image in pixels.
        num_threads (int) : Number of threads.

    Returns:
        Tuple[numpy.ndarray, int, int] : A tuple of three elements ('I0', 'n0',
        'm0'). The elements are the following:

        * 'I0' : Reference image array.
        * 'n0' : The lower bounds of the vertical detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.
        * 'm0' : The lower bounds of the horizontal detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.

    Notes:
        The pixel mapping `u` maps the intensity measurements from the detector
        plane to the reference plane as follows:

        .. math::
            i_{ref}[n, i, j] = u[0, i, j] + di[n], \; j_{ref}[n, i, j] = u[1, i, j] + dj[n]

        The reference image profile :math:`I_{ref}[ii, jj]` is obtained with the
        kernel regression extimator as follows:

        .. math::

            I_{ref}[i, j] = \frac{\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n,
            i^{\prime}, j^{\prime}], j - j_{ref}[n, i^{\prime}, j^{\prime}], h]
            \; W[i^{\prime}, j^{\prime}] I[n, i^{\prime}, j^{\prime}]}
            {\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n, i^{\prime}, j^{\prime}], 
            j - j_{ref}[n, i^{\prime}, j^{\prime}], h] \; W^2[i^{\prime}, j^{\prime}]}

        where :math:`K[i, j, h] = \frac{1}{\sqrt{2 \pi}} \exp(-\frac{i^2 + j^2}{h})`
        is the Gaussian kernel.
    """
    if ds_y <= 0.0 or ds_x <= 0.0:
        raise ValueError('Sampling intervals must be positive')

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int N = I_n.shape[0], Y = I_n.shape[1], X = I_n.shape[2]
    cdef int i, j, t, k

    cdef float_t n0 = -min_float(&u[0, 0, 0], Y * X) + max_float(&di[0], N)
    cdef float_t m0 = -min_float(&u[1, 0, 0], Y * X) + max_float(&dj[0], N)
    cdef int Y0 = <int>((max_float(&u[0, 0, 0], Y * X) - min_float(&di[0], N) + n0) / ds_y) + 1
    cdef int X0 = <int>((max_float(&u[1, 0, 0], Y * X) - min_float(&dj[0], N) + m0) / ds_x) + 1
        
    cdef np.npy_intp *shape = [num_threads, Y0, X0]
    cdef float_t[:, :, ::1] I0_buf = np.PyArray_ZEROS(3, shape, type_num, 0)
    cdef float_t[:, :, ::1] W0_buf = np.PyArray_ZEROS(3, shape, type_num, 0)
    cdef np.ndarray I0 = np.PyArray_SimpleNew(2, shape + 1, type_num)

    if Y0 > 1:
        for i in prange(N, schedule='guided', num_threads=num_threads, nogil=True):
            t = openmp.omp_get_thread_num()
            KR_frame_2d(I0_buf[t], W0_buf[t], I_n[i], W, u, di[i] - n0, dj[i] - m0, ds_y, ds_x, h)
    else:
        for i in prange(N, schedule='guided', num_threads=num_threads, nogil=True):
            t = openmp.omp_get_thread_num()
            KR_frame_1d(I0_buf[t], W0_buf[t], I_n[i], W, u, dj[i] - m0, ds_x, h)

    cdef float_t[:, ::1] _I0 = I0
    cdef float_t I0_sum, W0_sum
    for k in prange(X0, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(Y0):
            I0_sum = 0.0; W0_sum = 0.0
            for i in range(<int>num_threads):
                I0_sum = I0_sum + I0_buf[i, j, k]
                W0_sum = W0_sum + W0_buf[i, j, k]

            _I0[j, k] = I0_sum / W0_sum if W0_sum > 0.0 else 1.0

    if return_nm0:
        return I0, <int>n0, <int>m0
    else:
        return I0

cdef double pm_loss(uint_t[:, :, ::1] I_n, float_t W, float_t sigma, float_t[:, ::1] I0,
                    float_t[::1] di, float_t[::1] dj, int j, int k, float_t u_y, float_t u_x,
                    double ds_y, double ds_x) nogil:
    """Return fraction of variance unexplained between the validation set I and trained
    profile I0. Find the predicted values at the points (y, x) with bilinear interpolation.
    """
    cdef int N = I_n.shape[0], Y0 = I0.shape[0], X0 = I0.shape[1]
    cdef int i, y0, y1, x0, x1
    cdef double y, x, dy, dx, I0_bi, err = 0.0

    for i in range(N):
        y = (u_y - di[i]) / ds_y
        x = (u_x - dj[i]) / ds_x

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
        err += sigma + Huber_loss((<double>I_n[i, j, k] - W * I0_bi) / sigma) * sigma
    
    return err / N

def pm_errors(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, ::1] I0 not None,
              float_t[:, ::1] sigma not None, float_t[:, :, ::1] u not None, float_t[::1] di not None,
              float_t[::1] dj not None, double ds_y, double ds_x, unsigned num_threads=1) -> np.ndarray:
    r"""Return the residuals for the pixel mapping fit.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        I0 (numpy.ndarray) : Reference image of the sample.
        sigma (numpy.ndarray) : The standard deviation of `I_n`.
        u (numpy.ndarray) : The discrete geometrical mapping of the detector
            plane to the reference image.
        di (numpy.ndarray) : Sample's translations along the vertical detector
            axis in pixels.
        dj (numpy.ndarray) : Sample's translations along the fast detector
            axis in pixels.
        ds_y (float) : Sampling interval of reference image in pixels along
            the vertical axis.
        ds_x (float) : Sampling interval of reference image in pixels along
            the horizontal axis.
        num_threads (int): Number of threads.

    Returns:
        numpy.ndarray : Residual profile of the pixel mapping fit.

    See Also:
        :func:`pyrost.bin.pm_gsearch` : Description of error metric which
        is being minimized.

    Notes:
        The error metric is given by:

        .. math::

            \varepsilon_{pm}[i, j] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] - di[n], u[1, i, j] - dj[n]]}{\sigma}
            \right)

        where :math:`f(x)` is L1 norm, L2 norm or Huber loss function.
    """
    if ds_y <= 0.0 or ds_x <= 0.0:
        raise ValueError('Sampling intervals must be positive')

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k

    cdef np.ndarray errs = np.PyArray_SimpleNew(2, <np.npy_intp *>I_n.shape + 1, type_num)
    cdef float_t[:, ::1] _errs = errs

    for k in prange(X, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(Y):
            if W[j, k]:
                _errs[j, k] = pm_loss(I_n, W[j, k], sigma[j, k], I0, di, dj, j, k,
                                      u[0, j, k], u[1, j, k], ds_y, ds_x)
            else:
                _errs[j, k] = 0.0

    return errs

cdef void pm_gsearcher(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, float_t[:, ::1] sigma,
                       float_t[:, :, ::1] u, float_t[:, ::1] derrs, float_t[::1] di, float_t[::1] dj, int j, int k,
                       double[::1] sw, int[::1] gsize, double ds_y, double ds_x) nogil:
    cdef int ii, jj, kk
    cdef double err, err0, err_min=FLOAT_MAX
    cdef double uy_min = 0.0, ux_min = 0.0, sgm_min = sigma[j, k], ux, uy, cs
    cdef double dsw_y = 2.0 * sw[0] / (gsize[0] - 1) if gsize[0] > 1 else 0.0
    cdef double dsw_x = 2.0 * sw[1] / (gsize[1] - 1) if gsize[1] > 1 else 0.0
    cdef double dsw_s = 2.0 * sw[2] / (gsize[2] - 1) if gsize[2] > 1 else 0.0

    err0 = pm_loss(I_n, W[j, k], sigma[j, k], I0, di, dj, j, k, u[0, j, k],
                   u[1, j, k], ds_y, ds_x)

    for ii in range(gsize[0]):
        uy = dsw_y * (ii - 0.5 * (gsize[0] - 1))
        for jj in range(gsize[1]):
            ux = dsw_x * (jj - 0.5 * (gsize[1] - 1))
            for kk in range(gsize[2]):
                cs = 1.0 + dsw_s * (kk - 0.5 * (gsize[2] - 1))
                err = pm_loss(I_n, W[j, k], cs * sigma[j, k], I0, di, dj, j, k, u[0, j, k] + uy,
                              u[1, j, k] + ux, ds_y, ds_x)

                if err < err_min:
                    uy_min = uy; ux_min = ux; sgm_min = cs * sigma[j, k]; err_min = err

    u[0, j, k] += uy_min; u[1, j, k] += ux_min; sigma[j, k] = sgm_min
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_gsearch(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, ::1] I0 not None,
               float_t[:, ::1] sigma not None, float_t[:, :, ::1] u0 not None, float_t[::1] di not None,
               float_t[::1] dj not None, object search_window not None, object grid_size not None,
               double ds_y, double ds_x, unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error
    (MSE). Perform a grid search within the search window of `sw_y`,
    `sw_x` size along the vertical and fast axes accordingly in order to
    minimize the MSE at each point of the detector grid separately.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        I0 (numpy.ndarray) : Reference image of the sample.
        sigma (float) : The standard deviation of `I_n`.
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

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k
    cdef np.ndarray sw = sim.normalize_sequence(search_window, 3, np.NPY_FLOAT64)
    cdef np.ndarray gsize = sim.normalize_sequence(grid_size, 3, np.NPY_INT32)
    cdef double[::1] _sw = sw
    cdef int[::1] _gsize = gsize

    cdef np.npy_intp *u_shape = [2, Y, X]
    cdef np.ndarray u = np.PyArray_SimpleNew(3, u_shape, type_num)
    cdef np.ndarray derr = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef np.ndarray sgm = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef float_t[:, :, ::1] _u = u
    cdef float_t[:, ::1] _sgm = sgm
    cdef float_t[:, ::1] _derr = derr

    for k in prange(X, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(Y):
            _u[0, j, k] = u0[0, j, k]; _u[1, j, k] = u0[1, j, k]; _sgm[j, k] = sigma[j, k]
            if W[j, k] > 0.0:
                pm_gsearcher(I_n, W, I0, _sgm, _u, _derr, di, dj, j, k,
                             _sw, _gsize, ds_y, ds_x)

    return u, sgm, derr

cdef void pm_rsearcher(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] sigma,
                       float_t[:, ::1] I0, gsl_rng *r, float_t[:, :, ::1] u, float_t[:, ::1] derrs,
                       float_t[::1] di, float_t[::1] dj, int j, int k, double[::1] sw,
                       unsigned N, double ds_y, double ds_x) nogil:
    cdef double err, err0, err_min=FLOAT_MAX
    cdef double uy_min = 0.0, ux_min = 0.0, sig_min = sigma[j, k], ux, uy, cs
    cdef int ii

    err0 = pm_loss(I_n, W[j, k], sigma[j, k], I0, di, dj, j, k,
                   u[0, j, k], u[1, j, k], ds_y, ds_x)

    for ii in range(<int>N):
        uy = 2.0 * sw[0] * (gsl_rng_uniform(r) - 0.5)
        ux = 2.0 * sw[1] * (gsl_rng_uniform(r) - 0.5)
        cs = 1.0 + 2.0 * sw[2] * (gsl_rng_uniform(r) - 0.5)

        err = pm_loss(I_n, W[j, k], cs * sigma[j, k], I0, di, dj, j, k,
                      u[0, j, k] + uy, u[1, j, k] + ux, ds_y, ds_x)
        if err < err_min:
            uy_min = uy; ux_min = ux; sig_min = cs * sigma[j, k]; err_min = err

    u[0, j, k] += uy_min; u[1, j, k] += ux_min; sigma[j, k] = sig_min
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_rsearch(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None,
               float_t[:, ::1] sigma not None, float_t[:, ::1] I0 not None,
               float_t[:, :, ::1] u0 not None, float_t[::1] di not None,
               float_t[::1] dj not None, object search_window not None,
               unsigned n_trials, unsigned long seed, double ds_y, double ds_x,
               unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error (MSE).
    Perform a random search within the search window of `sw_y`, `sw_x` size
    along the vertical and fast axes accordingly in order to minimize the MSE
    at each point of the detector grid separately.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        sigma (numpy.ndarray) : The standard deviation of `I_n`.
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
        n_trials (int) : Number of points generated at each pixel of the detector grid.
        seed (int) : Specify seed for the random number generation.
        ds_y (float) : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x (float) : Sampling interval of reference image in pixels along the
            horizontal axis.
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
    if n_trials <= 1:
        raise ValueError('n_tirals must be more than 1')

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k
    cdef np.ndarray sw = sim.normalize_sequence(search_window, 3, np.NPY_FLOAT64)
    cdef double[::1] _sw = sw

    cdef np.npy_intp *u_shape = [2, Y, X]
    cdef np.ndarray u = np.PyArray_SimpleNew(3, u_shape, type_num)
    cdef np.ndarray derr = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef np.ndarray sgm = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef float_t[:, :, ::1] _u = u
    cdef float_t[:, ::1] _derr = derr
    cdef float_t[:, ::1] _sgm = sgm

    cdef gsl_rng *r_master = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(r_master, seed)
    cdef unsigned long thread_seed
    cdef gsl_rng *r

    with nogil, parallel(num_threads=num_threads):
        r = gsl_rng_alloc(gsl_rng_mt19937)
        thread_seed = gsl_rng_get(r_master)
        gsl_rng_set(r, thread_seed)

        for k in prange(X, schedule='guided'):
            for j in range(Y):
                _u[0, j, k] = u0[0, j, k]; _u[1, j, k] = u0[1, j, k]
                _sgm[j, k] = sigma[j, k]
                if W[j, k] > 0.0:
                    pm_rsearcher(I_n, W, _sgm, I0, r, _u, _derr, di, dj, j, k,
                                 _sw, n_trials, ds_y, ds_x)

        gsl_rng_free(r)

    gsl_rng_free(r_master)

    return u, sgm, derr

cdef void pm_devolver(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] sigma,
                      float_t[:, ::1] I0, gsl_rng *r, float_t[:, :, ::1] u,
                      float_t[:, ::1] derrs, float_t[::1] di, float_t[::1] dj,
                      int j, int k, double[::1] sw, unsigned NP, unsigned n_iter,
                      double CR, double F, double ds_y, double ds_x) nogil:
    cdef double err0, err, err_min = FLOAT_MAX
    cdef int ii, jj, n, a, b
    cdef double pt_min[3]
    cdef double *pop = <double *>malloc(3 * NP * sizeof(double))
    cdef double *cost = <double *>malloc(NP * sizeof(double))
    cdef double *new_pop = <double *>malloc(3 * NP * sizeof(double))

    err0 = pm_loss(I_n, W[j, k], sigma[j, k], I0, di, dj, j, k,
                   u[0, j, k], u[1, j, k], ds_y, ds_x)

    for ii in range(<int>NP):
        pop[3 * ii] = 2.0 * sw[0] * (gsl_rng_uniform(r) - 0.5)
        pop[3 * ii + 1] = 2.0 * sw[1] * (gsl_rng_uniform(r) - 0.5)
        pop[3 * ii + 2] = 1.0 + 2.0 * sw[2] * (gsl_rng_uniform(r) - 0.5)

        cost[ii] = pm_loss(I_n, W[j, k], pop[3 * ii + 2] * sigma[j, k], I0, di, dj, j, k,
                           u[0, j, k] + pop[3 * ii], u[1, j, k] + pop[3 * ii + 1],
                           ds_y, ds_x)
        
        if cost[ii] < err_min:
            pt_min[0] = pop[3 * ii]; pt_min[1] = pop[3 * ii + 1]; pt_min[2] = pop[3 * ii + 2]
            err_min = cost[ii]

    for n in range(<int>n_iter):
        for ii in range(<int>NP):
            a = gsl_rng_uniform_int(r, NP)
            while a == ii:
                a = gsl_rng_uniform_int(r, NP)
            
            b = gsl_rng_uniform_int(r, NP)
            while b == ii or b == a:
                b = gsl_rng_uniform_int(r, NP)

            jj = gsl_rng_uniform_int(r, 3)
            if gsl_rng_uniform(r) < CR:
                new_pop[3 * ii + jj] = pt_min[jj] + F * (pop[3 * a + jj] - pop[3 * b + jj])
                if new_pop[3 * ii + jj] > sw[jj]: new_pop[3 * ii + jj] = sw[jj]
                if new_pop[3 * ii + jj] < -sw[jj]: new_pop[3 * ii + jj] = -sw[jj]
            else:
                new_pop[3 * ii + jj] = pop[3 * ii + jj]

            jj = (jj + 1) % 3
            if gsl_rng_uniform(r) < CR:
                new_pop[3 * ii + jj] = pt_min[jj] + F * (pop[3 * a + jj] - pop[3 * b + jj])
                if new_pop[3 * ii + jj] > sw[jj]: new_pop[3 * ii + jj] = sw[jj]
                if new_pop[3 * ii + jj] < -sw[jj]: new_pop[3 * ii + jj] = -sw[jj]
            else:
                new_pop[3 * ii + jj] = pop[3 * ii + jj]

            jj = (jj + 1) % 3
            new_pop[3 * ii + jj] = pt_min[jj] + F * (pop[3 * a + jj] - pop[3 * b + jj])
            if new_pop[3 * ii + jj] > sw[jj]: new_pop[3 * ii + jj] = sw[jj]
            if new_pop[3 * ii + jj] < -sw[jj]: new_pop[3 * ii + jj] = -sw[jj]

            err = pm_loss(I_n, W[j, k], new_pop[3 * ii + 2] * sigma[j, k], I0, di, dj, j, k,
                          u[0, j, k] + new_pop[3 * ii], u[1, j, k] + new_pop[3 * ii + 1],
                          ds_y, ds_x)

            if err < cost[ii]:
                cost[ii] = err
                if err < err_min:
                    pt_min[0] = new_pop[3 * ii]
                    pt_min[1] = new_pop[3 * ii + 1]
                    pt_min[2] = new_pop[3 * ii + 2]
                    err_min = err
            else:
                new_pop[3 * ii] = pop[3 * ii]
                new_pop[3 * ii + 1] = pop[3 * ii + 1]
                new_pop[3 * ii + 2] = pop[3 * ii + 2]
            
        for ii in range(3 * <int>NP):
            pop[ii] = new_pop[ii]

    free(pop); free(new_pop); free(cost)

    u[0, j, k] += pt_min[0]; u[1, j, k] += pt_min[1]; sigma[j, k] = pt_min[2] * sigma[j, k]
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_devolution(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None,
                  float_t[:, ::1] sigma not None, float_t[:, ::1] I0 not None,
                  float_t[:, :, ::1] u0 not None, float_t[::1] di not None,
                  float_t[::1] dj not None, object search_window not None,
                  unsigned pop_size, unsigned n_iter, unsigned long seed, double ds_y,
                  double ds_x, double F=0.75, double CR=0.7, unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error (MSE). Perform
    a differential evolution within the search window of `sw_y`, `sw_x` size along
    the vertical and fast axes accordingly in order to minimize the MSE at each
    point of the detector grid separately.

    Args:
        I_n (numpy.ndarray) : Measured intensity frames.
        W (numpy.ndarray) : Measured frames' whitefield.
        sigma (numpy.ndarray) : The standard deviation of `I_n`.
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
        pop_size (int) : The total population size. Must be greater or equal to 4.
        n_iter (int) : The maximum number of generations over which the entire
            population is evolved.
        seed (int) : Specify seed for the random number generation.
        ds_y (float) : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x (float) : Sampling interval of reference image in pixels along the
            horizontal axis.
        F (float) : The mutation constant. In the literature this is also known as
            differential weight. If specified as a float it should be in the
            range [0, 2].
        CR (float) : The recombination constant, should be in the range [0, 1]. In
            the literature this is also known as the crossover probability.
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
    if pop_size < 4:
        raise ValueError('Population size must be greater or equal to 4.')
    if F < 0.0 or F > 2.0:
        raise ValueError('The mutation constant F must be in the interval [0.0, 2.0].')
    if CR < 0.0 or CR > 1.0:
        raise ValueError('The recombination constant CR must be in the interval [0.0, 1.0].')

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k
    cdef np.ndarray sw = sim.normalize_sequence(search_window, 3, np.NPY_FLOAT64)
    cdef double[::1] _sw = sw

    cdef np.npy_intp *u_shape = [2, Y, X]
    cdef np.ndarray u = np.PyArray_SimpleNew(3, u_shape, type_num)
    cdef np.ndarray derr = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef np.ndarray sgm = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef float_t[:, :, ::1] _u = u
    cdef float_t[:, ::1] _derr = derr
    cdef float_t[:, ::1] _sgm = sgm

    cdef gsl_rng *r_master = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(r_master, seed)
    cdef unsigned long thread_seed
    cdef gsl_rng *r

    with nogil, parallel(num_threads=num_threads):
        r = gsl_rng_alloc(gsl_rng_mt19937)
        thread_seed = gsl_rng_get(r_master)
        gsl_rng_set(r, thread_seed)

        for k in prange(X, schedule='guided'):
            for j in range(Y):
                _u[0, j, k] = u0[0, j, k]; _u[1, j, k] = u0[1, j, k]
                _sgm[j, k] = sigma[j, k]
                if W[j, k] > 0.0:
                    pm_devolver(I_n, W, _sgm, I0, r, _u, _derr, di, dj, j, k,
                                _sw, pop_size, n_iter, CR, F, ds_y, ds_x)

        gsl_rng_free(r)

    gsl_rng_free(r_master)

    return u, sgm, derr

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
