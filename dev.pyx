#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
import speckle_tracking as st
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

    Parameters
    ----------
    I_n : numpy.ndarray
        Measured intensity frames.
    W : numpy.ndarray
        Measured frames' whitefield.
    I0 : numpy.ndarray
        Reference image of the sample.
    u0 : numpy.ndarray
        Initial pixel mapping.
    di : numpy.ndarray
        Sample's translations along the vertical detector axis
        in pixels.
    dj : numpy.ndarray
        Sample's translations along the fast detector axis
        in pixels.
    sw_y : int
        Search window size in pixels along the vertical detector
        axis.
    sw_x : int
        Search window size in pixels along the fast detector
        axis.
    ds_y : float
        Sampling interval of reference image in pixels along the vertical axis.
    ds_x : float
        Sampling interval of reference image in pixels along the horizontal axis.
    sigma : float
        The standard deviation of :code:`I_n`.
    loss : {'Epsilon', 'Huber', 'L1', 'L2'}, optional
        Choose between the following loss functions:

        * 'Epsilon': Epsilon loss function (epsilon = 0.5)
        * 'Huber' : Huber loss function (k = 1.345)
        * 'L1' : L1 norm loss function.
        * 'L2' : L2 norm loss function.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    u : numpy.ndarray
        Updated pixel mapping array.
    derr : numpy.ndarray
        Error decrease for each pixel in the detector grid.
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

cdef void pm_rsearcher(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, gsl_rng *r, float_t[:, :, ::1] u,
                       float_t[:, ::1] derrs, float_t[::1] di, float_t[::1] dj, int j, int k, double sw_y, double sw_x,
                       unsigned N, double ds_y, double ds_x, double sigma, loss_func f) nogil:
    cdef double err, err0, err_min=FLOAT_MAX, uy_min = 0.0, ux_min = 0.0, ux, uy
    cdef int ii

    err0 = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k],
                      u[1, j, k], ds_y, ds_x, sigma, f)

    for ii in range(<int>N):
        uy = 2.0 * sw_y * (gsl_rng_uniform(r) - 0.5)
        ux = 2.0 * sw_x * (gsl_rng_uniform(r) - 0.5)

        err = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k] + uy,
                         u[1, j, k] + ux, ds_y, ds_x, sigma, f)
        if err < err_min:
            uy_min = uy; ux_min = ux; err_min = err

    u[0, j, k] += uy_min; u[1, j, k] += ux_min
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_rsearch(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, ::1] I0 not None,
               float_t[:, :, ::1] u0 not None, float_t[::1] di not None, float_t[::1] dj not None,
               double sw_y, double sw_x, unsigned n_trials, unsigned long seed, double ds_y, double ds_x, double sigma,
               str loss='Huber', unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error
    (MSE). Perform a random search within the search window of `sw_y`,
    `sw_x` size along the vertical and fast axes accordingly in order to
    minimize the MSE at each point of the detector grid separately.

    Parameters
    ----------
    I_n : numpy.ndarray
        Measured intensity frames.
    W : numpy.ndarray
        Measured frames' whitefield.
    I0 : numpy.ndarray
        Reference image of the sample.
    u0 : numpy.ndarray
        Initial pixel mapping.
    di : numpy.ndarray
        Sample's translations along the vertical detector axis
        in pixels.
    dj : numpy.ndarray
        Sample's translations along the fast detector axis
        in pixels.
    sw_y : int
        Search window size in pixels along the vertical detector
        axis.
    sw_x : int
        Search window size in pixels along the horizontal detector
        axis.
    n_trials : int
        Number of points generated at each pixel of the detector grid.
    seed : int
        Specify seed for the random number generation.
    ds_y : float
        Sampling interval of reference image in pixels along the vertical axis.
    ds_x : float
        Sampling interval of reference image in pixels along the horizontal axis.
    sigma : float
        The standard deviation of :code:`I_n`.
    loss : {'Epsilon', 'Huber', 'L1', 'L2'}, optional
        Choose between the following loss functions:

        * 'Epsilon': Epsilon loss function (epsilon = 0.5)
        * 'Huber' : Huber loss function (k = 1.345)
        * 'L1' : L1 norm loss function.
        * 'L2' : L2 norm loss function.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    u : numpy.ndarray
        Updated pixel mapping array.
    derr : numpy.ndarray
        Error decrease for each pixel in the detector grid.
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
                if W[j, k] > 0.0:
                    pm_rsearcher(I_n, W, I0, r, _u, _derr, di, dj, j, k, sw_y, sw_x,
                                 n_trials, ds_y, ds_x, sigma, f)

        gsl_rng_free(r)

    gsl_rng_free(r_master)

    return u, derr

cdef void pm_devolver(uint_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, gsl_rng *r, float_t[:, :, ::1] u,
                      float_t[:, ::1] derrs, float_t[::1] di, float_t[::1] dj, int j, int k, double sw_y, double sw_x,
                      unsigned NP, unsigned n_iter, double CR, double F, double ds_y, double ds_x, double sigma, loss_func f) nogil:
    cdef double err0, err, err_min = FLOAT_MAX
    cdef int ii, jj, n, a, b
    cdef double u_min[2]
    cdef double sw[2]
    cdef double *pop = <double *>malloc(2 * NP * sizeof(double))
    cdef double *cost = <double *>malloc(NP * sizeof(double))
    cdef double *new_pop = <double *>malloc(2 * NP * sizeof(double))

    sw[0] = sw_y; sw[1] = sw_x
    err0 = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k],
                      u[1, j, k], ds_y, ds_x, sigma, f)

    for ii in range(<int>NP):
        pop[2 * ii] = 2.0 * sw_y * (gsl_rng_uniform(r) - 0.5)
        pop[2 * ii + 1] = 2.0 * sw_x * (gsl_rng_uniform(r) - 0.5)

        cost[ii] = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k] + pop[2 * ii],
                             u[1, j, k] + pop[2 * ii + 1], ds_y, ds_x, sigma, f)
        
        if cost[ii] < err_min:
            u_min[0] = pop[2 * ii]; u_min[1] = pop[2 * ii + 1]; err_min = cost[ii]

    for n in range(<int>n_iter):
        for ii in range(<int>NP):
            a = gsl_rng_uniform_int(r, NP)
            while a == ii:
                a = gsl_rng_uniform_int(r, NP)
            
            b = gsl_rng_uniform_int(r, NP)
            while b == ii or b == a:
                b = gsl_rng_uniform_int(r, NP)

            jj = gsl_rng_uniform_int(r, 2)
            if gsl_rng_uniform(r) < CR:
                new_pop[2 * ii + jj] = u_min[jj] + F * (pop[2 * a + jj] - pop[2 * b + jj])
                if new_pop[2 * ii + jj] > sw[jj]: new_pop[2 * ii + jj] = sw[jj]
                if new_pop[2 * ii + jj] < -sw[jj]: new_pop[2 * ii + jj] = -sw[jj]
            else:
                new_pop[2 * ii + jj] = pop[2 * ii + jj]
            jj = (jj + 1) % 2
            new_pop[2 * ii + jj] = u_min[jj] + F * (pop[2 * a + jj] - pop[2 * b + jj])
            if new_pop[2 * ii + jj] > sw[jj]: new_pop[2 * ii + jj] = sw[jj]
            if new_pop[2 * ii + jj] < -sw[jj]: new_pop[2 * ii + jj] = -sw[jj]

            err = FVU_interp(I_n, W[j, k], I0, di, dj, j, k, u[0, j, k] + new_pop[2 * ii],
                             u[1, j, k] + new_pop[2 * ii + 1], ds_y, ds_x, sigma, f)

            if err < cost[ii]:
                cost[ii] = err
                if err < err_min:
                    u_min[0] = new_pop[2 * ii]; u_min[1] = new_pop[2 * ii + 1]; err_min = err
            else:
                new_pop[2 * ii] = pop[2 * ii]; new_pop[2 * ii + 1] = pop[2 * ii + 1]
            
        for ii in range(2 * <int>NP):
            pop[ii] = new_pop[ii]

    free(pop); free(new_pop); free(cost)

    u[0, j, k] += u_min[0]; u[1, j, k] += u_min[1]
    derrs[j, k] = err0 - err_min if err_min < err0 else 0.0

def pm_devolution(uint_t[:, :, ::1] I_n not None, float_t[:, ::1] W not None, float_t[:, ::1] I0 not None,
                  float_t[:, :, ::1] u0 not None, float_t[::1] di not None, float_t[::1] dj not None,
                  double sw_y, double sw_x, unsigned pop_size, unsigned n_iter, unsigned long seed,
                  double ds_y, double ds_x, double sigma, double F=0.75, double CR=0.7, str loss='Huber',
                  unsigned num_threads=1):
    r"""Update the pixel mapping by minimizing mean-squared-error
    (MSE). Perform a differential evolution within the search window of `sw_y`,
    `sw_x` size along the vertical and fast axes accordingly in order to
    minimize the MSE at each point of the detector grid separately.

    Parameters
    ----------
    I_n : numpy.ndarray
        Measured intensity frames.
    W : numpy.ndarray
        Measured frames' whitefield.
    I0 : numpy.ndarray
        Reference image of the sample.
    u0 : numpy.ndarray
        Initial pixel mapping.
    di : numpy.ndarray
        Sample's translations along the vertical detector axis
        in pixels.
    dj : numpy.ndarray
        Sample's translations along the fast detector axis
        in pixels.
    sw_y : int
        Search window size in pixels along the vertical detector
        axis.
    sw_x : int
        Search window size in pixels along the horizontal detector
        axis.
    pop_size : int
        The total population size. Must be greater or equal to 4.
    n_iter : int
        The maximum number of generations over which the entire population
        is evolved.
    seed : int
        Specify seed for the random number generation.
    ds_y : float
        Sampling interval of reference image in pixels along the vertical axis.
    ds_x : float
        Sampling interval of reference image in pixels along the horizontal axis.
    sigma : float
        The standard deviation of :code:`I_n`.
    F : float, optional
        The mutation constant. In the literature this is also known as
        differential weight. If specified as a float it should be in the
        range [0, 2].
    CR : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability.
    loss : {'Epsilon', 'Huber', 'L1', 'L2'}, optional
        Choose between the following loss functions:

        * 'Epsilon': Epsilon loss function (epsilon = 0.5)
        * 'Huber' : Huber loss function (k = 1.345)
        * 'L1' : L1 norm loss function.
        * 'L2' : L2 norm loss function.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    u : numpy.ndarray
        Updated pixel mapping array.
    derr : numpy.ndarray
        Error decrease for each pixel in the detector grid.
    """
    if ds_y <= 0.0 or ds_x <= 0.0:
        raise ValueError('Sampling intervals must be positive')
    if pop_size < 4:
        raise ValueError('Population size must be greater or equal to 4.')
    if F < 0.0 or F > 2.0:
        raise ValueError('The mutation constant F must be in the interval [0.0, 2.0].')
    if CR < 0.0 or CR > 1.0:
        raise ValueError('The recombination constant CR must be in the interval [0.0, 1.0].')

    cdef loss_func f = choose_loss(loss)

    cdef int type_num = np.PyArray_TYPE(W.base)
    cdef int Y = I_n.shape[1], X = I_n.shape[2], j, k

    cdef np.npy_intp *u_shape = [2, Y, X]
    cdef np.ndarray u = np.PyArray_SimpleNew(3, u_shape, type_num)
    cdef np.ndarray derr = np.PyArray_ZEROS(2, u_shape + 1, type_num, 0)
    cdef float_t[:, :, ::1] _u = u
    cdef float_t[:, ::1] _derr = derr

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
                if W[j, k] > 0.0:
                    pm_devolver(I_n, W, I0, r, _u, _derr, di, dj, j, k, sw_y, sw_x,
                                pop_size, n_iter, CR, F, ds_y, ds_x, sigma, f)

        gsl_rng_free(r)

    gsl_rng_free(r_master)

    return u, derr

def fourier_basis(float_t[::1] x not None, object inp_shape not None, object out_shape not None,
                  int num_threads=1, np.ndarray res=None, pyfftw.FFTW fftw_obj=None) -> np.ndarray:
    """Return a set of fourier functions using the discrete cosine transform.

    Parameters
    ----------
    x : np.ndarray
        Set of fourier coefficient.
    inp_shape : sequence of int
        Shape of fourier coefficients.
    out_shape : sequence of int
        Shape of output image.
    num_threads : int, optional
        Number of threads.
    res : np.ndarray, optional
        The output is stored here if it's provided.
    fftw_obj : FFTW
        FFTW object for fourier computations.
    
    Returns
    -------
    out : np.ndarray
        Output fourier image, duplicates `res` if
        `res` is not None.
    """
    cdef int type_num = np.PyArray_TYPE(x.base)
    cdef np.ndarray[np.int64_t, ndim=1] ishape = sim.normalize_sequence(inp_shape, 2, np.NPY_INT64)
    cdef np.ndarray[np.int64_t, ndim=1] oshape = sim.normalize_sequence(out_shape, 2, np.NPY_INT64)

    if (ishape[0] * ishape[1]) != x.size:
        raise ValueError('x size must be equal to the input shape size')

    cdef np.npy_intp *dims = [oshape[0], oshape[1]]
    cdef np.ndarray[np.complex128_t, ndim=2] inp
    cdef np.ndarray[np.complex128_t, ndim=2] out
    if fftw_obj is None:
        inp = np.PyArray_ZEROS(2, dims, np.NPY_COMPLEX128, 0)
        out = np.PyArray_SimpleNew(2, dims, np.NPY_COMPLEX128)
        fftw_obj = pyfftw.FFTW(inp, out, axes=(0, 1), threads=num_threads)
    else:
        inp = fftw_obj._input_array
        out = fftw_obj._output_array
        np.PyArray_FILLWBYTE(inp, 0)

    cdef int i, j, ii, jj
    cdef int Y = <int>ishape[0], X = <int>ishape[1]
    cdef int Y0 = Y // 2 + 1, X0 = X // 2 + 1
    for i in range(Y0):
        for j in range(X0):
            ii = 2 * i; jj = 2 * j
            if (0 <= ii < Y) and (0 <= jj < X):
                inp[i, j] = <double>x[ii * X + jj] # [::2, ::2]
            else:
                inp[i, j] = 0.0
    fftw_obj._execute()

    cdef int YY = oshape[0], XX = oshape[1]
    if res is None:
        res = np.PyArray_ZEROS(2, dims, type_num, 0)
    else:
        np.PyArray_FILLWBYTE(res, 0)

    cdef float_t[:, ::1] _res = res
    for i in range(YY):
        for j in range(XX):
            _res[i, j] += 0.25 * (out[i, j].real + out[i, XX - 1 - j].real +
                                  out[YY - 1 - i, j].real + out[YY - 1 - i, XX - 1 - j].real)

    for i in range(Y0):
        for j in range(X0):
            ii = 2 * i; jj = 2 * j - 1
            if (0 <= ii < Y) and (0 <= jj < X):
                inp[i, j] = <double>x[ii * X + jj] # [::2, 1::2]
            else:
                inp[i, j] = 0.0
    fftw_obj._execute()

    for i in range(YY):
        for j in range(XX):
            _res[i, j] += 0.25 * (out[i, j].imag - out[i, XX - 1 - j].imag +
                                  out[YY - 1 - i, j].imag - out[YY - 1 - i, XX - 1 - j].imag)

    for i in range(Y0):
        for j in range(X0):
            ii = 2 * i - 1; jj = 2 * j
            if (0 <= ii < Y) and (0 <= jj < X):
                inp[i, j] = <double>x[ii * X + jj] # [1::2, ::2]
            else:
                inp[i, j] = 0.0
    fftw_obj._execute()

    for i in range(YY):
        for j in range(XX):
            _res[i, j] += 0.25 * (out[i, j].imag + out[i, XX - 1 - j].imag -
                                  out[YY - 1 - i, j].imag - out[YY - 1 - i, XX - 1 - j].imag)

    for i in range(Y0):
        for j in range(X0):
            ii = 2 * i - 1; jj = 2 * j - 1
            if (0 <= ii < Y) and (0 <= jj < X):
                inp[i, j] = <double>x[ii * X + jj] # [1::2, 1::2]
            else:
                inp[i, j] = 0.0
    fftw_obj._execute()

    for i in range(YY):
        for j in range(XX):
            _res[i, j] += 0.25 * (out[i, j].real - out[i, XX - 1 - j].real -
                                  out[YY - 1 - i, j].real + out[YY - 1 - i, XX - 1 - j].real)

    return res

def ct_integrate(float_t[:, ::1] sx_arr not None, float_t[:, ::1] sy_arr not None, int num_threads=1) -> np.ndarray:
    """Perform the Fourier Transform wavefront reconstruction [FTI]_
    with antisymmetric derivative integration [ASDI]_.

    Parameters
    ----------
    sx_arr : numpy.ndarray
        Array of gradient values along the horizontal axis.
    sy_arr : numpy.ndarray
        Array of gradient values along the vertical axis.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    w : numpy.ndarray
        Reconstructed wavefront.

    References
    ----------
    .. [FTI] C. Kottler, C. David, F. Pfeiffer, and O. Bunk,
             "A two-directional approach for grating based
             differential phase contrast imaging using hard x-rays,"
             Opt. Express 15, 1175-1181 (2007).
    .. [ASDI] Pierre Bon, Serge Monneret, and Benoit Wattellier,
              "Noniterative boundary-artifact-free wavefront
              reconstruction from its derivatives," Appl. Opt. 51,
              5698-5704 (2012).
    """
    cdef int type_num = np.PyArray_TYPE(sx_arr.base)
    cdef np.npy_intp a = sx_arr.shape[0], b = sx_arr.shape[1]
    cdef int i, j, ii, jj
    cdef np.npy_intp *asdi_shape = [2 * a, 2 * b]
    
    cdef np.ndarray[np.complex128_t, ndim=2] sfy_asdi = np.PyArray_SimpleNew(2, asdi_shape, np.NPY_COMPLEX128)
    cdef pyfftw.FFTW fftw_obj = pyfftw.FFTW(sfy_asdi, sfy_asdi, axes=(0, 1), threads=num_threads)
    for i in range(a):
        for j in range(b):
            sfy_asdi[i, j] = -sy_arr[a - i - 1, b - j - 1]
    for i in range(a):
        for j in range(b):
            sfy_asdi[i + a, j] = sy_arr[i, b - j - 1]
    for i in range(a):
        for j in range(b):
            sfy_asdi[i, j + b] = -sy_arr[a - i - 1, j]
    for i in range(a):
        for j in range(b):
            sfy_asdi[i + a, j + b] = sy_arr[i, j]
    fftw_obj._execute()

    cdef np.ndarray[np.complex128_t, ndim=2] sfx_asdi = np.PyArray_SimpleNew(2, asdi_shape, np.NPY_COMPLEX128)
    fftw_obj._update_arrays(sfx_asdi, sfx_asdi)
    for i in range(a):
        for j in range(b):
            sfx_asdi[i, j] = -sx_arr[a - i - 1, b - j - 1]
    for i in range(a):
        for j in range(b):
            sfx_asdi[i + a, j] = -sx_arr[i, b - j - 1]
    for i in range(a):
        for j in range(b):
            sfx_asdi[i, j + b] = sx_arr[a - i - 1, j]
    for i in range(a):
        for j in range(b):
            sfx_asdi[i + a, j + b] = sx_arr[i, j]
    fftw_obj._execute()

    cdef pyfftw.FFTW ifftw_obj = pyfftw.FFTW(sfx_asdi, sfx_asdi, direction='FFTW_BACKWARD', axes=(0, 1), threads=num_threads)
    cdef double xf, yf, norm = 1.0 / <double>np.PyArray_SIZE(sfx_asdi)
    for i in range(2 * a):
        yf = 0.5 * <double>i / a - i // a
        for j in range(2 * b):
            xf = 0.5 * <double>j / b - j // b
            sfx_asdi[i, j] = norm * (sfy_asdi[i, j] * yf + sfx_asdi[i, j] * xf) / (2j * pi * (xf * xf + yf * yf))
    sfx_asdi[0, 0] = 0.0 + 0.0j
    ifftw_obj._execute()

    return np.asarray(sfx_asdi.real[a:, b:], dtype=sx_arr.base.dtype)

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
    wl - wavelength
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
