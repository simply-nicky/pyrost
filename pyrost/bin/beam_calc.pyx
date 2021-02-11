cimport numpy as np
import numpy as np
from .beam_calc cimport *
from libc.math cimport sqrt, cos, sin, pi, erf
from libc.time cimport time_t, time
from libc.string cimport memcpy
from cython.parallel import prange
cimport openmp

cdef extern from "fft.c":

    int good_size_real(int n)
    int good_size_cmplx(int n)

    ctypedef struct cfft_plan_i
    cfft_plan_i* make_cfft_plan(int length) nogil
    void destroy_cfft_plan(cfft_plan_i* plan) nogil

    int cfft_forward(cfft_plan_i* plan, double* c, double fct) nogil
    int cfft_backward(cfft_plan_i* plan, double* c, double fct) nogil

    ctypedef struct rfft_plan_i
    rfft_plan_i* make_rfft_plan(int length) nogil
    void destroy_rfft_plan(rfft_plan_i* plan) nogil

    int rfft_forward(rfft_plan_i* plan, double* c, double fct) nogil
    int rfft_backward(rfft_plan_i* plan, double* c, double fct) nogil

ctypedef np.complex128_t complex_t
ctypedef np.uint64_t uint_t
ctypedef np.npy_bool bool_t

ctypedef fused numeric:
    np.float64_t
    np.float32_t
    np.uint64_t
    np.int64_t

DEF X_TOL = 4.320005384913445 # Y_TOL = 1e-9

cdef int fft(complex_t* arr, int n) nogil:
    cdef double* dptr = <double*>(arr)
    if not dptr:
        return -1
    cdef cfft_plan_i* plan = make_cfft_plan(n)
    if not plan:
        return -1
    cdef int fail = cfft_forward(plan, dptr, 1.0)
    if fail:
        return -1
    if plan:
        destroy_cfft_plan(plan)
    return 0

cdef int ifft(complex_t* arr, int n) nogil:
    cdef double* dptr = <double*>(arr)
    if not dptr:
        return -1
    cdef cfft_plan_i* plan = make_cfft_plan(n)
    if not plan:
        return -1
    cdef double fct = 1.0 / n
    cdef int fail = cfft_backward(plan, dptr, fct)
    if fail:
        return -1
    if plan:
        destroy_cfft_plan(plan)
    return 0

cdef int rfft(complex_t* res, double* arr, int n) nogil:
    cdef double* rptr = <double*>(res)
    cdef double* dptr = <double*>(arr)
    if not dptr:
        return -1
    cdef rfft_plan_i* plan = make_rfft_plan(n)
    if not plan:
        return -1
    memcpy(<char *>(rptr + 1), dptr, n * sizeof(double))
    cdef int fail = rfft_forward(plan, rptr + 1, 1.0)
    rptr[0] = rptr[1]; rptr[1] = 0.0
    if not n % 2:
        rptr[n + 1] = 0.0
    if fail:
        return -1
    if plan:
        destroy_rfft_plan(plan)
    return 0

cdef int irfft(double* res, complex_t* arr, int n) nogil:
    cdef double* dptr = <double*>(arr)
    cdef double* rptr = <double*>(res)
    if not dptr:
        return -1
    cdef rfft_plan_i* plan = make_rfft_plan(n)
    if not plan:
        return -1
    cdef double fct = 1.0 / n
    memcpy(<char *>(rptr + 1), dptr + 2, (n - 1) * sizeof(double))
    rptr[0] = dptr[0]
    cdef int fail = rfft_backward(plan, rptr, fct)
    if fail:
        return -1
    if plan:
        destroy_rfft_plan(plan)
    return 0

cdef void rsc_type1_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h,
                      double dx0, double dx, double z, double wl) nogil:
    cdef:
        int n = u.shape[0], i
        double ph, dist
        complex_t u0
    for i in range(n):
        dist = (dx0 * (i - n // 2))**2 + z**2
        ph = 2 * pi / wl * sqrt(dist)
        k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + 1j * cos(ph)) / dist**0.75
    fft(&u[0], n)
    fft(&k[0], n)
    for i in range(n):
        ph = pi * (<double>(i) / n - (2 * i) // n)**2 * dx / dx0 * n
        u[i] = u[i] * k[i] * (cos(ph) - 1j * sin(ph))
        k[i] = cos(ph) - 1j * sin(ph)
        h[i] = cos(ph) + 1j * sin(ph)
    ifft(&u[0], n)
    ifft(&h[0], n)
    for i in range(n):
        u[i] = u[i] * h[i]
    fft(&u[0], n)
    for i in range(n // 2):
        u0 = u[i] * k[i]; u[i] = u[i + n // 2] * k[i + n // 2]
        u[i + n // 2] = u0

cdef void rsc_type2_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h,
                      double dx0, double dx, double z, double wl) nogil:
    cdef:
        int n = u.shape[0], i
        double ph, ph2, dist
    for i in range(n):
        dist = (dx * (i - n // 2))**2 + z**2
        ph = 2 * pi / wl * sqrt(dist)
        k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + 1j * cos(ph)) / dist**0.75
    fft(&k[0], n)
    for i in range(n):
        ph = pi * (<double>(i) / n - (2 * i) // n)**2 * dx0 / dx * n
        ph2 = pi * (i - n // 2)**2 * dx0 / dx / n
        u[i] = u[i] * (cos(ph2) + 1j * sin(ph2))
        k[i] = k[i] * (cos(ph) + 1j * sin(ph))
        h[i] = cos(ph2) - 1j * sin(ph2)
    fft(&u[0], n)
    fft(&h[0], n)
    for i in range(n):
        u[i] = u[i] * h[i]
    ifft(&u[0], n)
    for i in range(n):
        u[i] = u[i] * k[i]
    ifft(&u[0], n)

cdef void rsc_wp_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h, complex_t[::1] u0,
                   double dx0, double dx, double z, double wl) nogil:
    cdef:
        int a = u0.shape[0], n = u.shape[0], i
    for i in range((n - a) // 2 + 1):
        u[i] = 0; u[n - 1 - i] = 0
    for i in range(a):
        u[i + (n - a) // 2] = u0[i]
    if dx0 >= dx:
        rsc_type1_c(u, k, h, dx0, dx, z, wl)
    else:
        rsc_type2_c(u, k, h, dx0, dx, z, wl)

def rsc_wp(complex_t[::1] u0, double dx0, double dx, double z, double wl):
    r"""Wavefront propagator based on Rayleigh-Sommerfeld convolution
    method [RSC]_. Propagates a wavefront `u0` by `z` distance
    downstream.

    Parameters
    ----------
    u0 : numpy.ndarray
        Initial wavefront.
    dx0 : float
        Sampling interval at the plane upstream [um].
    dx : float
        Sampling interval at the plane downstream [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].

    Returns
    -------
    u : numpy.ndarray
        Propagated wavefront.

    Notes
    -----
    The Rayleigh–Sommerfeld diffraction integral transform is defined as:

    .. math::
        u^{\prime}(x^{\prime}) = \frac{z}{j \sqrt{\lambda}} \int_{-\infty}^{+\infty}
        u(x) \mathrm{exp} \left[-j k r(x, x^{\prime}) \right] dx
    
    with

    .. math::
        r(x, x^{\prime}) = \left[ (x - x^{\prime})^2 + z^2 \right]^{1 / 2}

    References
    ----------
    .. [RSC] V. Nascov and P. C. Logofătu, "Fast computation algorithm
             for the Rayleigh-Sommerfeld diffraction formula using
             a type of scaled convolution," Appl. Opt. 48, 4310-4319
             (2009).
    """
    cdef:
        int a = u0.shape[0], i
        double alpha = dx0 / dx if dx0 <= dx else dx / dx0
        int n = good_size_cmplx(2 * (<int>(a * (1 + alpha) // 2) + 1))
        complex_t[::1] u = np.empty(n, dtype=np.complex128)
        complex_t[::1] k = np.empty(n, dtype=np.complex128)
        complex_t[::1] h = np.empty(n, dtype=np.complex128)
    rsc_wp_c(u, k, h, u0, dx0, dx, z, wl)
    return np.asarray(u[(n - a) // 2:(n + a) // 2])

cdef void fhf_wp_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] u0,
                   double dx0, double dx, double z, double wl) nogil:
    cdef:
        int a = u0.shape[0], n = u.shape[0], i
        double ph0 = 2 * pi / wl * z, ph1
        double alpha = dx0 * dx / wl / z
        complex_t h0 = -(sin(ph0) + 1j * cos(ph0)) / sqrt(wl * z) * dx0, w0, w1
    for i in range((n - a) // 2 + 1):
        u[i] = 0; u[n - 1 - i] = 0
    for i in range(a):
        u[i + (n - a) // 2] = u0[i]
    for i in range(n):
        ph0 = pi * (i - n // 2)**2 * alpha
        k[i] = cos(ph0) - 1j * sin(ph0)
        u[i] = u[i] * (cos(ph0) + 1j * sin(ph0))
    fft(&u[0], n)
    fft(&k[0], n)
    for i in range(n):
        u[i] = u[i] * k[i]
    ifft(&u[0], n)
    for i in range(n // 2):
        ph0 = pi * i**2 * alpha
        ph1 = pi * (i - n // 2)**2 * alpha
        w0 = (cos(ph0) - 1j * sin(ph0)) * u[i]
        w1 = (cos(ph1) - 1j * sin(ph1)) * u[i + n // 2]
        u[i] = h0 * (cos(ph1 / dx0 * dx) - 1j * sin(ph1 / dx0 * dx)) * w1
        u[i + n // 2] = h0 * (cos(ph0 / dx0 * dx) - 1j * sin(ph0 / dx0 * dx)) * w0

def fhf_wp(complex_t[::1] u0, double dx0, double dx, double z, double wl):
    r"""One-dimensional discrete form of Fraunhofer diffraction
    performed by the means of Fast Fourier transform.

    Parameters
    ----------
    u0 : numpy.ndarray
        Wavefront at the plane upstream.
    dx0 : float
        Sampling interval at the plane upstream [um].
    dx : float
        Sampling interval at the plane downstream [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].

    Returns
    -------
    numpy.ndarray
        Wavefront at the plane downstream.

    Notes
    -----
    The Fraunhofer integral transform is defined as:

    .. math::
        u^{\prime}(x^{\prime}) = \frac{e^{-j k z}}{j \sqrt{\lambda z}}
        e^{-\frac{j k}{2 z} x^{\prime 2}} \int_{-\infty}^{+\infty} u(x)
        e^{j\frac{2 \pi}{\lambda z} x x^{\prime}} dx
    """
    cdef:
        int a = u0.shape[0]
        int n = good_size_cmplx(2 * a - 1)
        complex_t[::1] u = np.empty(n, dtype=np.complex128)
        complex_t[::1] k = np.empty(n, dtype=np.complex128)
    fhf_wp_c(u, k, u0, dx0, dx, z, wl)
    return np.asarray(u[(n - a) // 2:(n + a) // 2])

def fhf_wp_scan(complex_t[:, ::1] u0, double dx0, double dx, double z, double wl):
    """One-dimensional discrete form of Fraunhofer diffraction
    performed by the means of Fast Fourier transform. The transform
    is applied to a set of wavefronts `u0`.

    Parameters
    ----------
    u0 : numpy.ndarray
        Set of wavefronts at the plane upstream.
    dx0 : float
        Sampling interval at the plane upstream [um].
    dx : float
        Sampling interval at the plane downstream [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].

    Returns
    -------
    numpy.ndarray
        Set of wavefronts at the plane downstream.

    See Also
    --------
    fhf_wp : Description of the Fraunhofer diffraction
        integral transform.
    """
    cdef:
        int a = u0.shape[0], b = u0.shape[1], i, t
        int n = good_size_cmplx(2 * b - 1)
        int max_threads = openmp.omp_get_max_threads()
        complex_t[:, ::1] u = np.empty((a, n), dtype=np.complex128)
        complex_t[:, ::1] k = np.empty((max_threads, n), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        fhf_wp_c(u[i], k[t], u0[i], dx0, dx, z, wl)
    return np.asarray(u[:, (n - b) // 2:(n + b) // 2], order='C')

def bar_positions(double x0, double x1, double b_dx, double rd):
    """Generate a coordinate array of randomized barcode's bar positions.

    Parameters
    ----------
    x0 : float
        Barcode's lower bound along the x axis [um].
    x1 : float
        Barcode's upper bound along the x axis [um].
    b_dx : float
        Average bar's size [um].
    rd : float
        Random deviation of barcode's bar positions (0.0 - 1.0).

    Returns
    -------
    bx_arr : numpy.ndarray
        Array of barcode's bar coordinates.
    """
    cdef:
        int br_n = 2 * (<int>((x1 - x0) / 2 / b_dx) + 1) if x1 > x0 else 0, i
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        double[::1] b_steps = np.empty(br_n, dtype=np.float64)
        time_t t = time(NULL)
    gsl_rng_set(r, t)
    if br_n:
        for i in range(br_n):
            b_steps[i] = x0 + b_dx * (i + 2 * rd * (gsl_rng_uniform_pos(r) - 0.5))
    gsl_rng_free(r)
    return np.asarray(b_steps)

cdef int binary_search(double[::1] values, int l, int r, double x) nogil:
    cdef int m = l + (r - l) // 2
    if l <= r:
        if x == values[m]:
            return m
        elif x > values[m] and x <= values[m + 1]:
            return m + 1
        elif x < values[m]:
            return binary_search(values, l, m, x)
        else:
            return binary_search(values, m + 1, r, x)

cdef int searchsorted(double[::1] values, double x) nogil:
    cdef int r = values.shape[0]
    if x < values[0]:
        return 0
    elif x > values[r - 1]:
        return r
    else:
        return binary_search(values, 0, r, x)

cdef void barcode_c(double[::1] b_tr, double[::1] b_steps, double b_atn, double b_sgm,
                    double blk_atn, double dx, double step) nogil:
    cdef:
        int a = b_tr.shape[0], b = b_steps.shape[0], i, j0, j
        double b_dx = (b_steps[b - 1] - b_steps[0]) / b
        int bb = <int>(X_TOL * sqrt(2) * b_sgm / b_dx + 1)
        double tr, xx, x0, x1
    for i in range(a):
        xx = dx * (i - a // 2) + step
        j0 = searchsorted(b_steps, xx) # even '-', odd '+'
        tr = 0
        for j in range(j0 - bb, j0 + bb + 1):
            if j > 0 and j < b - 1:
                x0 = (xx - b_steps[j - 1]) / sqrt(2) / b_sgm
                x1 = (xx - b_steps[j]) / sqrt(2) / b_sgm
                tr += b_atn * (b_steps[j] - b_steps[j - 1]) / b_dx * 0.5 * (0.5 - j % 2) * (erf(x0) - erf(x1))
        tr -= (0.25 * b_atn + 0.5 * blk_atn) * erf((xx - b_steps[0]) / sqrt(2 + 2 * (blk_atn / b_atn)**2) / b_sgm)
        tr += (0.25 * b_atn + 0.5 * blk_atn) * erf((xx - b_steps[b - 1]) / sqrt(2 + 2 * (blk_atn / b_atn)**2) / b_sgm)
        b_tr[i] = sqrt(1 + tr)
        
def barcode_profile(double[::1] b_steps, double b_atn, double b_sgm, double blk_atn,
                    double dx, double step, int n_f, int n_x):
    r"""Return an array of barcode's transmission profile.
    Barcode is scanned accross the x axis with `ss` step size
    and `nf` number of steps.

    Parameters
    ----------
    b_steps : numpy.ndarray
        Coordinates of barcode's bar positions [um].    
    b_atn : float
        Barcode's bar attenuation coefficient (0.0 - 1.0).
    b_sgm : float
        Bar's blurriness [um].
    blk_atn : float
        Barcode's bulk attenuation coefficient (0.0 - 1.0).
    dx : float
        Sampling interval [um].
    step : float
        Scan's step size [um].
    n_f : int
        Scan's number of frames.
    n_x : int
        Size of the wavefront array.
    
    Returns
    -------
    b_tr : numpy.ndarray
        Array of barcode's transmission profiles.

    Notes
    -----
    Barcode's transmission profile is simulated with a set
    of error functions:
    
    .. math::
        \begin{multline}
            T_{b}(x) = 1 - \frac{T_{bulk}}{2} \left\{
            \mathrm{erf}\left[ \frac{x - x_{bar}[0]}{\sqrt{2} \sigma} \right] +
            \mathrm{erf}\left[ \frac{x_{bar}[n - 1] - x}{\sqrt{2} \sigma} \right]
            \right\} -\\
            \frac{T_{bar}}{4} \sum_{i = 1}^{n - 2} \left\{
            2 \mathrm{erf}\left[ \frac{x - x_{bar}[i]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{bar}[i - 1]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{bar}[i + 1]}{\sqrt{2} \sigma} \right]
            \right\}
        \end{multline}
    
    where :math:`x_{bar}` is an array of bar coordinates.
    """
    cdef:
        int i
        double[:, ::1] b_tr = np.empty((n_f, n_x), dtype=np.float64)
    for i in prange(n_f, schedule='guided', nogil=True):
        barcode_c(b_tr[i], b_steps, b_atn, b_sgm, blk_atn, dx, i * step)
    return np.asarray(b_tr)

cdef void fft_convolve_c(double[::1] b1, complex_t[::1] c1, double[::1] a1, complex_t[::1] c2) nogil:
    cdef:
        int a = a1.shape[0], n = b1.shape[0], i
        double b0
    for i in range((n - a) // 2 + 1):
        b1[i] = a1[0]; b1[n - 1 - i] = a1[a - 1]
    for i in range(a):
        b1[i + (n - a) // 2] = a1[i]
    rfft(&c1[0], &b1[0], n)
    for i in range(n // 2 + 1):
        c1[i] = c1[i] * c2[i]
    irfft(&b1[0], &c1[0], n)
    for i in range(n // 2):
        b0 = b1[i]; b1[i] = b1[i + n // 2]
        b1[i + n // 2] = b0

def fft_convolve(double[::1] a1, double[::1] a2):
    """Convolve two one-dimensional arrays using FFT. The output
    size is the same size as `a1`.

    Parameters
    ----------
    a1 : numpy.ndarray
        First input array.
    a2 : numpy.ndarray
        Second input array.

    Returns
    -------
    out : numpy.ndarray
        A one-dimensional array containing the discrete linear
        convolution of `a1` with `a2`.
    """
    cdef:
        int a = a1.shape[0], b = a2.shape[0], i
        int n = good_size_real(a + b - 1)
        double[::1] b1 = np.empty(n, dtype=np.float64)
        double[::1] b2 = np.empty(n, dtype=np.float64)
        complex_t[::1] c1 = np.empty(n // 2 + 1, dtype=np.complex128)
        complex_t[::1] c2 = np.empty(n // 2 + 1, dtype=np.complex128)
    for i in range((n - b) // 2 + 1):
        b2[i] = a2[0]; b2[n - 1 - i] = a2[b - 1]
    for i in range(b):
        b2[i + (n - b) // 2] = a2[i]
    rfft(&c2[0], &b2[0], n)
    fft_convolve_c(b1, c1, a1, c2)
    return np.asarray(b1[(n - a) // 2:(n + a) // 2])

def fft_convolve_scan(double[:, ::1] a1, double[::1] a2):
    """Convolve a set of one-dimensional arrays `a1` with `a2` using
    FFT. The output size is the same size as `a1`.

    Parameters
    ----------
    a1 : numpy.ndarray
        Set of input arrays.
    a2 : numpy.ndarray
        Second input array.

    Returns
    -------
    out : numpy.ndarray
        A set of one-dimensional arrays containing the discrete
        linear convolution of `a1` with `a2`.
    """
    cdef:
        int nf = a1.shape[0], a = a1.shape[1], b = a2.shape[0], i, t
        int n = good_size_real(a + b - 1)
        int max_threads = openmp.omp_get_max_threads()
        double[:, ::1] b1 = np.empty((nf, n), dtype=np.float64)
        double[::1] b2 = np.empty(n, dtype=np.float64)
        complex_t[:, ::1] c1 = np.empty((max_threads, n // 2 + 1), dtype=np.complex128)
        complex_t[::1] c2 = np.empty(n // 2 + 1, dtype=np.complex128)
    for i in range((n - b) // 2 + 1):
        b2[i] = a2[0]; b2[n - 1 - i] = a2[b - 1]
    for i in range(b):
        b2[i + (n - b) // 2] = a2[i]
    rfft(&c2[0], &b2[0], n)
    for i in prange(nf, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        fft_convolve_c(b1[i], c1[t], a1[i], c2)
    return np.asarray(b1[:, (n - a) // 2:(n + a) // 2], order='C')

cdef void make_frame_nc(uint_t[:, ::1] frame, double[::1] i_x, double[::1] i_ss, double dx) nogil:
    cdef:
        int a = i_x.shape[0], ss = frame.shape[0], fs = frame.shape[1], i, j, jj, j0, j1
        double i_fs = 0
    j1 = <int>(0.5 * a / fs)
    for jj in range(j1):
        i_fs += i_x[jj] * dx
    for i in range(ss):
        frame[i, 0] = <int>(i_fs * i_ss[i])
    i_fs = 0
    for jj in range(j1):
        i_fs += i_x[a - 1 - jj] * dx
    for i in range(ss):
        frame[i, fs - 1] = <int>(i_fs * i_ss[i])
    for j in range(1, fs - 1):
        i_fs = 0
        j0 = <int>((j - 0.5) * a // fs)
        j1 = <int>((j + 0.5) * a // fs)
        for jj in range(j0, j1):
            i_fs += i_x[jj] * dx
        for i in range(ss):
            frame[i, j] = <int>(i_fs * i_ss[i])

cdef void make_frame_c(uint_t[:, ::1] frame, double[::1] i_x, double[::1] i_ss, double dx, unsigned long seed) nogil:
    cdef:
        int a = i_x.shape[0], ss = frame.shape[0], fs = frame.shape[1], i, j, jj, j0, j1
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        double i_fs = 0
    gsl_rng_set(r, seed)
    j1 = <int>(0.5 * a / fs)
    for jj in range(j1):
        i_fs += i_x[jj] * dx
    for i in range(ss):
        frame[i, 0] = gsl_ran_poisson(r, i_fs * i_ss[i])
    i_fs = 0
    for jj in range(j1):
        i_fs += i_x[a - 1 - jj] * dx
    for i in range(ss):
        frame[i, fs - 1] = gsl_ran_poisson(r, i_fs * i_ss[i])
    for j in range(1, fs - 1):
        i_fs = 0
        j0 = <int>((j - 0.5) * a // fs)
        j1 = <int>((j + 0.5) * a // fs)
        for jj in range(j0, j1):
            i_fs += i_x[jj] * dx
        for i in range(ss):
            frame[i, j] = gsl_ran_poisson(r, i_fs * i_ss[i])

def make_frames(double[:, ::1] i_x, double[::1] i_y, double dx, double dy,
                int ss, int fs, bool_t apply_noise):
    """Generate intensity frames with Poisson noise
    from one-dimensional intensity profiles `i_x`
    and `i_y` convoluted with the source's rocking
    curves `sc_x` and `sc_y`.

    Parameters
    ----------
    i_x : numpy.ndarray
        Intensity profile along the x axis.
    i_y : numpy.ndarray
        Intensity profile along the y axis.
    dx : float
        Sampling interval along the x axis [um].
    dy : float
        Sampling interval along the y axis [um].
    ss : int
        Detector's size along the slow axis.
    fs : int
        Detector's size along the fast axis.
    apply_noise : bool
        Apply Poisson noise if it's True.

    Returns
    -------
    frames : numpy.ndarray
        Intensity frames.
    """
    cdef:
        int nf = i_x.shape[0], a = i_x.shape[1], b = i_y.shape[0], i, ii, i0, i1
        uint_t[:, :, ::1] frames = np.zeros((nf, ss, fs), dtype=np.uint64)
        double[::1] i_ss = np.zeros(ss, dtype=np.float64)
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        time_t t = time(NULL)
        unsigned long seed
    i1 = <int>(0.5 * b // ss)
    for ii in range(i1):
        i_ss[0] += i_y[ii] * dy
        i_ss[ss - 1] += i_y[b - 1 - ii] * dy
    for i in range(1, ss - 1):
        i0 = <int>((i - 0.5) * b // ss)
        i1 = <int>((i + 0.5) * b // ss)
        for ii in range(i0, i1):
            i_ss[i] += i_y[ii] * dy
    if apply_noise:
        gsl_rng_set(r, t)
        for i in prange(nf, schedule='guided', nogil=True):
            seed = gsl_rng_get(r)
            make_frame_c(frames[i], i_x[i], i_ss, dx, seed)
        gsl_rng_free(r)
    else:
        for i in prange(nf, schedule='guided', nogil=True):
            make_frame_nc(frames[i], i_x[i], i_ss, dx)
    return np.asarray(frames)

cdef numeric wirthselect(numeric[::1] array, int k) nogil:
    cdef:
        int l = 0, m = array.shape[0] - 1, i, j
        numeric x, tmp 
    while l < m: 
        x = array[k] 
        i = l; j = m 
        while 1: 
            while array[i] < x: i += 1 
            while x < array[j]: j -= 1 
            if i <= j: 
                tmp = array[i]; array[i] = array[j]; array[j] = tmp
                i += 1; j -= 1 
            if i > j: break 
        if j < k: l = i 
        if k < i: m = j 
    return array[k]

def make_whitefield(numeric[:, :, ::1] data, bool_t[:, ::1] mask):
    """Generate a whitefield using the median filtering.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    mask : numpy.ndarray
        Bad pixel mask.

    Returns
    -------
    wf : numpy.ndarray
        Whitefield.
    """
    if numeric is np.float64_t:
        dtype = np.float64
    elif numeric is np.float32_t:
        dtype = np.float32
    else:
        dtype = np.uint64
    cdef:
        int a = data.shape[0], b = data.shape[1], c = data.shape[2], i, j, k
        int max_threads = openmp.omp_get_max_threads()
        numeric[:, ::1] wf = np.empty((b, c), dtype=dtype)
        numeric[:, ::1] array = np.empty((max_threads, a), dtype=dtype)
    for j in prange(b, schedule='guided', nogil=True):
        i = openmp.omp_get_thread_num()
        for k in range(c):
            if mask[j, k]:
                array[i] = data[:, j, k]
                wf[j, k] = wirthselect(array[i], a // 2)
            else:
                wf[j, k] = 0
    return np.asarray(wf)
