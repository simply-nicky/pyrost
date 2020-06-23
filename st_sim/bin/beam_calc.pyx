cimport numpy as cnp
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, exp, pi, tanh
from cython.parallel import prange
cimport openmp

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

cdef float_t lens_re(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1]
        float_t wl = (<float_t*> params)[2], f = (<float_t*> params)[3]
        float_t alpha = (<float_t*> params)[4]
    return cos(pi * xx**2 / wl * (1 / f - 1 / z) + 2 * pi / wl / z * x * xx - alpha * 1e9 * (xx / f)**3)

cdef float_t lens_im(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1]
        float_t wl = (<float_t*> params)[2], f = (<float_t*> params)[3]
        float_t alpha = (<float_t*> params)[4]
    return -sin(pi * xx**2 / wl * (1 / f - 1 / z) + 2 * pi / wl / z * x * xx - alpha * 1e9 * (xx / f)**3)

cdef float_t aperture_re(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1], wl = (<float_t*> params)[2]
    return cos(pi / wl / z * (x - xx)**2)

cdef float_t aperture_im(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1], wl = (<float_t*> params)[2]
    return sin(pi / wl / z * (x - xx)**2)

cdef float_t gsl_quad(gsl_function func, float_t a, float_t b, float_t eps_abs, float_t eps_rel, int_t limit) nogil:
    cdef:
        float_t result, error
        gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(limit)
    gsl_integration_qag(&func, a, b, eps_abs, eps_rel, limit, GSL_INTEG_GAUSS51, W, &result, &error)
    gsl_integration_workspace_free(W)
    return result

cdef complex_t lens_wp(float_t x, float_t defoc, float_t wl, float_t f, float_t alpha, float_t ap) nogil:
    cdef:
        float_t ph = 2 * pi * (f + defoc) / wl + pi * x**2 / wl / (f + defoc), re, im
        float_t params[5]
        int_t fn = <int_t> (ap**2 / wl / (f + defoc))
        gsl_function func
    params[0] = x; params[1] = f + defoc; params[2] = wl; params[3] = f; params[4] = alpha
    func.function = &lens_re; func.params = params
    re = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    func.function = &lens_im
    im = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    return re + 1j * im
    
cdef complex_t aperture_wp(float_t x, float_t z, float_t wl, float_t ap) nogil:
    cdef:
        float_t re, im
        float_t params[3]
        int_t fn = <int_t> (ap**2 / wl / z)
        gsl_function func
    params[0] = x; params[1] = z; params[2] = wl
    func.function = &aperture_re; func.params = params
    re = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    func.function = &aperture_im
    im = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    return re + 1j * im

cdef complex_t frn_wp(complex_t[::1] wf0, float_t[::1] x_arr, float_t xx, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = wf0.shape[0], i
        float_t ph0, ph1
        complex_t wf
    ph0 = 2 * pi / wl / dist * x_arr[0] * xx
    ph1 = 2 * pi / wl / dist * x_arr[1] * xx
    wf = (wf0[0] * (cos(ph0) - 1j * sin(ph0)) + wf0[1] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[1] - x_arr[0])
    for i in range(2, a):
        ph0 = ph1
        ph1 = 2 * pi / wl / dist * x_arr[i] * xx
        wf += (wf0[i - 1] * (cos(ph0) - 1j * sin(ph0)) + wf0[i] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[i] - x_arr[i - 1])
    return wf

cdef void frn_1d(complex_t[::1] wf1, complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = xx_arr.shape[0], i
    for i in range(a):
        wf1[i] = frn_wp(wf0, x_arr, xx_arr[i], dist, wl)

def fraunhofer_1d(complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl):
    """
    1D Fraunhofer diffraction calculation (without the coefficient)

    wf0 - wavefront at the plane downstream
    x_arr - coordinates at the plane downstream [um]
    xx_arr - coordinates at the plane upstream [um]
    dist - distance between planes [um]
    wl - wavelength [um]
    """
    cdef:
        int_t a = xx_arr.shape[0]
        complex_t[::1] wf = np.empty((a,), dtype=np.complex128)
    frn_1d(wf, wf0, x_arr, xx_arr, dist, wl)
    return np.asarray(wf)

def fraunhofer_2d(complex_t[:, ::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl):
    """
    1D Fraunhofer diffraction calculation for an array of wavefronts (without the coefficient)

    wf0 - an array of wavefronts at the plane downstream
    x_arr - coordinates at the plane downstream [um]
    xx_arr - coordinates at the plane upstream [um]
    dist - distance between planes [um]
    wl - wavelength [um]
    """
    cdef:
        int_t a = wf0.shape[0], b = xx_arr.shape[0], i
        complex_t[:, ::1] wf = np.empty((a, b), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True):
        frn_1d(wf[i], wf0[i], x_arr, xx_arr, dist, wl)
    return np.asarray(wf)

def aperture_wf(float_t[::1] x_arr, float_t z, float_t wl, float_t ap):
    """
    Aperture wavefront calculation by dint of Fresnel diffraction (without the coefficient)

    x_arr - coordinates at the plane downstream [um]
    z - propagation distance [um]
    wl - wavelength [um]
    ap - aperture's size [um]
    """
    cdef:
        int_t a = x_arr.shape[0], i
        complex_t[::1] wave_arr = np.empty((a,), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True, chunksize=10):
        wave_arr[i] = aperture_wp(x_arr[i], z, wl, ap)
    return np.asarray(wave_arr)

def lens_wf(float_t[::1] x_arr, float_t defoc, float_t wl, float_t f, float_t alpha, float_t ap):
    """
    Lens wavefront caluclation by dint of Fresnel diffraction (without the coefficient)

    x_arr - coordinates at the plane downstream [um]
    defoc - defocus [um]
    wl - wavelength [um]
    f - focal distance [um]
    alpha - third order abberations [rad/mrad^3]
    ap - lens' size [um]
    """
    cdef:
        int_t a = x_arr.shape[0], i
        complex_t[::1] wave_arr = np.empty((a,), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True, chunksize=10):
        wave_arr[i] = lens_wp(x_arr[i], defoc, wl, f, alpha, ap)
    return np.asarray(wave_arr)

def barcode_steps(float_t beam_dx, float_t bar_size, float_t rnd_div, float_t step_size, int_t n_frames):
    """
    Barcode bars' coordinates generation with random deviation

    beam_dx - incident beam size [um]
    bar_size - mean bar size [um]
    rnd_div - random deviation (0.0 - 1.0)
    step_size - scan step size [um]
    n_frames - number of frames of a scan
    """
    cdef:
        int_t bs_n = (<int_t>((beam_dx + step_size * n_frames) / bar_size) // 2 + 1) * 2, i
        float_t rnd
        float_t[::1] bs = np.empty(bs_n, dtype=np.float64)
    for i in range(bs_n):
        rnd = rand() / <float_t>(RAND_MAX)
        bs[i] = (i + 0.5) * bar_size + (rnd - 0.5) * rnd_div * bar_size / 2
    return np.asarray(bs)

def barcode(float_t[::1] x_arr, float_t[::1] bsteps, float_t b_sigma, float_t atn, float_t step_size, int_t n_frames):
    """
    Barcode transmission array for a scan

    x_arr - coordinates [um]
    bsteps - bar coordinates array [um]
    b_sigma - bar haziness width [um]
    atn - bar attenuation (0.0 - 1.0)
    step_size - scan step size [um]
    n_frames - number of frames of a scan
    """
    cdef:
        int_t a = x_arr.shape[0], aa = bsteps.shape[0], i, j, ii
        float_t tr, bs_x0, bs_x1
        float_t[:, ::1] bs_t = np.empty((n_frames, a), dtype=np.float64)
    for i in range(a):
        for j in range(n_frames):
            tr = 0
            for ii in range(aa / 2):
                bs_x0 = ((x_arr[i] - x_arr[0]) - bsteps[2 * ii] + j * step_size) / 2 / b_sigma
                bs_x1 = -((x_arr[i] - x_arr[0]) - bsteps[2 * ii + 1] + j * step_size) / 2 / b_sigma
                tr += 0.5 * tanh(bs_x0) * tanh(bs_x1) + 0.5
            bs_t[j, i] = 1 - atn + atn * tr
    return np.asarray(bs_t)

cdef void make_frame_c(uint_t[:, ::1] frame, complex_t[::1] wf_x, complex_t[::1] wf_y) nogil:
    cdef:
        int_t b = wf_y.shape[0], c = wf_x.shape[0], j, k
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        complex_t wf
    for j in range(b):
        for k in range(c):
            wf = wf_x[k] * wf_y[j]
            frame[j, k] = gsl_ran_poisson(r, wf.real**2 + wf.imag**2)
    gsl_rng_free(r)

def make_frames(complex_t[:, ::1] wf_x, complex_t[::1] wf_y):
    cdef:
        int_t a = wf_x.shape[0], b = wf_y.shape[0], c = wf_x.shape[1], i
        uint_t[:, :, ::1] frames = np.empty((a, b, c), dtype=np.uint64)
    for i in prange(a, schedule='guided', nogil=True):
        make_frame_c(frames[i], wf_x[i], wf_y)
    return np.asarray(frames)

cdef uint_t wirthselect_uint(uint_t[:] array, int k) nogil:
    cdef:
        int_t l = 0, m = array.shape[0] - 1, i, j
        uint_t x, tmp 
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

def make_whitefield(uint_t[:, :, ::1] data, uint8_t[:, ::1] mask):
    cdef:
        int_t a = data.shape[0], b = data.shape[1], c = data.shape[2], i, j, k
        int_t max_threads = openmp.omp_get_max_threads()
        uint_t[:, ::1] wf = np.empty((b, c), dtype=np.uint64)
        uint_t[:, ::1] array = np.empty((max_threads, a), dtype=np.uint64)
    for j in prange(b, schedule='guided', nogil=True):
        i = openmp.omp_get_thread_num()
        for k in range(c):
            if mask[j, k]:
                array[i] = data[:, j, k]
                wf[j, k] = wirthselect_uint(array[i], a // 2)
            else:
                wf[j, k] = 0
    return np.asarray(wf)