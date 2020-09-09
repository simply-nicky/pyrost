cimport numpy as cnp
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, exp, pi, erf
from cython.parallel import prange
cimport openmp

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

cdef float_t lens_re(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], wl = (<float_t*> params)[1]
        float_t f = (<float_t*> params)[2], df = (<float_t*> params)[3]
        float_t a = (<float_t*> params)[4]
        float_t ph, ph_ab
    ph = -pi * xx**2 / wl * df / f / (f + df) - 2 * pi / wl / (f + df) * x * xx
    ph_ab = -a * 1e9 * (xx / f)**3
    return cos(ph + ph_ab)

cdef float_t lens_im(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], wl = (<float_t*> params)[1]
        float_t f = (<float_t*> params)[2], df = (<float_t*> params)[3]
        float_t a = (<float_t*> params)[4]
        float_t ph, ph_ab
    ph = -pi * xx**2 / wl * df / f / (f + df) - 2 * pi / wl / (f + df) * x * xx
    ph_ab = -a * 1e9 * (xx / f)**3
    return sin(ph + ph_ab)

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

cdef complex_t lens_wp(float_t x, float_t wl, float_t ap, float_t f, float_t df, float_t a) nogil:
    cdef:
        float_t re, im, ph = pi / wl / (f + df) * x**2
        float_t params[5]
        int_t fn = <int_t> (ap**2 / wl / (f + df))
        gsl_function func
    params[0] = x; params[1] = wl; params[2] = f
    params[3] = df; params[4] = a
    func.function = &lens_re; func.params = params
    re = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    func.function = &lens_im
    im = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    return (re + 1j * im) * (cos(ph) + 1j * sin(ph))
    
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

cdef complex_t fhf_wp(complex_t[::1] wf0, float_t[::1] x_arr, float_t xx, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = wf0.shape[0], i
        float_t ph0, ph1, ph = pi / wl / dist * xx**2
        complex_t wf = 0 + 0j
    ph0 = 2 * pi / wl / dist * x_arr[0] * xx
    ph1 = 2 * pi / wl / dist * x_arr[1] * xx
    wf = (wf0[0] * (cos(ph0) - 1j * sin(ph0)) + wf0[1] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[1] - x_arr[0])
    for i in range(2, a):
        ph0 = ph1
        ph1 = 2 * pi / wl / dist * x_arr[i] * xx
        wf += (wf0[i - 1] * (cos(ph0) - 1j * sin(ph0)) + wf0[i] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[i] - x_arr[i - 1])
    return wf * (cos(ph) + 1j * sin(ph))

cdef void fhf_1d(complex_t[::1] wf1, complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = xx_arr.shape[0], i
    for i in range(a):
        wf1[i] = fhf_wp(wf0, x_arr, xx_arr[i], dist, wl)

cdef complex_t fnl_wp(complex_t[::1] wf0, float_t[::1] x_arr, float_t xx, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = wf0.shape[0], i
        float_t ph0, ph1
        complex_t wf
    ph0 = pi / wl / dist * (x_arr[0] - xx)**2
    ph1 = pi / wl / dist * (x_arr[1] - xx)**2
    wf = (wf0[0] * (cos(ph0) + 1j * sin(ph0)) + wf0[1] * (cos(ph1) + 1j * sin(ph1))) / 2 * (x_arr[1] - x_arr[0])
    for i in range(2, a):
        ph0 = ph1
        ph1 = pi / wl / dist * (x_arr[i] - xx)**2
        wf += (wf0[i - 1] * (cos(ph0) + 1j * sin(ph0)) + wf0[i] * (cos(ph1) + 1j * sin(ph1))) / 2 * (x_arr[i] - x_arr[i - 1])
    return wf

cdef void fnl_1d(complex_t[::1] wf1, complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl) nogil:
    cdef:
        int_t a = xx_arr.shape[0], i
    for i in range(a):
        wf1[i] = fnl_wp(wf0, x_arr, xx_arr[i], dist, wl)

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
    fhf_1d(wf, wf0, x_arr, xx_arr, dist, wl)
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
        fhf_1d(wf[i], wf0[i], x_arr, xx_arr, dist, wl)
    return np.asarray(wf)

def fresnel_1d(complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl):
    """
    1D Fresnel diffraction calculation (without the coefficient)

    wf0 - wavefront at the plane downstream
    x_arr - coordinates at the plane downstream [um]
    xx_arr - coordinates at the plane upstream [um]
    dist - distance between planes [um]
    wl - wavelength [um]
    """
    cdef:
        int_t a = xx_arr.shape[0]
        complex_t[::1] wf = np.empty((a,), dtype=np.complex128)
    fnl_1d(wf, wf0, x_arr, xx_arr, dist, wl)
    return np.asarray(wf)

def fresnel_2d(complex_t[:, ::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t dist, float_t wl):
    """
    1D Fresnel diffraction calculation for an array of wavefronts (without the coefficient)

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
        fnl_1d(wf[i], wf0[i], x_arr, xx_arr, dist, wl)
    return np.asarray(wf)

def aperture(float_t[::1] x_arr, float_t z, float_t wl, float_t ap):
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

def lens(float_t[::1] x_arr, float_t wl, float_t ap, float_t focus,
         float_t defoc, float_t alpha):
    """
    Lens wavefront calculation by dint of Fresnel diffraction (without the coefficient)
    with third order polinomial abberations

    x_arr - coordinates at the plane downstream [um]
    wl - wavelength [um]
    ap - lens' size [um]
    focus - focal distance [um]
    defoc - defocus [um]
    alpha - abberations coefficient [rad/mrad^3]
    """
    cdef:
        int_t a = x_arr.shape[0], i
        complex_t[::1] wave_arr = np.empty((a,), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True, chunksize=10):
        wave_arr[i] = lens_wp(x_arr[i], wl, ap, focus, defoc, alpha) 
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
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        float_t bs_min = max(1 - rnd_div, 0), bs_max = 1 + rnd_div
        float_t[::1] bs = np.empty(bs_n, dtype=np.float64)
    bs[0] = bar_size * (0.5 + bs_min + (bs_max - bs_min) * gsl_rng_uniform_pos(r))
    for i in range(1, bs_n):
        bs[i] = bs[i - 1] + bar_size * (bs_min + (bs_max - bs_min) * gsl_rng_uniform_pos(r))
    return np.asarray(bs)

cdef void bc_1d(float_t[::1] bc_t, float_t[::1] x_arr, float_t[::1] bsteps,
                float_t b_sigma, float_t atn, float_t step) nogil:
    cdef:
        int_t a = x_arr.shape[0], aa = bsteps.shape[0], i, ii
        float_t tr, bs_x0, bs_x1, t
    for i in range(a):
        tr = 0
        for ii in range(aa / 2):
            bs_x0 = ((x_arr[i] - x_arr[0]) - bsteps[2 * ii] + step) / 2 / b_sigma
            bs_x1 = -((x_arr[i] - x_arr[0]) - bsteps[2 * ii + 1] + step) / 2 / b_sigma
            tr += 0.5 * erf(bs_x0) * erf(bs_x1) + 0.5
        bc_t[i] = sqrt(1 - atn + atn * tr)

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
        int_t a = x_arr.shape[0], i
        float_t[:, ::1] bc_t = np.empty((n_frames, a), dtype=np.float64)
    for i in prange(n_frames, schedule='guided', nogil=True):
        bc_1d(bc_t[i], x_arr, bsteps, b_sigma, atn, i * step_size)
    return np.asarray(bc_t)

cdef float_t convolve_c(float_t[::1] a1, float_t[::1] a2, int_t k) nogil:
    cdef:
        int_t a = a1.shape[0], b = a2.shape[0]
        int_t i0 = max(k - b // 2, 0), i1 = min(k - b//2 + b, a), i
        float_t x = 0
    for i in range(i0, i1):
        x += a1[i] * a2[k + b//2 - i]
    return x

cdef void make_frame_c(uint_t[:, ::1] frame, float_t[::1] i_x, float_t[::1] i_y,
                       float_t[::1] sc, float_t pix_size, unsigned long seed) nogil:
    cdef:
        int_t b = i_y.shape[0], c = i_x.shape[0], j, k
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        float_t i_xs
    gsl_rng_set(r, seed)
    for k in range(c):
        i_xs = convolve_c(i_x, sc, k)
        for j in range(b):
            frame[j, k] = gsl_ran_poisson(r, i_xs * i_y[j] * pix_size**2)
    gsl_rng_free(r)

def make_frames(float_t[:, ::1] i_x, float_t[::1] i_y, float_t[::1] sc_x, float_t[::1] sc_y, float_t pix_size):
    """
    Generate intensity frames with Poisson noise from x and y coordinate wavefront profiles

    i_x, i_y - x and y coordinate intensity profiles
    sc_x, sc_y - source rocking curve along x- and y-axes
    pix_size - pixel size [um]
    """
    cdef:
        int_t a = i_x.shape[0], b = i_y.shape[0], c = i_x.shape[1], i
        uint_t[:, :, ::1] frames = np.empty((a, b, c), dtype=np.uint64)
        float_t[::1] i_ys = np.empty(b, dtype=np.float64)
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        unsigned long seed
    for i in range(b):
        i_ys[i] = convolve_c(i_y, sc_y, i)
    for i in prange(a, schedule='guided', nogil=True):
        seed = gsl_rng_get(r)
        make_frame_c(frames[i], i_x[i], i_ys, sc_x, pix_size, seed)
    gsl_rng_free(r)
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
    """
    Return whitefield based on median filtering of the stack of frames

    data - stack of frames
    mask - bad pixel mask
    """
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