cimport numpy as np
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, pi, erf
from cython.parallel import prange
cimport openmp
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

ctypedef np.complex128_t complex_t
ctypedef np.float64_t float_t
ctypedef np.uint64_t uint_t
ctypedef np.npy_bool bool_t

ctypedef fused numeric:
    np.float64_t
    np.float32_t
    np.uint64_t
    np.int64_t

DEF X_TOL = 4.320005384913445 # Y_TOL = 1e-9

cdef float_t gsl_quad(gsl_function func, float_t a, float_t b, float_t eps_abs, float_t eps_rel, int limit) nogil:
    cdef:
        float_t result, error
        gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(limit)
    gsl_integration_qag(&func, a, b, eps_abs, eps_rel, limit, GSL_INTEG_GAUSS51, W, &result, &error)
    gsl_integration_workspace_free(W)
    return result

cdef float_t lens_re(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], wl = (<float_t*> params)[1]
        float_t f = (<float_t*> params)[2], df = (<float_t*> params)[3]
        float_t a = (<float_t*> params)[4], xc = (<float_t*> params)[5]
        float_t ph, ph_ab
    ph = -pi * xx**2 / wl * df / f / (f + df) - 2 * pi / wl / (f + df) * x * xx
    ph_ab = -a * 1e9 * ((xx - xc) / f)**3
    return cos(ph + ph_ab)

cdef float_t lens_im(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], wl = (<float_t*> params)[1]
        float_t f = (<float_t*> params)[2], df = (<float_t*> params)[3]
        float_t a = (<float_t*> params)[4], xc = (<float_t*> params)[5]
        float_t ph, ph_ab
    ph = -pi * xx**2 / wl * df / f / (f + df) - 2 * pi / wl / (f + df) * x * xx
    ph_ab = -a * 1e9 * ((xx - xc) / f)**3
    return sin(ph + ph_ab)

cdef complex_t lens_wp_c(float_t x, float_t wl, float_t ap, float_t f,
                       float_t df, float_t a, float_t xc) nogil:
    cdef:
        float_t re, im, ph = pi / wl / (f + df) * x**2
        float_t params[6]
        int fn = <int> (ap**2 / wl / (f + df))
        gsl_function func
    params[0] = x; params[1] = wl; params[2] = f
    params[3] = df; params[4] = a; params[5] = xc
    func.function = &lens_re; func.params = params
    re = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    func.function = &lens_im
    im = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    return (re + 1j * im) * (cos(ph) + 1j * sin(ph))

def lens_wp(float_t[::1] x_arr, float_t wl, float_t ap, float_t focus,
            float_t defoc, float_t alpha, float_t xc):
    r"""Calculate beam's wavefront propagation from the lens'
    to the sample. The sample's plane is `focus` + `defoc`
    downstream from the lens' plane. Lens' phase profile is
    assumed to be quadratic with third order abberations.

    Parameters
    ----------
    x_arr : numpy.ndarray
        Coordinates at the sample's plane [um].
    wl : float
        Incoming beam's wavelength [um].
    ap : float
        Lens' aperture size along the x axis[um].
    focus : float
        Lens' focal distance [um].
    defoc : float
        Defocus distance [um].
    alpha : float
        Third order abberations coefficient [rad/mrad^3].
    xc : float
        Center point of the lens' abberations [um].

    Returns
    -------
    numpy.ndarray
        Beam's wavefront at the sample's plane.

    Notes
    -----
    The exit-surface at the lens' plane:

    .. math::
        U_0(x_0) = \Pi(a_x x_0) \exp
        \left[ -\frac{j \pi x_0^2}{\lambda f} + j \alpha
        \left( \frac{x_0 - x_c}{f} \right)^3 \right]

    Wavefront :math:`U_0` propagates to the sample's plane which
    is :math:`f + z_1` downstream from the lens. According to
    the Fresnel diffraction theory (without the normalizing
    coefficient before the integral):

    .. math::
        U(x) = \int_{-a_x / 2}^{a_x / 2}
        e^{-\frac{j k z_1 x_0^2 }{2f(z_1 + f)}}
        e^{j\alpha\left(\frac{x_0 - x_c}{f}\right)^3} 
        e^{j\frac{2 \pi}{\lambda z} x x_0} dx_0
    """
    cdef:
        int a = x_arr.shape[0], i
        complex_t[::1] lens_wf = np.empty((a,), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True, chunksize=10):
        lens_wf[i] = lens_wp_c(x_arr[i], wl, ap, focus, defoc, alpha, xc) 
    return np.asarray(lens_wf)

cdef float_t aperture_re(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1], wl = (<float_t*> params)[2]
    return cos(pi / wl / z * (x - xx)**2)

cdef float_t aperture_im(float_t xx, void* params) nogil:
    cdef:
        float_t x = (<float_t*> params)[0], z = (<float_t*> params)[1], wl = (<float_t*> params)[2]
    return sin(pi / wl / z * (x - xx)**2)
    
cdef complex_t aperture_wp_c(float_t x, float_t z, float_t wl, float_t ap) nogil:
    cdef:
        float_t re, im
        float_t params[3]
        int fn = <int> (ap**2 / wl / z)
        gsl_function func
    params[0] = x; params[1] = z; params[2] = wl
    func.function = &aperture_re; func.params = params
    re = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    func.function = &aperture_im
    im = gsl_quad(func, -ap / 2, ap / 2, 1e-9, 1e-7, 1000 * fn)
    return re + 1j * im

def aperture_wp(float_t[::1] x_arr, float_t z, float_t wl, float_t ap):
    r"""Calculate beam's wavefront propagation from the hard-edged
    aperture to the sample. The sample's plane is `z` downstream
    from the aperture's plane.

    Parameters
    ----------
    x_arr : numpy.ndarray
        Coordinates at the sample's plane [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].
    ap : float
        Aperture's size [um].

    Returns
    -------
    numpy.ndarray
        Beam's wavefront at the sample's plane.

    Notes
    -----
    The exit surface at the aperture's plane:

    .. math::
        U_0 = \Pi(a_x x_0)
    
    Wavefront :math:`U_0` propagates to the sample's plane
    which is :math:`z` downstream from the lens. According to
    the Fresnel diffraction theory (without the normalizing
    coefficient before the integral):

    .. math::
        U(x) = \int_{-a_x / 2}^{a_x / 2} e^{-\frac{jk x_0^2}{2z}}
        e^{j\frac{2 \pi}{\lambda z} x x_0} dx_0
    """
    cdef:
        int a = x_arr.shape[0], i
        complex_t[::1] ap_wf = np.empty((a,), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True, chunksize=10):
        ap_wf[i] = aperture_wp_c(x_arr[i], z, wl, ap)
    return np.asarray(ap_wf)

cdef complex_t fhf_wp(complex_t[::1] wf0, float_t[::1] x_arr, float_t xx, float_t z, float_t wl) nogil:
    cdef:
        int a = wf0.shape[0], i
        float_t ph0, ph1, ph = pi / wl / z * xx**2
        complex_t wf = 0 + 0j
    ph0 = 2 * pi / wl / z * x_arr[0] * xx
    ph1 = 2 * pi / wl / z * x_arr[1] * xx
    wf = (wf0[0] * (cos(ph0) - 1j * sin(ph0)) + wf0[1] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[1] - x_arr[0])
    for i in range(2, a):
        ph0 = ph1
        ph1 = 2 * pi / wl / z * x_arr[i] * xx
        wf += (wf0[i - 1] * (cos(ph0) - 1j * sin(ph0)) + wf0[i] * (cos(ph1) - 1j * sin(ph1))) / 2 * (x_arr[i] - x_arr[i - 1])
    return wf * (cos(ph) + 1j * sin(ph))

cdef void fhf_1d(complex_t[::1] wf1, complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t z, float_t wl) nogil:
    cdef:
        int a = xx_arr.shape[0], i
    for i in range(a):
        wf1[i] = fhf_wp(wf0, x_arr, xx_arr[i], z, wl)

def fraunhofer_1d(complex_t[::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t z, float_t wl):
    r"""One-dimensional discrete form of Fraunhofer diffraction
    integral transform.

    Parameters
    ----------
    wf0 : numpy.ndarray
        Wavefront at the plane upstream.
    x_arr : numpy.ndarray
        Coordinates at the plane upstream [um].
    xx_arr : numpy.ndarray
        Coordinates at the plane downstream [um].
    z : float
        Distance between the planes [um].
    wl : float
        Beam's wavelength [um].

    Returns
    -------
    numpy.ndarray
        Wavefront at the plane downstream.

    Notes
    -----
    The Fraunhofer integral transform is defined as (without the
    normalizing coefficient before the integral):

    .. math::
        U(x) = e^{-\frac{j k x^2}{2 z}} \int_{-\infty}^{+\infty}
        U_0(x_0) e^{j\frac{2 \pi}{\lambda z} x x_0} dx_0
    """
    cdef:
        int a = xx_arr.shape[0]
        complex_t[::1] wf = np.empty(a, dtype=np.complex128)
    fhf_1d(wf, wf0, x_arr, xx_arr, z, wl)
    return np.asarray(wf)

def fraunhofer_1d_scan(complex_t[:, ::1] wf0, float_t[::1] x_arr, float_t[::1] xx_arr, float_t z, float_t wl):
    """One-dimensional discrete form of Fraunhofer diffraction
    integral transform applied to an array of wavefronts `wf0`.

    Parameters
    ----------
    wf0 : numpy.ndarray
        Array of wavefronts at the plane upstream.
    x_arr : numpy.ndarray
        Coordinates at the plane upstream [um].
    xx_arr : numpy.ndarray
        Coordinates at the plane downstream [um].
    z : float
        Distance between the planes [um].
    wl : float
        Beam's wavelength [um].

    Returns
    -------
    numpy.ndarray
        Array of wavefronts at the plane downstream.

    See Also
    --------
    fraunhofer_1d : Description of the Fraunhofer diffraction
        integral transform.
    """
    cdef:
        int a = wf0.shape[0], b = xx_arr.shape[0], i
        complex_t[:, ::1] wf = np.empty((a, b), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True):
        fhf_1d(wf[i], wf0[i], x_arr, xx_arr, z, wl)
    return np.asarray(wf)

def barcode_steps(float_t x0, float_t x1, float_t br_dx, float_t rd):
    """Generate a coordinate array of barcode's bar positions.

    Parameters
    ----------
    x0 : float
        Barcode's lower bound along the x axis [um].
    x1 : float
        Barcode's upper bound along the x axis [um].
    br_dx : float
        Average bar's size [um].
    rd : float
        Random deviation of barcode's bar positions (0.0 - 1.0).

    Returns
    -------
    bx_arr : numpy.ndarray
        Array of barcode's bar coordinates.
    """
    cdef:
        int br_n = <int>((x1 - x0) / 2 / br_dx) * 2 if x1 - x0 > 0 else 0, i
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        float_t bs_min = max(1 - rd, 0), bs_max = min(1 + rd, 2)
        float_t[::1] bx_arr = np.empty(br_n, dtype=np.float64)
        timespec ts
    clock_gettime(CLOCK_REALTIME, &ts)
    gsl_rng_set(r, ts.tv_sec + ts.tv_nsec)
    if br_n:
        bx_arr[0] = x0 + br_dx * ((bs_max - bs_min) * gsl_rng_uniform_pos(r) - 1)
        for i in range(1, br_n):
            bx_arr[i] = bx_arr[i - 1] + br_dx * (bs_min + (bs_max - bs_min) * gsl_rng_uniform_pos(r))
    gsl_rng_free(r)
    return np.asarray(bx_arr)

cdef int binary_search(float_t[::1] values, int l, int r, float_t x) nogil:
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

cdef int searchsorted(float_t[::1] values, float_t x) nogil:
    cdef int r = values.shape[0]
    if x < values[0]:
        return 0
    elif x > values[r - 1]:
        return r
    else:
        return binary_search(values, 0, r, x)

cdef void barcode_c(float_t[::1] br_tr, float_t[::1] x_arr, float_t[::1] bx_arr,
                     float_t sgm, float_t atn0, float_t atn, float_t step) nogil:
    cdef:
        int a = x_arr.shape[0], b = bx_arr.shape[0], i, j0, j
        float_t br_dx = (bx_arr[b - 1] - bx_arr[0]) / b
        int bb = <int>(X_TOL * sqrt(2) * sgm / br_dx + 1)
        float_t tr, xx, x0, x1
    for i in range(a):
        xx = x_arr[i] + step
        j0 = searchsorted(bx_arr, xx) # even '-', odd '+'
        tr = 0
        for j in range(j0 - bb, j0 + bb + 1):
            if j >= 1 and j < b:
                x0 = (xx - bx_arr[j - 1]) / sqrt(2) / sgm
                x1 = (xx - bx_arr[j]) / sqrt(2) / sgm
                tr += atn * 0.5 * (0.5 - j % 2) * (erf(x0) - erf(x1))
        tr -= (0.25 * atn + 0.5 * atn0) * erf((xx - bx_arr[0]) / sqrt(2 + 2 * (atn0 / atn)**2) / sgm)
        tr += (0.25 * atn + 0.5 * atn0) * erf((xx - bx_arr[b - 1]) / sqrt(2 + 2 * (atn0 / atn)**2) / sgm)
        br_tr[i] = sqrt(1 + tr)
        
def barcode_profile(float_t[::1] x_arr, float_t[::1] bx_arr, float_t sgm,
                    float_t atn0, float_t atn, float_t ss, int nf):
    r"""Return an array of barcode's transmission profile.
    Barcode is scanned accross the x axis with `ss` step size
    and `nf` number of steps.

    Parameters
    ----------
    x_arr : numpy.ndarray
        Coordinates at the sample's plane [um].
    bx_arr : numpy.ndarray
        Coordinates of barcode's bar positions [um].    
    sgm : float
        Bar's blurriness [um].
    atn0 : float
        Barcode's bulk attenuation coefficient (0.0 - 1.0).
    atn : float
        Barcode's bar attenuation coefficient (0.0 - 1.0).
    ss : float
        Scan's step size [um].
    nf : int
        Scan's number of frames.
    
    Returns
    -------
    br_tr : numpy.ndarray
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
        int a = x_arr.shape[0], i
        float_t[:, ::1] br_tr = np.empty((nf, a), dtype=np.float64)
    for i in prange(nf, schedule='guided', nogil=True):
        barcode_c(br_tr[i], x_arr, bx_arr, sgm, atn0, atn, i * ss)
    return np.asarray(br_tr)

cdef float_t convolve_c(float_t[::1] a1, float_t[::1] a2, int k) nogil:
    cdef:
        int a = a1.shape[0], b = a2.shape[0]
        int i0 = max(k - b // 2, 0), i1 = min(k - b//2 + b, a), i
        float_t x = 0
    for i in range(i0, i1):
        x += a1[i] * a2[k + b//2 - i]
    return x

cdef void make_frame_c(uint_t[:, ::1] frame, float_t[::1] i_x, float_t[::1] i_y,
                       float_t[::1] sc, float_t pix_size, unsigned long seed) nogil:
    cdef:
        int b = i_y.shape[0], c = i_x.shape[0], j, k
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        float_t i_xs
    gsl_rng_set(r, seed)
    for k in range(c):
        i_xs = convolve_c(i_x, sc, k)
        for j in range(b):
            frame[j, k] = gsl_ran_poisson(r, i_xs * i_y[j] * pix_size**2)
    gsl_rng_free(r)

def make_frames(float_t[:, ::1] i_x, float_t[::1] i_y, float_t[::1] sc_x, float_t[::1] sc_y, float_t pix_size):
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
    sc_x : numpy.ndarray
        Source's rocking curve along the x axis.
    sc_y : numpy.ndarray
        Source's rocking curve along the y axis.
    pix_size : float
        Pixel's size [um].

    Returns
    -------
    frames : numpy.ndarray
        Intensity frames.
    """
    cdef:
        int a = i_x.shape[0], b = i_y.shape[0], c = i_x.shape[1], i
        uint_t[:, :, ::1] frames = np.empty((a, b, c), dtype=np.uint64)
        float_t[::1] i_ys = np.empty(b, dtype=np.float64)
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        timespec ts
        unsigned long seed
    clock_gettime(CLOCK_REALTIME, &ts)
    gsl_rng_set(r, ts.tv_sec + ts.tv_nsec)
    for i in range(b):
        i_ys[i] = convolve_c(i_y, sc_y, i)
    for i in prange(a, schedule='guided', nogil=True):
        seed = gsl_rng_get(r)
        make_frame_c(frames[i], i_x[i], i_ys, sc_x, pix_size, seed)
    gsl_rng_free(r)
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
