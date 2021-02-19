#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cimport numpy as np
import numpy as np
import speckle_tracking as st
from pyrost.bin import make_reference, mse_total, update_pixel_map_gs
from cython.parallel import prange
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
cimport openmp
from libc.math cimport sqrt, exp, pi, erf, floor, ceil
from libc.time cimport time, time_t
from libc.string cimport memcpy

cdef extern from "pyrost/bin/fft.c":

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

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.complex128_t complex_t
ctypedef np.npy_bool bool_t
ctypedef np.uint64_t uint_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF X_TOL = 4.320005384913445 # Y_TOL = 1e-9
DEF NO_VAR = -1.0  

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

def rfft_python(double[::1] arr):
    cdef:
        int n = arr.shape[0], fail = 0
        complex_t[::1] res = np.empty(n // 2 + 1, dtype=np.complex128)
    fail = rfft(&res[0], &arr[0], n)
    if fail:
        raise RuntimeError('FFT failed')
    return np.asarray(res)

def fft_python(complex_t[::1] arr):
    cdef:
        int n = arr.shape[0], fail = 0
        complex_t[::1] res = np.empty(n, dtype=np.complex128)
    res[...] = arr
    fail = fft(&res[0], n)
    if fail:
        raise RuntimeError('RFFT failed')
    return np.asarray(res)

def ifft_python(complex_t[::1] arr):
    cdef:
        int n = arr.shape[0], fail = 0
        complex_t[::1] res = np.empty(n, dtype=np.complex128)
    res[...] = arr
    fail = ifft(&res[0], n)
    if fail:
        raise RuntimeError('IFFT failed')
    return np.asarray(res)

def irfft_python(complex_t[::1] arr):
    cdef:
        int n = arr.shape[0], fail = 0
        double[::1] res = np.empty(2 * (n - 1), dtype=np.float64)
    fail = irfft(&res[0], &arr[0], 2 * (n - 1))
    if fail:
        raise RuntimeError('IRFFT failed')
    return np.asarray(res)

def st_update(I_n, dij, basis, x_ps, y_ps, z, df, sw_max=100, n_iter=5, filter=None):
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
    W = st.make_whitefield(I_n, M)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis,
                                            x_ps, y_ps, z,
                                            df, verbose=False)
    I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=False, verbose=False)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=False)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                                search_window=[1, sw_max], subpixel=True,
                                fill_bad_pix=True, integrate=False,
                                quadratic_refinement=True, verbose=False,
                                filter=filter)[0]

        # make reference image
        I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=False)

        # update translations
        dij_pix = st.update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]
    return {'u':u, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

def pixel_translations(basis, dij, df, z):
    dij_pix = (basis * dij[:, None]).sum(axis=-1)
    dij_pix /= (basis**2).sum(axis=-1) * df / z
    dij_pix -= dij_pix.mean(axis=0)
    return np.ascontiguousarray(dij_pix[:, 0]), np.ascontiguousarray(dij_pix[:, 1])

def str_update(I_n, W, dij, basis, x_ps, y_ps, z, df, sw_max=100, n_iter=5, l_scale=2.5):
    """
    Robust version of Andrew's speckle tracking update algorithm
    
    I_n - measured data
    W - whitefield
    basis - detector plane basis vectors
    x_ps, y_ps - x and y pixel sizes
    z - distance between the sample and the detector
    df - defocus distance
    sw_max - pixel mapping search window size
    n_iter - number of iterations
    """
    I_n = I_n.astype(np.float64)
    W = W.astype(np.float64)
    u0 = np.indices(W.shape, dtype=np.float64)
    di, dj = pixel_translations(basis, dij, df, z)
    I0, n0, m0 = make_reference(I_n=I_n, W=W, u=u0, di=di, dj=dj, ls=l_scale, sw_fs=0, sw_ss=0)

    es = []
    for i in range(n_iter):

        # calculate errors
        es.append(mse_total(I_n=I_n, W=W, I0=I0, u=u0, di=di - n0, dj=dj - m0, ls=l_scale))

        # update pixel map
        u = update_pixel_map_gs(I_n=I_n, W=W, I0=I0, u0=u0, di=di - n0, dj=dj - m0,
                                sw_ss=0, sw_fs=sw_max, ls=l_scale)
        sw_max = int(np.max(np.abs(u - u0)))
        u0 = u0 + gaussian_filter(u - u0, (0, 0, l_scale))

        # make reference image
        I0, n0, m0 = make_reference(I_n=I_n, W=W, u=u0, di=di, dj=dj, ls=l_scale, sw_ss=0, sw_fs=0)
        I0 = gaussian_filter(I0, (0, l_scale))
    return {'u':u0, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

def phase_fit(pixel_ab, x_ps, z, df, wl, max_order=2, pixels=None):
    def errors(fit, x, y):
        return np.polyval(fit[:max_order + 1], x - fit[max_order + 1]) - y

    # Apply ROI
    if pixels is None:
        pixels = np.arange(pixel_ab.shape[0])
    else:
        pixel_ab = pixel_ab[pixels]
    x_arr = pixels * x_ps / z * 1e3

    # Calculate initial argument
    x0 = np.zeros(max_order + 2)
    u0 = gaussian_filter(pixel_ab, pixel_ab.shape[0] / 10)
    if np.median(np.gradient(np.gradient(u0))) > 0:
        idx = np.argmin(u0)
    else:
        idx = np.argmax(u0)
    x0[max_order + 1] = x_arr[idx]
    lb = -np.inf * np.ones(max_order + 2)
    ub = np.inf * np.ones(max_order + 2)
    lb[max_order + 1] = x_arr.min()
    ub[max_order + 1] = x_arr.max()
        
    # Perform least squares fitting
    fit = least_squares(errors, x0, bounds=(lb, ub), loss='cauchy', jac='3-point',
                        args=(x_arr, pixel_ab), xtol=1e-14, ftol=1e-14)
    if np.linalg.det(fit.jac.T.dot(fit.jac)):
        cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
        err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
    else:
        err = np.zeros(fit.x.size)

    # Convert the fit
    ang_fit = fit.x * x_ps / z
    ang_fit[:max_order + 1] /= np.geomspace((x_ps / z)**max_order, 1, max_order + 1)
    ph_fit = np.zeros(max_order + 3)
    ph_fit[:max_order + 1] = ang_fit[:max_order + 1] * 2 * np.pi / wl * df / np.linspace(max_order + 1, 1, max_order + 1)
    ph_fit[max_order + 2] = ang_fit[max_order + 1]
    ph_fit[max_order + 1] = -np.polyval(ph_fit[:max_order + 2], pixels * x_ps / z - ph_fit[max_order + 2]).mean()

    # evaluating errors
    r_sq = 1 - np.sum(errors(fit.x, pixels, pixel_ab)**2) / np.sum((pixel_ab.mean() - pixel_ab)**2)
    return {'pixels': pixels, 'pix_fit': fit.x, 'ang_fit': ang_fit,
            'pix_err': err, 'ph_fit': ph_fit, 'r_sq': r_sq, 'fit': fit}

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

cdef double barcode_c(double[::1] bar_pos, double xx, double b_atn, double b_sgm, double blk_atn) nogil:
    cdef:
        int b = bar_pos.shape[0], i, j0, j
        double b_dx = (bar_pos[b - 1] - bar_pos[0]) / b
        int bb = <int>(X_TOL * sqrt(2) * b_sgm / b_dx + 1)
        double tr, x0, x1
    j0 = searchsorted(bar_pos, xx) # even '-', odd '+'
    tr = 0
    for j in range(j0 - bb, j0 + bb + 1):
        if j > 0 and j < b - 1:
            x0 = (xx - bar_pos[j - 1]) / sqrt(2) / b_sgm
            x1 = (xx - bar_pos[j]) / sqrt(2) / b_sgm
            tr += b_atn * (bar_pos[j] - bar_pos[j - 1]) / b_dx * 0.5 * (0.5 - j % 2) * (erf(x0) - erf(x1))
    tr -= (0.25 * b_atn + 0.5 * blk_atn) * erf((xx - bar_pos[0]) / sqrt(2 + 2 * (blk_atn / b_atn)**2) / b_sgm)
    tr += (0.25 * b_atn + 0.5 * blk_atn) * erf((xx - bar_pos[b - 1]) / sqrt(2 + 2 * (blk_atn / b_atn)**2) / b_sgm)
    return sqrt(1 + tr)
        
def barcode_profile(double[::1] bar_pos, double[::1] x_arr, double b_atn, double b_sgm, double blk_atn):
    cdef:
        int a = x_arr.shape[0], i
        double[::1] b_tr = np.empty(a, dtype=np.float64)
    for i in range(a):
        b_tr[i] = barcode_c(bar_pos, x_arr[i], b_atn, b_sgm, blk_atn)
    return np.asarray(b_tr)

cdef float_t rbf(float_t dsq, float_t ls) nogil:
    return exp(-dsq / 2 / ls**2) / sqrt(2 * pi)

cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                 float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
        int i, ss0, ss1, fs0, fs1
        float_t SS_res = 0, SS_tot = 0, ss, fs, dss, dfs, I0_bi
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss <= 0:
            dss = 0; ss0 = 0; ss1 = 0
        elif ss >= aa - 1:
            dss = 0; ss0 = aa - 1; ss1 = aa - 1
        else:
            dss = ss - floor(ss)
            ss0 = <int>(floor(ss)); ss1 = ss0 + 1
        if fs <= 0:
            dfs = 0; fs0 = 0; fs1 = 0
        elif fs >= bb - 1:
            dfs = 0; fs0 = bb - 1; fs1 = bb - 1
        else:
            dfs = fs - floor(fs)
            fs0 = <int>(floor(fs)); fs1 = fs0 + 1
        I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                (1 - dss) * dfs * I0[ss0, fs1] + \
                dss * (1 - dfs) * I0[ss1, fs0] + \
                dss * dfs * I0[ss1, fs1]
        SS_res += (I[i] - I0_bi)**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res; m_ptr[1] = SS_tot
    if m_ptr[2] >= 0:
        m_ptr[2] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef void krig_data_c(float_t[::1] I, float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, :, ::1] u,
                      int j, int k, float_t ls) nogil:
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, jj, kk
        int djk = <int>(ceil(2 * ls))
        int jj0 = j - djk if j - djk > 0 else 0
        int jj1 = j + djk if j + djk < b else b
        int kk0 = k - djk if k - djk > 0 else 0
        int kk1 = k + djk if k + djk < c else c
        float_t w0 = 0, rss = 0, r
    for i in range(a + 1):
        I[i] = 0
    for jj in range(jj0, jj1):
        for kk in range(kk0, kk1):
            r = rbf((u[0, jj, kk] - u[0, j, k])**2 + (u[1, jj, kk] - u[1, j, k])**2, ls)
            w0 += r * W[jj, kk]**2
            rss += W[jj, kk]**3 * r**2
            for i in range(a):
                I[i] += I_n[i, jj, kk] * W[jj, kk] * r
    if w0:
        for i in range(a):
            I[i] /= w0
        I[a] = rss / w0**2

def krig_data(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, :, ::1] u,
              int j, int k, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, jj, kk
        float_t[::1] I = np.zeros(a + 1, dtype=dtype)
        int djk = <int>(ceil(2 * ls))
        int jj0 = j - djk if j - djk > 0 else 0
        int jj1 = j + djk if j + djk < b else b
        int kk0 = k - djk if k - djk > 0 else 0
        int kk1 = k + djk if k + djk < c else c
        float_t w0 = 0, rss = 0, r
    for jj in range(jj0, jj1):
        for kk in range(kk0, kk1):
            r = rbf((u[0, jj, kk] - u[0, j, k])**2 + (u[1, jj, kk] - u[1, j, k])**2, ls)
            w0 += r * W[jj, kk]**2
            rss += W[jj, kk]**3 * r**2
            for i in range(a):
                I[i] += I_n[i, jj, kk] * W[jj, kk] * r
    if w0:
        for i in range(a):
            I[i] /= w0
        I[a] = rss / w0**2
    return np.asarray(I)

# cdef void mse_diff_bi(float_t* m_ptr, float_t[:, :, ::1] SS_m, float_t[:, ::1] I,
#                       float_t[:, ::1] rss, float_t[:, ::1] I0, float_t[:, :, ::1] u,
#                       float_t di0, float_t dj0, float_t di, float_t dj) nogil:
#     cdef:
#         int b = I.shape[0], c = I.shape[1], j, k
#         int ss_0, fs_0, ss_1, fs_1
#         int aa = I0.shape[0], bb = I0.shape[1]
#         float_t ss0, fs0, ss1, fs1, dss, dfs
#         float_t mse = 0, mse_var = 0, I0_bi, res_0, tot_0, res, tot, SS_res, SS_tot
#     for j in range(b):
#         for k in range(c):
#             ss0 = u[0, j, k] - di0; fs0 = u[1, j, k] - dj0
#             ss1 = u[0, j, k] - di; fs1 = u[1, j, k] - dj
#             if ss0 <= 0:
#                 dss = 0; ss_0 = 0; ss_1 = 0
#             elif ss0 >= aa - 1:
#                 dss = 0; ss_0 = aa - 1; ss_1 = aa - 1
#             else:
#                 dss = ss0 - floor(ss0)
#                 ss_0 = <int>(floor(ss0)); ss_1 = ss_0 + 1
#             if fs0 <= 0:
#                 dfs = 0; fs_0 = 0; fs_1 = 0
#             elif fs0 >= bb - 1:
#                 dfs = 0; fs_0 = bb - 1; fs_1 = bb - 1
#             else:
#                 dfs = fs0 - floor(fs0)
#                 fs_0 = <int>(floor(fs0)); fs_1 = fs_0 + 1
#             I0_bi = (1 - dss) * (1 - dfs) * I0[ss_0, fs_0] + \
#                     (1 - dss) * dfs * I0[ss_0, fs_1] + \
#                     dss * (1 - dfs) * I0[ss_1, fs_0] + \
#                     dss * dfs * I0[ss_1, fs_1]
#             res_0 = (I[j, k] - I0_bi)**2
#             tot_0 = (I[j, k] - 1)**2

#             if ss1 <= 0:
#                 dss = 0; ss_0 = 0; ss_1 = 0
#             elif ss1 >= aa - 1:
#                 dss = 0; ss_0 = aa - 1; ss_1 = aa - 1
#             else:
#                 dss = ss1 - floor(ss1)
#                 ss_0 = <int>(floor(ss1)); ss_1 = ss_0 + 1
#             if fs1 <= 0:
#                 dfs = 0; fs_0 = 0; fs_1 = 0
#             elif fs1 >= bb - 1:
#                 dfs = 0; fs_0 = bb - 1; fs_1 = bb - 1
#             else:
#                 dfs = fs1 - floor(fs1)
#                 fs_0 = <int>(floor(fs1)); fs_1 = fs_0 + 1
#             I0_bi = (1 - dss) * (1 - dfs) * I0[ss_0, fs_0] + \
#                     (1 - dss) * dfs * I0[ss_0, fs_1] + \
#                     dss * (1 - dfs) * I0[ss_1, fs_0] + \
#                     dss * dfs * I0[ss_1, fs_1]
#             res = (I[j, k] - I0_bi)**2
#             tot = (I[j, k] - 1)**2

#             SS_res = SS_m[0, j, k] - res_0 + res; SS_tot = SS_m[1, j, k] - tot_0 + tot
#             mse += SS_res / SS_tot / b / c
#             mse_var += 4 * rss[j, k] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3) / b**2 / c**2
#     m_ptr[0] = mse; m_ptr[1] = mse_var

# cdef void ut_surface_c(float_t[:, ::1] mse_m, float_t[:, ::1] mse_var, float_t[:, :, ::1] SS_m,
#                        float_t[:, ::1] I, float_t[:, ::1] rss, float_t[:, ::1] I0, float_t[:, :, ::1] u,
#                        float_t di, float_t dj, int sw_ss, int sw_fs) nogil:
#     cdef:
#         int ii, jj
#         float_t m_ptr[2]
#     for ii in range(-sw_ss, sw_ss + 1):
#         for jj in range(-sw_fs, sw_fs + 1):
#             mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, di, dj, di + ii, dj + jj)
#             mse_m[ii + sw_ss, jj + sw_fs] = m_ptr[0]
#             mse_var[ii + sw_ss, jj + sw_fs] = m_ptr[1]

# def ut_surface(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
#                float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, int sw_ss, int sw_fs,
#                float_t ls):
#     dtype = np.float64 if float_t is np.float64_t else np.float32
#     cdef:
#         int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j, k, t
#         int max_threads = openmp.omp_get_max_threads()
#         float_t[:, :, ::1] I = np.empty((b, c, a + 1), dtype=dtype)
#         float_t[:, :, ::1] I_buf = np.empty((max_threads + 1, b, c), dtype=dtype)
#         float_t[:, :, ::1] SS_m = np.empty((2, b, c), dtype=dtype)
#         float_t[:, :, ::1] mse_m = np.empty((a, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
#         float_t[:, :, ::1] mse_var = np.empty((a, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
#         float_t m_ptr[3]
#     m_ptr[2] = NO_VAR
#     for k in prange(c, schedule='guided', nogil=True):
#         for j in range(b):
#             krig_data_c(I[j, k], I_n, W, u, j, k, ls)
#             mse_bi(m_ptr, I[j, k], I0, di, dj, u[0, j, k], u[1, j, k])
#             SS_m[0, j, k] = m_ptr[0]; SS_m[1, j, k] = m_ptr[1]
#     I_buf[max_threads] = I[:, :, a]
#     for i in prange(a, schedule='guided', nogil=True):
#         t = openmp.omp_get_thread_num()
#         I_buf[t] = I[:, :, i]
#         ut_surface_c(mse_m[i], mse_var[i], SS_m, I_buf[t], I_buf[max_threads],
#                      I0, u, di[i], dj[i], sw_ss, sw_fs)
#     return np.asarray(mse_m), np.asarray(mse_var)

cdef void upm_surface_c(float_t[:, ::1] mse_m, float_t[:, ::1] mse_var, float_t[::1] I, float_t[:, ::1] I0,
                   float_t[::1] di, float_t[::1] dj, float_t u_ss, float_t u_fs, int sw_ss, int sw_fs) nogil:
    cdef:
        int ss, fs
        float_t mv_ptr[3]
    for ss in range(-sw_ss, sw_ss + 1):
        for fs in range(-sw_fs, sw_fs + 1):
            mse_bi(mv_ptr, I, I0, di, dj, u_ss + ss, u_fs + fs)
            mse_m[ss + sw_ss, fs + sw_fs] = mv_ptr[0] / mv_ptr[1]
            mse_var[ss + sw_ss, fs + sw_fs] = mv_ptr[2]

def upm_surface(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
           float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj,
           int sw_ss, int sw_fs, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
        float_t[:, :, :, ::1] mse_m = np.empty((b, c, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
        float_t[:, :, :, ::1] mse_var = np.empty((b, c, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u, j, k, ls)
            upm_surface_c(mse_m[j, k], mse_var[j, k], I[t], I0, di, dj, u[0, j, k], u[1, j, k], sw_ss, sw_fs)
    return np.asarray(mse_m), np.asarray(mse_var)

# cdef void subpixel_ref_2d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
#     cdef:
#         float_t dss = 0, dfs = 0, det, dd
#         float_t f00, f01, f10, f11, f12, f21, f22
#     f00 = mse_m[0]; f01 = mse_m[1]; f10 = mse_m[2]
#     f11 = mse_m[3]; f12 = mse_m[4]; f21 = mse_m[5]
#     f22 = mse_m[6]

#     det = 4 * (f21 + f01 - 2 * f11) * (f12 + f10 - 2 * f11) - \
#           (f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12)**2
#     if det != 0:
#         dss = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f12 - f10) - \
#                2 * (f12 + f10 - 2 * f11) * (f21 - f01)) / det * mu / 2
#         dfs = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f21 - f01) - \
#                2 * (f21 + f01 - 2 * f11) * (f12 - f10)) / det * mu / 2
#         dd = sqrt(dfs**2 + dss**2)
#         if dd > 1:
#             dss /= dd; dfs /= dd
    
#     x[0] += dss; x[1] += dfs

# cdef void subpixel_ref_1d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
#     cdef:
#         float_t dfs = 0, det, dd
#         float_t f10, f11, f12
#     f10 = mse_m[0]; f11 = mse_m[1]; f12 = mse_m[2]

#     det = 4 * (f12 + f10 - 2 * f11)
#     if det != 0:
#         dfs = (f10 - f12) / det * mu
#         dd = sqrt(dfs**2)
#         if dd > 1:
#             dfs /= dd

#     x[1] += dfs

# cdef void update_pm_c(float_t[::1] I, float_t[:, ::1] I0, float_t[::1] u,
#                        float_t[::1] di, float_t[::1] dj, int sw_ss, int sw_fs) nogil:
#     cdef:
#         int ss_min = -sw_ss, fs_min = -sw_fs, ss_max = -sw_ss, fs_max = -sw_fs, ss, fs
#         float_t mse_min = FLOAT_MAX, mse_max = -FLOAT_MAX, mse, mse_var, l1, mu
#         float_t mv_ptr[3]
#         float_t mse_m[7]
#     for ss in range(-sw_ss, sw_ss + 1):
#         for fs in range(-sw_fs, sw_fs + 1):
#             mse_bi(mv_ptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
#             mse = mv_ptr[0] / mv_ptr[1]
#             if mse < mse_min:
#                 mse_min = mse; mse_var = mv_ptr[2]; ss_min = ss; fs_min = fs; 
#             if mse > mse_max:
#                 mse_max = mse; ss_max = ss; fs_max = fs
#     u[0] += ss_min; u[1] += fs_min
#     l1 = 2 * (mse_max - mse_min) / ((ss_max - ss_min)**2 + (fs_max - fs_min)**2)
#     mu = (3 * mse_var**0.5 / l1)**0.33
#     mu = mu if mu > 2 else 2
#     if sw_ss:
#         mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1] - mu / 2)
#         mse_m[0] = mv_ptr[0] / mv_ptr[1]
#         mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1])
#         mse_m[1] = mv_ptr[0] / mv_ptr[1]
#         mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
#         mse_m[2] = mv_ptr[0] / mv_ptr[1]
#         mse_m[3] = mse_min
#         mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
#         mse_m[4] = mv_ptr[0] / mv_ptr[1]
#         mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1])
#         mse_m[5] = mv_ptr[0] / mv_ptr[1]
#         mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1] + mu / 2)
#         mse_m[6] = mv_ptr[0] / mv_ptr[1]
#         subpixel_ref_2d(u, mse_m, mu)
#     else:
#         mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
#         mse_m[0] = mv_ptr[0] / mv_ptr[1]
#         mse_m[1] = mse_min
#         mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
#         mse_m[2] = mv_ptr[0] / mv_ptr[1]
#         subpixel_ref_1d(u, mse_m, mu)

# def update_pixel_map_gs(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
#                         float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
#                         int sw_ss, int sw_fs, float_t ls):
#     dtype = np.float64 if float_t is np.float64_t else np.float32
#     cdef:
#         int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
#         int aa = I0.shape[0], bb = I0.shape[1], j, k, t
#         int max_threads = openmp.omp_get_max_threads()
#         float_t[::1, :, :] u = np.empty((2, b, c), dtype=dtype, order='F')
#         float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
#     for k in prange(c, schedule='guided', nogil=True):
#         t = openmp.omp_get_thread_num()
#         for j in range(b):
#             krig_data_c(I[t], I_n, W, u0, j, k, ls)
#             u[:, j, k] = u0[:, j, k]
#             update_pm_c(I[t], I0, u[:, j, k], di, dj, sw_ss, sw_fs)
#     return np.asarray(u, order='C')

# cdef void update_t_c(float_t[:, :, ::1] SS_m, float_t[:, ::1] I, float_t[:, ::1] rss, float_t[:, ::1] I0,
#                      float_t[:, :, ::1] u, float_t[::1] dij, int sw_ss, int sw_fs) nogil:
#     cdef:
#         int ii, jj
#         int ss_min = -sw_ss, fs_min = -sw_fs, ss_max = -sw_ss, fs_max = -sw_fs
#         float_t mse_min = FLOAT_MAX, mse_var = FLOAT_MAX, mse_max = -FLOAT_MAX, l1, mu
#         float_t m_ptr[2]
#         float_t mse_m[7]
#     for ii in range(-sw_ss, sw_ss + 1):
#         for jj in range(-sw_fs, sw_fs + 1):
#             mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + ii, dij[1] + jj)
#             if m_ptr[0] < mse_min:
#                 mse_min = m_ptr[0]; mse_var = m_ptr[1]; ss_min = ii; fs_min = jj
#             if m_ptr[0] > mse_max:
#                 mse_max = m_ptr[0]; ss_max = ii; fs_max = jj
#     dij[0] += ss_min; dij[1] += fs_min
#     l1 = 2 * (mse_max - mse_min) / ((ss_max - ss_min)**2 + (fs_max - fs_min)**2)
#     mu = (3 * mse_var**0.5 / l1)**0.33
#     mu = mu if mu > 2 else 2
#     if sw_ss:
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] - mu / 2, dij[1] - mu / 2)
#         mse_m[0] = m_ptr[0]
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] - mu / 2, dij[1])
#         mse_m[1] = m_ptr[0]
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] - mu / 2)
#         mse_m[2] = m_ptr[0]
#         mse_m[3] = mse_min
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] + mu / 2)
#         mse_m[4] = m_ptr[0]
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + mu / 2, dij[1])
#         mse_m[5] = m_ptr[0]
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + mu / 2, dij[1] + mu / 2)
#         mse_m[6] = m_ptr[0]
#         subpixel_ref_2d(dij, mse_m, mu)
#     else:
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] - mu / 2)
#         mse_m[0] = m_ptr[0]
#         mse_m[1] = mse_min
#         mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] + mu / 2)
#         mse_m[2] = m_ptr[0]
#         subpixel_ref_1d(dij, mse_m, mu)

# def update_translations_gs(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
#                            float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj,
#                            int sw_ss, int sw_fs, float_t ls):
#     dtype = np.float64 if float_t is np.float64_t else np.float32
#     cdef:
#         int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j, k, t
#         int max_threads = openmp.omp_get_max_threads()
#         float_t[:, :, ::1] I = np.empty((b, c, a + 1), dtype=dtype)
#         float_t[:, :, ::1] I_buf = np.empty((max_threads + 1, b, c), dtype=dtype)
#         float_t[:, :, ::1] SS_m = np.empty((3, b, c), dtype=dtype)
#         float_t[:, ::1] dij = np.empty((a, 2), dtype=dtype)
#         float_t m_ptr[3]
#     m_ptr[2] = NO_VAR
#     for k in prange(c, schedule='guided', nogil=True):
#         for j in range(b):
#             krig_data_c(I[j, k], I_n, W, u, j, k, ls)
#             mse_bi(m_ptr, I[j, k], I0, di, dj, u[0, j, k], u[1, j, k])
#             SS_m[0, j, k] = m_ptr[0]; SS_m[1, j, k] = m_ptr[1]
#     I_buf[max_threads] = I[:, :, a]
#     for i in prange(a, schedule='guided', nogil=True):
#         t = openmp.omp_get_thread_num()
#         I_buf[t] = I[:, :, i]; dij[i, 0] = di[i]; dij[i, 1] = dj[i]
#         update_t_c(SS_m, I_buf[t], I_buf[max_threads], I0, u, dij[i], sw_ss, sw_fs)
#     return np.asarray(dij)
