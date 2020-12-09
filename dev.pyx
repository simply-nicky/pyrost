#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
cimport openmp
from libc.math cimport sqrt, cos, sin, exp, pi, erf, sinh, floor, ceil
from libc.time cimport time, time_t
from cython.parallel import prange
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from pyrost.bin import make_reference
import speckle_tracking as st

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.complex128_t complex_t
ctypedef np.npy_bool bool_t
ctypedef np.uint64_t uint_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF NO_VAR = -1.0

def st_update(I_n, W, dij, basis, x_ps, y_ps, z, df, sw_ss, sw_fs, ls, roi=None, n_iter=5):
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
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis, x_ps,
                                            y_ps, z, df, verbose=False)
    I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, ls, roi=roi)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, ls=ls,
                                    roi=roi, subpixel=True, verbose=False)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                                sw_ss, sw_fs, ls, roi=roi)

        # make reference image
        I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, ls, roi=roi)
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

def ab_model(pix, coeff):
    return coeff[0] + coeff[1] * (pix - coeff[3]) + coeff[2] * (pix - coeff[3])**2

def ab_errors(coeff, data):
    return ab_model(data[:, 0], coeff) - data[:, 1]

def ph_model(theta, coeff):
    return coeff[0] + coeff[1] * (theta - coeff[4]) + \
           coeff[2] * (theta - coeff[4])**2 + coeff[3] * (theta - coeff[4])**3

def ph_errors(coeff, data):
    return ph_model(data[:, 0], coeff) - data[:, 1]

def phase_fit(u, x_ps, z, df, wl, l_scale=5, max_order=4, roi=None):
    # calculate the phase
    if roi is None:
        roi = (0, u.shape[2])
    u_pix = (u - np.indices((u.shape[1], u.shape[2])))[1, 0, roi[0]:roi[1]]
    ang = u_pix * x_ps / z
    phase = np.cumsum(ang) * x_ps * df / z * 2 * np.pi / wl
    pix = np.arange(u.shape[2])[roi[0]:roi[1]]
    x, theta = pix * x_ps, pix * x_ps / z
    data = np.stack((pix, gaussian_filter(u_pix, l_scale)), axis=-1)

    # find a min/max argument
    u0 = gaussian_filter(u_pix, u_pix.shape[0] / 10)
    if np.median(np.gradient(np.gradient(u0))) > 0:
        idx = np.argmin(u0)
    else:
        idx = np.argmax(u0)

        
    # fit the model to the data
    bounds = ([-np.inf, -np.inf, -np.inf, 0],
              [np.inf, np.inf, np.inf, u_pix.shape[0]])
    fit = least_squares(ab_errors, np.array([0, 0, 0, pix[idx]]), args=(data,),
                          xtol=1e-14, ftol=1e-14, bounds=bounds, loss='cauchy')
    ang_fit = np.array([fit.x[0], fit.x[1] / (x_ps / z),
                        fit.x[2] / (x_ps / z)**2, fit.x[3]]) * x_ps / z
    ph_fit = np.zeros(5)
    ph_fit[1:] = ang_fit; ph_fit[1:4] *= 2 * np.pi / wl * df / (np.arange(3) + 1)
    ph_fit[0] = np.mean(phase - ph_model(theta, ph_fit))

    # evaluating errors
    r_sq = 1 - np.sum(ab_errors(fit.x, data)**2) / np.sum((data[:, 1] - data[:, 1].mean())**2)
    return {'pix': pix, 'theta': theta, 'u_pix': u_pix, 'angles': ang, 'phase': phase,
            'fit': fit, 'ang_fit': ang_fit, 'ph_fit': ph_fit, 'r_sq': r_sq}

cdef float_t convolve_c(float_t[::1] a1, float_t[::1] a2, int k) nogil:
    cdef:
        int a = a1.shape[0], b = a2.shape[0]
        int i0 = max(k - b // 2, 0), i1 = min(k - b//2 + b, a), i
        float_t x = 0
    for i in range(i0, i1):
        x += a1[i] * a2[k + b//2 - i]
    return x

cdef void make_frame_nc(uint_t[:, ::1] frame, float_t[::1] i_x, float_t[::1] i_y,
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
    
cdef void make_frame_c(uint_t[:, ::1] frame, float_t[::1] i_x, float_t[::1] i_y,
                       float_t[::1] sc, float_t pix_size) nogil:
    cdef:
        int b = i_y.shape[0], c = i_x.shape[0], j, k
        float_t i_xs
    for k in range(c):
        i_xs = convolve_c(i_x, sc, k)
        for j in range(b):
            frame[j, k] = <uint_t>(i_xs * i_y[j] * pix_size**2)

def make_frames(float_t[:, ::1] i_x, float_t[::1] i_y, float_t[::1] sc_x, float_t[::1] sc_y, float_t pix_size,
                bool_t noise):
    """
    Generate intensity frames with Poisson noise from x and y coordinate wavefront profiles

    i_x, i_y - x and y coordinate intensity profiles
    sc_x, sc_y - source rocking curve along x- and y-axes
    pix_size - pixel size [um]
    """
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = i_x.shape[0], b = i_y.shape[0], c = i_x.shape[1], i
        uint_t[:, :, ::1] frames = np.empty((a, b, c), dtype=np.uint64)
        float_t[::1] i_ys = np.empty(b, dtype=dtype)
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        time_t t = time(NULL)
        unsigned long seed
    gsl_rng_set(r, t)
    for i in range(b):
        i_ys[i] = convolve_c(i_y, sc_y, i)
    for i in prange(a, schedule='guided', nogil=True):
        seed = gsl_rng_get(r)
        if noise:
            make_frame_nc(frames[i], i_x[i], i_ys, sc_x, pix_size, seed)
        else:
            make_frame_c(frames[i], i_x[i], i_ys, sc_x, pix_size)
    gsl_rng_free(r)
    return np.asarray(frames)

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

cdef float_t rbf(float_t dsq, float_t ls) nogil:
    return exp(-dsq / 2 / ls**2) / sqrt(2 * pi)

# cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
#                  float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
#     cdef:
#         int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
#         int i, ss0, ss1, fs0, fs1
#         float_t SS_res = 0, SS_tot = 0, ss, fs, dss, dfs, I0_bi
#     for i in range(a):
#         ss = ux - di[i]
#         fs = uy - dj[i]
#         if ss <= 0:
#             dss = 0; ss0 = 0; ss1 = 0
#         elif ss >= aa - 1:
#             dss = 0; ss0 = aa - 1; ss1 = aa - 1
#         else:
#             dss = ss - floor(ss)
#             ss0 = <int>(floor(ss)); ss1 = ss0 + 1
#         if fs <= 0:
#             dfs = 0; fs0 = 0; fs1 = 0
#         elif fs >= bb - 1:
#             dfs = 0; fs0 = bb - 1; fs1 = bb - 1
#         else:
#             dfs = fs - floor(fs)
#             fs0 = <int>(floor(fs)); fs1 = fs0 + 1
#         I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
#                 (1 - dss) * dfs * I0[ss0, fs1] + \
#                 dss * (1 - dfs) * I0[ss1, fs0] + \
#                 dss * dfs * I0[ss1, fs1]
#         SS_res += (I[i] - I0_bi)**2
#         SS_tot += (I[i] - 1)**2
#     m_ptr[0] = SS_res; m_ptr[1] = SS_tot
#     if m_ptr[2] >= 0:
#         m_ptr[2] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                 float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
        int i, ss0, ss1, fs0, fs1
        float_t SS_res = 0, SS_tot = 0, ss, fs, dss, dfs, I0_bi
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss == 0 and fs > 0 and fs < bb - 1:
            dss = 0; ss0 = 0; ss1 = 0
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

cdef void mse_nobi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                   float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
        int i, ss0, fs0
        float_t SS_res = 0, SS_tot = 0, ss, fs
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss >= 0 and ss <= aa - 1 and fs >= 0 and fs <= bb - 1:
            ss0 = <int>(floor(ss))
            fs0 = <int>(floor(fs))
            SS_res += (I[i] - I0[ss0, fs0])**2
            SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res; m_ptr[1] = SS_tot
    if m_ptr[2] > 0:
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

cdef void mse_diff_bi(float_t* m_ptr, float_t[:, :, ::1] SS_m, float_t[:, ::1] I,
                      float_t[:, ::1] rss, float_t[:, ::1] I0, float_t[:, :, ::1] u,
                      float_t di0, float_t dj0, float_t di, float_t dj) nogil:
    cdef:
        int b = I.shape[0], c = I.shape[1], j, k
        int ss_0, fs_0, ss_1, fs_1
        int aa = I0.shape[0], bb = I0.shape[1]
        float_t ss0, fs0, ss1, fs1, dss, dfs
        float_t mse = 0, mse_var = 0, I0_bi, res_0, tot_0, res, tot, SS_res, SS_tot
    for j in range(b):
        for k in range(c):
            ss0 = u[0, j, k] - di0; fs0 = u[1, j, k] - dj0
            ss1 = u[0, j, k] - di; fs1 = u[1, j, k] - dj
            if ss0 <= 0:
                dss = 0; ss_0 = 0; ss_1 = 0
            elif ss0 >= aa - 1:
                dss = 0; ss_0 = aa - 1; ss_1 = aa - 1
            else:
                dss = ss0 - floor(ss0)
                ss_0 = <int>(floor(ss0)); ss_1 = ss_0 + 1
            if fs0 <= 0:
                dfs = 0; fs_0 = 0; fs_1 = 0
            elif fs0 >= bb - 1:
                dfs = 0; fs_0 = bb - 1; fs_1 = bb - 1
            else:
                dfs = fs0 - floor(fs0)
                fs_0 = <int>(floor(fs0)); fs_1 = fs_0 + 1
            I0_bi = (1 - dss) * (1 - dfs) * I0[ss_0, fs_0] + \
                    (1 - dss) * dfs * I0[ss_0, fs_1] + \
                    dss * (1 - dfs) * I0[ss_1, fs_0] + \
                    dss * dfs * I0[ss_1, fs_1]
            res_0 = (I[j, k] - I0_bi)**2
            tot_0 = (I[j, k] - 1)**2

            if ss1 <= 0:
                dss = 0; ss_0 = 0; ss_1 = 0
            elif ss1 >= aa - 1:
                dss = 0; ss_0 = aa - 1; ss_1 = aa - 1
            else:
                dss = ss1 - floor(ss1)
                ss_0 = <int>(floor(ss1)); ss_1 = ss_0 + 1
            if fs1 <= 0:
                dfs = 0; fs_0 = 0; fs_1 = 0
            elif fs1 >= bb - 1:
                dfs = 0; fs_0 = bb - 1; fs_1 = bb - 1
            else:
                dfs = fs1 - floor(fs1)
                fs_0 = <int>(floor(fs1)); fs_1 = fs_0 + 1
            I0_bi = (1 - dss) * (1 - dfs) * I0[ss_0, fs_0] + \
                    (1 - dss) * dfs * I0[ss_0, fs_1] + \
                    dss * (1 - dfs) * I0[ss_1, fs_0] + \
                    dss * dfs * I0[ss_1, fs_1]
            res = (I[j, k] - I0_bi)**2
            tot = (I[j, k] - 1)**2

            SS_res = SS_m[0, j, k] - res_0 + res; SS_tot = SS_m[1, j, k] - tot_0 + tot
            mse += SS_res / SS_tot / b / c
            mse_var += 4 * rss[j, k] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3) / b**2 / c**2
    m_ptr[0] = mse; m_ptr[1] = mse_var

cdef void ut_surface_c(float_t[:, ::1] mse_m, float_t[:, ::1] mse_var, float_t[:, :, ::1] SS_m,
                       float_t[:, ::1] I, float_t[:, ::1] rss, float_t[:, ::1] I0, float_t[:, :, ::1] u,
                       float_t di, float_t dj, int sw_ss, int sw_fs) nogil:
    cdef:
        int ii, jj
        float_t m_ptr[2]
    for ii in range(-sw_ss, sw_ss + 1):
        for jj in range(-sw_fs, sw_fs + 1):
            mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, di, dj, di + ii, dj + jj)
            mse_m[ii + sw_ss, jj + sw_fs] = m_ptr[0]
            mse_var[ii + sw_ss, jj + sw_fs] = m_ptr[1]

def ut_surface(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
               float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, int sw_ss, int sw_fs,
               float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] I = np.empty((b, c, a + 1), dtype=dtype)
        float_t[:, :, ::1] I_buf = np.empty((max_threads + 1, b, c), dtype=dtype)
        float_t[:, :, ::1] SS_m = np.empty((2, b, c), dtype=dtype)
        float_t[:, :, ::1] mse_m = np.empty((a, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
        float_t[:, :, ::1] mse_var = np.empty((a, 2 * sw_ss + 1, 2 * sw_fs + 1), dtype=dtype)
        float_t m_ptr[3]
    m_ptr[2] = NO_VAR
    for k in prange(c, schedule='guided', nogil=True):
        for j in range(b):
            krig_data_c(I[j, k], I_n, W, u, j, k, ls)
            mse_bi(m_ptr, I[j, k], I0, di, dj, u[0, j, k], u[1, j, k])
            SS_m[0, j, k] = m_ptr[0]; SS_m[1, j, k] = m_ptr[1]
    I_buf[max_threads] = I[:, :, a]
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        I_buf[t] = I[:, :, i]
        ut_surface_c(mse_m[i], mse_var[i], SS_m, I_buf[t], I_buf[max_threads],
                     I0, u, di[i], dj[i], sw_ss, sw_fs)
    return np.asarray(mse_m), np.asarray(mse_var)

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

cdef void subpixel_ref_2d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
    cdef:
        float_t dss = 0, dfs = 0, det, dd
        float_t f00, f01, f10, f11, f12, f21, f22
    f00 = mse_m[0]; f01 = mse_m[1]; f10 = mse_m[2]
    f11 = mse_m[3]; f12 = mse_m[4]; f21 = mse_m[5]
    f22 = mse_m[6]

    det = 4 * (f21 + f01 - 2 * f11) * (f12 + f10 - 2 * f11) - \
          (f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12)**2
    if det != 0:
        dss = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f12 - f10) - \
               2 * (f12 + f10 - 2 * f11) * (f21 - f01)) / det * mu / 2
        dfs = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f21 - f01) - \
               2 * (f21 + f01 - 2 * f11) * (f12 - f10)) / det * mu / 2
        dd = sqrt(dfs**2 + dss**2)
        if dd > 1:
            dss /= dd; dfs /= dd
    
    x[0] += dss; x[1] += dfs

cdef void subpixel_ref_1d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
    cdef:
        float_t dfs = 0, det, dd
        float_t f10, f11, f12
    f10 = mse_m[0]; f11 = mse_m[1]; f12 = mse_m[2]

    det = 4 * (f12 + f10 - 2 * f11)
    if det != 0:
        dfs = (f10 - f12) / det * mu
        dd = sqrt(dfs**2)
        if dd > 1:
            dfs /= dd

    x[1] += dfs

cdef void update_pm_c(float_t[::1] I, float_t[:, ::1] I0, float_t[::1] u,
                       float_t[::1] di, float_t[::1] dj, int sw_ss, int sw_fs) nogil:
    cdef:
        int ss_min = -sw_ss, fs_min = -sw_fs, ss_max = -sw_ss, fs_max = -sw_fs, ss, fs
        float_t mse_min = FLOAT_MAX, mse_max = -FLOAT_MAX, mse, mse_var, l1, mu
        float_t mv_ptr[3]
        float_t mse_m[7]
    for ss in range(-sw_ss, sw_ss + 1):
        for fs in range(-sw_fs, sw_fs + 1):
            mse_bi(mv_ptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
            mse = mv_ptr[0] / mv_ptr[1]
            if mse < mse_min:
                mse_min = mse; mse_var = mv_ptr[2]; ss_min = ss; fs_min = fs; 
            if mse > mse_max:
                mse_max = mse; ss_max = ss; fs_max = fs
    u[0] += ss_min; u[1] += fs_min
    l1 = 2 * (mse_max - mse_min) / ((ss_max - ss_min)**2 + (fs_max - fs_min)**2)
    mu = (3 * mse_var**0.5 / l1)**0.33
    mu = mu if mu > 2 else 2
    if sw_ss:
        mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1] - mu / 2)
        mse_m[0] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1])
        mse_m[1] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
        mse_m[2] = mv_ptr[0] / mv_ptr[1]
        mse_m[3] = mse_min
        mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
        mse_m[4] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1])
        mse_m[5] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1] + mu / 2)
        mse_m[6] = mv_ptr[0] / mv_ptr[1]
        subpixel_ref_2d(u, mse_m, mu)
    else:
        mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
        mse_m[0] = mv_ptr[0] / mv_ptr[1]
        mse_m[1] = mse_min
        mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
        mse_m[2] = mv_ptr[0] / mv_ptr[1]
        subpixel_ref_1d(u, mse_m, mu)

def update_pixel_map_gs(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
                        float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
                        int sw_ss, int sw_fs, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[::1, :, :] u = np.empty((2, b, c), dtype=dtype, order='F')
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u0, j, k, ls)
            u[:, j, k] = u0[:, j, k]
            update_pm_c(I[t], I0, u[:, j, k], di, dj, sw_ss, sw_fs)
    return np.asarray(u, order='C')

cdef void update_t_c(float_t[:, :, ::1] SS_m, float_t[:, ::1] I, float_t[:, ::1] rss, float_t[:, ::1] I0,
                     float_t[:, :, ::1] u, float_t[::1] dij, int sw_ss, int sw_fs) nogil:
    cdef:
        int ii, jj
        int ss_min = -sw_ss, fs_min = -sw_fs, ss_max = -sw_ss, fs_max = -sw_fs
        float_t mse_min = FLOAT_MAX, mse_var = FLOAT_MAX, mse_max = -FLOAT_MAX, l1, mu
        float_t m_ptr[2]
        float_t mse_m[7]
    for ii in range(-sw_ss, sw_ss + 1):
        for jj in range(-sw_fs, sw_fs + 1):
            mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + ii, dij[1] + jj)
            if m_ptr[0] < mse_min:
                mse_min = m_ptr[0]; mse_var = m_ptr[1]; ss_min = ii; fs_min = jj
            if m_ptr[0] > mse_max:
                mse_max = m_ptr[0]; ss_max = ii; fs_max = jj
    dij[0] += ss_min; dij[1] += fs_min
    l1 = 2 * (mse_max - mse_min) / ((ss_max - ss_min)**2 + (fs_max - fs_min)**2)
    mu = (3 * mse_var**0.5 / l1)**0.33
    mu = mu if mu > 2 else 2
    if sw_ss:
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] - mu / 2, dij[1] - mu / 2)
        mse_m[0] = m_ptr[0]
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] - mu / 2, dij[1])
        mse_m[1] = m_ptr[0]
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] - mu / 2)
        mse_m[2] = m_ptr[0]
        mse_m[3] = mse_min
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] + mu / 2)
        mse_m[4] = m_ptr[0]
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + mu / 2, dij[1])
        mse_m[5] = m_ptr[0]
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0] + mu / 2, dij[1] + mu / 2)
        mse_m[6] = m_ptr[0]
        subpixel_ref_2d(dij, mse_m, mu)
    else:
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] - mu / 2)
        mse_m[0] = m_ptr[0]
        mse_m[1] = mse_min
        mse_diff_bi(m_ptr, SS_m, I, rss, I0, u, dij[0], dij[1], dij[0], dij[1] + mu / 2)
        mse_m[2] = m_ptr[0]
        subpixel_ref_1d(dij, mse_m, mu)

def update_translations_gs(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
                           float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj,
                           int sw_ss, int sw_fs, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] I = np.empty((b, c, a + 1), dtype=dtype)
        float_t[:, :, ::1] I_buf = np.empty((max_threads + 1, b, c), dtype=dtype)
        float_t[:, :, ::1] SS_m = np.empty((3, b, c), dtype=dtype)
        float_t[:, ::1] dij = np.empty((a, 2), dtype=dtype)
        float_t m_ptr[3]
    m_ptr[2] = NO_VAR
    for k in prange(c, schedule='guided', nogil=True):
        for j in range(b):
            krig_data_c(I[j, k], I_n, W, u, j, k, ls)
            mse_bi(m_ptr, I[j, k], I0, di, dj, u[0, j, k], u[1, j, k])
            SS_m[0, j, k] = m_ptr[0]; SS_m[1, j, k] = m_ptr[1]
    I_buf[max_threads] = I[:, :, a]
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        I_buf[t] = I[:, :, i]; dij[i, 0] = di[i]; dij[i, 1] = dj[i]
        update_t_c(SS_m, I_buf[t], I_buf[max_threads], I0, u, dij[i], sw_ss, sw_fs)
    return np.asarray(dij)

def mse_frame(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
              float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, ::1] mptr = NO_VAR * np.ones((max_threads, 3), dtype=dtype)
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
        float_t[:, ::1] mse_f = np.empty((b, c), dtype=dtype)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u, j, k, ls)
            mse_bi(&mptr[t, 0], I[t], I0, di, dj, u[0, j, k], u[1, j, k])
            mse_f[j, k] = mptr[t, 0] / mptr[t, 1]
    return np.asarray(mse_f)

def mse_total(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
              float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t err = 0
        float_t[:, ::1] mptr = NO_VAR * np.ones((max_threads, 3), dtype=dtype)
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u, j, k, ls)
            mse_bi(&mptr[t, 0], I[t], I0, di, dj, u[0, j, k], u[1, j, k])
            err += mptr[t, 0] / mptr[t, 1]
    return err / b / c