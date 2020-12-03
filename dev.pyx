#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
cimport openmp
from libc.math cimport sqrt, cos, sin, exp, pi, erf, sinh, floor, ceil
from libc.time cimport time, time_t
from cython.parallel import prange
from scipy.ndimage import gaussian_filter
from pyrost.bin import update_pixel_map_gs, make_reference, total_mse
import speckle_tracking as st

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.complex128_t complex_t
ctypedef np.npy_bool bool_t
ctypedef np.uint64_t uint_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF MU_C = 1.681792830507429
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
        es.append(total_mse(I_n=I_n, W=W, I0=I0, u=u0, di=di - n0, dj=dj - m0, ls=l_scale))

        # update pixel map
        u = update_pixel_map_gs(I_n=I_n, W=W, I0=I0, u0=u0, di=di - n0, dj=dj - m0,
                                sw_ss=0, sw_fs=sw_max, ls=l_scale)
        sw_max = int(np.max(np.abs(u - u0)))
        u0 = u0 + gaussian_filter(u - u0, (0, 0, l_scale))

        # make reference image
        I0, n0, m0 = make_reference(I_n=I_n, W=W, u=u0, di=di, dj=dj, ls=l_scale, sw_ss=0, sw_fs=0)
        I0 = gaussian_filter(I0, (0, l_scale))
    return {'u':u0, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

cdef float_t bprd_varc(float_t br_dx, float_t sgm, float_t atn) nogil:
    cdef:
        int a = <int>(br_dx / sgm + 1), i, n
        float_t var = 0
    for i in range(-a, a):
        n = 1 + 2 * i
        var += (atn * sin(pi * n / 2)**2 / pi / n)**2 * exp(-(pi * sgm * n / br_dx)**2)
    return var

cdef float_t bnprd_varc(float_t br_dx, float_t sgm, float_t atn) nogil:
    cdef:
        float_t br_rt = br_dx / 2 / sgm
        float_t exp_term = 4 * exp(-br_rt**2 / 4) - exp(-br_rt**2) - 3
    return atn**2 / 4 * (2 * erf(br_rt / 2) - erf(br_rt) + exp_term / sqrt(pi) / br_rt)

def bprd_var(float_t br_dx, float_t[::1] sgm_arr, float_t atn):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = sgm_arr.shape[0], i
        float_t[::1] var_arr = np.empty(a, dtype=dtype)
    for i in range(a):
        var_arr[i] = bprd_varc(br_dx, sgm_arr[i], atn)
    return np.asarray(var_arr)

def bnprd_var(float_t br_dx, float_t[::1] sgm_arr, float_t atn):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = sgm_arr.shape[0], i
        float_t[::1] var_arr = np.empty(a, dtype=dtype)
    for i in range(a):
        var_arr[i] = bnprd_varc(br_dx, sgm_arr[i], atn)
    return np.asarray(var_arr)

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
            ss = ss; dss = ss - floor(ss)
            ss0 = <int>(floor(ss)); ss1 = ss0 + 1
        if fs <= 0:
            dfs = 0; fs0 = 0; fs1 = 0
        elif fs >= bb - 1:
            dfs = 0; fs0 = bb - 1; fs1 = bb - 1
        else:
            fs = fs; dfs = fs - floor(fs)
            fs0 = <int>(floor(fs)); fs1 = fs0 + 1
        I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                (1 - dss) * dfs * I0[ss0, fs1] + \
                dss * (1 - dfs) * I0[ss1, fs0] + \
                dss * dfs * I0[ss1, fs1]
        SS_res += (I[i] - I0_bi)**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] >= 0:
        m_ptr[1] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef void mse_nobi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                   float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
        int i, ss0, fs0
        float_t SS_res = 0, SS_tot = 0, ss, fs
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss <= 0:
            ss0 = 0
        elif ss >= aa - 1:
            ss0 = aa - 1
        else:
            ss0 = <int>(floor(ss))
        if fs <= 0:
            fs0 = 0
        elif fs >= bb - 1:
            fs0 = bb - 1
        else:
            fs0 = <int>(floor(fs))
        SS_res += (I[i] - I0[ss0, fs0])**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] > 0:
        m_ptr[1] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

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
    print(jj0, jj1, kk0, kk1)
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

def subpixel_refinement_2d(float_t[::1] I, float_t[:, ::1] I0, float_t[:] u0,
                           float_t[::1] di, float_t[::1] dj, float_t l1):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        float_t[::1] u = np.empty(2, dtype=dtype)
        float_t dss = 0, dfs = 0, det, mu, dd
        float_t f22, f11, f00, f21, f01, f12, f10
        float_t mv_ptr[2]
    u[...] = u0
    
    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1])
    f11 = mv_ptr[0]
    print('mse_var = %f' % mv_ptr[1])
    mu = MU_C * mv_ptr[1]**0.25 / sqrt(l1)
    mu = mu if mu > 2 else 2
    print('mu = %f' % mu)
    mv_ptr[1] = NO_VAR

    mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1] - mu / 2)
    f00 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0] - mu / 2, u[1])
    f01 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
    f10 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
    f12 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1])
    f21 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0] + mu / 2, u[1] + mu / 2)
    f22 = mv_ptr[0]

    print('f21 = %f, f01 = %f' % (f21, f01))

    det = 4 * (f21 + f01 - 2 * f11) * (f12 + f10 - 2 * f11) - \
          (f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12)**2
    print('det = %f' % det)
    if det != 0:
        dss = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f12 - f10) - \
               2 * (f12 + f10 - 2 * f11) * (f21 - f01)) / det * mu / 2
        dfs = ((f22 + f00 + 2 * f11 - f01 - f21 - f10 - f12) * (f21 - f01) - \
               2 * (f21 + f01 - 2 * f11) * (f12 - f10)) / det * mu / 2
        dd = sqrt(dfs**2 + dss**2)
        if dd > 1:
            dss /= dd; dfs /= dd
    print('dss = %f, dfs = %f' % (dss, dfs))
    u[0] += dss; u[1] += dfs
    return np.asarray(u)

def subpixel_refinement_1d(float_t[::1] I, float_t[:, ::1] I0, float_t[:] u0,
                           float_t[::1] di, float_t[::1] dj, float_t l1):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        float_t[::1] u = np.empty(2, dtype=dtype)
        float_t dfs = 0, det, mu, dd
        float_t f11, f12, f10
        float_t mv_ptr[2]
    u[...] = u0
    
    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1])
    f11 = mv_ptr[0]
    print('mse_var = %f' % mv_ptr[1])
    mu = MU_C * mv_ptr[1]**0.25 / sqrt(l1)
    mu = mu if mu > 2 else 2
    print('mu = %f' % mu)
    mv_ptr[1] = NO_VAR

    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] - mu / 2)
    f10 = mv_ptr[0]
    mse_bi(mv_ptr, I, I0, di, dj, u[0], u[1] + mu / 2)
    f12 = mv_ptr[0]

    print('f12 = %f, f10 = %f' % (f12, f10))

    det = 4 * (f12 + f10 - 2 * f11)
    print('det = %f' % det)
    if det != 0:
        dfs = (f10 - f12) / det * mu
        dd = sqrt(dfs**2)
        if dd > 1:
            dfs /= dd
    print('dfs = %f' % dfs)
    u[1] += dfs
    return np.asarray(u)

cdef void mse_surface_c(float_t[:, ::1] mse_m, float_t[:, ::1] mse_var, float_t[::1] I, float_t[:, ::1] I0,
                        float_t[::1] di, float_t[::1] dj, float_t u_ss, float_t u_fs, int* bnds) nogil:
    cdef:
        int ss, fs
        int sslb = -bnds[0] if bnds[0] < u_ss - bnds[2] else <int>(bnds[2] - u_ss)
        int ssub = bnds[0] if bnds[0] < bnds[3] - u_ss else <int>(bnds[3] - u_ss)
        int fslb = -bnds[1] if bnds[1] < u_fs - bnds[4] else <int>(bnds[4] - u_fs)
        int fsub = bnds[1] if bnds[1] < bnds[5] - u_fs else <int>(bnds[5] - u_fs)
        float_t mv_ptr[2]
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            mse_bi(mv_ptr, I, I0, di, dj, u_ss + ss, u_fs + fs)
            mse_m[ss + bnds[0], fs + bnds[1]] = mv_ptr[0]
            mse_var[ss + bnds[0], fs + bnds[1]] = mv_ptr[1]

def mse_2d(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
           float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj,
           int sw_ss, int sw_fs, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
        float_t[:, :, :, ::1] mse_m = np.empty((b, c, 2 * sw_ss, 2 * sw_fs), dtype=dtype)
        float_t[:, :, :, ::1] mse_var = np.empty((b, c, 2 * sw_ss, 2 * sw_fs), dtype=dtype)
        int bnds[6] # sw_ss, sw_fs, di0, di1, dj0, dj1
    bnds[0] = sw_ss if sw_ss >= 1 else 1; bnds[1] = sw_fs if sw_fs >= 1 else 1
    bnds[2] = <int>(min_float(&di[0], a)); bnds[3] = <int>(max_float(&di[0], a)) + aa
    bnds[4] = <int>(min_float(&dj[0], a)); bnds[5] = <int>(max_float(&dj[0], a)) + bb
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u, j, k, ls)
            mse_surface_c(mse_m[j, k], mse_var[j, k], I[t], I0, di, dj, u[0, j, k], u[1, j, k], bnds)
    return np.asarray(mse_m), np.asarray(mse_var)
        
cdef void init_newton_c(float_t[::1] sptr, float_t[::1] I, float_t[:, ::1] I0,
                        float_t[::1] u, float_t[::1] di, float_t[::1] dj, int* bnds) nogil:
    cdef:
        int sslb = -bnds[0] if bnds[0] < u[0] - bnds[2] else <int>(bnds[2] - u[0])
        int ssub = bnds[0] if bnds[0] < bnds[3] - u[0] else <int>(bnds[3] - u[0])
        int fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int>(bnds[4] - u[1])
        int fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int>(bnds[5] - u[1])
        int ss, fs, ss_max = sslb, fs_max = fslb
        float_t mse_min = FLOAT_MAX, mse_max = -FLOAT_MAX, l1 = 0, d0, l, dist
        float_t mptr[2]
    mptr[1] = NO_VAR; sptr[2] = 0
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            mse_bi(mptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
            if mptr[0] < mse_min:
                mse_min = mptr[0]; sptr[0] = ss; sptr[1] = fs
            if mptr[0] > mse_max:
                mse_max = mptr[0]; ss_max = ss; fs_max = fs
    d0 = (ss_max - sptr[0])**2 + (fs_max - sptr[1])**2
    l1 = 2 * (mse_max - mse_min) / d0
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            dist = (ss - sptr[0])**2 + (fs - sptr[1])**2
            if dist > d0 / 4 and dist < d0:
                mse_bi(mptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
                l = 2 * (mptr[0] - mse_min) / dist
                if l > l1:
                    l1 = l
    sptr[2] = l1

cdef void newton_1d_c(float_t[::1] sptr, float_t[::1] I, float_t[:, ::1] I0, float_t[::1] u,
                      float_t[::1] di, float_t[::1] dj, int* bnds, int max_iter, float_t x_tol) nogil:
    cdef:
        int fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int>(bnds[4] - u[1]), k
        int fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int>(bnds[5] - u[1])
        float_t ss, fs, mu, dfs
        float_t mptr0[2]
        float_t mptr1[2]
        float_t mptr2[2]
    if sptr[2] == 0:
        init_newton_c(sptr, I, I0, u, di, dj, &bnds[0])
    ss = sptr[0]; fs = sptr[1]; mptr1[1] = NO_VAR; mptr2[1] = NO_VAR
    for k in range(max_iter):
        mse_bi(mptr0, I, I0, di, dj, u[0] + ss, u[1] + fs)
        mu = MU_C * mptr0[1]**0.25 / sqrt(sptr[2])
        mse_bi(mptr1, I, I0, di, dj, u[0] + ss, u[1] + fs - mu / 2)
        mse_bi(mptr2, I, I0, di, dj, u[0] + ss, u[1] + fs + mu / 2)
        dfs = -(mptr2[0] - mptr1[0]) / mu / sptr[2]
        fs += dfs
        if dfs < x_tol and dfs > -x_tol:
            u[1] += fs; sptr[1] = fs
            break
        if fs >= fsub or fs < fslb:
            u[1] += sptr[1]
            break
    else:
        u[1] += fs; sptr[1] = fs

def upm_newton_1d(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, float_t[:, :, ::1] u0,
                  float_t[::1] di, float_t[::1] dj, int sw_fs, float_t ls,
                  int max_iter=500, float_t x_tol=1e-12):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[::1, :, :] u = np.empty((2, b, c), dtype=dtype, order='F')
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
        float_t[:, ::1] sptr = np.zeros((max_threads, 3), dtype=dtype) # ss, fs, l1
        int bnds[6] # sw_ss, sw_fs, di0, di1, dj0, dj1
    bnds[0] = 1; bnds[1] = sw_fs if sw_fs >= 1 else 1
    bnds[2] = <int>(min_float(&di[0], a)); bnds[3] = <int>(max_float(&di[0], a)) + aa
    bnds[4] = <int>(min_float(&dj[0], a)); bnds[5] = <int>(max_float(&dj[0], a)) + bb
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u0, j, k, ls)
            u[:, j, k] = u0[:, j, k]
            newton_1d_c(sptr[t], I[t], I0, u[:, j, k], di, dj, bnds, max_iter, x_tol)
    return np.asarray(u)

def init_newton(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
               float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
               int sw_fs, float_t ls):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[::1, :, :] u = np.empty((2, b, c), dtype=dtype, order='F')
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=dtype)
        float_t[:, ::1] sptr = np.zeros((max_threads, 3), dtype=dtype) # ss, fs, l1
        float_t[:, ::1] l1 = np.empty((b, c), dtype=dtype)
        int bnds[6] # sw_ss, sw_fs, di0, di1, dj0, dj1
    bnds[0] = 1; bnds[1] = sw_fs if sw_fs >= 1 else 1
    bnds[2] = <int>(min_float(&di[0], a)); bnds[3] = <int>(max_float(&di[0], a)) + aa
    bnds[4] = <int>(min_float(&dj[0], a)); bnds[5] = <int>(max_float(&dj[0], a)) + bb
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u0, j, k, ls)
            u[:, j, k] = u0[:, j, k]
            init_newton_c(sptr[t], I[t], I0, u[:, j, k], di, dj, bnds)
            l1[j, k] = sptr[t, 2]
    return np.asarray(l1)

def ct_integrate(float_t[:, ::1] sx_arr, float_t[:, ::1] sy_arr):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = sx_arr.shape[0], b = sx_arr.shape[1], i, j, ii, jj
        float_t[:, ::1] s_asdi = np.empty((2 * a, 2 * b), dtype=dtype)
        complex_t[:, ::1] sf_asdi = np.empty((2 * a, 2 * b), dtype=np.complex128)
        float_t xf, yf
    for i in range(a):
        for j in range(b):
            s_asdi[i, j] = -sx_arr[a - i - 1, b - j - 1]
    for i in range(a):
        for j in range(b):
            s_asdi[i + a, j] = sx_arr[i, b - j - 1]
    for i in range(a):
        for j in range(b):
            s_asdi[i, j + b] = -sx_arr[a - i - 1, j]
    for i in range(a):
        for j in range(b):
            s_asdi[i + a, j + b] = sx_arr[i, j]
    cdef np.ndarray[np.complex128_t, ndim=2] sfx_asdi = np.fft.fft2(s_asdi)
    for i in range(a):
        for j in range(b):
            s_asdi[i, j] = -sy_arr[a - i - 1, b - j - 1]
    for i in range(a):
        for j in range(b):
            s_asdi[i + a, j] = -sy_arr[i, b - j - 1]
    for i in range(a):
        for j in range(b):
            s_asdi[i, j + b] = sy_arr[a - i - 1, j]
    for i in range(a):
        for j in range(b):
            s_asdi[i + a, j + b] = sy_arr[i, j]
    cdef np.ndarray[np.complex128_t, ndim=2] sfy_asdi = np.fft.fft2(s_asdi)
    for i in range(2 * a):
        xf = <float_t>(i) / 2 / a - i // a
        for j in range(2 * b):
            yf = <float_t>(j) / 2 / b - j // b
            sf_asdi[i, j] = (xf * sfx_asdi[i, j] + yf * sfy_asdi[i, j]) / (2j * pi * (xf**2 + yf**2))
    sf_asdi[0, 0] = 0
    return np.asarray(np.fft.ifft2(sf_asdi).real[a:, b:], dtype=dtype)