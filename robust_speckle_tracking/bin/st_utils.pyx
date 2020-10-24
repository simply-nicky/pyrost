cimport numpy as cnp
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, exp, pi, floor
from cython.parallel import prange
cimport openmp
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t
ctypedef cnp.npy_bool bool_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF MU_C = 1.681792830507429
DEF NO_VAR = -1.0
DEF MIN_VAR = 0.01

cdef float_t wirthselect_float(float_t[:] array, int k) nogil:
    cdef:
        int_t l = 0, m = array.shape[0] - 1, i, j
        float_t x, tmp 
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

def make_whitefield_st(float_t[:, :, ::1] data, bool_t[:, ::1] mask):
    """
    Return whitefield based on median filtering of the stack of frames

    data - stack of frames
    mask - bad pixel mask
    """
    cdef:
        int_t a = data.shape[0], b = data.shape[1], c = data.shape[2], i, j, k
        int_t max_threads = openmp.omp_get_max_threads()
        float_t[:, ::1] wf = np.empty((b, c), dtype=np.float64)
        float_t[:, ::1] array = np.empty((max_threads, a), dtype=np.float64)
    for j in prange(b, schedule='guided', nogil=True):
        i = openmp.omp_get_thread_num()
        for k in range(c):
            if mask[j, k]:
                array[i] = data[:, j, k]
                wf[j, k] = wirthselect_float(array[i], a // 2)
            else:
                wf[j, k] = 0
    return np.asarray(wf)

cdef float_t min_float(float_t* array, int_t a) nogil:
    cdef:
        int_t i
        float_t mv = array[0]
    for i in range(a):
        if array[i] < mv:
            mv = array[i]
    return mv

cdef float_t max_float(float_t* array, int_t a) nogil:
    cdef:
        int_t i
        float_t mv = array[0]
    for i in range(a):
        if array[i] > mv:
            mv = array[i]
    return mv

cdef float_t rbf(float_t dx, float_t ls) nogil:
    return exp(-dx**2 / 2 / ls**2) / sqrt(2 * pi) / ls

cdef void rbf_mc(float_t[:, ::1] rm, float_t ls) nogil:
    cdef:
        int_t a = rm.shape[0], b = rm.shape[1], i, j
        float_t dr, rv
    for i in range(a):
        for j in range(b):
            dr = sqrt((a // 2 - i)**2 + (b // 2 - j)**2)
            rv = rbf(dr, ls)
            rm[i, j] = rv

cdef void frame_reference(float_t[:, ::1] I0, float_t[:, ::1] w0, float_t[:, ::1] I, float_t[:, ::1] W,
                          float_t[:, :, ::1] u, float_t di, float_t dj, float_t ls) nogil:
    cdef:
        int_t b = I.shape[0], c = I.shape[1], j, k, jj, kk, j0, k0
        int_t aa = I0.shape[0], bb = I0.shape[1], jj0, jj1, kk0, kk1
        int_t dn = <int_t>(ceil(4 * ls))
        float_t ss, fs
    for j in range(b):
        for k in range(c):
            ss = u[0, j, k] - di
            fs = u[1, j, k] - dj
            j0 = <int_t>(ss) + 1
            k0 = <int_t>(fs) + 1
            jj0 = j0 - dn if j0 - dn > 0 else 0
            jj1 = j0 + dn if j0 + dn < aa else aa
            kk0 = k0 - dn if k0 - dn > 0 else 0
            kk1 = k0 + dn if k0 + dn < bb else bb
            for jj in range(jj0, jj1):
                for kk in range(kk0, kk1):
                    I0[jj, kk] += I[j, k] * W[j, k] * rbf(ss - jj, ls) * rbf(fs - kk, ls)
                    w0[jj, kk] += W[j, k]**2 * rbf(ss - jj, ls) * rbf(fs - kk, ls)

def make_reference(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, :, ::1] u,
                   float_t[::1] di, float_t[::1] dj, float_t ls=2.5):
    """
    Return a reference image 

    I_n - measured data
    W - whitefield
    u - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j
        float_t n0 = -min_float(&u[0, 0, 0], b * c) + max_float(&di[0], a)
        float_t m0 = -min_float(&u[1, 0, 0], b * c) + max_float(&dj[0], a)
        int_t aa = <int_t>(max_float(&u[0, 0, 0], b * c) - min_float(&di[0], a) + n0) + 1
        int_t bb = <int_t>(max_float(&u[1, 0, 0], b * c) - min_float(&dj[0], a) + m0) + 1
        float_t[:, ::1] I0 = np.zeros((aa, bb), dtype=np.float64)
        float_t[:, ::1] w0 = np.zeros((aa, bb), dtype=np.float64)
    for i in range(a):
        di[i] = di[i] - n0; dj[i] = dj[i] - m0
        frame_reference(I0, w0, I_n[i], W, u, di[i], dj[i], ls)
    for i in range(aa):
        for j in range(bb):
            if w0[i, j]:
                I0[i, j] /= w0[i, j]
    return np.asarray(I0), np.asarray(di), np.asarray(dj)

cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                 float_t W, float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, ss1, fs0, fs1
        float_t SS_res = 0, SS_tot = 0, ss, fs, dss, dfs, I0_bi, var
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss >= 0 and ss <= aa - 1 and fs >= 0 and fs <= bb - 1:
            dss = ss - floor(ss)
            dfs = fs - floor(fs)
            if dss:
                ss0 = <int_t>(floor(ss)); ss1 = ss0 + 1
            else:
                ss0 = <int_t>(ss); ss1 = ss0
            if dfs:
                fs0 = <int_t>(floor(fs)); fs1 = fs0 + 1
            else:
                fs0 = <int_t>(fs); fs1 = fs0
            I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                    (1 - dss) * dfs * I0[ss0, fs1] + \
                    dss * (1 - dfs) * I0[ss1, fs0] + \
                    dss * dfs * I0[ss1, fs1]
            SS_res += (I[i] - W * I0_bi)**2
            SS_tot += (I[i] - W)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] != NO_VAR:
        var = 4 * W * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)
        m_ptr[1] = var if var > MIN_VAR else MIN_VAR
         
cdef void mse_nobi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                   float_t W, float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, fs0
        float_t SS_res = 0, SS_tot = 0, ss, fs, var
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss >= 0 and ss <= aa - 1 and fs >= 0 and fs <= bb - 1:
            ss0 = <int_t>(floor(ss))
            fs0 = <int_t>(floor(fs))
            SS_res += (I[i] - W * I0[ss0, fs0])**2
            SS_tot += (I[i] - W)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] != NO_VAR:
        var = 4 * W * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)   
        m_ptr[1] = var if var > MIN_VAR else MIN_VAR

cdef void convolve_I(float_t[::1] I, float_t[:, :, ::1] I_n, float_t[:, ::1] rm, int_t j, int_t k) nogil:
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], dn = rm.shape[0] // 2, i, jj, kk
        int_t jj0 = j - dn if j - dn > 0 else 0
        int_t jj1 = j + dn if j + dn < b else b
        int_t kk0 = k - dn if k - dn > 0 else 0
        int_t kk1 = k + dn if k + dn < c else c
        float_t Isum, rsum
    for i in range(a):
        Isum = 0; rsum = 0
        for jj in range(jj0, jj1):
            for kk in range(kk0, kk1):
                Isum += I_n[i, jj, kk] * rm[jj - j + dn, kk - k + dn]
                rsum += rm[jj - j + dn, kk - k + dn]
        I[i] = Isum / rsum
        
cdef void mse_min_c(float_t[::1] I, float_t W, float_t[:, ::1] I0, float_t[:] u,
                    float_t[::1] di, float_t[::1] dj, int_t* bnds) nogil:
    cdef:
        int_t sslb = -bnds[0] if bnds[0] < u[0] - bnds[2] else <int_t>(bnds[2] - u[0])
        int_t ssub = bnds[0] if bnds[0] < bnds[3] - u[0] else <int_t>(bnds[3] - u[0])
        int_t fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int_t>(bnds[4] - u[1])
        int_t fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int_t>(bnds[5] - u[1])
        int_t ss_min = sslb, fs_min = fslb, ss, fs
        float_t mse_min = FLOAT_MAX
        float_t mv_ptr[2]
    mv_ptr[1] = NO_VAR
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            mse_bi(mv_ptr, I, I0, W, di, dj, u[0] + ss, u[1] + fs)
            if mv_ptr[0] < mse_min:
                mse_min = mv_ptr[0]; ss_min = ss; fs_min = fs
    u[0] += ss_min; u[1] += fs_min

def upm_search(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
               float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
               uint_t wss, uint_t wfs, float_t ls=5.):
    """
    Update the pixel map

    I_n - measured data
    W - whitefield
    I0 - reference image
    u0 - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    wss, wfs - search window size along slow and fast axis
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int_t aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int_t dn = <int_t>(ceil(2 * ls)), max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] u = np.empty((2, b, c), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
        float_t[:, ::1] rm = np.empty((2 * dn + 1, 2 * dn + 1), dtype=np.float64)
        int_t bnds[6] # wss, wfs, di0, di1, dj0, dj1
    bnds[0] = wss; bnds[1] = wfs
    bnds[2] = <int_t>(min_float(&di[0], a)); bnds[3] = <int_t>(max_float(&di[0], a)) + aa
    bnds[4] = <int_t>(min_float(&dj[0], a)); bnds[5] = <int_t>(max_float(&dj[0], a)) + bb
    u[...] = u0
    rbf_mc(rm, ls)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            convolve_I(I[t], I_n, rm, j, k)
            mse_min_c(I[t], W[j, k], I0, u[:, j, k], di, dj, bnds)
    return np.asarray(u)

cdef void init_stars_c(float_t[::1] sptr, float_t[::1] I, float_t W, float_t[:, ::1] I0,
                       float_t[:] u, float_t[::1] di, float_t[::1] dj, int_t* bnds) nogil:
    cdef:
        int_t sslb = -bnds[0] if bnds[0] < u[0] - bnds[2] else <int_t>(bnds[2] - u[0])
        int_t ssub = bnds[0] if bnds[0] < bnds[3] - u[0] else <int_t>(bnds[3] - u[0])
        int_t fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int_t>(bnds[4] - u[1])
        int_t fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int_t>(bnds[5] - u[1])
        int_t ss, fs
        float_t mse_min = FLOAT_MAX, l1, dist
        float_t mptr[2]
    mptr[1] = NO_VAR; sptr[2] = 0
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            mse_bi(mptr, I, I0, W, di, dj, u[0] + ss, u[1] + fs)
            if mptr[0] < mse_min:
                mse_min = mptr[0]; sptr[0] = ss; sptr[1] = fs
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            dist = (ss - sptr[0])**2 + (fs - sptr[1])**2
            if dist > 10**2 and dist < 100**2:
                mse_bi(mptr, I, I0, W, di, dj, u[0] + ss, u[1] + fs)
                l1 = 2 * (mptr[0] - mse_min) / dist
                if l1 > sptr[2]:
                    sptr[2] = l1

cdef void stars_1d_c(float_t[::1] sptr, float_t[::1] I, float_t W, float_t[:, ::1] I0,
                     float_t[:] u, float_t[::1] di, float_t[::1] dj, int_t* bnds,
                     float_t x_tol, int_t max_iter, float_t h) nogil:
    cdef:
        int_t k
        float_t ss, fs, vfs, mu, dfs
        timespec ts
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        int_t fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int_t>(bnds[4] - u[1])
        int_t fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int_t>(bnds[5] - u[1])
        float_t mptr0[2]
        float_t mptr1[2]
    if sptr[2] <= 0:
        init_stars_c(sptr, I, W, I0, u, di, dj, bnds)
    clock_gettime(CLOCK_REALTIME, &ts)
    gsl_rng_set(r, ts.tv_sec + ts.tv_nsec)
    ss = sptr[0]; mptr1[1] = NO_VAR; fs = sptr[1]
    for k in range(1, max_iter):
        mse_bi(mptr0, I, I0, W, di, dj, u[0] + ss, u[1] + fs)
        vfs = gsl_ran_gaussian(r, 1.0)
        mu = MU_C * mptr0[1]**0.25 / sqrt(sptr[2])
        if fs + vfs * mu >= fsub or fs + vfs * mu < fslb:
            u[1] += sptr[1]
            break
        mse_bi(mptr1, I, I0, W, di, dj, u[0] + ss, u[1] + fs + vfs * mu)
        dfs = -h * (mptr1[0] - mptr0[0]) / mu * vfs
        if dfs < x_tol and dfs > -x_tol:
            u[1] += fs + dfs; sptr[1] = fs + dfs
            break
        fs += dfs
        if fs >= fsub or fs < fslb:
            u[1] += sptr[1]
            break
    else:
        u[1] += fs; sptr[1] = fs
    gsl_rng_free(r)

def upm_stars(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0, float_t[:, :, ::1] u0,
              float_t[::1] di, float_t[::1] dj, int_t wss, int_t wfs, float_t ls=5.,
              float_t x_tol=1e-12, int_t max_iter=1000, float_t h=3.):
    """
    Update the pixel map based on the STARS algorithm
    https://arxiv.org/abs/1507.03332

    I_n - measured data
    W - whitefield
    I0 - reference image
    u0 - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    wss, wfs - search window size along slow and fast axis
    ls - length scale in pixels
    x_tol - argument tolerance
    max_iter - maximum number of iterations
    h - step size in STARS algorithm
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int_t aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int_t dn = <int_t>(ceil(2 * ls)), max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] u = np.empty((2, b, c), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
        float_t[:, ::1] rm = np.empty((2 * dn + 1, 2 * dn + 1), dtype=np.float64)
        float_t[:, ::1] sptr = np.zeros((max_threads, 3), dtype=np.float64)
        int_t bnds[6] # wss, wfs, di0, di1, dj0, dj1
    bnds[0] = wss; bnds[1] = wfs
    bnds[2] = <int_t>(min_float(&di[0], a)); bnds[3] = <int_t>(max_float(&di[0], a)) + aa
    bnds[4] = <int_t>(min_float(&dj[0], a)); bnds[5] = <int_t>(max_float(&dj[0], a)) + bb
    u[...] = u0
    rbf_mc(rm, ls)
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            convolve_I(I[t], I_n, rm, j, k)
            stars_1d_c(sptr[t], I[t], W[j, k], I0, u[:, j, k], di, dj, bnds, x_tol, max_iter, h)
    return np.asarray(u)

def total_mse(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
              float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, float_t ls=5.):
    """
    Return total mean squared error

    I_n - measured data
    W - whitefield
    I0 - reference image
    u - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], j, k, t
        int_t dn = <int_t>(ceil(2 * ls)), max_threads = openmp.omp_get_max_threads()
        float_t[:, ::1] rm = np.empty((2 * dn + 1, 2 * dn + 1), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
        float_t [:, ::1] mptr = NO_VAR * np.ones((max_threads, 2), dtype=np.float64)
        float_t err = 0
    rbf_mc(rm, ls)
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            convolve_I(I[t], I_n, rm, j, k)
            mse_bi(&mptr[t, 0], I[t], I0, W[j, k], di, dj, u[0, j, k], u[1, j, k])
            err += mptr[t, 0]
    return err / b / c