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

cdef float_t rbf(float_t dsq, float_t ls) nogil:
    return exp(-dsq / 2 / ls**2) / sqrt(2 * pi)

cdef void frame_reference(float_t[:, ::1] I0, float_t[:, ::1] w0, float_t[:, ::1] I, float_t[:, ::1] W,
                          float_t[:, :, ::1] u, float_t di, float_t dj, float_t ls) nogil:
    cdef:
        int_t b = I.shape[0], c = I.shape[1], j, k, jj, kk, j0, k0
        int_t aa = I0.shape[0], bb = I0.shape[1], jj0, jj1, kk0, kk1
        int_t dn = <int_t>(ceil(4 * ls))
        float_t ss, fs, r
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
                    r = rbf((jj - ss)**2 + (kk - fs)**2, ls)
                    I0[jj, kk] += I[j, k] * W[j, k] * r
                    w0[jj, kk] += W[j, k]**2 * r

def make_reference(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, :, ::1] u, float_t[::1] di,
                   float_t[::1] dj, float_t ls, int_t wfs=0, bool_t return_nm0=True):
    """
    Generate a reference image based on the pixel mapping and the measured data using the simple kriging

    I_n - measured frames
    W - whitefield
    u - pixel mapping between the data at the detector's plane
        and the reference image at the reference plane
    di, dj - sample translations along slow and fast axis in pixels
    ls - length scale in pixels
    return_nm0 - flag to return n0, m0, lower pixel bounds of the reference image at the reference plane
    sw_max - return extended reference image based on search window size
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j, k, t
        float_t n0 = -min_float(&u[0, 0, 0], b * c) + max_float(&di[0], a)
        float_t m0 = -min_float(&u[1, 0, 0], b * c) + max_float(&dj[0], a) + wfs
        int_t aa = <int_t>(max_float(&u[0, 0, 0], b * c) - min_float(&di[0], a) + n0) + 1
        int_t bb = <int_t>(max_float(&u[1, 0, 0], b * c) - min_float(&dj[0], a) + m0) + 1 + wfs
        int_t max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] I = np.zeros((max_threads, aa, bb), dtype=np.float64)
        float_t[:, :, ::1] w = np.zeros((max_threads, aa, bb), dtype=np.float64)
        float_t[::1] Is = np.empty(max_threads, dtype=np.float64)
        float_t[::1] ws = np.empty(max_threads, dtype=np.float64)
        float_t[:, ::1] I0 = np.zeros((aa, bb), dtype=np.float64)
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        frame_reference(I[t], w[t], I_n[i], W, u, di[i] - n0, dj[i] - m0, ls)
    for k in prange(bb, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(aa):
            Is[t] = 0; ws[t] = 0
            for i in range(max_threads):
                Is[t] = Is[t] + I[i, j, k]
                ws[t] = ws[t] + w[i, j, k]
            if ws[t]:
                I0[j, k] = Is[t] / ws[t]
            else:
                I0[j, k] = 0
    if return_nm0:
        return np.asarray(I0), <int_t>(n0), <int_t>(m0)
    else:
        return np.asarray(I0)

cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                 float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, ss1, fs0, fs1
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
            ss0 = <int_t>(floor(ss)); ss1 = ss0 + 1
        if fs <= 0:
            dfs = 0; fs0 = 0; fs1 = 0
        elif fs >= bb - 1:
            dfs = 0; fs0 = bb - 1; fs1 = bb - 1
        else:
            fs = fs; dfs = fs - floor(fs)
            fs0 = <int_t>(floor(fs)); fs1 = fs0 + 1
        I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                (1 - dss) * dfs * I0[ss0, fs1] + \
                dss * (1 - dfs) * I0[ss1, fs0] + \
                dss * dfs * I0[ss1, fs1]
        SS_res += (I[i] - I0_bi)**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] >= 0:
        m_ptr[1] = 4 * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef void mse_nobi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                   float_t[::1] di, float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, fs0
        float_t SS_res = 0, SS_tot = 0, ss, fs
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss <= 0:
            ss0 = 0
        elif ss >= aa - 1:
            ss0 = aa - 1
        else:
            ss0 = <int_t>(floor(ss))
        if fs <= 0:
            fs0 = 0
        elif fs >= bb - 1:
            fs0 = bb - 1
        else:
            fs0 = <int_t>(floor(fs))
        SS_res += (I[i] - I0[ss0, fs0])**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] > 0:
        m_ptr[1] = 4 * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef float_t krig_data_c(float_t[::1] I, float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, :, ::1] u,
                           int_t j, int_t k, float_t ls) nogil:
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, jj, kk
        int_t djk = <int_t>(ceil(2 * ls))
        int_t jj0 = j - djk if j - djk > 0 else 0
        int_t jj1 = j + djk if j + djk < b else b
        int_t kk0 = k - djk if k - djk > 0 else 0
        int_t kk1 = k + djk if k + djk < c else c
        float_t w0 = 0, rss = 0, r
    for i in range(a):
        I[i] = 0
    for jj in range(jj0, jj1):
        for kk in range(kk0, kk1):
            r = rbf((u[0, jj, kk] - u[0, j, k])**2 + (u[1, jj, kk] - u[1, j, k])**2, ls)
            w0 += r * W[jj, kk]**2
            rss += W[jj, kk]**4 * r**2
            for i in range(a):
                I[i] += I_n[i, jj, kk] * W[jj, kk] * r
    for i in range(a):
        I[i] /= w0
    return rss / w0**2
        
cdef void mse_min_c(float_t[::1] I, float_t[:, ::1] I0, float_t[:] u,
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
            mse_bi(mv_ptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
            if mv_ptr[0] < mse_min:
                mse_min = mv_ptr[0]; ss_min = ss; fs_min = fs
    u[0] += ss_min; u[1] += fs_min

def upm_search(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
               float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
               uint_t wss, uint_t wfs, float_t ls):
    """
    Update the pixel map using the brute-force search along the search window
    The update procedure minimizes the mean-squared-error (MSE) between the measured frames and the reference image

    I_n - measured frames
    W - whitefield
    I0 - reference image
    u0 - pixel mapping between the data at the detector's plane
         and the reference image at the reference plane
    di, dj - the sample translations along the slow and fast axes in pixels
    wss, wfs - search window size along slow and fast axis in pixels
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int_t aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int_t max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] u = np.empty((2, b, c), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
        int_t bnds[6] # wss, wfs, di0, di1, dj0, dj1
    bnds[0] = wss; bnds[1] = wfs
    bnds[2] = <int_t>(min_float(&di[0], a)); bnds[3] = <int_t>(max_float(&di[0], a)) + aa
    bnds[4] = <int_t>(min_float(&dj[0], a)); bnds[5] = <int_t>(max_float(&dj[0], a)) + bb
    u[...] = u0
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u0, j, k, ls)
            mse_min_c(I[t], I0, u[:, j, k], di, dj, bnds)
    return np.asarray(u)

cdef void init_upm_c(float_t[::1] sptr, float_t[::1] I, float_t[:, ::1] I0,
                       float_t[:] u, float_t[::1] di, float_t[::1] dj, int_t* bnds) nogil:
    cdef:
        int_t sslb = -bnds[0] if bnds[0] < u[0] - bnds[2] else <int_t>(bnds[2] - u[0])
        int_t ssub = bnds[0] if bnds[0] < bnds[3] - u[0] else <int_t>(bnds[3] - u[0])
        int_t fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int_t>(bnds[4] - u[1])
        int_t fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int_t>(bnds[5] - u[1])
        int_t ss, fs, ss_max = sslb, fs_max = fslb
        float_t mse_min = FLOAT_MAX, mse_max = -FLOAT_MAX, l1 = 0, l, dist
        float_t mptr[2]
    mptr[1] = NO_VAR; sptr[2] = 0
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            mse_bi(mptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
            if mptr[0] < mse_min:
                mse_min = mptr[0]; sptr[0] = ss; sptr[1] = fs
            if mptr[0] > mse_max:
                mse_max = mptr[0]; ss_max = ss; fs_max = fs
    l1 = 2 * (mse_max - mse_min) / ((ss_max - sptr[0])**2 + (fs_max - sptr[1])**2)
    for ss in range(sslb, ssub):
        for fs in range(fslb, fsub):
            dist = (ss - sptr[0])**2 + (fs - sptr[1])**2
            if dist > 10**2 and dist < 100**2:
                mse_bi(mptr, I, I0, di, dj, u[0] + ss, u[1] + fs)
                l = 2 * (mptr[0] - mse_min) / dist
                if l > l1:
                    l1 = l
    sptr[2] = l1

cdef void newton_1d_c(float_t[::1] sptr, float_t[::1] I, float_t[:, ::1] I0, float_t[:] u,
                      float_t[::1] di, float_t[::1] dj, int_t* bnds, int_t max_iter, float_t x_tol) nogil:
    cdef:
        int_t k
        float_t ss, fs, mu, dfs
        int_t fslb = -bnds[1] if bnds[1] < u[1] - bnds[4] else <int_t>(bnds[4] - u[1])
        int_t fsub = bnds[1] if bnds[1] < bnds[5] - u[1] else <int_t>(bnds[5] - u[1])
        float_t mptr0[2]
        float_t mptr1[2]
        float_t mptr2[2]
    if sptr[2] == 0:
        init_upm_c(sptr, I, I0, u, di, dj, &bnds[0])
    ss = sptr[0]; fs = sptr[1]; mptr1[1] = NO_VAR; mptr2[1] = NO_VAR
    for k in range(max_iter):
        mse_bi(mptr0, I, I0, di, dj, u[0] + ss, u[1] + fs)
        mu = MU_C * (mptr0[1] * sptr[3])**0.25 / sqrt(sptr[2])
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
                  float_t[::1] di, float_t[::1] dj, int_t wss, int_t wfs, float_t ls,
                  int_t max_iter=500, float_t x_tol=1e-12):
    """
    Update the pixel map based on the Newton's method
    The update procedure minimizes the mean-squared-error (MSE) between the measured frames and the reference image

    I_n - measured frames
    W - whitefield
    I0 - reference image
    u0 - pixel mapping between the data at the detector's plane
         and the reference image at the reference plane
    di, dj - sample translations along slow and fast axis in pixels
    wss, wfs - search window size along slow and fast axis
    ls - length scale in pixels
    x_tol - tolerance for termination by the change of the solution
    max_iter - maximum number of iterations
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int_t aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int_t max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] u = np.empty((2, b, c), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
        float_t[:, ::1] sptr = np.zeros((max_threads, 4), dtype=np.float64) # ss, fs, l1, rss
        int_t bnds[6] # wss, wfs, di0, di1, dj0, dj1
    bnds[0] = wss; bnds[1] = wfs
    bnds[2] = <int_t>(min_float(&di[0], a)); bnds[3] = <int_t>(max_float(&di[0], a)) + aa
    bnds[4] = <int_t>(min_float(&dj[0], a)); bnds[5] = <int_t>(max_float(&dj[0], a)) + bb
    u[...] = u0
    for k in prange(c, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            sptr[t, 3] = krig_data_c(I[t], I_n, W, u0, j, k, ls)
            newton_1d_c(sptr[t], I[t], I0, u[:, j, k], di, dj, bnds, max_iter, x_tol)
    return np.asarray(u)

def total_mse(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
              float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj, float_t ls):
    """
    Return the total mean squared error

    I_n - measured frames
    W - whitefield
    I0 - reference image
    u - pixel mapping between the data at the detector's plane
        and the reference image at the reference plane
    di, dj - sample translations along slow and fast axis in pixels
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2]
        int_t aa = I0.shape[0], bb = I0.shape[1], j, k, t
        int_t max_threads = openmp.omp_get_max_threads()
        float_t err = 0
        float_t[:, ::1] mptr = NO_VAR * np.ones((max_threads, 2), dtype=np.float64)
        float_t[:, ::1] I = np.empty((max_threads, a), dtype=np.float64)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], I_n, W, u, j, k, ls)
            mse_bi(&mptr[t, 0], I[t], I0, di, dj, u[0, j, k], u[1, j, k])
            err += mptr[t, 0]
    return err / b / c