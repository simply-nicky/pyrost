cimport numpy as cnp
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, exp, pi, floor
from cython.parallel import prange
cimport openmp

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t
ctypedef cnp.npy_bool bool_t
DEF FLOAT_MAX = 1.7976931348623157e+308

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

cdef float_t mse_bi(float_t[:] I, float_t[:, ::1] I0, float_t W, float_t[::1] di,
                    float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, ss1, fs0, fs1
        float_t mse = 0, var = 0, ss, fs, dss, dfs, I0_bi
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
            mse += (I[i] - W * I0_bi)**2
            var += (I[i] - W)**2
    return mse / var

cdef float_t mse_nobi(float_t[:] I, float_t[:, ::1] I0, float_t W, float_t[::1] di,
                      float_t[::1] dj, float_t ux, float_t uy) nogil:
    cdef:
        int_t a = I.shape[0], aa = I0.shape[0], bb = I0.shape[1]
        int_t i, ss0, fs0
        float_t mse = 0, var = 0, ss, fs
    for i in range(a):
        ss = ux - di[i]
        fs = uy - dj[i]
        if ss >= 0 and ss <= aa - 1 and fs >= 0 and fs <= bb - 1:
            ss0 = <int_t>(floor(ss))
            fs0 = <int_t>(floor(fs))
            mse += (I[i] - W * I0[ss0, fs0])**2
            var += (I[i] - W)**2
    return mse / var

cdef void search_2d(float_t[:] I, float_t[:, ::1] I0, float_t[::1] di, float_t[::1] dj,
                    float_t[:] u, float_t W, int_t i0, int_t i1, int_t j0, int_t j1,
                    float_t dss, float_t dfs) nogil:
    cdef:
        int_t j = j0, i_min = 0, j_min = 0, i, jj
        float_t mse_min = FLOAT_MAX, mse = 0, df_ss, df_fs
    for i in range(i0, i1):
        if i - i0:
            df_ss = (mse_bi(I, I0, W, di, dj, u[0] + i - 0.5 + dss, u[1] + j) - \
                     mse_bi(I, I0, W, di, dj, u[0] + i - 0.5 - dss, u[1] + j)) / 2 / dss
            mse += df_ss
            if mse < mse_min:
                mse_min = mse; i_min = i; j_min = j
        if (i - i0) % 2:
            for jj in range(j0 + 1, j1):
                j = j1 + j0 - jj - 1
                df_fs = (mse_bi(I, I0, W, di, dj, u[0] + i, u[1] + j - 0.5 + dfs) - \
                         mse_bi(I, I0, W, di, dj, u[0] + i, u[1] + j - 0.5 - dfs)) / 2 / dfs
                mse += df_fs
                if mse < mse_min:
                    mse_min = mse; i_min = i; j_min = j
        else:
            for j in range(j0 + 1, j1):
                df_fs = (mse_bi(I, I0, W, di, dj, u[0] + i, u[1] + j - 0.5 + dfs) - \
                         mse_bi(I, I0, W, di, dj, u[0] + i, u[1] + j - 0.5 - dfs)) / 2 / dfs
                mse += df_fs
                if mse < mse_min:
                    mse_min = mse; i_min = i; j_min = j
    u[0] += i_min; u[1] += j_min

def update_pixel_map(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
                     float_t[:, :, ::1] u0, float_t[::1] di, float_t[::1] dj,
                     float_t dss, float_t dfs, uint_t wss=1, uint_t wfs=100):
    """
    Update the pixel map

    I_n - measured data
    W - whitefield
    I0 - reference image
    u0 - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    dss, dfs - average sample translations along slow and fast axis in pixels
    wss, wfs - search window size along slow and fast axis
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], j, k
        int_t aa = I0.shape[0], bb = I0.shape[1]
        float_t di0 = min_float(&di[0], a), di1 = max_float(&di[0], a)
        float_t dj0 = min_float(&dj[0], a), dj1 = max_float(&dj[0], a)
        float_t i0, i1, j0, j1
        float_t[:, :, ::1] u = np.empty((2, b, c), dtype=np.float64)
    u[...] = u0
    for j in prange(b, schedule='guided', nogil=True):
        for k in range(c):
            # Define window search bounds
            i0 = -<float_t>(wss) if wss < u[0, j, k] - di0 else di0 - u[0, j, k]
            i1 = <float_t>(wss) if wss < di1 - u[0, j, k] + aa else di1 - u[0, j, k] + aa
            j0 = -<float_t>(wfs) if wfs < u[1, j, k] - dj0 else dj0 - u[1, j, k]
            j1 = <float_t>(wfs) if wfs < dj1 - u[1, j, k] + bb else dj1 - u[1, j, k] + bb
            
            # Execute pixel map search
            search_2d(I_n[:, j, k], I0, di, dj, u[:, j, k], W[j, k],
                      <int_t>(i0), <int_t>(i1), <int_t>(j0), <int_t>(j1), dss, dfs)
    return np.asarray(u)

def total_mse(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] I0,
              float_t[:, :, ::1] u, float_t[::1] di, float_t[::1] dj):
    """
    Return total mean squared error

    I_n - measured data
    W - whitefield
    I0 - reference image
    u - pixel map
    di, dj - sample translations along slow and fast axis in pixels
    """
    cdef:
        int_t b = I_n.shape[1], c = I_n.shape[2], j, k
        float_t err = 0
    for j in range(b):
        for k in range(c):
            err += mse_bi(I_n[:, j, k], I0, W[j, k], di, dj, u[0, j, k], u[1, j, k])
    return err / b / c