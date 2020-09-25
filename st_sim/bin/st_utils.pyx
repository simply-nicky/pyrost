cimport numpy as cnp
import numpy as np
from cython_gsl cimport *
from libc.math cimport sqrt, cos, sin, exp, pi
from cython.parallel import prange
cimport openmp

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.float32_t float32_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

cdef float32_t wirthselect_float(float32_t[:] array, int k) nogil:
    cdef:
        int_t l = 0, m = array.shape[0] - 1, i, j
        float32_t x, tmp 
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

def make_whitefield_st(float32_t[:, :, ::1] data, uint8_t[:, ::1] mask):
    """
    Return whitefield based on median filtering of the stack of frames

    data - stack of frames
    mask - bad pixel mask
    """
    cdef:
        int_t a = data.shape[0], b = data.shape[1], c = data.shape[2], i, j, k
        int_t max_threads = openmp.omp_get_max_threads()
        float32_t[:, ::1] wf = np.empty((b, c), dtype=np.float32)
        float32_t[:, ::1] array = np.empty((max_threads, a), dtype=np.float32)
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

def make_reference(float_t[:, :, ::1] I_n, float_t[:, ::1] W, float_t[:, ::1] dij, float_t[:, :, ::1] u, float_t ls):
    """
    Return a reference image 

    I_n - measured data
    W - whitefield
    dij - sample translations
    u - pixel map
    ls - length scale in pixels
    """
    cdef:
        int_t a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, j
        float_t n0 = -min_float(&u[0, 0, 0], b * c) + max_float(&dij[0, 0], a)
        float_t m0 = -min_float(&u[1, 0, 0], b * c) + max_float(&dij[1, 0], a)
        int_t aa = <int_t>(max_float(&u[0, 0, 0], b * c) - min_float(&dij[0, 0], a) + n0) + 1
        int_t bb = <int_t>(max_float(&u[1, 0, 0], b * c) - min_float(&dij[1, 0], a) + m0) + 1
        float_t[:, ::1] I0 = np.zeros((aa, bb), dtype=np.float64)
        float_t[:, ::1] w0 = np.zeros((aa, bb), dtype=np.float64)
    for i in range(a):
        frame_reference(I0, w0, I_n[i], W, u, dij[0, i] - n0, dij[1, i] - m0, ls)
    for i in range(aa):
        for j in range(bb):
            if w0[i, j]:
                I0[i, j] /= w0[i, j]
    return np.asarray(I0), n0, m0