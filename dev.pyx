#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from libc.math cimport log
from libc.string cimport memcmp
from libc.math cimport sqrt, exp, pi, floor, ceil
cimport openmp
from libc.math cimport sqrt, exp, pi, erf, floor, ceil, sin, cos
from libc.time cimport time, time_t
from libc.string cimport memcpy

cdef extern from "pyrost/bin/fft.c":

    int good_size_real(int n)
    int good_size_cmplx(int n)

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def init_fftw():
    fftw_init_threads()
    
    if Py_AtExit(fftw_cleanup_threads) != 0:
        raise ImportError('Failed to register the fftw library shutdown callback')

init_fftw()

def next_fast_len(np.npy_intp target, str backend='fftw'):
    if backend == 'fftw':
        return next_fast_len_fftw(target)
    elif backend == 'numpy':
        return good_size(target)
    else:
        raise ValueError('{:s} is invalid backend'.format(backend))

cdef int extend_mode_to_code(str mode) except -1:
    if mode == 'constant':
        return EXTEND_CONSTANT
    elif mode == 'nearest':
        return EXTEND_NEAREST
    elif mode == 'mirror':
        return EXTEND_MIRROR
    elif mode == 'reflect':
        return EXTEND_REFLECT
    elif mode == 'wrap':
        return EXTEND_WRAP
    else:
        raise RuntimeError('boundary mode not supported')

def extend_line(inp: np.ndarray, new_size: cython.ulong, axis: cython.int=-1,
                mode: str='reflect', cval: cython.double=0.0, num_threads: cython.uint=1) -> np.ndarray:
    inp = np.PyArray_GETCONTIGUOUS(inp)
    inp = np.PyArray_Cast(inp, np.NPY_FLOAT64)

    cdef np.npy_intp isize = inp.size
    cdef int ndim = inp.ndim
    axis = axis if axis >= 0 else ndim + axis
    cdef np.npy_intp istride = np.PyArray_STRIDE(inp, axis) / sizeof(double)
    cdef np.npy_intp *dims = inp.shape
    dims[axis] = new_size
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef int _mode = extend_mode_to_code(mode)

    with nogil:
        extend_line_double(_out, _inp, _mode, cval, new_size, isize, istride, num_threads)
    return out

cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

def gaussian_kernel(sigma: double, order: cython.uint=0, truncate: cython.double=4.) -> np.ndarray:
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0])
    return out

def gaussian_filter(inp: np.ndarray, sigma: object, order: object=0, mode: str='reflect',
                    cval: cython.double=0., truncate: cython.double=4., backend: str='fftw',
                    num_threads: cython.uint=1) -> np.ndarray:
    inp = np.PyArray_GETCONTIGUOUS(inp)
    inp = np.PyArray_Cast(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    if sigmas is None:
        return None
    cdef np.ndarray orders = normalize_sequence(order, ndim, np.NPY_UINT32)
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            gauss_filter_fftw(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate, num_threads)
        elif backend == 'numpy':
            fail = gauss_filter_np(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate)
            if fail:
                raise RuntimeError('NumPy FFT exited with error')
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    return out

def gaussian_gradient_magnitude(inp: np.ndarray, sigma: object, mode: str='mirror', cval: cython.double=0.,
                                truncate: cython.double=4., backend: str='fftw', num_threads: cython.uint=1) -> np.ndarray:
    inp = np.PyArray_GETCONTIGUOUS(inp)
    inp = np.PyArray_Cast(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            gauss_grad_fftw(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate, num_threads)
        elif backend == 'numpy':
            fail = gauss_grad_np(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate)
            if fail:
                raise RuntimeError('NumPy FFT exited with error')
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    return out

def rsc_wp(wft: np.ndarray, dx0: cython.double, dx: cython.double, z: cython.double, wl: cython.double,
           axis: cython.int=-1, backend: str='fftw', num_threads: cython.uint=1) -> np.ndarray:
    wft = np.PyArray_GETCONTIGUOUS(wft)
    wft = np.PyArray_Cast(wft, np.NPY_COMPLEX128)

    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    cdef np.npy_intp npts = np.PyArray_DIM(wft, axis)
    cdef int howmany = isize / npts
    if axis != ndim - 1:
        wft = <np.ndarray>np.PyArray_SwapAxes(wft, axis, ndim - 1)
    cdef np.npy_intp *dims = wft.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    cdef int fail = 0
    with nogil:
        if backend == 'fftw':
            rsc_fftw(_out, _inp, howmany, npts, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = rsc_np(_out, _inp, howmany, npts, dx0, dx, z, wl)
            if fail:
                raise RuntimeError('NumPy FFT exited with error')
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if axis != ndim - 1:
        out = <np.ndarray>np.PyArray_SwapAxes(wft, axis, ndim - 1)
    return out

def fraunhofer_wp(wft: np.ndarray, dx0: cython.double, dx: cython.double, z: cython.double,
                  wl: cython.double, axis: cython.int=-1, backend: str='fftw', num_threads: cython.uint=1) -> np.ndarray:
    wft = np.PyArray_GETCONTIGUOUS(wft)
    wft = np.PyArray_Cast(wft, np.NPY_COMPLEX128)

    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    cdef np.npy_intp npts = np.PyArray_DIM(wft, axis)
    cdef int howmany = isize / npts
    if axis != ndim - 1:
        wft = <np.ndarray>np.PyArray_SwapAxes(wft, axis, ndim - 1)
    cdef np.npy_intp *dims = wft.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    cdef int fail = 0
    with nogil:
        if backend == 'fftw':
            fraunhofer_fftw(_out, _inp, howmany, npts, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = fraunhofer_np(_out, _inp, howmany, npts, dx0, dx, z, wl)
            if fail:
                raise RuntimeError('NumPy FFT exited with error')
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if axis != ndim - 1:
        out = <np.ndarray>np.PyArray_SwapAxes(wft, axis, ndim - 1)
    return out

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: cython.int=-1, mode: str='constant',
                 cval: cython.double=0.0, backend: str='fftw', num_threads: cython.uint=1) -> np.ndarray:
    array = np.PyArray_GETCONTIGUOUS(array)
    array = np.PyArray_Cast(array, np.NPY_FLOAT64)
    kernel = np.PyArray_GETCONTIGUOUS(kernel)
    kernel = np.PyArray_Cast(kernel, np.NPY_FLOAT64)

    cdef np.npy_intp isize = np.PyArray_SIZE(array)
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    cdef np.npy_intp npts = np.PyArray_DIM(array, axis)
    cdef int istride = np.PyArray_STRIDE(array, axis) / np.PyArray_ITEMSIZE(array)
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = array.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(array)
    cdef double *_krn = <double *>np.PyArray_DATA(kernel)
    cdef int fail = 0
    with nogil:
        if backend == 'fftw':
            fft_convolve_fftw(_out, _inp, _krn, isize, npts, istride, ksize, _mode, cval, num_threads)
        elif backend == 'numpy':
            fail = fft_convolve_np(_out, _inp, _krn, isize, npts, istride, ksize, _mode, cval)
            if fail:
                raise RuntimeError('NumPy FFT exited with error')
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    return out

cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray[double] layers, complex mt0,
                                   complex mt1, complex mt2, double sigma, unsigned num_threads):
    x_arr = np.PyArray_GETCONTIGUOUS(x_arr)
    x_arr = np.PyArray_Cast(x_arr, np.NPY_FLOAT64)
    layers = np.PyArray_GETCONTIGUOUS(layers)
    layers = np.PyArray_Cast(layers, np.NPY_FLOAT64)

    cdef np.npy_intp npts = np.PyArray_SIZE(x_arr)
    cdef np.npy_intp nlyr = np.PyArray_SIZE(layers)
    cdef int ndim = x_arr.ndim
    cdef np.npy_intp *dims = x_arr.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef double *_x = <double *>np.PyArray_DATA(x_arr)
    cdef double *_lyrs = <double *>np.PyArray_DATA(layers)
    with nogil:
        ml_profile(_out, _x, _lyrs, npts, nlyr, mt0, mt1, mt2, sigma, num_threads)
    return out

def bar_positions(x0: cython.double, x1: cython.double, b_dx: cython.double, rd: cython.double, seed: cython.uint) -> np.ndarray:
    cdef np.npy_intp size = 2 * (<np.npy_intp>((x1 - x0) / 2 / b_dx) + 1) if x1 > x0 else 0
    cdef np.npy_intp *dims = [size,]
    cdef np.ndarray[double] bars = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_bars = <double *>np.PyArray_DATA(bars)
    if size:
        with nogil:
            barcode_bars(_bars, size, x0, b_dx, rd, seed)
    return bars

def barcode_profile(x_arr: np.ndarray, bars: np.ndarray, bulk_atn: cython.double,
                    bar_atn: cython.double, bar_sigma: cython.double, num_threads: cython.uint) -> np.ndarray:
    cdef complex mt0 = -1j * log(1 - bulk_atn)
    cdef complex mt1 = -1j * log(1 - bar_atn)
    return ml_profile_wrapper(x_arr, bars, mt0, mt1, 0., bar_sigma, num_threads)

def mll_profile(x_arr: np.ndarray, layers: np.ndarray, complex mt0,
                complex mt1, sigma: cython.double, num_threads: cython.uint) -> np.ndarray:
    return ml_profile_wrapper(x_arr, layers, 0., mt0, mt1, sigma, num_threads)

def make_frames(pfx: np.ndarray, pfy: np.ndarray, wfx: np.ndarray, wfy: np.ndarray, dx: cython.double,
                dy: cython.double, seed: cython.long, num_threads: cython.uint) -> np.ndarray:
    pfx = np.PyArray_GETCONTIGUOUS(pfx)
    pfx = np.PyArray_Cast(pfx, np.NPY_FLOAT64)
    pfy = np.PyArray_GETCONTIGUOUS(pfy)
    pfy = np.PyArray_Cast(pfy, np.NPY_FLOAT64)
    wfx = np.PyArray_GETCONTIGUOUS(wfx)
    wfx = np.PyArray_Cast(wfx, np.NPY_FLOAT64)
    wfy = np.PyArray_GETCONTIGUOUS(wfy)
    wfy = np.PyArray_Cast(wfy, np.NPY_FLOAT64)

    cdef np.npy_intp *xdim = pfx.shape
    cdef np.npy_intp ypts = np.PyArray_SIZE(pfy)
    cdef np.npy_intp fs_size = np.PyArray_SIZE(wfx)
    cdef np.npy_intp ss_size = np.PyArray_SIZE(wfy)
    cdef np.npy_intp *dim = [xdim[0], ss_size, fs_size]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(3, dim, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_pfx = <double *>np.PyArray_DATA(pfx)
    cdef double *_pfy = <double *>np.PyArray_DATA(pfy)
    cdef double *_wfx = <double *>np.PyArray_DATA(wfx)
    cdef double *_wfy = <double *>np.PyArray_DATA(wfy)
    with nogil:
        frames(_out, _pfx, _pfy, _wfx, _wfy, dx, dy, xdim[1], ypts, dim[0], dim[1], dim[2], seed, num_threads)
    return out

def make_whitefield(data: np.ndarray, mask: np.ndarray, axis: cython.int=0, num_threads: cython.uint=1) -> np.ndarray:
    data = np.PyArray_GETCONTIGUOUS(data)
    mask = np.PyArray_GETCONTIGUOUS(mask)

    if not np.PyArray_ISBOOL(mask):
        raise TypeError('mask array must be of boolean type')
    cdef int ndim = data.ndim
    if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
        raise ValueError('mask and data arrays must have identical shapes')
    axis = axis if axis >= 0 else ndim + axis
    if axis != 0:
        data = <np.ndarray>np.PyArray_SwapAxes(data, axis, 0)
        mask = <np.ndarray>np.PyArray_SwapAxes(mask, axis, 0)
    cdef np.npy_intp *dims = data.shape
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim - 1, dims + 1, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef np.npy_intp frame_size = np.PyArray_SIZE(data) // dims[0]
    cdef np.npy_intp size = 0
    with nogil:
        if type_num == np.NPY_FLOAT64:
                whitefield(_out, _data, _mask, dims[0], frame_size, 8, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
                whitefield(_out, _data, _mask, dims[0], frame_size, 4, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
                whitefield(_out, _data, _mask, dims[0], frame_size, 4, compare_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if axis != 0:
        data = <np.ndarray>np.PyArray_SwapAxes(data, 0, axis)
        mask = <np.ndarray>np.PyArray_SwapAxes(mask, 0, axis)
    return out

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.uint64_t uint_t
ctypedef np.complex128_t complex_t

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

cdef void rsc_type1_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h,
                      double dx0, double dx, double z, double wl) nogil:
    cdef:
        int n = u.shape[0], i
        double ph, dist
        complex_t u0
    for i in range(n):
        dist = (dx0 * (i - n // 2))**2 + z**2
        ph = 2 * pi / wl * sqrt(dist)
        k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + 1j * cos(ph)) / dist**0.75
    fft(&u[0], n)
    fft(&k[0], n)
    for i in range(n):
        ph = pi * (<double>(i) / n - (2 * i) // n)**2 * dx / dx0 * n
        u[i] = u[i] * k[i] * (cos(ph) - 1j * sin(ph))
        k[i] = cos(ph) - 1j * sin(ph)
        h[i] = cos(ph) + 1j * sin(ph)
    ifft(&u[0], n)
    ifft(&h[0], n)
    for i in range(n):
        u[i] = u[i] * h[i]
    fft(&u[0], n)
    for i in range(n // 2):
        u0 = u[i] * k[i]; u[i] = u[i + n // 2] * k[i + n // 2]
        u[i + n // 2] = u0

cdef void rsc_type2_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h,
                      double dx0, double dx, double z, double wl) nogil:
    cdef:
        int n = u.shape[0], i
        double ph, ph2, dist
    for i in range(n):
        dist = (dx * (i - n // 2))**2 + z**2
        ph = 2 * pi / wl * sqrt(dist)
        k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + 1j * cos(ph)) / dist**0.75
    fft(&k[0], n)
    for i in range(n):
        ph = pi * (<double>(i) / n - (2 * i) // n)**2 * dx0 / dx * n
        ph2 = pi * (i - n // 2)**2 * dx0 / dx / n
        u[i] = u[i] * (cos(ph2) + 1j * sin(ph2))
        k[i] = k[i] * (cos(ph) + 1j * sin(ph))
        h[i] = cos(ph2) - 1j * sin(ph2)
    fft(&u[0], n)
    fft(&h[0], n)
    for i in range(n):
        u[i] = u[i] * h[i]
    ifft(&u[0], n)
    for i in range(n):
        u[i] = u[i] * k[i]
    ifft(&u[0], n)

cdef void rsc_wp_c(complex_t[::1] u, complex_t[::1] k, complex_t[::1] h, complex_t[::1] u0,
                   double dx0, double dx, double z, double wl) nogil:
    cdef:
        int a = u0.shape[0], n = u.shape[0], i
    for i in range((n - a) // 2 + 1):
        u[i] = 0; u[n - 1 - i] = 0
    for i in range(a):
        u[i + (n - a) // 2] = u0[i]
    if dx0 >= dx:
        rsc_type1_c(u, k, h, dx0, dx, z, wl)
    else:
        rsc_type2_c(u, k, h, dx0, dx, z, wl)

def rsc_wp_scan(complex_t[::1] u0, double dx0, double dx, double[::1] z_arr, double wl):
    cdef:
        int a = z_arr.shape[0], b = u0.shape[0], i, t
        int n = good_size_cmplx(2 * b - 1)
        int max_threads = openmp.omp_get_max_threads()
        complex_t[:, ::1] u = np.empty((a, n), dtype=np.complex128)
        complex_t[:, ::1] k = np.empty((max_threads, n), dtype=np.complex128)
        complex_t[:, ::1] h = np.empty((max_threads, n), dtype=np.complex128)
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        rsc_wp_c(u[i], k[t], h[t], u0, dx0, dx, z_arr[i], wl)
    return np.asarray(u[:, (n - b) // 2:(n + b) // 2], order='C')

def st_update(I_n, dij, basis, x_ps, y_ps, z, df, sw_max=100, n_iter=5,
              filter=None, update_translations=False, verbose=False):
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
    W = st.make_whitefield(I_n, M, verbose=verbose)
    u, dij_pix, res = st.generate_pixel_map(W.shape, dij, basis,
                                            x_ps, y_ps, z,
                                            df, verbose=verbose)
    I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=False, verbose=verbose)

    es = []
    for i in range(n_iter):

        # calculate errors
        error_total = st.calc_error(I_n, M, W, dij_pix, I0, u, n0, m0, subpixel=True, verbose=verbose)[0]

        # store total error
        es.append(error_total)

        # update pixel map
        u = st.update_pixel_map(I_n, M, W, I0, u, n0, m0, dij_pix,
                                search_window=[1, sw_max], subpixel=True,
                                fill_bad_pix=False, integrate=False,
                                quadratic_refinement=False, verbose=verbose,
                                filter=filter)[0]

        # make reference image
        I0, n0, m0 = st.make_object_map(I_n, M, W, dij_pix, u, subpixel=True, verbose=verbose)

        # update translations
        if update_translations:
            dij_pix = st.update_translations(I_n, M, W, I0, u, n0, m0, dij_pix)[0]
    return {'u':u, 'I0':I0, 'errors':es, 'n0': n0, 'm0': m0}

def pixel_translations(basis, dij, df, z):
    dij_pix = (basis * dij[:, None]).sum(axis=-1)
    dij_pix /= (basis**2).sum(axis=-1) * df / z
    dij_pix -= dij_pix.mean(axis=0)
    return np.ascontiguousarray(dij_pix[:, 0]), np.ascontiguousarray(dij_pix[:, 1])

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

# def phase_fit(pixel_ab, x_ps, z, df, wl, max_order=2, pixels=None):
#     def errors(fit, x, y):
#         return np.polyval(fit[:max_order + 1], x - fit[max_order + 1]) - y

#     # Apply ROI
#     if pixels is None:
#         pixels = np.arange(pixel_ab.shape[0])
#     else:
#         pixel_ab = pixel_ab[pixels]
#     x_arr = pixels * x_ps / z * 1e3

#     # Calculate initial argument
#     x0 = np.zeros(max_order + 2)
#     u0 = gaussian_filter(pixel_ab, pixel_ab.shape[0] / 10)
#     if np.median(np.gradient(np.gradient(u0))) > 0:
#         idx = np.argmin(u0)
#     else:
#         idx = np.argmax(u0)
#     x0[max_order + 1] = x_arr[idx]
#     lb = -np.inf * np.ones(max_order + 2)
#     ub = np.inf * np.ones(max_order + 2)
#     lb[max_order + 1] = x_arr.min()
#     ub[max_order + 1] = x_arr.max()
        
#     # Perform least squares fitting
#     fit = least_squares(errors, x0, bounds=(lb, ub), loss='cauchy', jac='3-point',
#                         args=(x_arr, pixel_ab), xtol=1e-14, ftol=1e-14)
#     if np.linalg.det(fit.jac.T.dot(fit.jac)):
#         cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
#         err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
#     else:
#         err = np.zeros(fit.x.size)

#     # Convert the fit
#     ang_fit = fit.x * x_ps / z
#     ang_fit[:max_order + 1] /= np.geomspace((x_ps / z)**max_order, 1, max_order + 1)
#     ph_fit = np.zeros(max_order + 3)
#     ph_fit[:max_order + 1] = ang_fit[:max_order + 1] * 2 * np.pi / wl * df / np.linspace(max_order + 1, 1, max_order + 1)
#     ph_fit[max_order + 2] = ang_fit[max_order + 1]
#     ph_fit[max_order + 1] = -np.polyval(ph_fit[:max_order + 2], pixels * x_ps / z - ph_fit[max_order + 2]).mean()

#     # evaluating errors
#     r_sq = 1 - np.sum(errors(fit.x, pixels, pixel_ab)**2) / np.sum((pixel_ab.mean() - pixel_ab)**2)
#     return {'pixels': pixels, 'pix_fit': fit.x, 'ang_fit': ang_fit,
#             'pix_err': err, 'ph_fit': ph_fit, 'r_sq': r_sq, 'fit': fit}
