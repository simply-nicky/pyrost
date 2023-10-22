import numpy as np
import cython
from libc.math cimport sqrt

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def rsc_wp(np.ndarray wft not None, double dx0, double dx, double z,
           double wl, int axis=-1, str backend='numpy', unsigned num_threads=1):
    wft = check_array(wft, np.NPY_COMPLEX128)

    cdef int fail = 0
    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp *dims = wft.shape
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    with nogil:
        if backend == 'fftw':
            fail = rsc_fftw(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = rsc_np(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def fraunhofer_wp(np.ndarray wft not None, double dx0, double dx, double z,
                  double wl, int axis=-1, str backend='numpy', unsigned num_threads=1):
    wft = check_array(wft, np.NPY_COMPLEX128)

    cdef int fail = 0
    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp *dims = wft.shape
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    with nogil:
        if backend == 'fftw':
            fail = fraunhofer_fftw(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = fraunhofer_np(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def bar_positions(double x0, double x1, double b_dx, double rd, long seed):
    cdef np.npy_intp size = 2 * (<np.npy_intp>((x1 - x0) / 2 / b_dx) + 1) if x1 > x0 else 0
    cdef np.npy_intp *dims = [size,]
    cdef np.ndarray[double] bars = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_bars = <double *>np.PyArray_DATA(bars)
    if size:
        with nogil:
            barcode_bars(_bars, size, x0, b_dx, rd, seed)
    return bars

cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray layers, complex t0,
                                   complex t1, double sigma, unsigned num_threads):
    x_arr = check_array(x_arr, np.NPY_FLOAT64)
    layers = check_array(layers, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef int ndim = x_arr.ndim
    cdef np.npy_intp *dims = x_arr.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)

    cdef np.npy_intp isize = np.PyArray_SIZE(x_arr)
    cdef np.npy_intp lsize = np.PyArray_SIZE(layers)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef double *_x = <double *>np.PyArray_DATA(x_arr)
    cdef double *_lyrs = <double *>np.PyArray_DATA(layers)
    with nogil:
        fail = ml_profile(_out, _x, isize, _lyrs, lsize, t0, t1, sigma, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def barcode_profile(np.ndarray x_arr not None, np.ndarray bars not None, double bulk_atn,
                    double bar_atn, double bar_sigma, unsigned num_threads=1):
    cdef complex t0 = sqrt(1.0 - bulk_atn)
    cdef complex t1 = sqrt(1.0 - bulk_atn - bar_atn)
    return ml_profile_wrapper(x_arr, bars, t0, t1, bar_sigma, num_threads)

def mll_profile(np.ndarray x_arr not None, np.ndarray layers not None, complex t0,
                complex t1, double sigma, unsigned num_threads=1):
    return ml_profile_wrapper(x_arr, layers, t0, t1, sigma, num_threads)

def make_frames(np.ndarray pfx not None, np.ndarray pfy not None, double dx, double dy,
                tuple shape, long seed, unsigned num_threads=1):
    pfx = check_array(pfx, np.NPY_FLOAT64)
    pfy = check_array(pfy, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef np.npy_intp *oshape = [pfx.shape[0], <np.npy_intp>(shape[0]), <np.npy_intp>(shape[1])]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(3, oshape, np.NPY_FLOAT64)
    cdef unsigned long *_ishape = [<unsigned long>(pfx.shape[0]), <unsigned long>(pfy.shape[0]),
                                   <unsigned long>(pfx.shape[1])]
    cdef unsigned long *_oshape = [<unsigned long>(oshape[0]), <unsigned long>(oshape[1]), <unsigned long>(oshape[2])]
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_pfx = <double *>np.PyArray_DATA(pfx)
    cdef double *_pfy = <double *>np.PyArray_DATA(pfy)
    with nogil:
        fail = frames(_out, _pfx, _pfy, dx, dy, _ishape, _oshape, seed, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out