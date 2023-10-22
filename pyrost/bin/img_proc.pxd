cimport numpy as np

ctypedef int (*rconvolve_func)(double*, double*, int, unsigned long*, double*,
                               unsigned long, int, int, double, unsigned)
ctypedef int (*cconvolve_func)(double complex*, double complex*, int, unsigned long*, double complex*,
                               unsigned long, int, int, double complex, unsigned)

cdef extern from "pocket_fft.h":
    unsigned long next_fast_len_fftw(unsigned long target) nogil
    unsigned long good_size(unsigned long n) nogil

cdef extern from "fft_functions.h":
    int rfft_convolve_fftw(double *out, double *inp, int ndim, unsigned long* dims, double *krn,
                           unsigned long ksize, int axis, int mode, double cval, unsigned threads) nogil

    int cfft_convolve_fftw(double complex *out, double complex *inp, int ndim, unsigned long* dims,
                           double complex *krn, unsigned long ksize, int axis, int mode, double complex cval,
                           unsigned threads) nogil

    int rfft_convolve_np(double *out, double *inp, int ndim, unsigned long* dims, double *krn,
                          unsigned long ksize, int axis, int mode, double cval, unsigned threads) nogil

    int cfft_convolve_np(double complex *out, double complex *inp, int ndim, unsigned long* dims,
                         double complex *krn, unsigned long ksize, int axis, int mode, double complex cval,
                         unsigned threads) nogil

    int gauss_kernel1d(double *out, double sigma, unsigned order, unsigned long ksize, int step) nogil

    int gauss_filter_r(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                       unsigned *order, int mode, double cval, double truncate, unsigned threads,
                       rconvolve_func fft_convolve) nogil

    int gauss_filter_c(double complex *out, double complex *inp, int ndim, unsigned long *dims,
                       double *sigma, unsigned *order, int mode, double complex cval, double truncate,
                       unsigned threads, cconvolve_func fft_convolve) nogil

    int gauss_grad_mag_r(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                         int mode, double cval, double truncate, unsigned threads,
                         rconvolve_func fft_convolve) nogil

    int gauss_grad_mag_c(double *out, double complex *inp, int ndim, unsigned long *dims,
                         double *sigma, int mode, double complex cval, double truncate, unsigned threads,
                         cconvolve_func fft_convolve) nogil

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

cdef extern from "median.h":
    double get_double(void *a) nogil
    double get_float(void *a) nogil
    double get_int(void *a) nogil
    double get_uint(void *a) nogil
    double get_ulong(void *a) nogil

    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, unsigned char *imask,
                        int ndim, unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                        unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                        unsigned threads) nogil

    int robust_mean_c "robust_mean" (double *out, void *inp, int ndim, unsigned long *dims,
                      unsigned long item_size, int axis, int (*compar)(void*, void*),
                      double (*getter)(void*), double r0, double r1, int n_iter, double lm,
                      unsigned threads) nogil 

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4

cdef inline int mode_to_code(str mode) except -1:
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
        raise RuntimeError(f'Invalid boundary mode: {mode}')

cdef inline np.ndarray check_array(np.ndarray array, int type_num):
    cdef np.ndarray out
    cdef int tn = np.PyArray_TYPE(array)
    if not np.PyArray_IS_C_CONTIGUOUS(array):
        array = np.PyArray_GETCONTIGUOUS(array)

    if tn != type_num:
        out = np.PyArray_SimpleNew(array.ndim, <np.npy_intp *>array.shape, type_num)
        np.PyArray_CastTo(out, array)
    else:
        out = array

    return out

cdef inline np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef inline np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = check_array(<np.ndarray>inp, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

cdef inline np.ndarray to_array(object inp, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, 1, type_num)
    elif np.PyArray_Check(inp):
        arr = check_array(<np.ndarray>inp, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    return arr