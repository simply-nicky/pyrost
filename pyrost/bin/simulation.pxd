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

    int rsc_np(double complex *out, double complex *inp, int ndim, unsigned long *dims,
               int axis, double dx0, double dx, double z, double wl, unsigned threads) nogil

    int rsc_fftw(double complex *out, double complex *inp, int ndim, unsigned long *dims,
                 int axis, double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fraunhofer_np(double complex *out, double complex *inp, int ndim, unsigned long *dims, int axis,
                      double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fraunhofer_fftw(double complex *out, double complex *inp, int ndim, unsigned long *dims, int axis,
                        double dx0, double dx, double z, double wl, unsigned threads) nogil

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

cdef extern from "routines.h":
    void barcode_bars(double *bars, unsigned long size, double x0, double b_dx, double rd, long seed) nogil

    int ml_profile(complex *out, double *inp, unsigned long isize, double *layers, unsigned long lsize, 
                   complex t0, complex t1, double sgm, unsigned threads) nogil

    int frames(double *out, double *pfx, double *pfy, double dx, double dy, unsigned long *ishape,
               unsigned long *oshape, long seed, unsigned threads) nogil

cdef extern from "median.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, int ndim,
                        unsigned long *dims, unsigned long item_size, unsigned long *fsize, int mode,
                        void *cval, int (*compar)(void*, void*), unsigned threads) nogil

cdef extern from "fftw3.h":
    void fftw_init_threads() nogil
    void fftw_cleanup_threads() nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4

cdef int extend_mode_to_code(str mode) except -1
cdef np.ndarray check_array(np.ndarray array, int type_num)
cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num)
cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num)
cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray layers, complex t0,
                                   complex t1, double sigma, unsigned num_threads)
