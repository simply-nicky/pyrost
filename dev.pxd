cdef extern from "Python.h":
    int Py_AtExit(void(*func)())

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
    ctypedef struct line:
        pass

    void barcode_bars(double *bars, unsigned long size, double x0, double b_dx, double rd, long seed) nogil

    int ml_profile(complex *out, double *inp, unsigned long isize, double *layers, unsigned long lsize, 
                   complex t0, complex t1, double sgm, unsigned threads) nogil

    int frames(double *out, double *pfx, double *pfy, double dx, double dy, unsigned long *ishape,
               unsigned long *oshape, long seed, unsigned threads) nogil

    void dot_double(void *out, line, line) nogil
    void dot_long(void *out, line, line) nogil

    int dot_c "dot" (void *out, void *inp1, int ndim1, unsigned long *dims1, int axis1, void *inp2, 
                     int ndim2, unsigned long *dims2, int axis2, unsigned long item_size,
                     void (*dot_func)(void*, line, line), unsigned threads) nogil

cdef extern from "median.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil

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

cdef extern from "gsl/gsl_rng.h":

    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    cdef gsl_rng_type *gsl_rng_mt19937

    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil

    unsigned long int gsl_rng_get(gsl_rng * r) nogil

    void gsl_rng_set(gsl_rng * r, unsigned long int seed) nogil

    void gsl_rng_free(gsl_rng * r) nogil

    double gsl_rng_uniform(gsl_rng * r) nogil
    unsigned long gsl_rng_uniform_int(gsl_rng * r, unsigned long n) nogil