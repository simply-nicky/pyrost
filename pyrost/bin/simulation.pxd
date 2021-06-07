cdef extern from "st_gaussian.h":
    void gauss_kernel1d(double *out, double sigma, unsigned order, unsigned long ksize) nogil
    
    void gauss_filter_fftw(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                           unsigned *order, int mode, double cval, double truncate, unsigned threads) nogil

    int gauss_filter_np(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                        unsigned *order, int mode, double cval, double truncate, unsigned threads) nogil

    void gauss_grad_fftw(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                         int mode, double cval, double truncate, unsigned threads) nogil

    int gauss_grad_np(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                      int mode, double cval, double truncate, unsigned threads) nogil

cdef extern from "st_waveprop_fftw.h":
    unsigned long next_fast_len_fftw(unsigned long target) nogil

    void rsc_fftw(complex *out, complex *inp, unsigned long isize, unsigned long npts, unsigned long istride,
                  double dx0, double dx, double z, double wl, unsigned threads) nogil

    void fraunhofer_fftw(complex *out, complex *inp, unsigned long isize, unsigned long npts, unsigned long istride,
                         double dx0, double dx, double z, double wl, unsigned threads) nogil

    void fft_convolve_fftw(double *out, double *inp, double *krn, unsigned long isize,
                           unsigned long npts, unsigned long istride, unsigned long ksize,
                           int mode, double cval, unsigned threads) nogil

cdef extern from "st_waveprop_np.h":
    unsigned long good_size(unsigned long n) nogil

    int rsc_np(complex *out, complex *inp, unsigned long isize, unsigned long npts, unsigned long istride,
               double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fraunhofer_np(complex *out, complex *inp, unsigned long isize, unsigned long npts, unsigned long istride,
                      double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fft_convolve_np(double *out, double *inp, double *krn, unsigned long isize,
                        unsigned long npts, unsigned long istride, unsigned long ksize,
                        int mode, double cval, unsigned threads) nogil

cdef extern from "st_utils.h":
    void barcode_bars(double *bars, unsigned long size, double x0, double b_dx, double rd, long seed) nogil

    void ml_profile(complex *out, double *inp, double *layers, unsigned long isize, unsigned long lsize,
                    unsigned long nlyr, complex mt0, complex mt1, complex mt2, double sgm, unsigned threads) nogil

    void frames(double *out, double *pfx, double *pfy, double dx, double dy, unsigned long *ishape,
                unsigned long *oshape, long seed, unsigned threads) nogil

    void whitefield(void *out, void *data, unsigned char *mask, unsigned long isize, unsigned long npts,
                    unsigned long istride, unsigned long size, int (*compar)(const void*, const void*),
                    unsigned threads) nogil

    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_long(void *a, void *b) nogil

cdef extern from "fftw3.h":
    void fftw_init_threads() nogil
    void fftw_cleanup_threads() nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4
