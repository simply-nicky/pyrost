#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cdef extern from "Python.h":
    int Py_AtExit(void(*func)())

cdef extern from "fft_functions.h":
    int fft_convolve_fftw(double *out, double *inp, double *krn, unsigned long isize,
                           unsigned long npts, unsigned long istride, unsigned long ksize,
                           int mode, double cval, unsigned threads) nogil

    int fft_convolve_np(double *out, double *inp, double *krn, unsigned long isize,
                        unsigned long npts, unsigned long istride, unsigned long ksize,
                        int mode, double cval, unsigned threads) nogil

cdef extern from "fftw3.h":
    void fftw_init_threads() nogil
    void fftw_cleanup_threads() nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4