cimport numpy as np
from .img_proc import check_array

cdef extern from "fft_functions.h":
    int rsc_np(double complex *out, double complex *inp, int ndim, unsigned long *dims,
               int axis, double dx0, double dx, double z, double wl, unsigned threads) nogil

    int rsc_fftw(double complex *out, double complex *inp, int ndim, unsigned long *dims,
                 int axis, double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fraunhofer_np(double complex *out, double complex *inp, int ndim, unsigned long *dims, int axis,
                      double dx0, double dx, double z, double wl, unsigned threads) nogil

    int fraunhofer_fftw(double complex *out, double complex *inp, int ndim, unsigned long *dims, int axis,
                        double dx0, double dx, double z, double wl, unsigned threads) nogil

cdef extern from "routines.h":
    void barcode_bars(double *bars, unsigned long size, double x0, double b_dx, double rd, long seed) nogil

    int ml_profile(complex *out, double *inp, unsigned long isize, double *layers, unsigned long lsize, 
                   complex t0, complex t1, double sgm, unsigned threads) nogil

    int frames(double *out, double *pfx, double *pfy, double dx, double dy, unsigned long *ishape,
               unsigned long *oshape, long seed, unsigned threads) nogil
