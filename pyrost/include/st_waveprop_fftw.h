#ifndef ST_WAVEPROP_H
#define ST_WAVEPROP_H
#include "st_include.h"

NOINLINE size_t next_fast_len_fftw(size_t target);

void rsc_fftw(double complex *out, const double complex *inp, size_t isize, size_t npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads);

void fraunhofer_fftw(double complex *out, const double complex *inp, size_t isize, size_t npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads);

void fft_convolve_fftw(double *out, const double *inp, const double *krn, size_t isize,
    size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode, double cval, unsigned threads);

#endif