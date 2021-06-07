#ifndef ST_WAVEPROP_NP_H
#define ST_WAVEPROP_NP_H
#include "st_include.h"

NOINLINE size_t good_size(size_t n);

int rsc_np(double complex *out, const double complex *inp, size_t isize, size_t npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads);

int fraunhofer_np(double complex *out, const double complex *inp, size_t isize, size_t npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads);

int fft_convolve_np(double *out, const double *inp, const double *krn, size_t isize, size_t npts,
    size_t istride, size_t ksize, EXTEND_MODE mode, double cval, unsigned threads);

#endif