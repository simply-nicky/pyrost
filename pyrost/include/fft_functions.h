#ifndef FFT_FUNCTIONS_H
#define FFT_FUNCTIONS_H
#include "include.h"
#include "array.h"

int rfft_convolve_np(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads);

int cfft_convolve_np(double complex *out, double complex *inp, int ndim, size_t *dims,
    double complex *krn, size_t ksize, int axis, EXTEND_MODE mode, double complex cval,
    unsigned threads);

int rfft_convolve_fftw(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads);

int cfft_convolve_fftw(double complex *out, double complex *inp, int ndim, size_t *dims,
    double complex *krn, size_t ksize, int axis, EXTEND_MODE mode, double complex cval,
    unsigned threads);

int rsc_np(double complex *out, double complex *inp, int ndim, size_t *dims,
    int axis, double dx0, double dx, double z, double wl, unsigned threads);

int rsc_fftw(double complex *out, double complex *inp, int ndim, size_t *dims,
    int axis, double dx0, double dx, double z, double wl, unsigned threads);

int fraunhofer_np(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads);

int fraunhofer_fftw(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads);

typedef int (*rconvolve_func)(double*, double*, int, size_t*, double*,
    size_t, int, EXTEND_MODE, double, unsigned);

typedef int (*cconvolve_func)(double complex*, double complex*, int, size_t*, double complex*,
    size_t, int, EXTEND_MODE, double complex, unsigned);

int gauss_kernel1d(double *out, double sigma, unsigned order, size_t ksize, int step);

int gauss_filter_r(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    rconvolve_func fft_convolve);

int gauss_filter_c(double complex *out, double complex *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double complex cval, double truncate, unsigned threads,
    cconvolve_func fft_convolve);

int gauss_grad_mag_r(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    rconvolve_func fft_convolve);

int gauss_grad_mag_c(double *out, double complex *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double complex cval, double truncate, unsigned threads,
    cconvolve_func fft_convolve);

#endif