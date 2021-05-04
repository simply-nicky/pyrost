#ifndef ST_GAUSSIAN_H
#define ST_GAUSSIAN_H
#include "st_include.h"
#include "st_utils.h"

void gauss_kernel1d(double *out, double sigma, unsigned order, size_t ksize);

void gauss_filter_fftw(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads);

int gauss_filter_np(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads);

void gauss_grad_fftw(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads);

int gauss_grad_np(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads);

#endif