#include "st_utils.h"

typedef int (*fft_func)(void *plan, double complex *inp);
typedef int (*rfft_func)(void *plan, double *inp, size_t npts);

int fft_convolve_np(double *out, const double *inp, const double *krn,
    size_t isize, size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode,
    double cval, unsigned threads);

int fft_convolve_fftw(double *out, const double *inp, const double *krn,
    size_t isize, size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode,
    double cval, unsigned threads);