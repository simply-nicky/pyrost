#ifndef POCKET_FFT_H
#define POCKET_FFT_H
#include "include.h"

struct cfft_plan_i;
typedef struct cfft_plan_i * cfft_plan;
struct rfft_plan_i;
typedef struct rfft_plan_i * rfft_plan;

cfft_plan make_cfft_plan (size_t length);

void destroy_cfft_plan (cfft_plan plan);

rfft_plan make_rfft_plan(size_t length);

void destroy_rfft_plan(rfft_plan plan);

NOINLINE size_t good_size(size_t n);

int fft_np(void *plan, double complex *inp);

int ifft_np(void *plan, double complex *inp);

int rfft_np(void *plan, double *inp, size_t npts);

int irfft_np(void *plan, double *inp, size_t npts);

size_t next_fast_len_fftw(size_t target);

int fft_fftw(void *plan, double complex *inp);

int ifft_fftw(void *plan, double complex *inp);

int rfft_fftw(void *plan, double *inp, size_t npts);

int irfft_fftw(void *plan, double *inp, size_t npts);

#endif