#ifndef ROUTINES_H
#define ROUTINES_H
#include "include.h"
#include "array.h"

void barcode_bars(double *bars, size_t size, double x0, double b_dx, double rd, long seed);

int ml_profile(double complex *out, double *inp, size_t isize, double *layers, size_t lsize, 
    double complex t0, double complex t1, double sgm, unsigned threads);

int frames(double *out, double *pfx, double *pfy, double dx, double dy, size_t *idims, size_t *odims,
    long seed, unsigned threads);

void dot_double(void *out, line line1, line line2);
void dot_long(void *out, line line1, line line2);

int dot(void *out, void *inp1, int ndim1, size_t *dims1, int axis1, void *inp2, int ndim2, size_t *dims2,
    int axis2, size_t item_size, void (*dot_func)(void*, line, line), unsigned threads);

#endif