#ifndef ST_UTILS_H
#define ST_UTILS_H
#include "st_include.h"

// Extend line routines
typedef enum
{
    EXTEND_CONSTANT = 0,
    EXTEND_NEAREST = 1,
    EXTEND_MIRROR = 2,
    EXTEND_REFLECT = 3,
    EXTEND_WRAP = 4
} EXTEND_MODE;

void extend_line_complex(double complex *out, const double complex *inp, EXTEND_MODE mode,
    double complex cval, size_t osize, size_t isize, size_t istride);
void extend_line_double(double *out, const double *inp, EXTEND_MODE mode, double cval, size_t osize,
    size_t isize, size_t istride);

// Speckle Tracking utility functions
NOINLINE int compare_double(const void *a, const void *b);
NOINLINE int compare_float(const void *a, const void *b);
NOINLINE int compare_long(const void *a, const void *b);

size_t binary_search(const void *key, const void *array, size_t l, size_t r, size_t size,
    int (*compar)(const void*, const void*));

size_t searchsorted(const void *key, const void *base, size_t npts, size_t size,
    int (*compar)(const void*, const void*));

void barcode_bars(double *bars, size_t size, double x0, double b_dx, double rd, long seed);

void ml_profile(double complex *out, const double *inp, const double *layers, size_t isize, size_t lsize, 
    size_t nlyr, double complex mt0, double complex mt1, double complex mt2, double sgm, unsigned threads);

void frames(double *out, const double *pfx, const double *pfy, double dx, double dy, size_t *ishape, size_t *oshape,
    long seed, unsigned threads);

void whitefield(void *out, const void *data, const unsigned char *mask, size_t isize,
    size_t npts, size_t istride, size_t size, int (*compar)(const void*, const void*), unsigned threads);

#endif