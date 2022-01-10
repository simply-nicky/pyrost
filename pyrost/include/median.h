#ifndef MEDIAN_H
#define MEDIAN_H
#include "include.h"
#include "array.h"

typedef struct footprint_s
{
    int ndim;
    int npts;
    int counter;
    int *offsets;
    int *coordinates;
    size_t item_size;
    void *data;
} footprint_s;
typedef struct footprint_s *footprint;

footprint init_footprint(int ndim, size_t item_size, size_t *fsize);
void free_footprint(footprint fpt);

void update_footprint(footprint fpt, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval);

// Comparing functions
int compare_double(const void *a, const void *b);
int compare_float(const void *a, const void *b);
int compare_int(const void *a, const void *b);
int compare_uint(const void *a, const void *b);
int compare_ulong(const void *a, const void *b);

int median(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
    int axis, int (*compar)(const void*, const void*), unsigned threads);

int median_filter(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
    size_t *fsize, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*), unsigned threads);

#endif