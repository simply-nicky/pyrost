#ifndef ARRAY_H
#define ARRAY_H
#include "include.h"

typedef struct array_s
{
    int ndim;
    size_t size;
    size_t item_size;
    size_t *dims;
    size_t *strides;
    void *data;
} array_s;
typedef struct array_s *array;

array new_array(int ndim, size_t *dims, size_t item_size, void *data);
void free_array(array arr);

void unravel_index(int *coord, int idx, array arr);
int ravel_index(int *coord, array arr);

typedef struct line_s
{
    size_t npts;
    size_t stride;
    size_t item_size;
    void *data;
} line_s;
typedef struct line_s *line;

line new_line(size_t npts, size_t stride, size_t item_size, void *data);
line init_line(array arr, int axis);
void update_line(line ln, array arr, int iter);

// -----------Extend line modes-----------
//
// EXTEND_CONSTANT: kkkkkkkk|abcd|kkkkkkkk
// EXTEND_NEAREST:  aaaaaaaa|abcd|dddddddd
// EXTEND_MIRROR:   cbabcdcb|abcd|cbabcdcb
// EXTEND_REFLECT:  abcddcba|abcd|dcbaabcd
// EXTEND_WRAP:     abcdabcd|abcd|abcdabcd
typedef enum
{
    EXTEND_CONSTANT = 0,
    EXTEND_NEAREST = 1,
    EXTEND_MIRROR = 2,
    EXTEND_REFLECT = 3,
    EXTEND_WRAP = 4
} EXTEND_MODE;

void extend_line(void *out, size_t osize, line inp, EXTEND_MODE mode, void *cval);
int extend_point(void *out, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval);

// Array search
size_t searchsorted(const void *key, const void *base, size_t npts, size_t size,
    int (*compar)(const void*, const void*));

#endif