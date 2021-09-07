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

#define UNRAVEL_INDEX(_coord, _idx, _arr)                       \
{                                                               \
    int _i = _idx, _n;                                          \
    for (_n = 0; _n < _arr->ndim; _n++)                         \
    {                                                           \
        (_coord)[_n] = _i / _arr->strides[_n];                  \
        _i -= (_coord)[_n] * _arr->strides[_n];                 \
    }                                                           \
}

#define RAVEL_INDEX(_coord, _idx, _arr)                         \
{                                                               \
    _idx = 0; int _n;                                           \
    for (_n = 0; _n < _arr->ndim; _n++)                         \
        _idx += _arr->strides[_n] * (_coord)[_n];               \
}

typedef struct line_s
{
    size_t npts;
    size_t stride;
    size_t item_size;
    size_t line_size;
    void *data, *first;
} line_s;
typedef struct line_s *line;

line new_line(size_t npts, size_t stride, size_t item_size, void *data);
line init_line(array arr, int axis);

#define UPDATE_LINE(_line, _iter)                               \
{                                                               \
    int _div;                                                   \
    _div = _iter / _line->stride;                               \
    _line->data = _line->first + _line->line_size * _div +      \
    (_iter - _div * _line->stride) * _line->item_size;          \
}

typedef struct slice_s
{
    array iter;
    size_t stride;
    void *first;
} slice_s;
typedef struct slice_s *slice;

slice init_slice(array arr, int axis);
void free_slice(slice slc);

#define UPDATE_SLICE(_slc, _iter)                               \
{                                                               \
    _slc->iter->data = _slc->first + _iter * _slc->stride;      \
}


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