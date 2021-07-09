#include "array.h"

array new_array(int ndim, size_t *dims, void *data)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_array: ndim must be positive."); return NULL;}

    array arr = (array)malloc(sizeof(struct array_s));
    if (!arr) {ERROR("new_array: not enough memory."); return NULL;}

    arr->ndim = ndim;
    arr->size = 1;
    for (int n = 0; n < ndim; n++) arr->size *= dims[n];

    arr->dims = dims;
    arr->strides = (size_t *)malloc(arr->ndim * sizeof(size_t));
    if (!arr->strides) {ERROR("new_array: not enough memory."); return NULL;}
    for (int n = 0; n < arr->ndim; n++)
    {
        arr->strides[n] = arr->size;
        for (int m = 0; m <= n; m++) arr->strides[n] /= arr->dims[m];
    }
    arr->data = data;
    return arr;
}

void free_array(array arr)
{
    free(arr->strides);
    free(arr);
}

// note: the line count over axis is given by: arr->size / arr->dims[axis]
// note: you can free the line just with: free(line)

line new_line(size_t npts, size_t stride, void *data)
{
    line ln = (line)malloc(sizeof(line_s));
    if (!ln) {ERROR("new_line: not enough memory."); return NULL;}

    ln->npts = npts;
    ln->stride = stride;
    ln->data = data;
    return ln;
}

line init_line(array arr, int axis)
{
    /* check parameters */
    if (axis < 0 || axis >= arr->ndim) {ERROR("init_line: invalid axis."); return NULL;}

    line ln = (line)malloc(sizeof(line_s));
    if (!ln) {ERROR("init_line: not enough memory."); return NULL;}

    ln->npts = arr->dims[axis];
    ln->stride = arr->strides[axis];
    ln->data = arr->data;
    return ln;
}

void update_line(line ln, array arr, int iter, size_t item_size)
{
    ln->data = arr->data + (ln->npts * ln->stride * (iter / ln->stride) + (iter % ln->stride)) * item_size;
}

void extend_line(void *out, size_t item_size, size_t osize, line inp, EXTEND_MODE mode, void *cval)
{
    int dsize = (int)osize - (int)inp->npts;
    int size1 = dsize - dsize / 2;
    int size2 = dsize - size1;
    for (int i = 0; i < (int)inp->npts; i++) memcpy(out + (i + size1) * item_size, inp->data + i * inp->stride * item_size, item_size);
    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case EXTEND_CONSTANT:
            for (int i = 0; i < size1; i++) memcpy(out + i * item_size, cval, item_size);
            for (int i = 0; i < size2; i++) memcpy(out + (inp->npts + size1 + i) * item_size, cval, item_size);
            break;
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:
            for (int i = 0; i < size1; i++) memcpy(out + i * item_size, inp->data, item_size);
            for (int i = 0; i < size2; i++)
            {memcpy(out + (inp->npts + size1 + i) * item_size, inp->data + (inp->npts - 1) * inp->stride * item_size, item_size);}
            break;
        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:
            for (int i = (int)inp->npts - size1 - 1; i < (int)inp->npts - 1; i++)
            {
                int fct = (i / ((int)inp->npts - 1)) % 2;
                int idx = ((int)inp->npts - 1) * (1 - fct) + (2 * fct - 1) * (i % ((int)inp->npts - 1));
                memcpy(out + (i - inp->npts + size1 + 1) * item_size, inp->data + idx * inp->stride * item_size, item_size);
            }
            for (int i = 1; i <= size2; i++)
            {
                int fct = (i / ((int)inp->npts - 1)) % 2;
                int idx = ((int)inp->npts - 1) * (1 - fct) + (2 * fct - 1) * (i % ((int)inp->npts - 1));
                memcpy(out + (inp->npts + size1 - 1 + i) * item_size, inp->data + idx * inp->stride * item_size, item_size);
            }
            break;
        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            for (int i = (int)inp->npts - size1; i < (int)inp->npts; i++)
            {
                int fct = (i / inp->npts) % 2;
                int idx = ((int)inp->npts - 1) * (1 - fct) + (2 * fct - 1) * (i % inp->npts);
                memcpy(out + (i - inp->npts + size1) * item_size, inp->data + idx * inp->stride * item_size, item_size);
            }
            for (int i = 0; i < size2; i++)
            {
                int fct = (i / inp->npts) % 2;
                int idx = ((int)inp->npts - 1) * (1 - fct) + (2 * fct - 1) * (i % inp->npts);
                memcpy(out + (inp->npts + size1 + i) * item_size, inp->data + idx * inp->stride * item_size, item_size);
            }
            break;
        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            for (int i = (int)inp->npts - size1; i < (int)inp->npts; i++)
            {memcpy(out + (i - inp->npts + size1) * item_size, inp->data + (i % inp->npts) * inp->stride * item_size, item_size);}
            for (int i = 0; i < size2; i++)
            {memcpy(out + (inp->npts + size1 + i) * item_size, inp->data + (i % inp->npts) * inp->stride * item_size, item_size);}
            break;
        default:
            ERROR("extend_line: invalid extend mode.");
    }
}

static size_t binary_search(const void *key, const void *array, size_t l, size_t r, size_t size,
    int (*compar)(const void*, const void*))
{
    if (l <= r)
    {
        size_t m = l + (r - l) / 2;
        int cmp0 = compar(key, array + m * size);
        int cmp1 = compar(key, array + (m + 1) * size);
        if (cmp0 == 0) return m;
        if (cmp0 > 0 && cmp1 < 0) return m + 1;
        if (cmp0 < 0) return binary_search(key, array, l, m, size, compar);
        return binary_search(key, array, m + 1, r, size, compar);
    }
    return 0;
}

size_t searchsorted(const void *key, const void *base, size_t npts, size_t size,
    int (*compar)(const void*, const void*))
{
    if (compar(key, base) < 0) return 0;
    if (compar(key, base + (npts - 1) * size) > 0) return npts;
    return binary_search(key, base, 0, npts, size, compar);
}