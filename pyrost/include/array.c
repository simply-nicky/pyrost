#include "array.h"

array new_array(int ndim, size_t *dims, size_t item_size, void *data)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_array: ndim must be positive."); return NULL;}

    array arr = (array)malloc(sizeof(struct array_s));
    if (!arr) {ERROR("new_array: not enough memory."); return NULL;}

    arr->ndim = ndim;
    arr->item_size = item_size;
    arr->size = 1;
    for (int n = 0; n < ndim; n++) arr->size *= dims[n];

    arr->dims = dims;
    arr->strides = (size_t *)malloc(arr->ndim * sizeof(size_t));
    if (!arr->strides) {ERROR("new_array: not enough memory."); return NULL;}
    size_t stride = 1;
    for (int n = arr->ndim - 1; n >= 0; n--)
    {
        arr->strides[n] = stride;
        stride *= arr->dims[n];
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

line new_line(size_t npts, size_t stride, size_t item_size, void *data)
{
    line ln = (line)malloc(sizeof(line_s));
    if (!ln) {ERROR("new_line: not enough memory."); return NULL;}

    ln->npts = npts;
    ln->stride = stride;
    ln->item_size = item_size;
    ln->line_size = ln->npts * ln->stride * ln->item_size;
    ln->data = ln->first = data;
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
    ln->item_size = arr->item_size;
    ln->line_size = ln->npts * ln->stride * ln->item_size;
    ln->data = ln->first = arr->data;
    return ln;
}

slice init_slice(array arr, int axis)
{
    /* check parameters */
    if (axis < 0 || axis >= arr->ndim) {ERROR("init_line: invalid axis."); return NULL;}
    if (arr->ndim < 2) {ERROR("init_slice: array must be 2D or higher"); return NULL;}

    slice slc = (slice)malloc(sizeof(slice_s));
    if (!slc) {ERROR("init_slice: not enough memory."); return NULL;}

    slc->iter = (array)malloc(sizeof(array_s));
    slc->iter->ndim = arr->ndim - 1;
    slc->iter->item_size = arr->item_size;
    slc->iter->dims = (size_t *)malloc(slc->iter->ndim * sizeof(size_t));
    slc->iter->strides = (size_t *)malloc(slc->iter->ndim * sizeof(size_t));
    for (int n = 0, m = 0; n < arr->ndim; n++)
    {
        if (n != axis)
        {
            slc->iter->dims[m] = arr->dims[n];
            slc->iter->strides[m] = arr->dims[n];
            m++;
        }
    }

    slc->iter->data = slc->first = arr->data;
    slc->stride = arr->strides[axis];

    return slc;
}

void free_slice(slice slc)
{
    free(slc->iter->dims);
    free_array(slc->iter);
    free(slc);
}

void extend_line(void *out, size_t osize, line inp, EXTEND_MODE mode, void *cval)
{
    int dsize = (int)osize - (int)inp->npts;
    int size_before = dsize - dsize / 2;
    int size_after = dsize - size_before;

    void *last = inp->data + inp->line_size;
    void *dst = out + size_before * inp->item_size;
    void *src = inp->data;

    int line_size = inp->npts;
    while(line_size--)
    {
        memcpy(dst, src, inp->item_size);
        dst += inp->item_size;
        src += inp->stride * inp->item_size;
    }

    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case EXTEND_CONSTANT:

            dst = out;
            while (size_before--)
            {
                memcpy(dst, cval, inp->item_size);
                dst += inp->item_size;
            }

            dst = out + (osize - size_after) * inp->item_size;
            while (size_after--)
            {
                memcpy(dst, cval, inp->item_size);
                dst += inp->item_size;
            }
            break;

        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            dst = out; src = inp->data;
            while (size_before--)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - inp->stride * inp->item_size;
            while (size_after--)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
            }
            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            dst = out + (size_before - 1) * inp->item_size;
            src = inp->data + inp->stride * inp->item_size;

            while (size_before-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = last - 2 * inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - 2 * inp->stride * inp->item_size;

            while (size_after-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src -= inp->item_size * inp->stride;
            }
            src = inp->data + inp->stride * inp->item_size;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            dst = out + (size_before - 1) * inp->item_size;
            src = inp->data;

            while (size_before-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = last - inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - inp->stride * inp->item_size;

            while (size_after-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src -= inp->item_size * inp->stride;
            }
            src = inp->data;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            dst = out + (size_before - 1) * inp->item_size;
            src = last - inp->stride * inp->item_size;

            while (size_before-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            src = last - inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = inp->data;

            while (size_after-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = inp->data;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        default:
            ERROR("extend_line: invalid extend mode.");
    }
}

int extend_point(void *out, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval)
{
    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == EXTEND_CONSTANT)
    {
            memcpy(out, cval, arr->item_size);
            return 1;
    }

    int *close = (int *)malloc(arr->ndim * sizeof(int));
    size_t dist;

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n]) close[n] = arr->dims[n] - 1;
                else if (coord[n] < 0) close[n] = 0;
                else close[n] = coord[n];
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = arr->dims[n] - 1;
                    dist = coord[n] - arr->dims[n] + 1;

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n];

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = arr->dims[n] - 1;
                    dist = coord[n] - arr->dims[n];

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n] - 1;

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = 0;
                    dist = coord[n] - arr->dims[n];

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else if (coord[n] < 0)
                {
                    close[n] = arr->dims[n] - 1;
                    dist = -coord[n] - 1;

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else close[n] = coord[n];
            }

            break;

        default:
            ERROR("extend_point: invalid extend mode.");
    }

    int index;
    RAVEL_INDEX(close, index, arr);
    free(close);

    if (((unsigned char *)mask->data)[index])
    {
        memcpy(out, arr->data + index * arr->item_size, arr->item_size);
        return 1;
    }
    else return 0;

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