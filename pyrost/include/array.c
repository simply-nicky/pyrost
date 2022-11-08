#include "array.h"

array new_array(int ndim, const size_t *dims, size_t item_size, void *data)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_array: ndim must be positive."); return NULL;}

    array arr = (array)malloc(sizeof(struct array_s));
    if (!arr) {ERROR("new_array: not enough memory."); return NULL;}

    arr->ndim = ndim;
    arr->item_size = item_size;
    arr->size = 1;
    arr->data = data;
    arr->dims = dims;
    arr->strides = MALLOC(size_t, arr->ndim);
    if (!arr->strides) {ERROR("new_array: not enough memory."); return NULL;}

    for (int n = arr->ndim - 1; n >= 0; n--)
    {
        arr->strides[n] = arr->size;
        arr->size *= dims[n];
    }

    return arr;
}

void free_array(array arr)
{
    DEALLOC(arr->strides);
    DEALLOC(arr);
}

// note: the line count over axis is given by: arr->size / arr->dims[axis]
// note: you can free the line just with: DEALLOC(line)

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

void extend_line(void *out, size_t osize, line inp, EXTEND_MODE mode, const void *cval)
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

int extend_point(void *out, int *coord, array arr, array mask, EXTEND_MODE mode, const void *cval)
{
    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == EXTEND_CONSTANT)
    {
            memcpy(out, cval, arr->item_size);
            return 1;
    }

    int *close = MALLOC(int, arr->ndim);
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
    RAVEL_INDEX(close, &index, arr);
    DEALLOC(close);

    if (*(unsigned char *)(mask->data + index * mask->item_size))
    {
        memcpy(out, arr->data + index * arr->item_size, arr->item_size);
        return 1;
    }
    else return 0;

}

/*----------------------------------------------------------------------------*/
/*--------------------------- Comparing functions ----------------------------*/
/*----------------------------------------------------------------------------*/

int compare_double(const void *a, const void *b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}

int compare_float(const void *a, const void *b)
{
    if (*(float*)a > *(float*)b) return 1;
    else if (*(float*)a < *(float*)b) return -1;
    else return 0;
}

int compare_int(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int compare_uint(const void *a, const void *b)
{
    if (*(unsigned int *)a > *(unsigned int *)b) return 1;
    else if (*(unsigned int *)a < *(unsigned int *)b) return -1;
    else return 0;
}

int compare_ulong(const void *a, const void *b)
{
    if (*(unsigned long *)a > *(unsigned long *)b) return 1;
    else if (*(unsigned long *)a < *(unsigned long *)b) return -1;
    else return 0;
}

int indirect_compare_double(const void *a, const void *b, void *data)
{
    double *dptr = data;
    if (dptr[*(size_t *)a] > dptr[*(size_t *)b]) return 1;
    else if (dptr[*(size_t *)a] < dptr[*(size_t *)b]) return -1;
    else return 0;
}

int indirect_compare_float(const void *a, const void *b, void *data)
{
    float *dptr = data;
    if (dptr[*(size_t *)a] > dptr[*(size_t *)b]) return 1;
    else if (dptr[*(size_t *)a] < dptr[*(size_t *)b]) return -1;
    else return 0;
}

int indirect_search_double(const void *key, const void *base, void *data)
{
    double *dptr = data;
    if (*(double *)key > dptr[*(size_t *)base]) return 1;
    else if (*(double *)key < dptr[*(size_t *)base]) return -1;
    else return 0;
}

int indirect_search_float(const void *key, const void *base, void *data)
{
    float *dptr = data;
    if (*(float *)key > dptr[*(size_t *)base]) return 1;
    else if (*(float *)key < dptr[*(size_t *)base]) return -1;
    else return 0;
}

static size_t binary_left(const void *key, const void *array, int l, int r, size_t size,
    int (*compar)(const void *, const void *))
{
    size_t m, out = 0;
    int cmp;
    while (l <= r)
    {
        m = l + (r - l) / 2;
        cmp = compar(key, array + m * size);

        // if m is less than key, all elements
        // in range [l, m] are also less
        // so we now search in [m + 1, r]
        if (cmp > 0)
        {
            l = m + 1;
            if (compar(key, array + (m + 1) * size) < 0) out = m + 1;
        }
        // if m is greater than key, all elements
        // in range [m + 1, r] are also greater
        // so we now search in [l, m - 1]
        else if (cmp < 0) r = m - 1;
        // if m is equal to key, we note down
        // the last found index then we search
        // for more in left side of m
        // so we now search in [l, m - 1]
        else {out = m; r = m - 1; }
    }
    return out;
}

static size_t binary_right(const void *key, const void *array, int l, int r, size_t size,
    int (*compar)(const void *, const void *))
{
    size_t m, out = 0;
    int cmp;
    while (l <= r)
    {
        m = l + (r - l) / 2;
        cmp = compar(key, array + m * size);

        // if m is less than key, then all elements
        // in range [l, m - 1] are also less
        // so we now search in [m + 1, r]
        if (cmp > 0)
        {
            l = m + 1;
            if (compar(key, array + (m + 1) * size) < 0) out = m + 1;
        }
        // if m is greater than key, then all
        // elements in range [m + 1, r] are
        // also greater so we now search in
        // [l, m - 1]
        else if (cmp < 0) r = m - 1;
        // if m is equal to key, we note down
        // the last found index then we search
        // for more in right side of m
        // so we now search in [m + 1, r]
        else {out = m; l = m + 1; }
    }
    return out;
}

size_t searchsorted(const void *key, const void *base, size_t npts, size_t size,
    SEARCH_SIDE side, int (*compar)(const void *, const void *))
{
    if (compar(key, base) < 0) return 0;
    if (compar(key, base + (npts - 1) * size) > 0) return npts;
    switch (side)
    {
        case SEARCH_LEFT: return binary_left(key, base, 0, npts, size, compar);
        case SEARCH_RIGHT: return binary_right(key, base, 0, npts, size, compar);
        default: ERROR("searchsorted: invalid extend mode."); return 0;
    }
}

static size_t binary_left_r(const void *key, const void *array, int l, int r, size_t size,
    int (*compar)(const void *, const void *, void *), void *arg)
{
    size_t m, out = 0;
    int cmp;
    while (l <= r)
    {
        m = l + (r - l) / 2;
        cmp = compar(key, array + m * size, arg);

        // if m is less than key, all elements
        // in range [l, m] are also less
        // so we now search in [m + 1, r]
        if (cmp > 0)
        {
            l = m + 1;
            if (compar(key, array + (m + 1) * size, arg) < 0) out = m + 1;
        }
        // if m is greater than key, all elements
        // in range [m + 1, r] are also greater
        // so we now search in [l, m - 1]
        else if (cmp < 0) r = m - 1;
        // if m is equal to key, we note down
        // the last found index then we search
        // for more in left side of m
        // so we now search in [l, m - 1]
        else {out = m; r = m - 1; }
    }
    return out;
}

static size_t binary_right_r(const void *key, const void *array, int l, int r, size_t size,
    int (*compar)(const void *, const void *, void *), void *arg)
{
    size_t m, out = 0;
    int cmp;
    while (l <= r)
    {
        m = l + (r - l) / 2;
        cmp = compar(key, array + m * size, arg);

        // if m is less than key, then all elements
        // in range [l, m - 1] are also less
        // so we now search in [m + 1, r]
        if (cmp > 0)
        {
            l = m + 1;
            if (compar(key, array + (m + 1) * size, arg) < 0) out = m + 1;
        }
        // if m is greater than key, then all
        // elements in range [m + 1, r] are
        // also greater so we now search in
        // [l, m - 1]
        else if (cmp < 0) r = m - 1;
        // if m is equal to key, we note down
        // the last found index then we search
        // for more in right side of m
        // so we now search in [m + 1, r]
        else {out = m; l = m + 1; }

    }
    return out;
}

size_t searchsorted_r(const void *key, const void *base, size_t npts, size_t size,
    SEARCH_SIDE side, int (*compar)(const void *, const void *, void *), void *arg)
{
    if (compar(key, base, arg) < 0) return 0;
    if (compar(key, base + (npts - 1) * size, arg) > 0) return npts;
    switch (side)
    {
        case SEARCH_LEFT: return binary_left_r(key, base, 0, npts, size, compar, arg);
        case SEARCH_RIGHT: return binary_right_r(key, base, 0, npts, size, compar, arg);
        default: ERROR("searchsorted_r: invalid extend mode."); return 0;
    }
}

/*----------------------------------------------------------------------------*/
/*------------------------------- Wirth select -------------------------------*/
/*----------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
    Function :  kth_smallest()
    In       :  array of elements, n elements in the array, rank k 
    Out      :  one element
    Job      :  find the kth smallest element in the array
    Notice   :  Buffer must be of size n

    Reference:
        Author: Wirth, Niklaus
        Title: Algorithms + data structures = programs
        Publisher: Englewood Cliffs: Prentice-Hall, 1976 Physical description: 366 p.
        Series: Prentice-Hall Series in Automatic Computation
---------------------------------------------------------------------------*/

void *wirthselect(void *inp, int k, int n, size_t size, int (*compar)(const void *, const void *))
{
    int i, j, l = 0, m = n - 1;
    void *buf = malloc(size);
    while (l < m)
    {
        memcpy(buf, inp + k * size, size);
        i = l; j = m;

        do
        {
            while (compar(buf, inp + i * size) > 0) i++;
            while (compar(buf, inp + j * size) < 0) j--;
            if (i <= j) 
            {
                SWAP_BUF(inp + i * size, inp + j * size, size);
                i++; j--;
            }
        } while (i <= j);
        if (j < k) l = i;
        if (k < i) m = j;
    }
    free(buf);
    
    return inp + k * size;
}

void *wirthselect_r(void *inp, int k, int n, size_t size, int (*compar)(const void *, const void *, void *), void *arg)
{
    int i, j, l = 0, m = n - 1;
    void *buf = malloc(size);
    while (l < m)
    {
        memcpy(buf, inp + k * size, size);
        i = l; j = m;

        do
        {
            while (compar(buf, inp + i * size, arg) > 0) i++;
            while (compar(buf, inp + j * size, arg) < 0) j--;
            if (i <= j) 
            {
                SWAP_BUF(inp + i * size, inp + j * size, size);
                i++; j--;
            }
        } while (i <= j);
        if (j < k) l = i;
        if (k < i) m = j;
    }
    free(buf);
    
    return inp + k * size;
}