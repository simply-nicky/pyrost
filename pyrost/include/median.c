#include "median.h"

footprint init_footprint(int ndim, size_t item_size, size_t *fsize)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_footprint: ndim must be positive."); return NULL;}

    array farr = new_array(ndim, fsize, 0, NULL);

    footprint fpt = (footprint)malloc(sizeof(struct footprint_s));
    fpt->ndim = farr->ndim;
    fpt->npts = farr->size;
    fpt->counter = 0;

    fpt->offsets = (int *)malloc(fpt->npts * fpt->ndim * sizeof(int));
    fpt->coordinates = (int *)malloc(fpt->npts * fpt->ndim * sizeof(int));

    fpt->item_size = item_size;
    fpt->data = malloc(fpt->npts * fpt->item_size);
    
    if (!fpt || !fpt->offsets || !fpt->coordinates || !fpt->data)
    {ERROR("new_footprint: not enough memory."); return NULL;}

    for (int i = 0; i < (int)farr->size; i++)
    {
        UNRAVEL_INDEX(fpt->offsets + ndim * i, i, farr);
        for (int n = 0; n < fpt->ndim; n++) fpt->offsets[ndim * i + n] -= farr->dims[n] / 2;
    }

    free_array(farr);
    return fpt;
}

void free_footprint(footprint fpt)
{
    free(fpt->coordinates);
    free(fpt->offsets);
    free(fpt->data);
    free(fpt);
}

void update_footprint(footprint fpt, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval)
{
    int extend, index;
    fpt->counter = 0;

    for (int i = 0; i < fpt->npts; i++)
    {
        extend = 0;

        for (int n = 0; n < fpt->ndim; n++)
        {
            fpt->coordinates[i * fpt->ndim + n] = coord[n] + fpt->offsets[i * fpt->ndim + n];
            extend |= (fpt->coordinates[i * fpt->ndim + n] >= (int)arr->dims[n]) ||
                (fpt->coordinates[i * fpt->ndim + n] < 0);
        }

        if (extend)
        {
            fpt->counter += extend_point(fpt->data + fpt->counter * fpt->item_size,
                &(fpt->coordinates[i * fpt->ndim]), arr, mask, mode, cval);
        }
        else
        {
            RAVEL_INDEX(fpt->coordinates + i * fpt->ndim, index, arr);
            memcpy(fpt->data + fpt->counter * fpt->item_size, arr->data + index * arr->item_size,
                arr->item_size);
            fpt->counter++;
        }
    }
}

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

static void wirthselect(void *data, void *key, int k, int l, int m, size_t size,
    int (*compar)(const void*, const void*))
{
    int i, j;
    while (l < m)
    {
        memcpy(key, data + k * size, size);
        i = l; j = m;

        do
        {
            while (compar(key, data + i * size) > 0) i++;
            while (compar(key, data + j * size) < 0) j--;
            if (i <= j) 
            {
                SWAP_BUF(data + i * size, data + j * size, size);
                i++; j--;
            }
        } while((i <= j));
        if (j < k) l = i;
        if (k < i) m = j;
    }
    
    key = data + k * size;
}

int median(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size, int axis,
    int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !dims) {ERROR("median: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median: invalid axis."); return -1;}
    if (threads == 0) {ERROR("median: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, data);
    array marr = new_array(ndim, dims, 1, mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        void *buffer = malloc(iarr->dims[axis] * iarr->item_size);
        void *key = malloc(iarr->item_size);

        line iline = init_line(iarr, axis);
        line mline = init_line(marr, axis);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(mline, i);

            int len = 0;
            for (int n = 0; n < (int)iline->npts; n++)
            {
                if (((unsigned char *)mline->data)[n * mline->stride])
                {memcpy(buffer + len++ * iline->item_size,
                    iline->data + n * iline->stride * iline->item_size, iline->item_size);}
            }

            if (len) 
            {
                wirthselect(buffer, key, len / 2, 0, len - 1, iline->item_size, compar);
                memcpy(out + i * iline->item_size, key, iline->item_size);
            }
            else memset(out + i * iline->item_size, 0, iline->item_size);

        }

        free(iline); free(mline);
        free(key); free(buffer);
    }

    free_array(iarr); free_array(marr);

    return 0;
}

int median_filter(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
    size_t *fsize, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !fsize || !cval)
    {ERROR("median_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
    if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, data);
    array marr = new_array(ndim, dims, 1, mask);

    #pragma omp parallel num_threads(threads)
    {
        footprint fpt = init_footprint(iarr->ndim, iarr->item_size, fsize);
        int *coord = (int *)malloc(iarr->ndim * sizeof(int));
        void *key = malloc(iarr->item_size);

        #pragma omp for
        for (int i = 0; i < (int)iarr->size; i++)
        {
            UNRAVEL_INDEX(coord, i, iarr);

            update_footprint(fpt, coord, iarr, marr, mode, cval);

            if (fpt->counter)
            {
                wirthselect(fpt->data, key, fpt->counter / 2, 0, fpt->counter - 1, fpt->item_size, compar);
                memcpy(out + i * fpt->item_size, key, fpt->item_size);
            }
            else memset(out + i * fpt->item_size, 0, fpt->item_size);
        }

        free_footprint(fpt); free(coord); free(key);
    }

    free_array(iarr); free_array(marr);

    return 0;
}