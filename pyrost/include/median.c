#include "median.h"

footprint init_footprint(int ndim, size_t item_size, size_t *fsize, unsigned char *fmask)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_footprint: ndim must be positive."); return NULL;}

    array farr = new_array(ndim, fsize, 0, NULL);

    footprint fpt = (footprint)malloc(sizeof(struct footprint_s));
    fpt->ndim = farr->ndim;

    fpt->npts = 0; fpt->counter = 0;
    for (int i = 0; i < (int)farr->size; i++) if (fmask[i]) fpt->npts++;

    if (fpt->npts == 0) {ERROR("new_footprint: zerro number of points in a footprint."); return NULL;}
    
    fpt->offsets = MALLOC(int, fpt->npts * fpt->ndim);
    fpt->coordinates = MALLOC(int, fpt->npts * fpt->ndim);

    fpt->item_size = item_size;
    fpt->data = malloc(fpt->npts * fpt->item_size);
    
    if (!fpt || !fpt->offsets || !fpt->coordinates || !fpt->data)
    {ERROR("new_footprint: not enough memory."); return NULL;}

    for (int i = 0, j = 0; i < (int)farr->size; i++)
    {
        if (fmask[i])
        {
            UNRAVEL_INDEX(fpt->offsets + ndim * j, &i, farr);
            for (int n = 0; n < fpt->ndim; n++) fpt->offsets[ndim * j + n] -= farr->dims[n] / 2;
            j++;
        }
    }

    free_array(farr);
    return fpt;
}

void free_footprint(footprint fpt)
{
    DEALLOC(fpt->coordinates);
    DEALLOC(fpt->offsets);
    DEALLOC(fpt->data);
    DEALLOC(fpt);
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
            fpt->counter += extend_point(GETP(fpt, 1, fpt->counter), fpt->coordinates + i * fpt->ndim,
                                         arr, mask, mode, cval);
        }
        else
        {
            RAVEL_INDEX(fpt->coordinates + i * fpt->ndim, &index, arr);
            if (GET(mask, unsigned char, index))
            {
                memcpy(GETP(fpt, 1, fpt->counter), GETP(arr, 1, index), arr->item_size);
                fpt->counter++;
            }
        }
    }
}

int median(void *out, void *inp, unsigned char *mask, int ndim, const size_t *dims, size_t item_size, int axis,
    int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !mask || !dims || !compar) {ERROR("median: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median: invalid axis."); return -1;}
    if (threads == 0) {ERROR("median: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, inp);

    if (!iarr->size) {free_array(iarr); return 0;}
    array marr = new_array(ndim, dims, 1, mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        void *buffer = malloc(iarr->dims[axis] * iarr->item_size);
        void *key;

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
                if (GET(mline, unsigned char, n * mline->stride))
                {
                    memcpy(buffer + len++ * iline->item_size, GETP(iline, iline->stride, n),
                           iline->item_size);
                }
            }

            if (len) 
            {
                key = wirthmedian(buffer, len, iline->item_size, compar);
                memcpy(out + i * iline->item_size, key, iline->item_size);
            }
            else memset(out + i * iline->item_size, 0, iline->item_size);

        }

        DEALLOC(iline); DEALLOC(mline); DEALLOC(buffer);
    }

    free_array(iarr); free_array(marr);

    return 0;
}

int median_filter(void *out, void *inp, unsigned char *mask, unsigned char *imask, int ndim, const size_t *dims,
    size_t item_size, size_t *fsize, unsigned char *fmask, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*),
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !mask || !dims || !fsize || !fmask || !cval || !compar)
    {ERROR("median_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
    if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, inp);

    if (!iarr->size) {free_array(iarr); return 0;}
    if (!imask) imask = mask;
    array imarr = new_array(ndim, dims, 1, imask);

    threads = (threads > iarr->size) ? iarr->size : threads;

    #pragma omp parallel num_threads(threads)
    {
        footprint fpt = init_footprint(iarr->ndim, iarr->item_size, fsize, fmask);
        int *coord = MALLOC(int, iarr->ndim);
        void *key;

        #pragma omp for schedule(guided)
        for (int i = 0; i < (int)iarr->size; i++)
        {
            if (mask[i])
            {
                UNRAVEL_INDEX(coord, &i, iarr);

                update_footprint(fpt, coord, iarr, imarr, mode, cval);

                if (fpt->counter)
                {
                    key = wirthmedian(fpt->data, fpt->counter, fpt->item_size, compar);
                    memcpy(out + i * fpt->item_size, key, fpt->item_size);
                }
                else memset(out + i * fpt->item_size, 0, fpt->item_size);
            }
            else memset(out + i * fpt->item_size, 0, fpt->item_size);
        }

        free_footprint(fpt); DEALLOC(coord);
    }

    free_array(iarr); free_array(imarr);

    return 0;
}

int robust_mean(double *out, void *inp, int ndim, const size_t *dims, size_t item_size, int axis,
                int (*compar)(const void*, const void*), double (*getter)(const void*),
                double r0, double r1, int n_iter, double lm, unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !compar)
    {ERROR("robust_mean: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("robust_mean: ndim must be positive."); return -1;}
    if (threads == 0) {ERROR("robust_mean: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, inp);

    if (iarr->size == 0) {free_array(iarr); return 0;}

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    void *theta = malloc(repeats * iarr->item_size);
    void *mask = (unsigned char *)malloc(iarr->size); memset(mask, 1, iarr->size);
    median(theta, inp, mask, ndim, dims, item_size, axis, compar, threads);
    DEALLOC(mask);

    #pragma omp parallel num_threads(threads)
    {
        int n, j, j0 = (int)(r0 * dims[axis]), j1 = (int)(r1 * dims[axis]);
        double mean, cumsum;

        line iline = init_line(iarr, axis);
        double *err = MALLOC(double, iline->npts);
        size_t *idxs = MALLOC(size_t, iline->npts);

        #pragma omp for
        for (int i = 0; i < repeats; i++)
        {
            UPDATE_LINE(iline, i);

            mean = getter(theta + i * iline->item_size);

            for (n = 0; n < n_iter; n++)
            {
                for (j = 0; j < (int)iline->npts; j++)
                {
                    err[j] = fabs(getter(GETP(iline, iline->stride, j)) - mean);
                }

                for (j = 0; j < (int)iline->npts; j++) idxs[j] = j;
                POSIX_QSORT_R(idxs, iline->npts, sizeof(size_t), indirect_compare_double, (void *)err);

                mean = 0.0f;
                for (j = j0; j < j1; j++) mean += getter(GETP(iline, iline->stride, idxs[j]));
                mean = mean / (j1 - j0);
            }

            for (j = 0; j < (int)iline->npts; j++)
            {
                err[j] = SQ(getter(GETP(iline, iline->stride, j)) - mean);
            }

            for (j = 0; j < (int)iline->npts; j++) idxs[j] = j;
            POSIX_QSORT_R(idxs, iline->npts, sizeof(size_t), indirect_compare_double, (void *)err);

            for (j = n = 0, out[i] = cumsum = 0.0f; j < (int)iline->npts; j++)
            {
                if (lm * cumsum > j * err[idxs[j]])
                {
                    out[i] += getter(GETP(iline, iline->stride, idxs[j]));
                    n++;
                }

                cumsum += err[idxs[j]];
            }
            if (n) out[i] = out[i] / n;
        }

        DEALLOC(iline); DEALLOC(err); DEALLOC(idxs);
    }

    free_array(iarr); DEALLOC(theta);

    return 0;
}