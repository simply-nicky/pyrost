#include "routines.h"

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

int compare_long(const void *a, const void *b)
{
    return (*(long *)a - *(long *)b);
}

void barcode_bars(double *bars, size_t size, double x0, double b_dx, double rd, long seed)
{
    if (seed >= 0)
    {
        gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(r, (unsigned long) seed);
        for (int i = 0; i < (int) size; i++)
        {
            bars[i] = x0 + b_dx * (i + 2 * rd * (gsl_rng_uniform_pos(r) - 0.5));
        }
        gsl_rng_free(r);
    }
    else {for (int i = 0; i < (int) size; i++) bars[i] = x0 + b_dx * i;}
}

int ml_profile(double complex *out, double *xcrd, size_t isize, double *layers, size_t lsize,
    double complex mt0, double complex mt1, double complex mt2, double sgm, unsigned threads)
{
    /* check parameters */
    if (!out || !xcrd || !layers) {ERROR("ml_profile: one of the arguments is NULL."); return -1;}
    if (threads == 0) {ERROR("ml_profile: threads must be positive."); return -1;}

    int b = 2 * (lsize / 2);

    #pragma omp parallel for num_threads(threads)
    for (int n = 0; n < (int) isize; n++)
    {
        double complex ref_idx = 0;
        int j0 = searchsorted(&xcrd[n], layers, lsize, sizeof(double), compare_double);
        if (j0 > 0 && j0 < b)
        {
            double x0 = (xcrd[n] - layers[j0 - 1]) / sqrt(2) / sgm;
            double x1 = (xcrd[n] - layers[j0]) / sqrt(2) / sgm;
            ref_idx += (mt1 - mt2) / 2 * ((double) (j0 % 2) - 0.5) * (erf(x0) - erf(x1));
            ref_idx -= (mt1 - mt2) / 4 * erf((xcrd[n] - layers[0]) / sqrt(2) / sgm);
            ref_idx += (mt1 - mt2) / 4 * erf((xcrd[n] - layers[b - 1]) / sqrt(2) / sgm);
        }
        ref_idx += (mt1 + mt0) / 2 * erf((xcrd[n] - layers[0]) / sqrt(2) / sgm);
        ref_idx -= (mt1 + mt0) / 2 * erf((xcrd[n] - layers[b - 1]) / sqrt(2) / sgm);
        out[n] = (cos(creal(ref_idx)) + sin(creal(ref_idx)) * I) * exp(-cimag(ref_idx));
    }

    return 0;
}

static void rebin_line_double(double *out, const double *inp, size_t osize, size_t isize, unsigned threads)
{
    double ratio = (double) isize / osize;
    threads = (threads > (unsigned) osize) ? (unsigned) osize : threads;
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int) osize; i++)
    {
        double lb, ub;
        out[i] = 0;
        int j0 = (int)(i * ratio);
        int j1 = (int)((i + 1) * ratio);
        for (int j = j0; (j <= j1) && (j < (int) isize); j++)
        {
            lb = ((double) j) > (i * ratio) ? j : i * ratio;
            ub = ((double) j + 1) < ((i + 1) * ratio) ? j + 1 : (i + 1) * ratio;
            out[i] += (ub - lb) * inp[j];
        }
    }
}

int frames(double *out, double *pfx, double *pfy, double dx, double dy, size_t *ishape, size_t *oshape,
    long seed, unsigned threads)
{
    /* check parameters */
    if (!out || !pfx || !pfy) {ERROR("frames: one of the arguments is NULL."); return -1;}
    if (dx <= 0 || dy <= 0) {ERROR("frames: dx and dy mus be positive."); return -1;}
    if (threads == 0) {ERROR("frames: threads must be positive."); return -1;}

    size_t nframes = ishape[0], ypts = ishape[1], xpts = ishape[2];
    size_t ss_size = oshape[1], fs_size = oshape[2];
    double *pfyss = (double *)malloc(ss_size * sizeof(double));
    double *pfxss = (double *)malloc(fs_size * sizeof(double));
    rebin_line_double(pfyss, pfy, ss_size, ypts, threads);
    for (int n = 0; n < (int) nframes; n++)
    {
        rebin_line_double(pfxss, pfx, fs_size, xpts, threads);
        if (seed >= 0)
        {
            #pragma omp parallel num_threads(threads)
            {
                gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(r, (unsigned long) seed);
                #pragma omp for
                for (int i = 0; i < (int) (fs_size * ss_size); i++)
                {
                    double val = pfxss[i % fs_size] * pfyss[i / fs_size] * dx * dy;
                    out[i] = gsl_ran_poisson(r, val);
                }
                gsl_rng_free(r);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < (int) (fs_size * ss_size); i++)
            {
                out[i] = pfxss[i % fs_size] * pfyss[i / fs_size] * dx * dy;
            }
        }
        out += fs_size * ss_size; pfx += xpts;
    }
    free(pfxss); free(pfyss);

    return 0;
}

static void *wirthselect(void *data, void *key, int k, int l, int m, size_t size,
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
    
    return data + k * size;
}

int median(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size, int axis,
    int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !dims) {ERROR("median: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median: invalid axis."); return -1;}
    if (threads == 0) {ERROR("median: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, data);
    array marr = new_array(ndim, dims, mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        unsigned char *buffer = (unsigned char *)malloc(iarr->dims[axis] * item_size);
        void *key = malloc(item_size);

        line iline = init_line(iarr, axis);
        line mline = init_line(marr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i, item_size);
            update_line(mline, marr, i, 1);

            int len = 0;
            for (int n = 0; n < (int)iline->npts; n++)
            {
                if (((unsigned char *)mline->data)[n * mline->stride])
                {memcpy(buffer + len++ * item_size, iline->data + n * iline->stride * item_size, item_size);}
            }
            if (len) 
            {
                void *median = wirthselect(buffer, key, (len & 1) ? (len / 2) : (len / 2 - 1),
                    0, len - 1, item_size, compar);
                memcpy(out + i * item_size, median, item_size);
            }
            else memset(out + i * item_size, 0, item_size);
        }
        free(key); free(buffer);
    }

    return 0;
}

int median_filter(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
    int axis, size_t window, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*),
    unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !cval) {ERROR("median_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median_filter: invalid axis."); return -1;}
    if (window == 0) {ERROR("median_filter: window must be positive."); return -1;}
    if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

    unsigned char mval = 1;
    array iarr = new_array(ndim, dims, data);
    array oarr = new_array(ndim, dims, out);
    array marr = new_array(ndim, dims, (void *)mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        void *inpbf = malloc((iarr->dims[axis] + window) * item_size);
        unsigned char *mbf = (unsigned char *)malloc(marr->dims[axis] + window);
        void *medbf = malloc(window * item_size);
        void *key = malloc(item_size);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        line mline = init_line(marr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i, item_size);
            update_line(oline, oarr, i, item_size);
            update_line(mline, marr, i, 1);

            extend_line(inpbf, item_size, iarr->dims[axis] + window, iline, mode, cval);
            extend_line((void *)mbf, 1, marr->dims[axis] + window, mline, mode, (void *)&mval);

            for (int j = 0; j < (int)iline->npts; j++)
            {
                int len = 0;
                for (int n = -(int)window / 2; n < (int)window / 2 + (int)window % 2; n++)
                {
                    if (mbf[n + j])
                    {memcpy(medbf + len++ * item_size, inpbf + (n + j) * item_size, item_size);}
                }
                if (len) 
                {
                    void *median = wirthselect(medbf, key, (len & 1) ? (len / 2) : (len / 2 - 1),
                        0, len - 1, item_size, compar);
                    memcpy(oline->data + j * oline->stride * item_size, median, item_size);
                }
                else memset(oline->data + j * oline->stride * item_size, 0, item_size);
            }
        }
        free(iline); free(oline); free(mline);
        free(key); free(medbf); free(mbf); free(inpbf);
    }

    return 0;
}

void dot_double(void *out, line line1, line line2)
{
    const int num = (int)line1->npts;
    const int is1 = (int)line1->stride;
    const double *ip1 = (double *)line1->data;
    const int is2 = (int)line2->stride;
    const double *ip2 = (double *)line2->data;

    *(double *)out = ddot_(&num, ip1, &is1, ip2, &is2);
}

void dot_long(void *out, line line1, line line2)
{
    long sum = 0;
    long *ip1 = (long *)line1->data;
    long *ip2 = (long *)line2->data;
    for (int i = 0; i < (int)line1->npts; i++, ip1 += line1->stride, ip2 += line2->stride)
    {sum += (*ip1) * (*ip2);}
    *(long *)out = sum;
}

int dot(void *out, void *inp1, int ndim1, size_t *dims1, int axis1, void *inp2, int ndim2, size_t *dims2,
    int axis2, size_t item_size, void (*dot_func)(void*, line, line), unsigned threads)
{
    /* check parameters */
    if (!out || !inp1 || !inp2 || !dims1 || !dims2)
    {ERROR("dot: one of the arguments is NULL."); return -1;}
    if (ndim1 <= 0 || ndim2 <= 0) {ERROR("dot: ndim1 and ndim2 must be positive."); return -1;}
    if (axis1 < 0 || axis1 >= ndim1) {ERROR("dot: invalid axis1."); return -1;}
    if (axis2 < 0 || axis2 >= ndim2) {ERROR("dot: invalid axis2."); return -1;}
    if (dims1[axis1] != dims2[axis2]) {ERROR("dot: incompatible shapes."); return -1;}
    if (threads == 0) {ERROR("dot: threads must be positive."); return -1;}

    array arr1 = new_array(ndim1, dims1, inp1);
    array arr2 = new_array(ndim2, dims2, inp2);

    int rep1 = arr1->size / arr1->dims[axis1];
    int rep2 = arr2->size / arr2->dims[axis2];
    int repeats = rep1 * rep2;
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        line line1 = init_line(arr1, axis1);
        line line2 = init_line(arr2, axis2);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(line1, arr1, i / rep2, item_size);
            update_line(line2, arr2, i % rep2, item_size);

            dot_func(out + i * item_size, line1, line2);
        }

        free(line1); free(line2);
    }

    free_array(arr1);
    free_array(arr2);

    return 0;
}