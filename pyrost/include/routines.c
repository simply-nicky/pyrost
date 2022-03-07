#include "routines.h"
#include "median.h"

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
    double complex t0, double complex t1, double sgm, unsigned threads)
{
    /* check parameters */
    if (!out || !xcrd || !layers) {ERROR("ml_profile: one of the arguments is NULL."); return -1;}
    if (threads == 0) {ERROR("ml_profile: threads must be positive."); return -1;}

    int b = 2 * (lsize / 2);
    int dj = (int)(4.0 * sgm * b / (layers[b - 1] - layers[0])) + 1;

    double x0, x1;
    int jj, j0, j1, j;

    #pragma omp parallel for num_threads(threads)
    for (int n = 0; n < (int) isize; n++)
    {
        out[n] = 1.0;
        jj = searchsorted(&xcrd[n], layers, lsize, sizeof(double), compare_double);
        j0 = 2 * ((jj - dj) / 2) - 1;
        j1 = 2 * ((jj + dj) / 2);
        for (j = j0; j < j1; j += 2)
        {
            if (j > 0 && j < b - 1)
            {
                x0 = (xcrd[n] - layers[j]) / sgm;
                x1 = (layers[j + 1] - xcrd[n]) / sgm;
                out[n] += 0.5 * (t1 - t0) * (tanh(x0) + tanh(x1));
            }
        }
        out[n] += 0.5 * (t0 - 1.0) * tanh((xcrd[n] - layers[0]) / sgm);
        out[n] += 0.5 * (t0 - 1.0) * tanh((layers[b - 1] - xcrd[n]) / sgm);
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
    size_t dety_size = oshape[1], detx_size = oshape[2];
    double *pfyss = (double *)malloc(dety_size * sizeof(double));
    double *pfxss = (double *)malloc(detx_size * sizeof(double));
    rebin_line_double(pfyss, pfy, dety_size, ypts, threads);

    for (int n = 0; n < (int) nframes; n++)
    {
        rebin_line_double(pfxss, pfx, detx_size, xpts, threads);
        if (seed >= 0)
        {
            gsl_rng *r_master = gsl_rng_alloc(gsl_rng_mt19937);
            gsl_rng_set(r_master, (unsigned long) seed);

            #pragma omp parallel num_threads(threads)
            {
                unsigned long thread_seed;
                gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);

                thread_seed = gsl_rng_get(r_master);
                gsl_rng_set(r, thread_seed);

                #pragma omp for
                for (int i = 0; i < (int) (detx_size * dety_size); i++)
                {
                    double val = pfxss[i % detx_size] * pfyss[i / detx_size] * dx * dy;
                    out[i] = gsl_ran_poisson(r, val);
                }
                gsl_rng_free(r);
            }

            gsl_rng_free(r_master);
        }
        else
        {
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < (int) (detx_size * dety_size); i++)
            {
                out[i] = pfxss[i % detx_size] * pfyss[i / detx_size] * dx * dy;
            }
        }
        out += detx_size * dety_size; pfx += xpts;
    }
    free(pfxss); free(pfyss);

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

    array arr1 = new_array(ndim1, dims1, item_size, inp1);
    array arr2 = new_array(ndim2, dims2, item_size, inp2);

    int rep1 = arr1->size / arr1->dims[axis1];
    int rep2 = arr2->size / arr2->dims[axis2];
    int repeats = rep1 * rep2;
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        line line1 = init_line(arr1, axis1);
        line line2 = init_line(arr2, axis2);
        int div;

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            div = i / rep2;
            UPDATE_LINE(line1, div);
            UPDATE_LINE(line2, i - div * rep2);

            dot_func(out + i * line1->item_size, line1, line2);
        }

        free(line1); free(line2);
    }

    free_array(arr1);
    free_array(arr2);

    return 0;
}