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

    #pragma omp parallel num_threads(threads)
    {
        double x0, x1;
        int jj, j0, j1, j;

        #pragma omp for
        for (int n = 0; n < (int) isize; n++)
        {
            out[n] = 1.0;
            jj = searchsorted(&xcrd[n], layers, lsize, sizeof(double), SEARCH_LEFT, compare_double);
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
    }

    return 0;
}

static void rebin_line_double(double *out, const double *inp, size_t osize, size_t isize, unsigned threads)
{
    double ratio = (double) isize / osize;
    threads = (threads > (unsigned) osize) ? (unsigned) osize : threads;

    #pragma omp parallel num_threads(threads)
    {
        double lb, ub;
        int j0, j1;

        #pragma omp for
        for (int i = 0; i < (int) osize; i++)
        {
            out[i] = 0;
            j0 = (int)(i * ratio);
            j1 = (int)((i + 1) * ratio);
            for (int j = j0; (j <= j1) && (j < (int) isize); j++)
            {
                lb = ((double) j) > (i * ratio) ? j : i * ratio;
                ub = ((double) j + 1) < ((i + 1) * ratio) ? j + 1 : (i + 1) * ratio;
                out[i] += (ub - lb) * inp[j];
            }
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