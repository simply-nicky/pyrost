#include "st_utils.h"

void extend_line_complex(double complex *out, const double complex *inp, EXTEND_MODE mode, double complex cval,
    size_t osize, size_t isize, size_t istride)
{
    int dsize = osize - isize;
    int size1 = dsize - dsize / 2;
    int size2 = dsize - size1;
    for (int i = 0; i < (int) isize; i++) out[i + size1] = inp[i * istride];
    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case EXTEND_CONSTANT:
            for (int i = 0; i < size1; i++) out[i] = cval;
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = cval;
            break;
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:
            for (int i = 0; i < size1; i++) out[i] = inp[0];
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = inp[(isize - 1) * istride];
            break;
            /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:
            for (int i = (int)(isize - size1 - 1); i < (int)(isize - 1); i++)
            {
                int fct = (i / (isize - 1)) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % (isize - 1));
                out[i - isize + size1 + 1] = inp[idx * istride];
            }
            for (int i = 1; i <= size2; i++)
            {
                int fct = (i / (isize - 1)) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % (isize - 1));
                out[isize + size1 - 1 + i] = inp[idx * istride];
            }
            break;
        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            for (int i = (int)(isize - size1); i < (int)(isize); i++)
            {
                int fct = (i / isize) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % isize);
                out[i - isize + size1] = inp[idx * istride];
            }
            for (int i = 0; i < size2; i++)
            {
                int fct = (i / isize) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % isize);
                out[isize + size1 + i] = inp[idx * istride];
            }
            break;
        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            for (int i = (int)(isize - size1); i < (int)isize; i++) out[i - isize + size1] = inp[(i % isize) * istride];
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = inp[(i % isize) * istride];
            break;
    }
}

void extend_line_double(double *out, const double *inp, EXTEND_MODE mode, double cval, size_t osize,
    size_t isize, size_t istride)
{
    int dsize = osize - isize;
    int size1 = dsize - dsize / 2;
    int size2 = dsize - size1;
    for (int i = 0; i < (int) isize; i++) out[i + size1] = inp[i * istride];
    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case EXTEND_CONSTANT:
            for (int i = 0; i < size1; i++) out[i] = cval;
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = cval;
            break;
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:
            for (int i = 0; i < size1; i++) out[i] = inp[0];
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = inp[(isize - 1) * istride];
            break;
            /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:
            for (int i = (int)(isize - size1 - 1); i < (int)(isize - 1); i++)
            {
                int fct = (i / (isize - 1)) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % (isize - 1));
                out[i - isize + size1 + 1] = inp[idx * istride];
            }
            for (int i = 1; i <= size2; i++)
            {
                int fct = (i / (isize - 1)) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % (isize - 1));
                out[isize + size1 - 1 + i] = inp[idx * istride];
            }
            break;
        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            for (int i = (int)(isize - size1); i < (int)(isize); i++)
            {
                int fct = (i / isize) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % isize);
                out[i - isize + size1] = inp[idx * istride];
            }
            for (int i = 0; i < size2; i++)
            {
                int fct = (i / isize) % 2;
                int idx = (isize - 1) * (1 - fct) + (2 * fct - 1) * (i % isize);
                out[isize + size1 + i] = inp[idx * istride];
            }
            break;
        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            for (int i = (int)(isize - size1); i < (int)isize; i++) out[i - isize + size1] = inp[(i % isize) * istride];
            for (int i = 0; i < size2; i++) out[isize + size1 + i] = inp[(i % isize) * istride];
            break;
    }
}

NOINLINE int compare_double(const void *a, const void *b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}

NOINLINE int compare_float(const void *a, const void *b)
{
    if (*(float*)a > *(float*)b) return 1;
    else if (*(float*)a < *(float*)b) return -1;
    else return 0;
}

NOINLINE int compare_long(const void *a, const void *b)
{
    return (*(long *)a - *(long *)b);
}

size_t binary_search(const void *key, const void *array, size_t l, size_t r, size_t size,
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

void ml_profile(double complex *out, const double *inp, const double *layers, size_t isize, size_t lsize, 
    size_t nlyr, double complex mt0, double complex mt1, double complex mt2, double sgm, unsigned threads)
{
    int b = 2 * (nlyr / 2);
    int repeats = lsize / nlyr;
    for (int i = 0; i < repeats; i++)
    {
        #pragma omp parallel for num_threads(threads)
        for (int n = 0; n < (int) isize; n++)
        {
            double complex ref_idx = 0;
            int j0 = searchsorted(&inp[n], &layers[i * nlyr], nlyr, sizeof(double), compare_double);
            if (j0 > 0 && j0 < b)
            {
                double x0 = (inp[n] - layers[j0 - 1 + i * nlyr]) / sqrt(2) / sgm;
                double x1 = (inp[n] - layers[j0 + i * nlyr]) / sqrt(2) / sgm;
                ref_idx += (mt1 - mt2) / 2 * ((double) (j0 % 2) - 0.5) * (erf(x0) - erf(x1));
                ref_idx -= (mt1 - mt2) / 4 * erf((inp[n] - layers[i * nlyr]) / sqrt(2) / sgm);
                ref_idx += (mt1 - mt2) / 4 * erf((inp[n] - layers[b - 1 + i * nlyr]) / sqrt(2) / sgm);
            }
            ref_idx += (mt1 + mt0) / 2 * erf((inp[n] - layers[i * nlyr]) / sqrt(2) / sgm);
            ref_idx -= (mt1 + mt0) / 2 * erf((inp[n] - layers[b - 1 + i * nlyr]) / sqrt(2) / sgm);
            out[n + i * isize] = (cos(creal(ref_idx)) + sin(creal(ref_idx)) * I) * exp(-cimag(ref_idx));
        }
    }
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

void frames(double *out, const double *pfx, const double *pfy, double dx, double dy, size_t *ishape, size_t *oshape,
    long seed, unsigned threads)
{
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
}

static void *wirthselect(void *data, size_t k, size_t l, size_t m, size_t size,
    int (*compar)(const void*, const void*))
{
    int i, j, n_iter = 0;
    int max_iter = (int)(m - l);
    void *key = malloc(size);
    while ((l < m) & (n_iter++ < max_iter))
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
        if (j < (int) k) l = i;
        if ((int) k < i) m = j;
    }
    free(key);
    return data + k * size;
}

void whitefield(void *out, const void *data, const unsigned char *mask, size_t isize,
    size_t npts, size_t istride, size_t size, int (*compar)(const void*, const void*), unsigned threads)
{
    int repeats = isize / npts;
    threads = (threads > (size_t) repeats) ? (size_t) repeats : threads;
    #pragma omp parallel num_threads(threads)
    {
        unsigned char *buffer = (unsigned char *)malloc(npts * size);
        #pragma omp for
        for (int i = 0; i < repeats; i++)
        {
            int len = 0;
            for (int n = 0; n < (int) npts; n++)
            {
                if (mask[n * istride + npts * istride * (i / istride) + (i % istride)])
                {
                    memcpy(buffer + len++ * size, data + (n * istride + npts * istride * (i / istride) + (i % istride)) * size, size);
                }
            }
            if (len) memcpy(out + (npts * istride * (i / istride) + (i % istride)) * size,
                wirthselect(buffer, (len & 1) ? (len / 2) : (len / 2 - 1), 0, len - 1, size, compar), size);
            else memset(out + (npts * istride * (i / istride) + (i % istride)) * size, 0, size);
        }
        free(buffer);
    }
}