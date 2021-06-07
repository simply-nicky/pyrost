#include "st_gaussian.h"
#include "st_waveprop_fftw.h"
#include "st_waveprop_np.h"

void gauss_kernel1d(double *out, double sigma, unsigned order, size_t ksize)
{
    int radius = (ksize - 1) / 2;
    double sum = 0;
    double sigma2 = sigma * sigma;
    for (int i = 0; i < (int) ksize; i++)
    {
        out[i] = exp(-0.5 * pow(i - radius, 2) / sigma2); sum += out[i];
    }
    for (int i = 0; i < (int) ksize; i++) out[i] /= sum;
    if (order)
    {
        double *q0 = (double *)calloc(order + 1, sizeof(double)); q0[0] = 1.;
        double *q1 = (double *)calloc(order + 1, sizeof(double));
        int idx; double qval;
        for (int k = 0; k < (int) order; k++)
        {
            for (int i = 0; i <= (int) order; i++)
            {
                qval = 0;
                for (int j = 0; j <= (int) order; j++)
                {
                    idx = j + (order + 1) * i;
                    if ((idx % (order + 2)) == 1) qval += q0[j] * (idx / (order + 2) + 1);
                    if ((idx % (order + 2)) == (order + 1)) qval -= q0[j] / sigma2; 
                }
                q1[i] = qval;
            }
            for (int i = 0; i <= (int) order; i++) q0[i] = q1[i];
        }
        free(q0);
        double fct;
        for (int i = 0; i < (int) ksize; i++)
        {
            fct = 0;
            for (int j = 0; j <= (int) order; j++) fct += pow(i - radius, j) * q1[j];
            out[i] *= fct;
        }
        free(q1);
    }
}

void gauss_filter_fftw(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads)
{
    size_t isize = 1;
    for (int i = 0; i < ndim; i++) isize *= dims[i];
    for (int i = 0; i < (int) isize; i++) out[i] = inp[i];
    for (int n = 0; n < ndim; n++)
    {
        if (sigma[n] > 1e-15)
        {
            size_t istride = isize;
            for (int i = 0; i <= n; i++) istride /= dims[i];
            size_t ksize = 2 * (size_t) (sigma[n] * truncate) + 1;
            double *krn = (double *)malloc(ksize * sizeof(double));
            gauss_kernel1d(krn, sigma[n], order[n], ksize);
            fft_convolve_fftw(out, out, krn, isize, dims[n], istride, ksize, mode, cval, threads);
            free(krn);
        }
    }
}

int gauss_filter_np(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads)
{
    int fail = 0;
    size_t isize = 1;
    for (int i = 0; i < ndim; i++) isize *= dims[i];
    for (int i = 0; i < (int) isize; i++) out[i] = inp[i];
    for (int n = 0; n < ndim; n++)
    {
        if (sigma[n] > 1e-15)
        {
            size_t istride = isize;
            for (int i = 0; i <= n; i++) istride /= dims[i];
            size_t ksize = 2 * (size_t) (sigma[n] * truncate) + 1;
            double *krn = (double *)malloc(ksize * sizeof(double));
            gauss_kernel1d(krn, sigma[n], order[n], ksize);
            fail |= fft_convolve_np(out, out, krn, isize, dims[n], istride, ksize, mode, cval, threads);
            free(krn);
        }
    }
    return fail;
}

void gauss_grad_fftw(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads)
{
    size_t isize = 1;
    for (int i = 0; i < ndim; i++) isize *= dims[i];

    double *g0[ndim], *g1[ndim];
    size_t *strides = (size_t *)malloc(ndim * sizeof(size_t));
    size_t *ksizes = (size_t *)malloc(ndim * sizeof(size_t));
    for (int n = 0; n < ndim; n++)
    {
        strides[n] = isize;
        for (int i = 0; i <= n; i++) strides[n] /= dims[i];

        if (sigma[n] > 1e-15)
        {
            ksizes[n] = 2 * (size_t) (sigma[n] * truncate) + 1;
            g0[n] = (double *)malloc(ksizes[n] * sizeof(double));
            g1[n] = (double *)malloc(ksizes[n] * sizeof(double));
            gauss_kernel1d(g0[n], sigma[n], 0, ksizes[n]);
            gauss_kernel1d(g1[n], sigma[n], 1, ksizes[n]);
        }
        else {g0[n] = NULL; g1[n] = NULL; ksizes[n] = 0;}
    }

    if (ksizes[0] != 0) fft_convolve_fftw(out, inp, g1[0], isize, dims[0],
        strides[0], ksizes[0], mode, cval, threads);
    else {for (int i = 0; i < (int) isize; i++) out[i] = inp[i];}
    for (int n = 1; n < ndim; n++)
    {
        if (ksizes[n] != 0) fft_convolve_fftw(out, out, g0[n], isize, dims[n],
            strides[n], ksizes[n], mode, cval, threads);
    }
    for (int i = 0; i < (int) isize; i++) out[i] = out[i] * out[i];

    double *tmp = (double *)malloc(isize * sizeof(double));
    for (int m = 1; m < ndim; m++)
    {
        if (ksizes[0] != 0) fft_convolve_fftw(tmp, inp, g0[0], isize, dims[0],
            strides[0], ksizes[0], mode, cval, threads);
        else {for (int i = 0; i < (int) isize; i++) tmp[i] = inp[i];}
        for (int n = 1; n < ndim; n++)
        {
            if (ksizes[n])
            {
                if (n == m) {fft_convolve_fftw(tmp, tmp, g1[n], isize, dims[n],
                    strides[n], ksizes[n], mode, cval, threads);}
                else {fft_convolve_fftw(tmp, tmp, g0[n], isize, dims[n],
                    strides[n], ksizes[n], mode, cval, threads);}
            }
        }
        for (int i = 0; i < (int) isize; i++) out[i] += tmp[i] * tmp[i];
    }
    free(tmp); free(ksizes); free(strides);
    for (int n = 0; n < ndim; n++) {free(g0[n]); free(g1[n]);}

    for (int i = 0; i < (int) isize; i++) out[i] = sqrt(out[i]);
}

int gauss_grad_np(double *out, const double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads)
{
    int fail = 0;
    size_t isize = 1;
    for (int i = 0; i < ndim; i++) isize *= dims[i];

    double *g0[ndim], *g1[ndim];
    size_t *strides = (size_t *)malloc(ndim * sizeof(size_t));
    size_t *ksizes = (size_t *)malloc(ndim * sizeof(size_t));
    for (int n = 0; n < ndim; n++)
    {
        strides[n] = isize;
        for (int i = 0; i <= n; i++) strides[n] /= dims[i];

        if (sigma[n] > 1e-15)
        {
            ksizes[n] = 2 * (size_t) (sigma[n] * truncate) + 1;
            g0[n] = (double *)malloc(ksizes[n] * sizeof(double));
            g1[n] = (double *)malloc(ksizes[n] * sizeof(double));
            gauss_kernel1d(g0[n], sigma[n], 0, ksizes[n]);
            gauss_kernel1d(g1[n], sigma[n], 1, ksizes[n]);
        }
        else {g0[n] = NULL; g1[n] = NULL; ksizes[n] = 0;}
    }

    if (ksizes[0] != 0) fail |= fft_convolve_np(out, inp, g1[0], isize, dims[0],
        strides[0], ksizes[0], mode, cval, threads);
    else for (int i = 0; i < (int) isize; i++) out[i] = inp[i];
    for (int n = 1; n < ndim; n++)
    {
        if (ksizes[n] != 0) fail |= fft_convolve_np(out, out, g0[n], isize, dims[n],
            strides[n], ksizes[n], mode, cval, threads);
    }
    for (int i = 0; i < (int) isize; i++) out[i] = out[i] * out[i];

    double *tmp = (double *)malloc(isize * sizeof(double));
    for (int m = 1; m < ndim; m++)
    {
        if (ksizes[0] != 0) fail |= fft_convolve_np(tmp, inp, g0[0], isize, dims[0],
            strides[0], ksizes[0], mode, cval, threads);
        else for (int i = 0; i < (int) isize; i++) tmp[i] = inp[i];
        for (int n = 1; n < ndim; n++)
        {
            if (ksizes[n])
            {
                if (n == m) {fail |= fft_convolve_np(tmp, tmp, g1[n], isize, dims[n],
                    strides[n], ksizes[n], mode, cval, threads);}
                else {fail |= fft_convolve_np(tmp, tmp, g0[n], isize, dims[n],
                    strides[n], ksizes[n], mode, cval, threads);}
            }
        }
        for (int i = 0; i < (int) isize; i++) out[i] += tmp[i] * tmp[i];
    }
    free(tmp); free(ksizes); free(strides);
    for (int n = 0; n < ndim; n++) {free(g0[n]); free(g1[n]);}

    for (int i = 0; i < (int) isize; i++) out[i] = sqrt(out[i]);
    return fail;
}