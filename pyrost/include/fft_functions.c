#include "fft_functions.h"
#include "pocket_fft.h"

static int fft_convolve_calc(void *rfft_plan, void *irfft_plan, double *out, double *inp, double *krn,
    int flen, int npts, size_t istride, rfft_func rfft, rfft_func irfft)
{
    int fail = 0;
    fail = rfft(rfft_plan, inp, flen);
    double re, im;
    for (int i = 0; i < (flen / 2 + 1); i++)
    {
        re = (inp[2 * i] * krn[2 * i] - inp[2 * i + 1] * krn[2 * i + 1]);
        im = (inp[2 * i] * krn[2 * i + 1] + inp[2 * i + 1] * krn[2 * i]);
        inp[2 * i] = re; inp[2 * i + 1] = im;
    }
    fail = irfft(irfft_plan, inp, flen);
    for (int i = 0; i < npts / 2; i++) out[i * istride] = inp[i + flen - npts / 2];
    for (int i = 0; i < npts / 2 + npts % 2; i++) out[(i + npts / 2) * istride] = inp[i];
    return fail;
}

int fft_convolve_np(double *out, const double *inp, const double *krn, size_t isize,
    size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode, double cval, unsigned threads)
{
    int fail = 0;
    int flen = good_size(npts + ksize - 1);
    int repeats = isize / npts;
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double *inpft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        double *krnft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        rfft_plan plan = make_rfft_plan(flen);

        extend_line_double(krnft, krn, EXTEND_CONSTANT, 0., flen, ksize, 1);
        fail |= rfft_np((void *)plan, krnft, flen);

        #pragma omp for
        for (int i = 0; i < repeats; i++)
        {
        extend_line_double(inpft, inp + npts * istride * (i / istride) + (i % istride),
            mode, cval, flen, npts, istride);
        fail |= fft_convolve_calc((void *)plan, (void *)plan,
            out + npts * istride * (i / istride) + (i % istride),
            inpft, krnft, flen, npts, istride, rfft_np, irfft_np);
        }

        destroy_rfft_plan(plan);
        free(inpft); free(krnft);    
    }

    return fail;
}

int fft_convolve_fftw(double *out, const double *inp, const double *krn, size_t isize,
    size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode, double cval, unsigned threads)
{
    int fail = 0;
    int flen = next_fast_len_fftw(npts + ksize - 1);
    int repeats = isize / npts;
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double *inpft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
        double *krnft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
        fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
        dim->n = flen; dim->is = 1; dim->os = 1;
        fftw_plan rfft_plan, irfft_plan;

        #pragma omp critical
        {
            rfft_plan = fftw_plan_guru_dft_r2c(1, dim, 0, NULL, inpft, (fftw_complex *)inpft,
                FFTW_ESTIMATE);
            irfft_plan = fftw_plan_guru_dft_c2r(1, dim, 0, NULL, (fftw_complex *)inpft,
                inpft, FFTW_ESTIMATE);
        }
        
        extend_line_double(krnft, krn, EXTEND_CONSTANT, 0., flen, ksize, 1);
        fail |= rfft_fftw((void *)rfft_plan, krnft, flen);

        #pragma omp for
        for (int i = 0; i < repeats; i++)
        {
            extend_line_double(inpft, inp + npts * istride * (i / istride) + (i % istride),
                mode, cval, flen, npts, istride);
            fail |= fft_convolve_calc((void *)rfft_plan, (void *)irfft_plan,
                out + npts * istride * (i / istride) + (i % istride),
                inpft, krnft, flen, npts, istride, rfft_fftw, irfft_fftw);
        }

        fftw_destroy_plan(rfft_plan);
        fftw_destroy_plan(irfft_plan);
        fftw_free(inpft); fftw_free(krnft);
        free(dim);
    }

    return fail;
}