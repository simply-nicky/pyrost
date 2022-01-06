#include "fft_functions.h"
#include "pocket_fft.h"

typedef int (*fft_func)(void *plan, double complex *inp);
typedef int (*rfft_func)(void *plan, double *inp, size_t npts);

static int rfft_convolve_calc(void *rfft_plan, void *irfft_plan, line out, double *inp,
    double *krn, size_t flen, rfft_func rfft, rfft_func irfft)
{
    int fail = 0;
    fail = rfft(rfft_plan, inp, flen);
    double re, im;
    for (int i = 0; i < (int)flen / 2 + 1; i++)
    {
        re = (inp[2 * i] * krn[2 * i] - inp[2 * i + 1] * krn[2 * i + 1]) / flen;
        im = (inp[2 * i] * krn[2 * i + 1] + inp[2 * i + 1] * krn[2 * i]) / flen;
        inp[2 * i] = re; inp[2 * i + 1] = im;
    }
    fail = irfft(irfft_plan, inp, flen);
    for (int i = 0; i < (int)out->npts / 2; i++) ((double *)out->data)[i * out->stride] = inp[i + flen - out->npts / 2];
    for (int i = 0; i < (int)out->npts / 2 + (int)out->npts % 2; i++) ((double *)out->data)[(i + out->npts / 2) * out->stride] = inp[i];
    return fail;
}

static int cfft_convolve_calc(void *fft_plan, void *ifft_plan, line out, double complex *inp,
    double complex *krn, size_t flen, fft_func fft, fft_func ifft)
{
    int fail = 0;
    fail = fft(fft_plan, inp);
    for (int i = 0; i < (int)flen; i++) inp[i] *= krn[i] / flen;
    fail = ifft(ifft_plan, inp);
    for (int i = 0; i < (int)out->npts / 2; i++) ((double complex *)out->data)[i * out->stride] = inp[i + flen - out->npts / 2];
    for (int i = 0; i < (int)out->npts / 2 + (int)out->npts % 2; i++) ((double complex *)out->data)[(i + out->npts / 2) * out->stride] = inp[i];
    return fail;
}

int rfft_convolve_np(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convole_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double), krn);
    
    int fail = 0;
    size_t flen = good_size(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double *inpft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        double *krnft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        rfft_plan plan = make_rfft_plan(flen);

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= rfft_np((void *)plan, krnft, flen);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= rfft_convolve_calc((void *)plan, (void *)plan, oline, inpft, krnft,
                flen, rfft_np, irfft_np);
        }

        free(iline); free(oline);
        destroy_rfft_plan(plan);
        free(inpft); free(krnft);    
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

int cfft_convolve_np(double complex *out, double complex *inp, int ndim, size_t *dims,
    double complex *krn, size_t ksize, int axis, EXTEND_MODE mode, double complex cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convole_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double complex), krn);
    
    int fail = 0;
    size_t flen = good_size(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *inpft = (double complex *)malloc(flen * sizeof(double complex));
        double complex *krnft = (double complex *)malloc(flen * sizeof(double complex));
        cfft_plan plan = make_cfft_plan(flen);

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= fft_np((void *)plan, krnft);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= cfft_convolve_calc((void *)plan, (void *)plan, oline, inpft, krnft,
                flen, fft_np, ifft_np);
        }

        free(iline); free(oline);
        destroy_cfft_plan(plan);
        free(inpft); free(krnft);    
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

int rfft_convolve_fftw(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convolve_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double), krn);
    
    int fail = 0;
    size_t flen = next_fast_len_fftw(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

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

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= rfft_fftw((void *)rfft_plan, krnft, flen);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= rfft_convolve_calc((void *)rfft_plan, (void *)irfft_plan, oline,
                inpft, krnft, flen, rfft_fftw, irfft_fftw);
        }

        free(iline); free(oline);
        fftw_destroy_plan(rfft_plan);
        fftw_destroy_plan(irfft_plan);
        free(dim); fftw_free(inpft); fftw_free(krnft);
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

int cfft_convolve_fftw(double complex *out, double complex *inp, int ndim, size_t *dims,
    double complex *krn, size_t ksize, int axis, EXTEND_MODE mode, double complex cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convolve_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double complex), krn);
    
    int fail = 0;
    size_t flen = next_fast_len_fftw(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *inpft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        double complex *krnft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
        dim->n = flen; dim->is = 1; dim->os = 1;
        fftw_plan fft_plan, ifft_plan;

        #pragma omp critical
        {
            fft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_FORWARD, FFTW_ESTIMATE);
            ifft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_BACKWARD, FFTW_ESTIMATE);
        }

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= fft_fftw((void *)fft_plan, krnft);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= cfft_convolve_calc((void *)fft_plan, (void *)ifft_plan, oline,
                inpft, krnft, flen, fft_fftw, ifft_fftw);
        }

        free(iline); free(oline);
        fftw_destroy_plan(fft_plan);
        fftw_destroy_plan(ifft_plan);
        free(dim); fftw_free(inpft); fftw_free(krnft);
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

static int rsc_type1_calc(void *fft_plan, void *ifft_plan, line out, double complex *inp,
    double complex *krn, int flen, double dx0, double dx, double z, double wl, fft_func fft, fft_func ifft)
{
    int fail = 0;
    double ph, dist;
    for (int i = 0; i < flen; i++)
    {
        dist = SQ(dx0 * (i - flen / 2)) + SQ(z);
        ph = 2 * M_PI / wl * sqrt(dist);
        krn[i] = -dx0 * z / sqrt(wl) * (sin(ph) + cos(ph) * I) / pow(dist, 0.75);
    }

    fail |= fft(fft_plan, inp);
    fail |= fft(fft_plan, krn);

    for (int i = 0; i < flen; i++)
    {
        ph = M_PI * SQ(((double) i / flen) - ((2 * i) / flen)) * dx / dx0 * flen;
        inp[i] *= krn[i] * (cos(ph) - sin(ph) * I) / flen;
        krn[i] = (cos(ph) + sin(ph) * I) / flen;
    }

    fail |= ifft(ifft_plan, inp);
    fail |= ifft(ifft_plan, krn);

    for (int i = 0; i < flen; i++) inp[i] *= krn[i];
    fail |= fft(fft_plan, inp);

    for (int i = 0; i < (int)out->npts / 2; i++)
    {
        ph = M_PI * SQ((double) (i - (int)out->npts / 2) / flen) * dx / dx0 * flen;
        ((double complex *)out->data)[i * out->stride] = inp[i + flen - (int)out->npts / 2] * (cos(ph) - sin(ph) * I);
    }

    for (int i = 0; i < (int)out->npts / 2 + (int)out->npts % 2; i++)
    {
        ph = M_PI * SQ((double) i / flen) * dx / dx0 * flen;
        ((double complex *)out->data)[(i + out->npts / 2) * out->stride] = inp[i] * (cos(ph) - sin(ph) * I);
    }
  return fail;
}

static int rsc_type2_calc(void *fft_plan, void *ifft_plan, line out, double complex *inp,
    double complex *krn, int flen, double dx0, double dx, double z, double wl, fft_func fft, fft_func ifft)
{
    int fail = 0;
    double ph, dist;
    for (int i = 0; i < flen; i++)
    {
        ph = M_PI * SQ(i - flen / 2) * dx0 / dx / flen;
        krn[i] = cos(ph) - sin(ph) * I;
        inp[i] *= cos(ph) + sin(ph) * I;
    }

    fail |= fft(fft_plan, inp);
    fail |= fft(fft_plan, krn);

    for (int i = 0; i < flen; i++)
    {
        inp[i] *= krn[i] / flen;
        dist = SQ(dx * (i - flen / 2)) + SQ(z);
        ph = 2 * M_PI / wl * sqrt(dist);
        krn[i] = -dx0 * z / sqrt(wl) * (sin(ph) + cos(ph) * I) / pow(dist, 0.75);
    }

    fail |= fft(ifft_plan, krn);
    fail |= ifft(ifft_plan, inp);

    for (int i = 0; i < flen; i++)
    {
        ph = M_PI * SQ(((double) i / flen) - ((2 * i) / flen)) * dx0 / dx * flen;
        inp[i] *= krn[i] * (cos(ph) + sin(ph) * I) / flen;
    }

    fail |= ifft(ifft_plan, inp);
    for (int i = 0; i < (int)out->npts; i++)
    {((double complex *)out->data)[i * out->stride] = inp[i + (flen - out->npts) / 2];}
    return fail;
}

typedef int (*rsc_func)(void*, void*, line, double complex*, double complex*, int,
    double, double, double, double, fft_func, fft_func);

int rsc_np(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims) {ERROR("rsc_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("rsc_np: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("rsc_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("rsc_np: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);

    int fail = 0;
    dx = fabs(dx); dx0 = fabs(dx0);
    rsc_func rsc_calc = (dx0 >= dx) ? rsc_type1_calc : rsc_type2_calc;
    double alpha = (dx0 <= dx) ? (dx0 / dx) : (dx / dx0);

    size_t flen = good_size((size_t) (iarr->dims[axis] * (1 + alpha)) + 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *krnft = (double complex *)malloc(flen * sizeof(double complex));
        double complex *inpft = (double complex *)malloc(flen * sizeof(double complex));
        cfft_plan plan = make_cfft_plan(flen);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, EXTEND_CONSTANT, (void *)&zerro);
            fail |= rsc_calc((void *)plan, (void *)plan, oline, inpft, krnft, flen,
                dx0, dx, z, wl, fft_np, ifft_np);
        }

        free(iline); free(oline);
        destroy_cfft_plan(plan);
        free(inpft); free(krnft);
    }

    free_array(iarr);
    free_array(oarr);

    return fail;
}

int rsc_fftw(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims) {ERROR("rsc_fftw: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("rsc_fftw: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("rsc_fftw: invalid axis."); return -1;}
    if (threads == 0) {ERROR("rsc_fftw: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);

    int fail = 0;
    dx = fabs(dx); dx0 = fabs(dx0);
    rsc_func rsc_calc = (dx0 >= dx) ? rsc_type1_calc : rsc_type2_calc;
    double alpha = (dx0 <= dx) ? (dx0 / dx) : (dx / dx0);

    size_t flen = next_fast_len_fftw((size_t) (iarr->dims[axis] * (1 + alpha)) + 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *krnft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        double complex *inpft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
        dim->n = flen; dim->is = 1; dim->os = 1;
        fftw_plan fft_plan, ifft_plan;

        #pragma omp critical
        {
            fft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_FORWARD, FFTW_ESTIMATE);
            ifft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_BACKWARD, FFTW_ESTIMATE);
        }

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, EXTEND_CONSTANT, (void *)&zerro);
            fail |= rsc_calc((void *)fft_plan, (void *)ifft_plan, oline, inpft, krnft, flen,
                dx0, dx, z, wl, fft_fftw, ifft_fftw);
        }

        free(iline); free(oline);
        fftw_destroy_plan(fft_plan);
        fftw_destroy_plan(ifft_plan);
        free(dim); free(inpft); free(krnft);
    }

    free_array(iarr);
    free_array(oarr);

    return fail;
}

int fraunhofer_calc(void *fft_plan, void *ifft_plan, line out, double complex *inp,
    double complex *krn, int flen, double dx0, double dx, double alpha, fft_func fft, fft_func ifft)
{
    int fail = 0;
    double ph;
    for (int i = 0; i < flen; i++)
    {
        ph = M_PI * SQ(i - flen / 2) * alpha;
        inp[i] *= cos(ph) + sin(ph) * I;
    }

    fail |= fft(fft_plan, inp);
    for (int i = 0; i < flen; i++) inp[i] *= krn[i] / flen;
    fail |= ifft(ifft_plan, inp);
    
    double complex w;
    for (int i = 0; i < (int)out->npts / 2; i++)
    {
        ph = M_PI * SQ(i - (int)out->npts / 2) * alpha;
        w = (cos(ph) - sin(ph) * I) * inp[i + flen - (int)out->npts / 2];
        ((double complex *)out->data)[i * out->stride] = (cos(ph / dx0 * dx) - sin(ph / dx0 * dx) * I) * w;
    }

    for (int i = 0; i < (int)out->npts / 2 + (int)out->npts % 2; i++)
    {
        ph = M_PI * SQ(i) * alpha;
        w = (cos(ph) - sin(ph) * I) * inp[i];
        ((double complex *)out->data)[(i + out->npts / 2) * out->stride] = (cos(ph / dx0 * dx) - sin(ph / dx0 * dx) * I) * w;
    }

    return fail;
}

int fraunhofer_np(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims) {ERROR("fraunhofer_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("fraunhofer_np: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fraunhofer_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fraunhofer_np: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);

    int fail = 0;
    dx = fabs(dx); dx0 = fabs(dx0);
    double alpha = (dx0 <= dx) ? (dx0 / dx) : (dx / dx0);
    size_t flen = good_size((size_t) (iarr->dims[axis] * (1 + alpha)) + 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *krnft = (double complex *)malloc(flen * sizeof(double complex));
        double complex *inpft = (double complex *)malloc(flen * sizeof(double complex));
        cfft_plan plan = make_cfft_plan(flen);

        double ph = 2 * M_PI / wl * z;
        double alpha = dx0 * dx / wl / z;
        double complex k0 = -(sin(ph) + cos(ph) * I) / sqrt(wl * z) * dx0;
        for (int i = 0; i < (int)flen; i++)
        {
            ph = M_PI * SQ(i - (int)flen / 2) * alpha;
            krnft[i] = k0 * (cos(ph) - sin(ph) * I);
        }
        fail |= fft_np(plan, krnft);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, EXTEND_CONSTANT, (void *)&zerro);
            fail |= fraunhofer_calc((void *)plan, (void *)plan, oline, inpft, krnft, flen,
                dx0, dx, alpha, fft_np, ifft_np);
        }

        free(iline); free(oline);
        destroy_cfft_plan(plan);
        free(inpft); free(krnft);
    }

    free_array(iarr);
    free_array(oarr);

    return fail;
}

int fraunhofer_fftw(double complex *out, double complex *inp, int ndim, size_t *dims, int axis,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims) {ERROR("fraunhofer_fftw: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("fraunhofer_fftw: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fraunhofer_fftw: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fraunhofer_fftw: threads must be positive."); return -1;}

    double complex zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double complex), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double complex), (void *)inp);

    int fail = 0;
    dx = fabs(dx); dx0 = fabs(dx0);
    double alpha = (dx0 <= dx) ? (dx0 / dx) : (dx / dx0);
    size_t flen = next_fast_len_fftw((size_t) (iarr->dims[axis] * (1 + alpha)) + 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double complex *krnft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        double complex *inpft = (double complex *)fftw_malloc(flen * sizeof(double complex));
        fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
        dim->n = flen; dim->is = 1; dim->os = 1;
        fftw_plan fft_plan, ifft_plan;

        #pragma omp critical
        {
            fft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_FORWARD, FFTW_ESTIMATE);
            ifft_plan = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)inpft,
                (fftw_complex *)inpft, FFTW_BACKWARD, FFTW_ESTIMATE);
        }

        double ph = 2.0 * M_PI / wl * z;
        double alpha = dx0 * dx / wl / z;
        double complex k0 = -(sin(ph) + cos(ph) * I) / sqrt(wl * z) * dx0;
        for (int i = 0; i < (int)flen; i++)
        {
            ph = M_PI * SQ(i - (int)flen / 2) * alpha;
            krnft[i] = k0 * (cos(ph) - sin(ph) * I);
        }
        fail |= fft_fftw(fft_plan, krnft);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(iline, i);
            UPDATE_LINE(oline, i);
            extend_line((void *)inpft, flen, iline, EXTEND_CONSTANT, (void *)&zerro);
            fail |= fraunhofer_calc((void *)fft_plan, (void *)ifft_plan, oline, inpft, krnft, flen,
                dx0, dx, alpha, fft_fftw, ifft_fftw);
        }

        free(iline); free(oline);
        fftw_destroy_plan(fft_plan);
        fftw_destroy_plan(ifft_plan);
        free(dim); free(inpft); free(krnft);
    }

    free_array(iarr);
    free_array(oarr);

    return fail;
}

int gauss_kernel1d(double *out, double sigma, unsigned order, size_t ksize, int step)
{
    /* check parameters */
    if (!out) {ERROR("gauss_kernel1d: out is NULL."); return -1;}
    if (sigma <= 0) {ERROR("gauss_kernel1d: sigma must be positive."); return -1;}
    if (!ksize) {ERROR("gauss_kernel1d: ksize must be positive."); return -1;}

    int radius = (ksize - 1) / 2;
    double sum = 0;
    double sigma2 = sigma * sigma;
    for (int i = 0; i < (int) ksize; i++)
    {
        out[i * step] = exp(-0.5 * SQ(i - radius) / sigma2); sum += out[i * step];
    }
    for (int i = 0; i < (int) ksize; i++) out[i * step] /= sum;
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
            out[i * step] *= fct;
        }
        free(q1);
    }
    return 0;
}

int gauss_filter_r(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    rconvolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma || !order)
    {ERROR("gauss_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_filter: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_filter: threads must be positive."); return -1;}

    int fail = 0;
    int axis = 0;
    while (sigma[axis] < 1e-15 && axis < ndim) axis++;
    if (axis < ndim)
    {
        size_t ksize = 2 * (size_t) (sigma[axis] * truncate) + 1;
        double *krn = (double *)malloc(ksize * sizeof(double));
        fail |= gauss_kernel1d(krn, sigma[axis], order[axis], ksize, 1);
        fail |= fft_convolve(out, inp, ndim, dims, krn, ksize, axis, mode, cval, threads);
        free(krn);

        for (int n = axis + 1; n < ndim; n++)
        {
            if (sigma[n] > 1e-15)
            {
                ksize = 2 * (size_t) (sigma[n] * truncate) + 1;
                krn = (double *)malloc(ksize * sizeof(double));
                fail |= gauss_kernel1d(krn, sigma[n], order[n], ksize, 1);
                fail |= fft_convolve(out, out, ndim, dims, krn, ksize, n, mode, cval, threads);
                free(krn);
            }
        }
    }
    else
    {
        size_t size = 1;
        for (int n = 0; n < ndim; n++) size *= dims[n];
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] = inp[i];
    }
    return fail;
}

int gauss_filter_c(double complex *out, double complex *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double complex cval, double truncate, unsigned threads,
    cconvolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma || !order)
    {ERROR("gauss_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_filter: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_filter: threads must be positive."); return -1;}

    int fail = 0;
    int axis = 0;
    while (sigma[axis] < 1e-15 && axis < ndim) axis++;
    if (axis < ndim)
    {
        size_t ksize = 2 * (size_t) (sigma[axis] * truncate) + 1;
        double complex *krn = (double complex *)calloc(ksize, sizeof(double complex));
        fail |= gauss_kernel1d((double *)krn, sigma[axis], order[axis], ksize, 2);
        fail |= fft_convolve(out, inp, ndim, dims, krn, ksize, axis, mode, cval, threads);
        free(krn);

        for (int n = axis + 1; n < ndim; n++)
        {
            if (sigma[n] > 1e-15)
            {
                ksize = 2 * (size_t) (sigma[n] * truncate) + 1;
                krn = (double complex *)calloc(ksize, sizeof(double complex));
                fail |= gauss_kernel1d((double *)krn, sigma[n], order[n], ksize, 2);
                fail |= fft_convolve(out, out, ndim, dims, krn, ksize, n, mode, cval, threads);
                free(krn);
            }
        }
    }
    else
    {
        size_t size = 1;
        for (int n = 0; n < ndim; n++) size *= dims[n];
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] = inp[i];
    }
    return fail;
}

int gauss_grad_mag_r(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    rconvolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma)
    {ERROR("gauss_grad_mag: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_grad_mag: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_grad_mag: threads must be positive."); return -1;}

    int fail = 0;
    size_t size = 1;
    unsigned *order = (unsigned *)malloc(ndim * sizeof(unsigned));
    for (int n = 0; n < ndim; n++) size *= dims[n];

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = 0.0;

    double *tmp = (double *)malloc(size * sizeof(double));
    for (int m = 0; m < ndim; m++)
    {
        for (int n = 0; n < ndim; n++) order[n] = (n == m) ? 1 : 0;
        fail |= gauss_filter_r(tmp, inp, ndim, dims, sigma, order, mode,
            cval, truncate, threads, fft_convolve);
        
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] += tmp[i] * tmp[i];
    }

    free(tmp); free(order);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = sqrt(out[i]);
    return fail;
}

int gauss_grad_mag_c(double *out, double complex *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double complex cval, double truncate, unsigned threads,
    cconvolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma)
    {ERROR("gauss_grad_mag: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_grad_mag: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_grad_mag: threads must be positive."); return -1;}

    int fail = 0;
    size_t size = 1;
    unsigned *order = (unsigned *)malloc(ndim * sizeof(unsigned));
    for (int n = 0; n < ndim; n++) size *= dims[n];

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = 0.0;

    double complex *tmp = (double complex *)malloc(size * sizeof(double complex));
    for (int m = 0; m < ndim; m++)
    {
        for (int n = 0; n < ndim; n++) order[n] = (n == m) ? 1 : 0;
        fail |= gauss_filter_c(tmp, inp, ndim, dims, sigma, order, mode,
            cval, truncate, threads, fft_convolve);
        
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] += SQ(creal(tmp[i])) + SQ(cimag(tmp[i]));
    }

    free(tmp); free(order);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = sqrt(out[i]);
    return fail;
}