#include "st_utils.h"
#include "st_waveprop_fftw.h"

#define LPRE_SIZE 585
static const size_t LPRE[LPRE_SIZE] = {18,    20,    21,    22,    24,    25,    26,    27,    28,    30,
                                       32,    33,    35,    36,    39,    40,    42,    44,    45,    48,
                                       49,    50,    52,    54,    55,    56,    60,    63,    64,
                                       65,    66,    70,    72,    75,    77,    78,    80,    81,
                                       84,    88,    90,    91,    96,    98,    99,    100,   104,
                                       105,   108,   110,   112,   117,   120,   125,   126,   128,
                                       130,   132,   135,   140,   144,   147,   150,   154,   156,
                                       160,   162,   165,   168,   175,   176,   180,   182,   189,
                                       192,   195,   196,   198,   200,   208,   210,   216,   220,
                                       224,   225,   231,   234,   240,   243,   245,   250,   252,
                                       256,   260,   264,   270,   273,   275,   280,   288,   294,
                                       297,   300,   308,   312,   315,   320,   324,   325,   330,
                                       336,   343,   350,   351,   352,   360,   364,   375,   378,
                                       384,   385,   390,   392,   396,   400,   405,   416,   420,
                                       432,   440,   441,   448,   450,   455,   462,   468,   480,
                                       486,   490,   495,   500,   504,   512,   520,   525,   528,
                                       539,   540,   546,   550,   560,   567,   576,   585,   588,
                                       594,   600,   616,   624,   625,   630,   637,   640,   648,
                                       650,   660,   672,   675,   686,   693,   700,   702,   704,
                                       720,   728,   729,   735,   750,   756,   768,   770,   780,
                                       784,   792,   800,   810,   819,   825,   832,   840,   864,
                                       875,   880,   882,   891,   896,   900,   910,   924,   936,
                                       945,   960,   972,   975,   980,   990,   1000,  1008,  1024,
                                       1029,  1040,  1050,  1053,  1056,  1078,  1080,  1092,  1100,
                                       1120,  1125,  1134,  1152,  1155,  1170,  1176,  1188,  1200,
                                       1215,  1225,  1232,  1248,  1250,  1260,  1274,  1280,  1296,
                                       1300,  1320,  1323,  1344,  1350,  1365,  1372,  1375,  1386,
                                       1400,  1404,  1408,  1440,  1456,  1458,  1470,  1485,  1500,
                                       1512,  1536,  1540,  1560,  1568,  1575,  1584,  1600,  1617,
                                       1620,  1625,  1638,  1650,  1664,  1680,  1701,  1715,  1728,
                                       1750,  1755,  1760,  1764,  1782,  1792,  1800,  1820,  1848,
                                       1872,  1875,  1890,  1911,  1920,  1925,  1944,  1950,  1960,
                                       1980,  2000,  2016,  2025,  2048,  2058,  2079,  2080,  2100,
                                       2106,  2112,  2156,  2160,  2184,  2187,  2200,  2205,  2240,
                                       2250,  2268,  2275,  2304,  2310,  2340,  2352,  2376,  2400,
                                       2401,  2430,  2450,  2457,  2464,  2475,  2496,  2500,  2520,
                                       2548,  2560,  2592,  2600,  2625,  2640,  2646,  2673,  2688,
                                       2695,  2700,  2730,  2744,  2750,  2772,  2800,  2808,  2816,
                                       2835,  2880,  2912,  2916,  2925,  2940,  2970,  3000,  3024,
                                       3072,  3080,  3087,  3120,  3125,  3136,  3150,  3159,  3168,
                                       3185,  3200,  3234,  3240,  3250,  3276,  3300,  3328,  3360,
                                       3375,  3402,  3430,  3456,  3465,  3500,  3510,  3520,  3528,
                                       3564,  3584,  3600,  3640,  3645,  3675,  3696,  3744,  3750,
                                       3773,  3780,  3822,  3840,  3850,  3888,  3900,  3920,  3960,
                                       3969,  4000,  4032,  4050,  4095,  4096,  4116,  4125,  4158,
                                       4160,  4200,  4212,  4224,  4312,  4320,  4368,  4374,  4375,
                                       4400,  4410,  4455,  4459,  4480,  4500,  4536,  4550,  4608,
                                       4620,  4680,  4704,  4725,  4752,  4800,  4802,  4851,  4860,
                                       4875,  4900,  4914,  4928,  4950,  4992,  5000,  5040,  5096,
                                       5103,  5120,  5145,  5184,  5200,  5250,  5265,  5280,  5292,
                                       5346,  5376,  5390,  5400,  5460,  5488,  5500,  5544,  5600,
                                       5616,  5625,  5632,  5670,  5733,  5760,  5775,  5824,  5832,
                                       5850,  5880,  5940,  6000,  6048,  6075,  6125,  6144,  6160,
                                       6174,  6237,  6240,  6250,  6272,  6300,  6318,  6336,  6370,
                                       6400,  6468,  6480,  6500,  6552,  6561,  6600,  6615,  6656,
                                       6720,  6750,  6804,  6825,  6860,  6875,  6912,  6930,  7000,
                                       7020,  7040,  7056,  7128,  7168,  7200,  7203,  7280,  7290,
                                       7350,  7371,  7392,  7425,  7488,  7500,  7546,  7560,  7644,
                                       7680,  7700,  7776,  7800,  7840,  7875,  7920,  7938,  8000,
                                       8019,  8064,  8085,  8100,  8125,  8190,  8192,  8232,  8250,
                                       8316,  8320,  8400,  8424,  8448,  8505,  8575,  8624,  8640,
                                       8736,  8748,  8750,  8775,  8800,  8820,  8910,  8918,  8960,
                                       9000,  9072,  9100,  9216,  9240,  9261,  9360,  9375,  9408,
                                       9450,  9477,  9504,  9555,  9600,  9604,  9625,  9702,  9720,
                                       9750,  9800,  9828,  9856,  9900,  9984,  10000};

static NOINLINE int compare_size_t(const void *a, const void *b)
{
    return (*(size_t *)a - *(size_t *)b);
}

static NOINLINE size_t search_lpre(size_t key)
{
    return searchsorted(&key, LPRE, LPRE_SIZE, sizeof(size_t), compare_size_t);
}

static size_t find_match(size_t target, size_t p7_11_13)
{
    size_t p5_7_11_13, p3_5_7_11_13;
    size_t match = 2 * target;
    while (p7_11_13 < target)
    {
        p5_7_11_13 = p7_11_13;
        while (p5_7_11_13 < target)
        {
            p3_5_7_11_13 = p5_7_11_13;
            while (p3_5_7_11_13 < target)
            {
                while (p3_5_7_11_13 < target) p3_5_7_11_13 *= 2;

                if (p3_5_7_11_13 == target) return p3_5_7_11_13;
                if (p3_5_7_11_13 < match) match = p3_5_7_11_13;

                while (!(p3_5_7_11_13 & 1)) p3_5_7_11_13 >>= 1;

                p3_5_7_11_13 *= 3;
                if (p3_5_7_11_13 == target) return p3_5_7_11_13;
            }
            if (p3_5_7_11_13 < match) match = p3_5_7_11_13;

            p5_7_11_13 *= 5;
            if (p5_7_11_13 == target) return p5_7_11_13;
        }
        if (p5_7_11_13 < match) match = p5_7_11_13;

        p7_11_13 *= 7;
        if (p7_11_13 == target) return p7_11_13;
    }
    if (p7_11_13 < match) return p7_11_13;
    return match;
}

NOINLINE size_t next_fast_len_fftw(size_t target)
{
    if (target <= 16) return target;
    if (!(target & (target - 1))) return target;
    if (target <= LPRE[LPRE_SIZE - 1]) return LPRE[search_lpre(target)];
    size_t match, best_match = 2 * target;

    match = find_match(target, 1);
    if (match < best_match) best_match = match;
    match = find_match(target, 11);
    if (match < best_match) best_match = match;
    match = find_match(target, 13);
    if (match < best_match) best_match = match;
    return best_match;
}

typedef void (*rsc_func)(fftw_plan, fftw_plan, double complex *, double complex *,
    double complex *, int, int, size_t, double, double, double, double, unsigned);

static void rsc_type1_fftw(fftw_plan fftp, fftw_plan ifftp, double complex *out,
    double complex *u, double complex *k, int flen, int npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for (int i = 0; i < flen; i++)
        {
            double dist = pow((dx0 * (i - flen / 2)), 2) + pow(z, 2);
            double ph = 2 * M_PI / wl * sqrt(dist);
            k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + cos(ph) * I) / pow(dist, 0.75);
        }
        #pragma omp master
        {
            fftw_execute_dft(fftp, (fftw_complex *)u, (fftw_complex *)u);
            fftw_execute_dft(fftp, (fftw_complex *)k, (fftw_complex *)k);
        }
        #pragma omp barrier
        #pragma omp for    
        for (int i = 0; i < flen; i++)
        {
            double ph = M_PI * pow((((double) i / flen) - ((2 * i) / flen)), 2) * dx / dx0 * flen;
            u[i] *= k[i] * (cos(ph) - sin(ph) * I) / flen;
            k[i] = (cos(ph) + sin(ph) * I) / flen;
        }
        #pragma omp master
        {
            fftw_execute_dft(ifftp, (fftw_complex *)u, (fftw_complex *)u);
            fftw_execute_dft(ifftp, (fftw_complex *)k, (fftw_complex *)k);
        }
        #pragma omp barrier
        #pragma omp for
        for (int i = 0; i < flen; i++) u[i] *= k[i];
        #pragma omp master
        {fftw_execute_dft(fftp, (fftw_complex *)u, (fftw_complex *)u);}
        #pragma omp barrier
        #pragma omp for
        for (int i = 0; i < npts / 2; i++) 
        {
            double ph = M_PI * pow((double) (i - npts / 2) / flen, 2) * dx / dx0 * flen;
            out[i * istride] = u[i + flen - npts / 2] * (cos(ph) - sin(ph) * I);
        }
        #pragma omp for
        for (int i = 0; i < npts / 2 + npts % 2; i++)
        {
            double ph = M_PI * pow((double) i / flen, 2) * dx / dx0 * flen;
            out[(i + npts / 2) * istride] = u[i] * (cos(ph) - sin(ph) * I);
        }
    }
}

static void rsc_type2_fftw(fftw_plan fftp, fftw_plan ifftp, double complex *out,
    double complex *u, double complex *k, int flen, int npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for (int i = 0; i < flen; i++)
        {
            double ph = M_PI * pow(i - flen / 2, 2) * dx0 / dx / flen;
            k[i] = cos(ph) - sin(ph) * I;
            u[i] *= cos(ph) + sin(ph) * I;
        }
        #pragma omp master
        {
            fftw_execute_dft(fftp, (fftw_complex *)u, (fftw_complex *)u);
            fftw_execute_dft(fftp, (fftw_complex *)k, (fftw_complex *)k);
        }
        #pragma omp barrier
        #pragma omp for    
        for (int i = 0; i < flen; i++)
        {
            u[i] *= k[i] / flen;
            double dist = pow((dx * (i - flen / 2)), 2) + pow(z, 2);
            double ph = 2 * M_PI / wl * sqrt(dist);
            k[i] = -dx0 * z / sqrt(wl) * (sin(ph) + cos(ph) * I) / pow(dist, 0.75);
        }
        #pragma omp master
        {
            fftw_execute_dft(fftp, (fftw_complex *)k, (fftw_complex *)k);
            fftw_execute_dft(ifftp, (fftw_complex *)u, (fftw_complex *)u);
        }
        #pragma omp barrier
        #pragma omp for
        for (int i = 0; i < flen; i++)
        {
            double ph = M_PI * pow((((double) i / flen) - ((2 * i) / flen)), 2) * dx0 / dx * flen;
            u[i] *= k[i] * (cos(ph) + sin(ph) * I) / flen;
        }
        #pragma omp master
        {fftw_execute_dft(ifftp, (fftw_complex *)u, (fftw_complex *)u);}
        #pragma omp barrier
        #pragma omp for
        for (int i = 0; i < npts; i++) out[i * istride] = u[i + (flen - npts) / 2];
    }
}

void rsc_fftw(double complex *out, const double complex *inp, size_t isize, size_t npts, size_t istride,
    double dx0, double dx, double z, double wl, unsigned threads)
{
    dx = fabs(dx); dx0 = fabs(dx0);
    double alpha = (dx0 <= dx) ? (dx0 / dx) : (dx / dx0);
    size_t flen = next_fast_len_fftw((size_t) (npts * (1 + alpha)) + 1);
    int repeats = isize / npts;
    double complex *u = (double complex *)fftw_malloc(flen * sizeof(double complex));
    double complex *k = (double complex *)fftw_malloc(flen * sizeof(double complex));
    fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    dim->n = flen; dim->is = 1; dim->os = 1;
    fftw_plan_with_nthreads(threads);
    fftw_plan fftp = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)u, (fftw_complex *)u, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan ifftp = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)u, (fftw_complex *)u, FFTW_BACKWARD, FFTW_ESTIMATE);
    rsc_func rsc_calc = (dx0 >= dx) ? rsc_type1_fftw : rsc_type2_fftw;

    for (int i = 0; i < repeats; i++)
    {
        extend_line_complex(u, inp + npts * istride * (i / istride) + (i % istride), EXTEND_CONSTANT, 0., flen, npts, istride);
        rsc_calc(fftp, ifftp, out + npts * istride * (i / istride) + (i % istride), u, k, flen, npts, istride, dx0, dx, z, wl, threads);
    }

    fftw_destroy_plan(fftp);
    fftw_destroy_plan(ifftp);
    fftw_free(u); fftw_free(k);
    free(dim);
}

static void fhf_fftw(fftw_plan fftp, fftw_plan ifftp, double complex *out,
    double complex *u, double complex *k, int flen, int npts, size_t istride,
    double dx0, double dx, double alpha, unsigned threads)
{
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for (int i = 0; i < flen; i++)
        {
            double ph = M_PI * pow(i - flen / 2, 2) * alpha;
            u[i] *= cos(ph) + sin(ph) * I;
        }
        #pragma omp master
        {fftw_execute(fftp);}
        #pragma omp barrier
        #pragma omp for
        for (int i = 0; i < flen; i++) u[i] *= k[i] / flen;
        #pragma omp master
        {fftw_execute(ifftp);}
        #pragma omp barrier
        #pragma omp for
        for (int n = 0; n < npts / 2; n++)
        {
            double ph = M_PI * pow(n - npts / 2, 2) * alpha;
            double complex w = (cos(ph) - sin(ph) * I) * u[n + flen - npts / 2];
            out[n * istride] = (cos(ph / dx0 * dx) - sin(ph / dx0 * dx) * I) * w;
        }
        #pragma omp for
        for (int n = 0; n < npts / 2 + npts % 2; n++)
        {
            double ph = M_PI * pow(n, 2) * alpha;
            double complex w = (cos(ph) - sin(ph) * I) * u[n];
            out[(n + npts / 2) * istride] = (cos(ph / dx0 * dx) - sin(ph / dx0 * dx) * I) * w;
        }
    }
}

void fraunhofer_fftw(double complex *out, const double complex *inp, size_t isize, size_t npts,
    size_t istride, double dx0, double dx, double z, double wl, unsigned threads)
{
    dx = fabs(dx); dx0 = fabs(dx0);
    int flen = next_fast_len_fftw(2 * npts - 1);
    int repeats = isize / npts;
    double complex *u = (double complex *)fftw_malloc(flen * sizeof(double complex));
    double complex *k = (double complex *)fftw_malloc(flen * sizeof(double complex));
    fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    dim->n = flen; dim->is = 1; dim->os = 1;
    fftw_plan_with_nthreads(threads);
    fftw_plan fftp = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)u, (fftw_complex *)u, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan ifftp = fftw_plan_guru_dft(1, dim, 0, NULL, (fftw_complex *)u, (fftw_complex *)u, FFTW_BACKWARD, FFTW_ESTIMATE);

    double ph = 2 * M_PI / wl * z;
    double alpha = dx0 * dx / wl / z;
    double complex k0 = -(sin(ph) + cos(ph) * I) / sqrt(wl * z) * dx0;
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < flen; i++)
    {
        double ph = M_PI * pow(i - flen / 2, 2) * alpha;
        k[i] = k0 * (cos(ph) - sin(ph) * I);
    }
    fftw_execute_dft(fftp, (fftw_complex *)k, (fftw_complex *)k);

    for (int i = 0; i < repeats; i++)
    {
        extend_line_complex(u, inp + npts * istride * (i / istride) + (i % istride), EXTEND_CONSTANT, 0., flen, npts, istride);
        fhf_fftw(fftp, ifftp, out + npts * istride * (i / istride) + (i % istride), u, k, flen, npts, istride, dx0, dx, alpha, threads);
    }

    fftw_destroy_plan(fftp);
    fftw_destroy_plan(ifftp);
    fftw_free(u); fftw_free(k);
    free(dim);
}

void fft_convolve_fftw(double *out, const double *inp, const double *krn, size_t isize,
    size_t npts, size_t istride, size_t ksize, EXTEND_MODE mode, double cval, unsigned threads)
{
    int flen = next_fast_len_fftw(npts + ksize - 1);
    int repeats = isize / npts;
    double *inpft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
    double *krnft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
    fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    dim->n = flen; dim->is = 1; dim->os = 1;
    fftw_plan_with_nthreads(threads);
    fftw_plan rfftp = fftw_plan_guru_dft_r2c(1, dim, 0, NULL, inpft, (fftw_complex *)inpft, FFTW_ESTIMATE);
    fftw_plan irfftp = fftw_plan_guru_dft_c2r(1, dim, 0, NULL, (fftw_complex *)inpft, inpft, FFTW_ESTIMATE);

    extend_line_double(krnft, krn, EXTEND_CONSTANT, 0., flen, ksize, 1);
    fftw_execute_dft_r2c(rfftp, (double *)krnft, (fftw_complex *)krnft);

    for (int i = 0; i < repeats; i++)
    {
        extend_line_double(inpft, inp + npts * istride * (i / istride) + (i % istride), mode, cval, flen, npts, istride);
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp master
            {fftw_execute(rfftp);}
            #pragma omp barrier
            #pragma omp for
            for (int j = 0; j < (flen / 2 + 1); j++)
            {
                double re = (inpft[2 * j] * krnft[2 * j] - inpft[2 * j + 1] * krnft[2 * j + 1]);
                double im = (inpft[2 * j] * krnft[2 * j + 1] + inpft[2 * j + 1] * krnft[2 * j]);
                inpft[2 * j] = re; inpft[2 * j + 1] = im;
            }
            #pragma omp master
            {fftw_execute(irfftp);}
            #pragma omp barrier
            #pragma omp for
            for (int n = 0; n < (int) npts / 2; n++) out[n * istride + npts * istride * (i / istride) + (i % istride)] = inpft[n + flen - npts / 2] / flen;
            #pragma omp for
            for (int n = 0; n < (int) (npts / 2 + npts % 2); n++) out[(n + npts / 2) * istride + npts * istride * (i / istride) + (i % istride)] = inpft[n] / flen;
        }
    }

    fftw_destroy_plan(rfftp);
    fftw_destroy_plan(irfftp);
    fftw_free(inpft); fftw_free(krnft);
    free(dim);
}