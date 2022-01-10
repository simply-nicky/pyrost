#include <stdio.h>
#include <stdlib.h>
#include "median.h"
#include "fastsum.h"

static int test_median();
static int test_fastsum();

int main(int argc, char *argv[])
{
    return test_fastsum();
}

static int test_median()
{
    int X = 16;
    int Y = 16;
    double *data = (double *)malloc(X * Y * sizeof(double));
    unsigned char *mask = (unsigned char *)malloc(X * Y);
    double *out = (double *)malloc(X * Y * sizeof(double));

    if (!data || !out)
    {
        printf("not enough memory\n");
        free(data); free(mask); free(out);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < X * Y; i++)
    {data[i] = (double)i; mask[i] = 1;}

    size_t dims[2] = {X, Y};
    size_t fsize[2] = {3, 3};
    double cval = 0.0;

    median_filter(out, data, mask, 2, dims, sizeof(double), fsize,
        EXTEND_MIRROR, &cval, compare_double, 1);

    printf("Result:\n");
    for (int i = 0; i < X; i++)
    {
        for (int j = 0; j < Y; j++) printf("%.0f ", out[i]);
        printf("\n");
    }
    printf("\n");

    free(data); free(mask); free(out);

    return EXIT_SUCCESS;
}

static int test_fastsum()
{
    int N_total = 20;
    int M_total = 30;
    int d = 2, i, n;
    int num_threads = 1;
    fastsum_plan my_plan;

    kernel k = gaussian;
    double param[1] = {0.2}; /* sigma */

    omp_set_num_threads(num_threads);
    fftw_init_threads();
    fastsum_init_guru(&my_plan, d, N_total, M_total, k, param, 0, 2, 2, 2);

    double r_max = 0.25 - 1.0 / 16.0;
    for (i = 0; i < my_plan.N_total; i++)
    {
        my_plan.alpha[i] = (double)rand() / (double)RAND_MAX;
        for (n = 0; n < my_plan.d; n++)
            my_plan.x[my_plan.d * i + n] = 2.0 * r_max * (double)rand() / (double)RAND_MAX - r_max;
    }
    for (i = 0; i < my_plan.M_total; i++)
    {
        for (n = 0; n < my_plan.d; n++)
            my_plan.y[my_plan.d * i + n] = 2.0 * r_max * (double)rand() / (double)RAND_MAX - r_max;
    }

    fastsum_exact(&my_plan);
    for (i = 0; i < my_plan.N_total; i++)
    {
        printf("%.3f ", my_plan.f[i]);
    }
    printf("\n");

    fastsum_precompute(&my_plan);
    fastsum_trafo(&my_plan);

    for (i = 0; i < my_plan.N_total; i++)
    {
        printf("%.3f ", my_plan.f[i]);
    }
    printf("\n");

    fastsum_finalize(&my_plan);

    return EXIT_SUCCESS;
}