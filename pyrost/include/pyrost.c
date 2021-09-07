#include "include.h"
#include "array.h"
#include "fastsum.h"

#define FIND_MIN_MAX(_min, _max, _arr, _len)        \
{                                                   \
    int _n;                                         \
    _min = _max = (_arr)[0];                        \
    for (_n = 1; _n < (int)(_len); _n++)            \
    {                                               \
        if (_min > (_arr)[_n]) _min = (_arr)[_n];   \
        if (_max < (_arr)[_n]) _max = (_arr)[_n];   \
    }                                               \
}

int make_reference_nfft(double **I0, int *X0, int *Y0, double *dX0, double *dY0,
    double *I_n, double *W, double *u, size_t *dims, double *di, double *dj,
    double ls, unsigned int threads)
{
    /* check parameters */
    if (!I_n || !W || !u || !dims || !di || !dj)
    {ERROR("make_reference_nfft: one of the arguments is NULL."); return -1;}
    if (ls <= 0) {ERROR("make_reference_nfft: ls must be positive."); return -1;}
    if (threads == 0) {ERROR("make_reference_nfft: threads must be positive."); return -1;}

    omp_set_num_threads(threads);

    int n_frames = dims[0];
    int frame_size = dims[1] * dims[2];

    double uy_min, uy_max, ux_min, ux_max;
    FIND_MIN_MAX(uy_min, uy_max, u, frame_size);
    FIND_MIN_MAX(ux_min, ux_max, u + frame_size, frame_size);
    if (uy_max == uy_min && ux_max == ux_min) {ERROR("make_reference_nfft: pixel map is constant."); return -1;}

    double di_max, di_min, dj_max, dj_min;
    FIND_MIN_MAX(di_min, di_max, di, dims[0]);
    FIND_MIN_MAX(dj_min, dj_max, dj, dims[0]);
    *dY0 = di_max - uy_min;
    *dX0 = dj_max - ux_min;
    
    *Y0 = uy_max - di_min + *dY0 + 1;
    *X0 = ux_max - dj_min + *dX0 + 1;
    size_t N0[2] = {(*Y0), (*X0)};
    *I0 = (double *)malloc(N0[0] * N0[1] * sizeof(double));

    fastsum_plan fs_plan;
    double r_max = 0.25 - 1.0 / 32.0;
    double sigma = 2 * r_max * ls / N0[1];
    int i, j;
    if (N0[0] == 1)
    {
        /* n = p = m = 4 yields 10^-6 accuracy */
        fastsum_init_guru(&fs_plan, 1, n_frames * frame_size, N0[1], gaussian, &sigma, NEARFIELD_BOXES, 4, 4, 4);

        for (i = 0; i < n_frames; i++)
        {
            for (j = 0; j < frame_size; j++)
            {
                fs_plan.alpha[i * frame_size + j] = (W[j] > 0) ? I_n[i * frame_size + j] / W[j] : 1.0;
                fs_plan.x[i * frame_size + j] = 2.0 * r_max * (u[frame_size + j] - dj[i] + (*dX0)) / N0[1] - r_max;
            }
        }

        for (i = 0; i < fs_plan.M_total; i++)
            fs_plan.y[i] = 2.0 * r_max * (double)i / (double)fs_plan.M_total - r_max;
    }
    else
    {
        /* n = p = m = 4 yields 10^-6 accuracy */
        fastsum_init_guru(&fs_plan, 2, n_frames * frame_size, N0[0] * N0[1], gaussian, &sigma, NEARFIELD_BOXES, 4, 4, 4);

        for (i = 0; i < n_frames; i++)
        {
            for (j = 0; j < frame_size; j++)
            {
                fs_plan.alpha[i * frame_size + j] = (W[j] > 0) ? I_n[i * frame_size + j] / W[j] : 1.0;
                fs_plan.x[2 * i * frame_size + 2 * j] = 2.0 * r_max * (u[j] - di[i] + (*dY0)) / N0[0] - r_max;
                fs_plan.x[2 * i * frame_size + 2 * j + 1] = 2.0 * r_max * (u[frame_size + j] - dj[i] + (*dX0)) / N0[1] - r_max;
            }
        }

        for (i = 0; i < (int)N0[0]; i++)
            for (j = 0; j < (int)N0[1]; j++)
            {
                fs_plan.y[2 * i * N0[1] + 2 * j] = 2.0 * r_max * (double)i / (double)N0[0] - r_max;
                fs_plan.y[2 * i * N0[1] + 2 * j + 1] = 2.0 * r_max * (double)j / (double)N0[1] - r_max;
            }
    }

    fastsum_precompute(&fs_plan);
    fastsum_trafo(&fs_plan);

    for (i = 0; i < fs_plan.M_total; i++) (*I0)[i] = creal(fs_plan.f[i]);

    fastsum_finalize(&fs_plan);

    return 0;
}