#ifndef PYROST_H
#define PYROST_H

int make_reference_nfft(double **I0, int *X0, int *Y0, double *dX0, double *dY0,
    double *I_n, double *W, double *u, size_t *dims, double *di, double *dj,
    double ls, unsigned int threads);

#endif