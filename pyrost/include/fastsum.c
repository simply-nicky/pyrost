/*
 * Copyright (c) 2002, 2017 Jens Keiner, Stefan Kunis, Daniel Potts
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOdouble A PARTICULAdouble PURPOSE.    See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/*! \file fastsum.c
 *  \brief Fast NFFT-based summation algorithm.
 *
 *  \author Markus Fenn
 *  \date 2003-2006
 */
#include "fastsum.h"

double complex gaussian(double x, int der, const double *param)    /* K(x)=exp(-x^2/c^2) */
{
  double c = param[0];
  double value = 0.0;

  switch (der)
  {
    case  0 : value = exp(-x*x/(c*c)); break;
    case  1 : value = -2.0*x/(c*c)*exp(-x*x/(c*c)); break;
    case  2 : value = 2.0*exp(-x*x/(c*c))*(-c*c+2.0*x*x)/(c*c*c*c); break;
    case  3 : value = -4.0*x*exp(-x*x/(c*c))*(-3.0*c*c+2.0*x*x)/(c*c*c*c*c*c); break;
    case  4 : value = 4.0*exp(-x*x/(c*c))*(3.0*c*c*c*c-12.0*c*c*x*x+4.0*x*x*x*x)/(c*c*c*c*c*c*c*c); break;
    case  5 : value = -8.0*x*exp(-x*x/(c*c))*(15.0*c*c*c*c-20.0*c*c*x*x+4.0*x*x*x*x)/pow(c,10.0); break;
    case  6 : value = 8.0*exp(-x*x/(c*c))*(-15.0*c*c*c*c*c*c+90.0*x*x*c*c*c*c-60.0*x*x*x*x*c*c+8.0*x*x*x*x*x*x)/pow(c,12.0); break;
    case  7 : value = -16.0*x*exp(-x*x/(c*c))*(-105.0*c*c*c*c*c*c+210.0*x*x*c*c*c*c-84.0*x*x*x*x*c*c+8.0*x*x*x*x*x*x)/pow(c,14.0); break;
    case  8 : value = 16.0*exp(-x*x/(c*c))*(105.0*c*c*c*c*c*c*c*c-840.0*x*x*c*c*c*c*c*c+840.0*x*x*x*x*c*c*c*c-224.0*x*x*x*x*x*x*c*c+16.0*x*x*x*x*x*x*x*x)/pow(c,16.0); break;
    case  9 : value = -32.0*x*exp(-x*x/(c*c))*(945.0*c*c*c*c*c*c*c*c-2520.0*x*x*c*c*c*c*c*c+1512.0*x*x*x*x*c*c*c*c-288.0*x*x*x*x*x*x*c*c+16.0*x*x*x*x*x*x*x*x)/pow(c,18.0); break;
    case 10 : value = 32.0*exp(-x*x/(c*c))*(-945.0*pow(c,10.0)+9450.0*x*x*c*c*c*c*c*c*c*c-12600.0*x*x*x*x*c*c*c*c*c*c+5040.0*x*x*x*x*x*x*c*c*c*c-720.0*x*x*x*x*x*x*x*x*c*c+32.0*pow(x,10.0))/pow(c,20.0); break;
    case 11 : value = -64.0*x*exp(-x*x/(c*c))*(-10395.0*pow(c,10.0)+34650.0*x*x*c*c*c*c*c*c*c*c-27720.0*x*x*x*x*c*c*c*c*c*c+7920.0*x*x*x*x*x*x*c*c*c*c-880.0*x*x*x*x*x*x*x*x*c*c+32.0*pow(x,10.0))/pow(c,22.0); break;
    case 12 : value = 64.0*exp(-x*x/(c*c))*(10395.0*pow(c,12.0)-124740.0*x*x*pow(c,10.0)+207900.0*x*x*x*x*c*c*c*c*c*c*c*c-110880.0*x*x*x*x*x*x*c*c*c*c*c*c+23760.0*x*x*x*x*x*x*x*x*c*c*c*c-2112.0*pow(x,10.0)*c*c+64.0*pow(x,12.0))/pow(c,24.0); break;
    default : value = 0.0;
  }

  return value;
}

/** max */
static int max_i(int a, int b)
{
    return a >= b ? a : b;
}

/** factorial */
static double fak(int n)
{
    if (n <= 1)
        return 1.0;
    else
        return (double)(n) * fak(n - 1);
}

/** binomial coefficient */
static double binom(int n, int m)
{
    return fak(n) / fak(m) / fak(n - m);
}

/** basis polynomial for regularized kernel */
static double BasisPoly(int m, int r, double xx)
{
    int k;
    double sum = 0.0;

    for (k = 0; k <= m - r; k++)
    {
        sum += binom(m + k, k) * pow((xx + 1.0) / 2.0, (double) k);
    }
    return sum * pow((xx + 1.0), (double) r) * pow(1.0 - xx, (double) (m + 1))
        / (double)(1 << (m + 1)) / fak(r); /* 1<<(m+1) = 2^(m+1) */
}

/** regularized kernel with K_I arbitrary and K_B smooth to zero */
double complex regkern(kernel k, double xx, int p, const double *param, double a, double b)
{
    int r;
    double complex sum = 0.0;

    if (xx < -0.5)
        xx = -0.5;
    if (xx > 0.5)
        xx = 0.5;
    if ((xx >= -0.5 + b && xx <= -a) || (xx >= a && xx <= 0.5 - b))
    {
        return k(xx, 0, param);
    }
    else if (xx < -0.5 + b)
    {
        sum = (k(-0.5, 0, param) + k(0.5, 0, param)) / 2.0
            * BasisPoly(p - 1, 0, 2.0 * xx / b + (1.0 - b) / b);
        for (r = 0; r < p; r++)
        {
            sum += pow(-b / 2.0, (double) r) * k(-0.5 + b, r, param)
                * BasisPoly(p - 1, r, -2.0 * xx / b + (b - 1.0) / b);
        }
        return sum;
    }
    else if ((xx > -a) && (xx < a))
    {
        for (r = 0; r < p; r++)
        {
            sum += pow(a, (double) r)
                * (k(-a, r, param) * BasisPoly(p - 1, r, xx / a)
                + k(a, r, param) * BasisPoly(p - 1, r, -xx / a)
                * (r & 1 ? -1 : 1));
        }
        return sum;
    }
    else if (xx > 0.5 - b)
    {
        sum = (k(-0.5, 0, param) + k(0.5, 0, param)) / 2.0
            * BasisPoly(p - 1, 0, -2.0 * xx / b + (1.0 - b) / b);
        for (r = 0; r < p; r++)
        {
            sum += pow(b / 2.0, (double) r) * k(0.5 - b, r, param)
                * BasisPoly(p - 1, r, 2.0 * xx / b - (1.0 - b) / b);
        }
        return sum;
    }
    return k(xx, 0, param);
}

/** regularized kernel with K_I arbitrary and K_B periodized
 *  (used in 1D)
 */
static double complex regkern1(kernel k, double xx, int p, const double *param, double a, double b)
{
    int r;
    double complex sum = 0.0;

    if (xx < -0.5) xx = -0.5;
    if (xx > 0.5) xx = 0.5;
    if ((xx >= -0.5 + b && xx <= -a) || (xx >= a && xx <= 0.5 - b))
    {
        return k(xx, 0, param);
    }
    else if ((xx > -a) && (xx < a))
    {
        for (r = 0; r < p; r++)
        {
            sum += pow(a, (double) r)
                * (k(-a, r, param) * BasisPoly(p - 1, r, xx / a)
                + k(a, r, param) * BasisPoly(p - 1, r, -xx / a)
                * (r & 1 ? -1 : 1));
        }
        return sum;
    }
    else if (xx < -0.5 + b)
    {
        for (r = 0; r < p; r++)
        {
            sum += pow(b, (double) r)
                * (k(0.5 - b, r, param) * BasisPoly(p - 1, r, (xx + 0.5) / b)
                + k(-0.5 + b, r, param) * BasisPoly(p - 1, r, -(xx + 0.5) / b)
                * (r & 1 ? -1 : 1));
        }
        return sum;
    }
    else if (xx > 0.5 - b)
    {
        for (r = 0; r < p; r++)
        {
            sum += pow(b, (double) r)
                * (k(0.5 - b, r, param) * BasisPoly(p - 1, r, (xx - 0.5) / b)
                + k(-0.5 + b, r, param) * BasisPoly(p - 1, r, -(xx - 0.5) / b)
                * (r & 1 ? -1 : 1));
        }
        return sum;
    }
    return k(xx, 0, param);
}

/** regularized kernel for even kernels with K_I even
 *    and K_B mirrored smooth to 1/2 (used in dD, d>1)
 */
static double complex regkern3(kernel k, double xx, int p, const double *param, double a, double b)
{
    int r;
    double complex sum = 0.0;

    xx = fabs(xx);

    if (xx >= 0.5)
    {
        /*return kern(typ,c,0,0.5);*/
        xx = 0.5;
    }
    /* else */
    if ((a <= xx) && (xx <= 0.5 - b))
    {
        return k(xx, 0, param);
    }
    else if (xx < a)
    {
        for (r = 0; r < p; r++)
        {
            sum += pow(-a, (double) r) * k(a, r, param)
                * (BasisPoly(p - 1, r, xx / a) + BasisPoly(p - 1, r, -xx / a));
        }
        /*sum=kern(typ,c,0,xx); */
        return sum;
    }
    else if ((0.5 - b < xx) && (xx <= 0.5))
    {
        sum = k(0.5, 0, param) * BasisPoly(p - 1, 0, -2.0 * xx / b + (1.0 - b) / b);
        /* sum=regkern2(typ,c,p,a,b, 0.5)*BasisPoly(p-1,0,-2.0*xx/b+(1.0-b)/b); */
        for (r = 0; r < p; r++)
        {
            sum += pow(b / 2.0, (double) r) * k(0.5 - b, r, param)
                * BasisPoly(p - 1, r, 2.0 * xx / b - (1.0 - b) / b);
        }
        return sum;
    }
    return 0.0;
}

/** cubic spline interpolation in near field with even kernels */
double complex kubintkern(const double x, const double complex *Add, const int Ad, const double a)
{
    double c, c1, c2, c3, c4;
    int r;
    double complex f0, f1, f2, f3;
    c = x * (double)(Ad) / a;
    r = (int)(lrint(c));
    r = abs(r);
    if (r == 0)
    {
        f0 = Add[r + 1];
        f1 = Add[r];
        f2 = Add[r + 1];
        f3 = Add[r + 2];
    }
    else
    {
        f0 = Add[r - 1];
        f1 = Add[r];
        f2 = Add[r + 1];
        f3 = Add[r + 2];
    }
    c = fabs(c);
    c1 = c - (double)(r);
    c2 = c1 + 1.0;
    c3 = c1 - 1.0;
    c4 = c1 - 2.0;

    return (-f0 * c1 * c3 * c4 / 6.0 + f1 * c2 * c3 * c4 / 2.0
        - f2 * c2 * c1 * c4 / 2.0 + f3 * c2 * c1 * c3 / 6.0);
}

/** cubic spline interpolation in near field with arbitrary kernels */
static double complex kubintkern1(const double x, const double complex *Add, const int Ad, const double a)
{
    double c, c1, c2, c3, c4;
    int r;
    double complex f0, f1, f2, f3;
    Add += 2;
    c = (x + a) * (double)(Ad) / 2.0 / a;
    r = (int)(lrint(c));
    r = abs(r);
    /*if (r==0) {f0=Add[r];f1=Add[r];f2=Add[r+1];f3=Add[r+2];}
     else */
    {
        f0 = Add[r - 1];
        f1 = Add[r];
        f2 = Add[r + 1];
        f3 = Add[r + 2];
    }
    c = fabs(c);
    c1 = c - (double)(r);
    c2 = c1 + 1.0;
    c3 = c1 - 1.0;
    c4 = c1 - 2.0;

    return (-f0 * c1 * c3 * c4 / 6.0 + f1 * c2 * c3 * c4 / 2.0
        - f2 * c2 * c1 * c4 / 2.0 + f3 * c2 * c1 * c3 / 6.0);
}

/** quicksort algorithm for source knots and associated coefficients */
static void quicksort(int d, int t, double *x, double complex *alpha, int *permutation_x_alpha, int N)
{
    int lpos = 0;
    int rpos = N - 1;
    /*double pivot=x[((N-1)/2)*d+t];*/
    double pivot = x[(N / 2) * d + t];

    int k;
    double temp1;
    double complex temp2;
    int temp_int;

    while (lpos <= rpos)
    {
        while (x[lpos * d + t] < pivot)
            lpos++;
        while (x[rpos * d + t] > pivot)
            rpos--;
        if (lpos <= rpos)
        {
            for (k = 0; k < d; k++)
            {
                temp1 = x[lpos * d + k];
                x[lpos * d + k] = x[rpos * d + k];
                x[rpos * d + k] = temp1;
            }
            temp2 = alpha[lpos];
            alpha[lpos] = alpha[rpos];
            alpha[rpos] = temp2;
            
            if (permutation_x_alpha)     /** store the permutation of x */
            {
                temp_int = permutation_x_alpha[lpos];
                permutation_x_alpha[lpos] = permutation_x_alpha[rpos];
                permutation_x_alpha[rpos] = temp_int;            
            }
            
            lpos++;
            rpos--;
        }
    }
    if (0 < rpos)
        quicksort(d, t, x, alpha, permutation_x_alpha, rpos + 1);
    if (lpos < N - 1)
        quicksort(d, t, x + lpos * d, alpha + lpos, permutation_x_alpha ? permutation_x_alpha + lpos : NULL, N - lpos);
}

/** initialize box-based search data structures */
static void BuildBox(fastsum_plan *ths)
{
    int t, l;
    int *box_index;
    double val[ths->d];

    box_index = (int *) nfft_malloc((size_t)(ths->box_count) * sizeof(int));
    for (t = 0; t < ths->box_count; t++)
        box_index[t] = 0;

    for (l = 0; l < ths->N_total; l++)
    {
        int ind = 0;
        for (t = 0; t < ths->d; t++)
        {
            val[t] = ths->x[ths->d * l + t] + 0.25 - ths->eps_B / 2.0;
            ind *= ths->box_count_per_dim;
            ind += (int) (val[t] / ths->eps_I);
        }
        box_index[ind]++;
    }

    ths->box_offset[0] = 0;
    for (t = 1; t <= ths->box_count; t++)
    {
        ths->box_offset[t] = ths->box_offset[t - 1] + box_index[t - 1];
        box_index[t - 1] = ths->box_offset[t - 1];
    }

    for (l = 0; l < ths->N_total; l++)
    {
        int ind = 0;
        for (t = 0; t < ths->d; t++)
        {
            val[t] = ths->x[ths->d * l + t] + 0.25 - ths->eps_B / 2.0;
            ind *= ths->box_count_per_dim;
            ind += (int) (val[t] / ths->eps_I);
        }

        ths->box_alpha[box_index[ind]] = ths->alpha[l];

        for (t = 0; t < ths->d; t++)
        {
            ths->box_x[ths->d * box_index[ind] + t] = ths->x[ths->d * l + t];
        }
        box_index[ind]++;
    }
    nfft_free(box_index);
}

/** inner computation function for box-based near field correction */
static inline double complex calc_SearchBox(int d, double *y, double *x, double complex *alpha, int start,
        int end_lt, const double complex *Add, const int Ad, int p, double a, const kernel k,
        const double *param, const unsigned flags)
{
    double complex result = 0.0;

    int m, l;
    double r;

    for (m = start; m < end_lt; m++)
    {
        if (d == 1)
        {
            r = y[0] - x[m];
        }
        else
        {
            r = 0.0;
            for (l = 0; l < d; l++)
                r += (y[l] - x[m * d + l]) * (y[l] - x[m * d + l]);
            r = sqrt(r);
        }
        if (fabs(r) < a)
        {
            result += alpha[m] * k(r, 0, param); /* alpha*(kern-regkern) */
            if (d == 1)
            {
                if (flags & EXACT_NEARFIELD)
                    result -= alpha[m] * regkern1(k, r, p, param, a, 1.0 / 16.0); /* exact value (in 1D)    */
                else
                    result -= alpha[m] * kubintkern1(r, Add, Ad, a); /* spline approximation */
            }
            else
            {
                if (flags & EXACT_NEARFIELD)
                    result -= alpha[m] * regkern(k, r, p, param, a, 1.0 / 16.0); /* exact value (in dD)    */
                else
                    result -= alpha[m] * kubintkern(r, Add, Ad, a); /* spline approximation */
            }
        }
    }
    return result;
}

/** box-based near field correction */
static double complex SearchBox(double *y, fastsum_plan *ths)
{
    double complex val = 0.0;
    int t;
    int y_multiind[ths->d];
    int multiindex[ths->d];
    int y_ind;

    for (t = 0; t < ths->d; t++)
    {
        y_multiind[t] = (int)(lrint((y[t] + 0.25 - ths->eps_B / 2.0) / ths->eps_I));
    }

    if (ths->d == 1)
    {
        for (y_ind = max_i(0, y_multiind[0] - 1);
                y_ind < ths->box_count_per_dim && y_ind <= y_multiind[0] + 1; y_ind++)
        {
            val += calc_SearchBox(ths->d, y, ths->box_x, ths->box_alpha,
                    ths->box_offset[y_ind], ths->box_offset[y_ind + 1], ths->Add, ths->Ad,
                    ths->p, ths->eps_I, ths->k, ths->kernel_param, ths->flags);
        }
    }
    else if (ths->d == 2)
    {
        for (multiindex[0] = max_i(0, y_multiind[0] - 1);
                multiindex[0] < ths->box_count_per_dim
                        && multiindex[0] <= y_multiind[0] + 1; multiindex[0]++)
            for (multiindex[1] = max_i(0, y_multiind[1] - 1);
                    multiindex[1] < ths->box_count_per_dim
                            && multiindex[1] <= y_multiind[1] + 1; multiindex[1]++)
            {
                y_ind = (ths->box_count_per_dim * multiindex[0]) + multiindex[1];
                val += calc_SearchBox(ths->d, y, ths->box_x, ths->box_alpha,
                        ths->box_offset[y_ind], ths->box_offset[y_ind + 1], ths->Add,
                        ths->Ad, ths->p, ths->eps_I, ths->k, ths->kernel_param, ths->flags);
            }
    }
    else if (ths->d == 3)
    {
        for (multiindex[0] = max_i(0, y_multiind[0] - 1);
                multiindex[0] < ths->box_count_per_dim
                        && multiindex[0] <= y_multiind[0] + 1; multiindex[0]++)
            for (multiindex[1] = max_i(0, y_multiind[1] - 1);
                    multiindex[1] < ths->box_count_per_dim
                            && multiindex[1] <= y_multiind[1] + 1; multiindex[1]++)
                for (multiindex[2] = max_i(0, y_multiind[2] - 1);
                        multiindex[2] < ths->box_count_per_dim
                                && multiindex[2] <= y_multiind[2] + 1; multiindex[2]++)
                {
                    y_ind = ((ths->box_count_per_dim * multiindex[0]) + multiindex[1])
                            * ths->box_count_per_dim + multiindex[2];
                    val += calc_SearchBox(ths->d, y, ths->box_x, ths->box_alpha,
                            ths->box_offset[y_ind], ths->box_offset[y_ind + 1], ths->Add,
                            ths->Ad, ths->p, ths->eps_I, ths->k, ths->kernel_param,
                            ths->flags);
                }
    }
    else
    {
        return 0.0/0.0; //exit(EXIT_FAILURE);
    }
    return val;
}

/** recursive sort of source knots dimension by dimension to get tree structure */
static void BuildTree(int d, int t, double *x, double complex *alpha, int *permutation_x_alpha, int N)
{
    if (N > 1)
    {
        int m = N / 2;

        quicksort(d, t, x, alpha, permutation_x_alpha, N);

        BuildTree(d, (t + 1) % d, x, alpha, permutation_x_alpha, m);
        BuildTree(d, (t + 1) % d, x + (m + 1) * d, alpha + (m + 1), permutation_x_alpha ? permutation_x_alpha + (m + 1) : NULL, N - m - 1);
    }
}

/** fast search in tree of source knots for near field computation*/
static double complex SearchTree(const int d, const int t, const double *x, const double complex *alpha,
        const double *xmin, const double *xmax, const int N, const kernel k, const double *param,
        const int Ad, const double complex *Add, const int p, const unsigned flags)
{
    if (N == 0)
    {
        return 0.0;
    }
    else
    {
        int m = N / 2;
        double Min = xmin[t];
        double Max = xmax[t];
        double Median = x[m * d + t];
        double a = fabs(Max - Min) / 2;
        int l;
        int E = 0;
        double r;

        if (Min > Median)
            return SearchTree(d, (t + 1) % d, x + (m + 1) * d, alpha + (m + 1), xmin,
                xmax, N - m - 1, k, param, Ad, Add, p, flags);
        else if (Max < Median)
            return SearchTree(d, (t + 1) % d, x, alpha, xmin, xmax, m, k, param, Ad,
                Add, p, flags);
        else
        {
            double complex result = 0.0;
            E = 0;

            for (l = 0; l < d; l++)
            {
                if (x[m * d + l] > xmin[l] && x[m * d + l] < xmax[l]) E++;
            }

            if (E == d)
            {
                if (d == 1)
                {
                    r = xmin[0] + a - x[m]; /* remember: xmin+a = y */
                }
                else
                {
                    r = 0.0;
                    for (l = 0; l < d; l++)
                        r += (xmin[l] + a - x[m * d + l]) * (xmin[l] + a - x[m * d + l]); /* remember: xmin+a = y */
                    r = sqrt(r);
                }
                if (fabs(r) < a)
                {
                    result += alpha[m] * k(r, 0, param); /* alpha*(kern-regkern) */
                    if (d == 1)
                    {
                        if (flags & EXACT_NEARFIELD)
                            result -= alpha[m] * regkern1(k, r, p, param, a, 1.0 / 16.0); /* exact value (in 1D)    */
                        else
                            result -= alpha[m] * kubintkern1(r, Add, Ad, a); /* spline approximation */
                    }
                    else
                    {
                        if (flags & EXACT_NEARFIELD)
                            result -= alpha[m] * regkern(k, r, p, param, a, 1.0 / 16.0); /* exact value (in dD)    */
                        else
                            result -= alpha[m] * kubintkern(r, Add, Ad, a); /* spline approximation */
                    }
                }
            }
            result += SearchTree(d, (t + 1) % d, x + (m + 1) * d, alpha + (m + 1), xmin,
                            xmax, N - m - 1, k, param, Ad, Add, p, flags);
            result += SearchTree(d, (t + 1) % d, x, alpha, xmin, xmax, m, k, param, Ad, Add,
                                        p, flags);
            return result;
        }
    }
}

static void fastsum_precompute_kernel(fastsum_plan *ths)
{
    int j, k, t;
    INT N[ths->d];
    int n_total;

    /** precompute spline values for near field */
    if (ths->eps_I > 0.0 && !(ths->flags & EXACT_NEARFIELD))
    {
        if (ths->d == 1)
            #pragma omp parallel for default(shared) private(k)
            for (k = -ths->Ad / 2 - 2; k <= ths->Ad / 2 + 2; k++)
                ths->Add[k + ths->Ad / 2 + 2] = regkern1(ths->k,
                        ths->eps_I * (double) k / (double)(ths->Ad) * 2.0, ths->p, ths->kernel_param,
                        ths->eps_I, ths->eps_B);
        else
            #pragma omp parallel for default(shared) private(k)
            for (k = 0; k <= ths->Ad + 2; k++)
                ths->Add[k] = regkern3(ths->k, ths->eps_I * (double) k / (double)(ths->Ad), ths->p,
                        ths->kernel_param, ths->eps_I, ths->eps_B);
    }

    /** precompute Fourier coefficients of regularised kernel*/
    n_total = 1;
    for (t = 0; t < ths->d; t++)
        n_total *= ths->n;

    #pragma omp parallel for default(shared) private(j,k,t)
    for (j = 0; j < n_total; j++)
    {
        if (ths->d == 1)
            ths->b[j] = regkern1(ths->k, (double) - (j / (double)(ths->n) - 0.5), ths->p,
                    ths->kernel_param, ths->eps_I, ths->eps_B) / (double)(n_total);
        else
        {
            k = j;
            ths->b[j] = 0.0;
            for (t = 0; t < ths->d; t++)
            {
                ths->b[j] += ((double) (k % (ths->n)) / (double)(ths->n) - 0.5)
                        * ((double) (k % (ths->n)) / (double)(ths->n) - 0.5);
                k = k / (ths->n);
            }
            ths->b[j] = regkern3(ths->k, sqrt(creal(ths->b[j])), ths->p, ths->kernel_param,
                    ths->eps_I, ths->eps_B) / (double)(n_total);
        }
    }

    for (t = 0; t < ths->d; t++)
        N[t] = ths->n;

    nfft_fftshift_complex(ths->b, (int)(ths->d), N);
    fftw_execute(ths->fft_plan);
    nfft_fftshift_complex(ths->b, (int)(ths->d), N);
}

void fastsum_init_guru_kernel(fastsum_plan *ths, int d, kernel k, double *param,
        unsigned flags, int nn, int p)
{
    int t;
    int N[d];
    int n_total;

    int nthreads = nfft_get_num_threads();

    ths->d = d;

    ths->k = k;
    ths->kernel_param = param;

    ths->flags = flags;

    ths->p = p;
    ths->eps_I = (double)ths->p / (double)nn; //** inner boundary */
    ths->eps_B = 1.0 / 16.0; //** outer boundary */

    /** init spline for near field computation */
    if (ths->eps_I > 0.0 && !(ths->flags & EXACT_NEARFIELD))
    {
        if (ths->d == 1)
        {
            ths->Ad = 4 * (ths->p) * (ths->p);
            ths->Add = (double complex *) nfft_malloc((size_t)(ths->Ad + 5) * (sizeof(double complex)));
        }
        else
        {
            ths->Ad = 2 * (ths->p) * (ths->p);
            ths->Add = (double complex *) nfft_malloc((size_t)(ths->Ad + 3) * (sizeof(double complex)));
        } /* multi-dimensional case */
    } /* !EXACT_NEARFIELD == spline approximation in near field AND eps_I > 0 */

    ths->n = nn;
    for (t = 0; t < d; t++)
    {
        N[t] = nn;
    }

    /** init d-dimensional FFTW plan */
    n_total = 1;
    for (t = 0; t < d; t++)
        n_total *= nn;

    ths->b = (double complex*) nfft_malloc((size_t)(n_total) * sizeof(double complex));
    ths->f_hat = (double complex*) nfft_malloc((size_t)(n_total) * sizeof(double complex));
    #pragma omp critical (nfft_omp_critical_fftw_plan)
    {
        fftw_plan_with_nthreads(nthreads);

        ths->fft_plan = fftw_plan_dft(d, N, ths->b, ths->b, FFTW_FORWARD,
                FFTW_ESTIMATE);
    }

    fastsum_precompute_kernel(ths);
}

void fastsum_init_guru_source_nodes(fastsum_plan *ths, int N_total, int nn_oversampled, int m)
{
    int t;
    int N[ths->d], n[ths->d];
    unsigned sort_flags_adjoint = 0U;

    if (ths->d > 1)
    {
        sort_flags_adjoint = NFFT_SORT_NODES | NFFT_OMP_BLOCKWISE_ADJOINT;
    }

    ths->N_total = N_total;

    ths->x = (double *) nfft_malloc((size_t)(ths->d * N_total) * (sizeof(double)));
    ths->alpha = (double complex *) nfft_malloc((size_t)(N_total) * (sizeof(double complex)));

    /** init d-dimensional NFFT plan */
    for (t = 0; t < ths->d; t++)
    {
        N[t] = ths->n;
        n[t] = nn_oversampled;
    }

    nfft_init_guru(&(ths->mv1), ths->d, N, N_total, n, m,
            sort_flags_adjoint | PRE_PHI_HUT | PRE_PSI | FFTW_INIT
            | ((ths->d == 1) ? FFT_OUT_OF_PLACE : 0U),
            FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    ths->mv1.x = ths->x;
    ths->mv1.f = ths->alpha;
    ths->mv1.f_hat = ths->f_hat;
    
    ths->box_offset = NULL;
    ths->box_alpha = NULL;
    ths->box_x = NULL;
    ths->permutation_x_alpha = NULL;

    if (ths->flags & NEARFIELD_BOXES)
    {
        if (ths->eps_I > 0.0)
        {
            ths->box_count_per_dim = (int)(lrint(floor((0.5 - ths->eps_B) / ths->eps_I))) + 1;
            ths->box_count = 1;
            for (t = 0; t < ths->d; t++)
                ths->box_count *= ths->box_count_per_dim;

            ths->box_offset = (int *) nfft_malloc((size_t)(ths->box_count + 1) * sizeof(int));

            ths->box_alpha = (double complex *) nfft_malloc((size_t)(ths->N_total) * (sizeof(double complex)));

            ths->box_x = (double *) nfft_malloc((size_t)(ths->d * ths->N_total) * sizeof(double));
        } /* eps_I > 0 */
    } /* NEARFIELD_BOXES */
    else
    {
        if ((ths->flags & STORE_PERMUTATION_X_ALPHA) && (ths->eps_I > 0.0))
        {
            ths->permutation_x_alpha = (int *) nfft_malloc((size_t)(ths->N_total) * (sizeof(int)));
            for (int i=0; i<ths->N_total; i++)
                ths->permutation_x_alpha[i] = i;
        }
    } /* search tree */
}

void fastsum_init_guru_target_nodes(fastsum_plan *ths, int M_total, int nn_oversampled, int m)
{
    int t;
    int N[ths->d], n[ths->d];
    unsigned sort_flags_trafo = 0U;

    if (ths->d > 1)
        sort_flags_trafo = NFFT_SORT_NODES;

    ths->M_total = M_total;

    ths->y = (double *) nfft_malloc((size_t)(ths->d * M_total) * (sizeof(double)));
    ths->f = (double complex *) nfft_malloc((size_t)(M_total) * (sizeof(double complex)));

    /** init d-dimensional NFFT plan */
    for (t = 0; t < ths->d; t++)
    {
        N[t] = ths->n;
        n[t] = nn_oversampled;
    }

    nfft_init_guru(&(ths->mv2), ths->d, N, M_total, n, m,
        sort_flags_trafo | PRE_PHI_HUT | PRE_PSI | FFTW_INIT
        | ((ths->d == 1) ? FFT_OUT_OF_PLACE : 0U),
        FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    ths->mv2.x = ths->y;
    ths->mv2.f = ths->f;
    ths->mv2.f_hat = ths->f_hat;
}

/** initialization of fastsum plan */
void fastsum_init_guru(fastsum_plan *ths, int d, int N_total, int M_total,
        kernel k, double *param, unsigned flags, int nn, int m, int p)
{
    fastsum_init_guru_kernel(ths, d, k, param, flags, nn, p);
    fastsum_init_guru_source_nodes(ths, N_total, 2 * nn, m);
    fastsum_init_guru_target_nodes(ths, M_total, 2 * nn, m);
}

/** finalization of fastsum plan */
void fastsum_finalize_source_nodes(fastsum_plan *ths)
{
    nfft_free(ths->x);
    nfft_free(ths->alpha);

    nfft_finalize(&(ths->mv1));

    if (ths->flags & NEARFIELD_BOXES)
    {
        if (ths->eps_I > 0.0)
        {
            nfft_free(ths->box_offset);
            nfft_free(ths->box_alpha);
            nfft_free(ths->box_x);
        }
    } /* NEARFIELD_BOXES */
    else
    {
        if (ths->permutation_x_alpha)
            nfft_free(ths->permutation_x_alpha);
    } /* search tree */
}

/** finalization of fastsum plan */
void fastsum_finalize_target_nodes(fastsum_plan *ths)
{
    nfft_free(ths->y);
    nfft_free(ths->f);

    nfft_finalize(&(ths->mv2));
}

/** finalization of fastsum plan */
void fastsum_finalize_kernel(fastsum_plan *ths)
{
    if (ths->eps_I > 0.0 && !(ths->flags & EXACT_NEARFIELD))
        nfft_free(ths->Add);

    #pragma omp critical (nfft_omp_critical_fftw_plan)
    {
        fftw_destroy_plan(ths->fft_plan);
    }

    nfft_free(ths->b);
    nfft_free(ths->f_hat);
}

/** finalization of fastsum plan */
void fastsum_finalize(fastsum_plan *ths)
{
    fastsum_finalize_target_nodes(ths);
    fastsum_finalize_source_nodes(ths);
    fastsum_finalize_kernel(ths);
}

/** direct computation of sums */
void fastsum_exact(fastsum_plan *ths)
{
    int j, k;
    int t;
    double r;

    #pragma omp parallel for default(shared) private(j,k,t,r)
    for (j = 0; j < ths->M_total; j++)
    {
        ths->f[j] = 0.0;
        for (k = 0; k < ths->N_total; k++)
        {
            if (ths->d == 1)
                r = ths->y[j] - ths->x[k];
            else
            {
                r = 0.0;
                for (t = 0; t < ths->d; t++)
                    r += (ths->y[j * ths->d + t] - ths->x[k * ths->d + t])
                        * (ths->y[j * ths->d + t] - ths->x[k * ths->d + t]);
                r = sqrt(r);
            }
            ths->f[j] += ths->alpha[k] * ths->k(r, 0, ths->kernel_param);
        }
    }
}

/** precomputation for fastsum */
void fastsum_precompute_source_nodes(fastsum_plan *ths)
{

    if (ths->eps_I > 0.0)
    {
        if (ths->flags & NEARFIELD_BOXES)
            BuildBox(ths);
        else
        /** sort source knots for search tree */
            BuildTree(ths->d, 0, ths->x, ths->alpha, ths->permutation_x_alpha, ths->N_total);
    } /* eps_I > 0 */

    /** precompute psi, the entries of the matrix B */
    if (ths->mv1.flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(ths->mv1));

    if (ths->mv1.flags & PRE_PSI)
        nfft_precompute_psi(&(ths->mv1));

    if (ths->mv1.flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(ths->mv1));

}

/** precomputation for fastsum */
void fastsum_precompute_target_nodes(fastsum_plan *ths)
{
    /** precompute psi, the entries of the matrix B */
    if (ths->mv2.flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(ths->mv2));

    if (ths->mv2.flags & PRE_PSI)
        nfft_precompute_psi(&(ths->mv2));

    if (ths->mv2.flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(ths->mv2));
}

/** precomputation for fastsum */
void fastsum_precompute(fastsum_plan *ths)
{
    fastsum_precompute_source_nodes(ths);
    fastsum_precompute_target_nodes(ths);
}

/** fast NFFT-based summation */
void fastsum_trafo(fastsum_plan *ths)
{
    int j, k, t;

    /** first step of algorithm */
    nfft_adjoint(&(ths->mv1));

    /** second step of algorithm */
    #pragma omp parallel for default(shared) private(k)
    for (k = 0; k < ths->mv2.N_total; k++)
        ths->mv2.f_hat[k] = ths->b[k] * ths->mv1.f_hat[k];

    /** third step of algorithm */
    nfft_trafo(&(ths->mv2));

    /** write far field to output */
    #pragma omp parallel for default(shared) private(j)
    for (j = 0; j < ths->M_total; j++)
        ths->f[j] = ths->mv2.f[j];

    if (ths->eps_I > 0.0)
    {
        /** add near field */
        #pragma omp parallel for default(shared) private(j,k,t)
        for (j = 0; j < ths->M_total; j++)
        {
            double ymin[ths->d], ymax[ths->d]; /** limits for d-dimensional near field box */

            if (ths->flags & NEARFIELD_BOXES)
                ths->f[j] += SearchBox(ths->y + ths->d * j, ths);
            else
            {
                for (t = 0; t < ths->d; t++)
                {
                    ymin[t] = ths->y[ths->d * j + t] - ths->eps_I;
                    ymax[t] = ths->y[ths->d * j + t] + ths->eps_I;
                }
                ths->f[j] += SearchTree(ths->d, 0, ths->x, ths->alpha, ymin, ymax, ths->N_total,
                         ths->k, ths->kernel_param, ths->Ad, ths->Add, ths->p, ths->flags);
            }
        }
    }
}

/* fastsum.c */