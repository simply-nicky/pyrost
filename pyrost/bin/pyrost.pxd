cimport numpy as np
from libc.math cimport sqrt, exp, pi, floor, ceil, fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef fused uint_t:
    np.uint64_t
    np.uint32_t

cdef extern from "gsl/gsl_rng.h":

    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    cdef gsl_rng_type *gsl_rng_mt19937

    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil

    unsigned long int gsl_rng_get(gsl_rng * r) nogil

    void gsl_rng_set(gsl_rng * r, unsigned long int seed) nogil

    void gsl_rng_free(gsl_rng * r) nogil

    double gsl_rng_uniform(gsl_rng * r) nogil
    unsigned long gsl_rng_uniform_int(gsl_rng * r, unsigned long n) nogil