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