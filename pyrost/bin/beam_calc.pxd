cdef extern from "gsl/gsl_math.h":

    ctypedef struct gsl_function:
        double (* function) (double x, void * params) nogil
        void * params

cdef extern from "gsl/gsl_integration.h":

    ctypedef struct gsl_integration_workspace
    ctypedef struct gsl_integration_qaws_table
    ctypedef struct gsl_integration_qawo_table
    ctypedef struct gsl_integration_cquad_workspace
    cdef enum:
        GSL_INTEG_GAUSS15 = 1
        GSL_INTEG_GAUSS21 = 2
        GSL_INTEG_GAUSS31 = 3
        GSL_INTEG_GAUSS41 = 4
        GSL_INTEG_GAUSS51 = 5
        GSL_INTEG_GAUSS61 = 6
    cdef enum gsl_integration_qawo_enum:
        GSL_INTEG_COSINE, GSL_INTEG_SINE

    gsl_integration_cquad_workspace *  gsl_integration_cquad_workspace_alloc (size_t n) nogil
    
    void  gsl_integration_cquad_workspace_free (gsl_integration_cquad_workspace * w) nogil

    gsl_integration_workspace *  gsl_integration_workspace_alloc(size_t n) nogil

    void  gsl_integration_workspace_free(gsl_integration_workspace * w) nogil

    int  gsl_integration_qag(gsl_function *f, double a, double b, double epsabs, double epsrel, size_t limit, int key, gsl_integration_workspace * workspace, double * result, double * abserr) nogil

cdef extern from "gsl/gsl_rng.h":

    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    cdef gsl_rng_type *gsl_rng_mt19937

    gsl_rng *gsl_rng_alloc ( gsl_rng_type * T) nogil

    unsigned long int gsl_rng_get ( gsl_rng * r) nogil

    void gsl_rng_set ( gsl_rng * r, unsigned long int seed) nogil

    void gsl_rng_free (gsl_rng * r) nogil

    double gsl_rng_uniform ( gsl_rng * r) nogil
    double gsl_rng_uniform_pos ( gsl_rng * r) nogil

cdef extern from "gsl/gsl_randist.h":

    unsigned int gsl_ran_poisson ( gsl_rng * r, double mu) nogil