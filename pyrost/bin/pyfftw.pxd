# Copyright 2014 Knowledge Economy Developments Ltd
#
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

cimport numpy as np
from libc.stdint cimport int64_t

ctypedef struct _fftw_iodim:
    int _n
    int _is
    int _os

ctypedef float cfloat[2]
ctypedef double cdouble[2]

cdef extern from "cpu.h":

    int simd_alignment()

cdef extern from "Python.h":
    int Py_AtExit(void(*func)())

cdef extern from 'fftw3.h':

    # Double precision plans
    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    # Single precision plans
    ctypedef struct fftwf_plan_struct:
        pass

    ctypedef fftwf_plan_struct *fftwf_plan

    # The stride info structure. I think that strictly
    # speaking, this should be defined with a type suffix
    # on fftw (ie fftw, fftwf or fftwl), but since the
    # definition is transparent and is defined as _fftw_iodim,
    # we ignore the distinction in order to simplify the code.
    ctypedef struct fftw_iodim:
        pass

    # Double precision complex planner
    fftw_plan fftw_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cdouble *_in, cdouble *_out,
            int sign, unsigned flags) nogil

    # Single precision complex planner
    fftwf_plan fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, cfloat *_out,
            int sign, unsigned flags) nogil

    # Double precision real to complex planner
    fftw_plan fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, cdouble *_out,
            unsigned flags) nogil

    # Single precision real to complex planner
    fftwf_plan fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, cfloat *_out,
            unsigned flags) nogil

    # Double precision complex to real planner
    fftw_plan fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cdouble *_in, double *_out,
            unsigned flags) nogil

    # Single precision complex to real planner
    fftwf_plan fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, float *_out,
            unsigned flags) nogil

    # Double precision real planner
    fftw_plan fftw_plan_guru_r2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, double *_out,
            int *kind, unsigned flags)

    # Single precision real planner
    fftwf_plan fftwf_plan_guru_r2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, float *_out,
            int *kind, unsigned flags)
    # Double precision complex new array execute
    void fftw_execute_dft(fftw_plan,
          cdouble *_in, cdouble *_out) nogil

    # Single precision complex new array execute
    void fftwf_execute_dft(fftwf_plan,
          cfloat *_in, cfloat *_out) nogil

    # Double precision real to complex new array execute
    void fftw_execute_dft_r2c(fftw_plan,
          double *_in, cdouble *_out) nogil

    # Single precision real to complex new array execute
    void fftwf_execute_dft_r2c(fftwf_plan,
          float *_in, cfloat *_out) nogil

    # Double precision complex to real new array execute
    void fftw_execute_dft_c2r(fftw_plan,
          cdouble *_in, double *_out) nogil

    # Single precision complex to real new array execute
    void fftwf_execute_dft_c2r(fftwf_plan,
          cfloat *_in, float *_out) nogil

    # Double precision real new array execute
    void fftw_execute_r2r(fftw_plan,
          double *_in, double *_out) nogil

    # Single precision real new array execute
    void fftwf_execute_r2r(fftwf_plan,
          float *_in, float *_out) nogil

    # Double precision plan destroyer
    void fftw_destroy_plan(fftw_plan)

    # Single precision plan destroyer
    void fftwf_destroy_plan(fftwf_plan)

    # Double precision set timelimit
    void fftw_set_timelimit(double seconds)

    # Single precision set timelimit
    void fftwf_set_timelimit(double seconds)

    # Threading routines
    # Double precision
    void fftw_init_threads()
    void fftw_plan_with_nthreads(int n)

    # Single precision
    void fftwf_init_threads()
    void fftwf_plan_with_nthreads(int n)

    # cleanup routines
    void fftw_cleanup()
    void fftwf_cleanup()
    void fftw_cleanup_threads()
    void fftwf_cleanup_threads()

    double FFTW_NO_TIMELIMIT

# Define function pointers that can act as a placeholder
# for whichever dtype is used (the problem being that fftw
# has different function names and signatures for all the
# different precisions and dft types).
ctypedef void * (*fftw_generic_plan_guru)(
        int rank, fftw_iodim *dims,
        int howmany_rank, fftw_iodim *howmany_dims,
        void *_in, void *_out,
        int *directions, unsigned flags) nogil

ctypedef void (*fftw_generic_execute)(void *_plan, void *_in, void *_out) nogil

ctypedef void (*fftw_generic_destroy_plan)(void *_plan)

ctypedef void (*fftw_generic_init_threads)()

ctypedef void (*fftw_generic_plan_with_nthreads)(int n)

ctypedef void (*fftw_generic_set_timelimit)(double seconds)

ctypedef bint (*validator)(np.ndarray input_array,
        np.ndarray output_array, int64_t *axes, int64_t *not_axes,
        int64_t axes_length)

# Direction enum
cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1
    # from fftw3.f 3.3.3; may not be valid for different versions of FFTW.
    FFTW_REDFT00  = 3
    FFTW_REDFT01  = 4
    FFTW_REDFT10  = 5
    FFTW_REDFT11  = 6
    FFTW_RODFT00  = 7
    FFTW_RODFT01  = 8
    FFTW_RODFT10  = 9
    FFTW_RODFT11  = 10

# Documented flags
cdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64
    FFTW_WISDOM_ONLY = 2097152

cdef class FFTW:
    # Each of these function pointers simply
    # points to a chosen fftw wrapper function
    cdef fftw_generic_plan_guru _fftw_planner
    cdef fftw_generic_execute _fftw_execute
    cdef fftw_generic_destroy_plan _fftw_destroy
    cdef fftw_generic_plan_with_nthreads _nthreads_plan_setter

    # The plan is typecast when it is created or used
    # within the wrapper functions
    cdef void *_plan

    cdef np.ndarray _input_array
    cdef np.ndarray _output_array
    cdef int *_direction
    cdef unsigned _flags

    cdef bint _simd_allowed
    cdef int _input_array_alignment
    cdef int _output_array_alignment
    cdef bint _use_threads

    cdef object _input_item_strides
    cdef object _input_strides
    cdef object _output_item_strides
    cdef object _output_strides
    cdef object _input_shape
    cdef object _output_shape
    cdef object _input_dtype
    cdef object _output_dtype
    cdef object _flags_used

    cdef double _normalisation_scaling
    cdef double _sqrt_normalisation_scaling

    cdef int _rank
    cdef _fftw_iodim *_dims
    cdef int _howmany_rank
    cdef _fftw_iodim *_howmany_dims

    cdef int64_t *_axes
    cdef int64_t *_not_axes

    cdef int64_t _total_size

    cdef bint _normalise_idft
    cdef bint _ortho

    cdef inline _update_arrays(self, np.ndarray new_input_array,
                               np.ndarray new_output_array):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self._input_array = new_input_array
        self._output_array = new_output_array

    cdef inline _execute(self):
        '''execute()

        Execute the planned operation, taking the correct kind of FFT of
        the input array (i.e. :attr:`FFTW.input_array`),
        and putting the result in the output array (i.e.
        :attr:`FFTW.output_array`).
        '''
        cdef void *input_pointer = (
                <void *>np.PyArray_DATA(self._input_array))
        cdef void *output_pointer = (
                <void *>np.PyArray_DATA(self._output_array))

        cdef void *plan = self._plan
        cdef fftw_generic_execute fftw_execute = self._fftw_execute
        with nogil:
            fftw_execute(plan, input_pointer, output_pointer)