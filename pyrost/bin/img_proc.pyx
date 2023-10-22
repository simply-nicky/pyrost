import numpy as np
from libc.string cimport memcmp
from libc.stdlib cimport malloc, free

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def next_fast_len(unsigned target, str backend='numpy'):
    if target < 0:
        raise ValueError('Target length must be positive')
    if backend == 'fftw':
        return next_fast_len_fftw(target)
    elif backend == 'numpy':
        return good_size(target)
    else:
        raise ValueError('{:s} is invalid backend'.format(backend))

def fft_convolve(np.ndarray array not None, np.ndarray kernel not None, int axis=-1,
                 str mode='constant', double cval=0.0, str backend='numpy',
                 unsigned num_threads=1):
    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = mode_to_code(mode)
    cdef np.npy_intp *dims = array.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef int type_num
    if np.PyArray_ISCOMPLEX(array) or np.PyArray_ISCOMPLEX(kernel):
        type_num = np.NPY_COMPLEX128
    else:
        type_num = np.NPY_FLOAT64

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = np.PyArray_DATA(out)
    cdef void *_inp
    cdef void *_krn

    if np.PyArray_ISCOMPLEX(array) or np.PyArray_ISCOMPLEX(kernel):
        array = check_array(array, np.NPY_COMPLEX128)
        kernel = check_array(kernel, np.NPY_COMPLEX128)

        _inp = np.PyArray_DATA(array)
        _krn = np.PyArray_DATA(kernel)

        with nogil:
            if backend == 'fftw':
                fail = cfft_convolve_fftw(<double complex *>_out, <double complex *>_inp, ndim,
                                          _dims, <double complex *>_krn, ksize, axis, _mode,
                                          <double complex>cval, num_threads)
            elif backend == 'numpy':
                fail = cfft_convolve_np(<double complex *>_out, <double complex *>_inp, ndim,
                                        _dims, <double complex *>_krn, ksize, axis, _mode,
                                        <double complex>cval, num_threads)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))
    else:
        array = check_array(array, np.NPY_FLOAT64)
        kernel = check_array(kernel, np.NPY_FLOAT64)

        _inp = np.PyArray_DATA(array)
        _krn = np.PyArray_DATA(kernel)

        with nogil:
            if backend == 'fftw':
                fail = rfft_convolve_fftw(<double *>_out, <double *>_inp, ndim, _dims,
                                          <double *>_krn, ksize, axis, _mode, cval, num_threads)
            elif backend == 'numpy':
                fail = rfft_convolve_np(<double *>_out, <double *>_inp, ndim, _dims,
                                        <double *>_krn, ksize, axis, _mode, cval, num_threads)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_kernel(double sigma, unsigned order=0, double truncate=4.0):
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0], 1)
    return out

def gaussian_filter(np.ndarray inp not None, object sigma not None, object order not None=0,
                    str mode='reflect', double cval=0.0, double truncate=4.0, str backend='numpy',
                    unsigned num_threads=1):
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.ndarray orders = normalize_sequence(order, ndim, np.NPY_UINT32)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = mode_to_code(mode)
    cdef np.npy_intp *dims = inp.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef int type_num
    if np.PyArray_ISCOMPLEX(inp):
        type_num = np.NPY_COMPLEX128
    else:
        type_num = np.NPY_FLOAT64

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = np.PyArray_DATA(out)

    cdef void *_inp
    if np.PyArray_ISCOMPLEX(inp):
        inp = check_array(inp, np.NPY_COMPLEX128)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_filter_c(<double complex *>_out, <double complex *>_inp,
                                      ndim, _dims, _sig, _ord, _mode, <double complex>cval,
                                      truncate, num_threads, cfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_filter_c(<double complex *>_out, <double complex *>_inp,
                                      ndim, _dims, _sig, _ord, _mode, <double complex>cval,
                                      truncate, num_threads, cfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    else:
        inp = check_array(inp, np.NPY_FLOAT64)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_filter_r(<double *>_out, <double *>_inp, ndim, _dims, _sig,
                                      _ord, _mode, cval, truncate, num_threads, rfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_filter_r(<double *>_out, <double *>_inp, ndim, _dims, _sig,
                                      _ord, _mode, cval, truncate, num_threads, rfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_gradient_magnitude(np.ndarray inp not None, object sigma not None, str mode='reflect',
                                double cval=0.0, double truncate=4.0, str backend='numpy',
                                unsigned num_threads=1):
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = mode_to_code(mode)
    cdef np.npy_intp *dims = inp.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)

    cdef void *_inp
    if np.PyArray_ISCOMPLEX(inp):
        inp = check_array(inp, np.NPY_COMPLEX128)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_grad_mag_c(_out, <double complex *>_inp, ndim, _dims, _sig,
                                        _mode, <double complex>cval, truncate, num_threads,
                                        cfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_grad_mag_c(_out, <double complex *>_inp, ndim, _dims, _sig,
                                        _mode, <double complex>cval, truncate, num_threads,
                                        cfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    else:
        inp = check_array(inp, np.NPY_FLOAT64)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_grad_mag_r(_out, <double *>_inp, ndim, _dims, _sig, _mode,
                                        cval, truncate, num_threads, rfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_grad_mag_r(_out, <double *>_inp, ndim, _dims, _sig, _mode,
                                        cval, truncate, num_threads, rfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))
    
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def median(np.ndarray inp not None, np.ndarray mask=None, object axis=0,
           unsigned num_threads=1):
    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(inp.ndim, inp.shape, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)
        if memcmp(inp.shape, mask.shape, inp.ndim * sizeof(np.npy_intp)):
            raise ValueError('mask and inp arrays must have identical shapes')

    cdef int ndim = 0, i, j, repeats = 1
    cdef np.ndarray axarr = to_array(axis, np.NPY_INT32)
    for i in range(axarr.size):
        axarr[i] = axarr[i] if axarr[i] >= 0 else inp.ndim + axarr[i]
        if axarr[i] >= inp.ndim:
            raise ValueError(f'Axis {axarr[i]:d} is out of bounds')

    for i in range(inp.ndim):
        j = 0
        while j < axarr.size and axarr[j] != i:
            j += 1

        if j == axarr.size:
            inp = np.PyArray_SwapAxes(inp, ndim, i)
            repeats *= inp.shape[ndim]
            ndim += 1

    cdef np.npy_intp *new_dims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    cdef np.npy_intp *odims = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        new_dims[i] = inp.shape[i]
        odims[i] = inp.shape[i]
    cdef int type_num = np.PyArray_TYPE(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, odims, type_num)
    if not out.size:
        return out

    new_dims[ndim] = inp.size / repeats
    new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    new_shape[0].ptr = new_dims; new_shape[0].len = ndim + 1

    inp = np.PyArray_Newshape(inp, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)
    mask = np.PyArray_Newshape(mask, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(mask):
        mask = np.PyArray_GETCONTIGUOUS(mask)
    free(odims); free(new_dims); free(new_shape)

    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned long *_dims = <unsigned long *>inp.shape

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 8, ndim, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 8, ndim, compare_ulong, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def median_filter(np.ndarray inp not None, object size=None, np.ndarray footprint=None,
                  np.ndarray mask=None, np.ndarray inp_mask=None, str mode='reflect', double cval=0.0,
                  unsigned num_threads=1):
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)

    cdef int ndim = inp.ndim
    cdef np.npy_intp *dims = inp.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if inp_mask is None:
        inp_mask = mask
    else:
        inp_mask = check_array(inp_mask, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalize_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    else:
        footprint = check_array(footprint, np.NPY_BOOL)

    if footprint.ndim != ndim:
        raise ValueError('footprint and size must have the same number of dimensions as the input')
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)

    if not out.size:
        return out

    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned char *_imask = <unsigned char *>np.PyArray_DATA(inp_mask)
    cdef int _mode = mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_ulong, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def robust_mean(np.ndarray inp not None, object axis=0, double r0=0.0, double r1=0.5,
                int n_iter=12, double lm=9.0, unsigned num_threads=1):
    cdef int ndim = 0, i, j, repeats = 1
    cdef np.ndarray axarr = to_array(axis, np.NPY_INT32)
    for i in range(axarr.size):
        axarr[i] = axarr[i] if axarr[i] >= 0 else inp.ndim + axarr[i]
        if axarr[i] >= inp.ndim:
            raise ValueError(f'Axis {axarr[i]:d} is out of bounds')

    for i in range(inp.ndim):
        j = 0
        while j < axarr.size and axarr[j] != i:
            j += 1

        if j == axarr.size:
            inp = np.PyArray_SwapAxes(inp, ndim, i)
            repeats *= inp.shape[ndim]
            ndim += 1

    cdef np.npy_intp *new_dims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    cdef np.npy_intp *odims = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        new_dims[i] = inp.shape[i]
        odims[i] = inp.shape[i]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, odims, np.NPY_FLOAT64)
    if not out.size:
        return out

    new_dims[ndim] = inp.size / repeats
    new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    new_shape[0].ptr = new_dims; new_shape[0].len = ndim + 1

    inp = np.PyArray_Newshape(inp, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)
    free(odims); free(new_dims); free(new_shape)

    cdef int type_num = np.PyArray_TYPE(inp)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>inp.shape

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 8, ndim, compare_double,
                                 get_double, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_float,
                                 get_float, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_INT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_int,
                                 get_int, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_uint,
                                 get_uint, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 8, ndim, compare_ulong,
                                 get_ulong, r0, r1, n_iter, lm, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out