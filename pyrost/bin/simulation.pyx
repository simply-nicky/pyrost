import numpy as np
import cython
from libc.string cimport memcmp
from libc.math cimport sqrt
from libc.stdlib cimport abort, malloc, free

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

# Helper functions

cdef int extend_mode_to_code(str mode) except -1:
    if mode == 'constant':
        return EXTEND_CONSTANT
    elif mode == 'nearest':
        return EXTEND_NEAREST
    elif mode == 'mirror':
        return EXTEND_MIRROR
    elif mode == 'reflect':
        return EXTEND_REFLECT
    elif mode == 'wrap':
        return EXTEND_WRAP
    else:
        raise RuntimeError('boundary mode not supported')

cdef np.ndarray check_array(np.ndarray array, int type_num):
    cdef np.ndarray out
    cdef int tn = np.PyArray_TYPE(array)
    if not np.PyArray_IS_C_CONTIGUOUS(array):
        array = np.PyArray_GETCONTIGUOUS(array)

    if tn != type_num:
        out = np.PyArray_SimpleNew(array.ndim, <np.npy_intp *>array.shape, type_num)
        np.PyArray_CastTo(out, array)
    else:
        out = array

    return out

cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray array, out
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        out = number_to_array(inp, rank, type_num)

    elif np.PyArray_Check(inp):
        array = <np.ndarray>inp
        tn = np.PyArray_TYPE(array)
        if tn != type_num:
            out = np.PyArray_SimpleNew(array.ndim, <np.npy_intp *>array.shape, type_num)
            np.PyArray_CastTo(out, array)
        else:
            out = array

    elif isinstance(inp, (list, tuple)):
        out = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)

    else:
        raise ValueError("Wrong sequence argument type")

    cdef np.npy_intp size = np.PyArray_SIZE(out)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return out

def next_fast_len(unsigned target, str backend='numpy'):
    r"""Find the next fast size of input data to fft, for zero-padding, etc.
    FFT algorithms gain their speed by a recursive divide and conquer strategy.
    This relies on efficient functions for small prime factors of the input length.
    Thus, the transforms are fastest when using composites of the prime factors handled
    by the fft implementation. If there are efficient functions for all radices <= n,
    then the result will be a number x >= target with only prime factors < n. (Also
    known as n-smooth numbers)

    Args:
        target (int) : Length to start searching from. Must be a positive integer.
        backend (str) : Find n-smooth number for the FFT implementation from the numpy
            ('numpy') or FFTW ('fftw') library.

    Returns:
        int : The smallest fast length greater than or equal to `target`.
    """
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
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Args:
        array (numpy.ndarray) : Input array.
        kernel (numpy.ndarray) : Kernel array.
        axis (int) : Array axis along which convolution is performed.
        mode (str) : The mode parameter determines how the input array is extended
            when the filter overlaps a border. Default value is 'constant'. The
            valid values and their behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by
              filling all values beyond the edge with the same constant value, defined
              by the `cval` parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by
              replicating the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by
              reflecting about the center of the last pixel. This mode is also sometimes
              referred to as whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by
              reflecting about the edge of the last pixel. This mode is also sometimes
              referred to as half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping
              around to the opposite edge.

        cval (float) :  Value to fill past edges of input if mode is 'constant'. Default
            is 0.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') library for the FFT
            implementation.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : A multi-dimensional array containing the discrete linear
        convolution of `array` with `kernel`.
    """
    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = extend_mode_to_code(mode)
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

def rsc_wp(np.ndarray wft not None, double dx0, double dx, double z,
           double wl, int axis=-1, str backend='numpy', unsigned num_threads=1):
    r"""Wavefront propagator based on Rayleigh-Sommerfeld convolution
    method [RSC]_. Propagates a wavefront `wft` by `z` distance downstream.
    You can choose between 'fftw' and 'numpy' backends for FFT calculations.

    Args: 
        wft (numpy.ndarray) : Initial wavefront.
        dx0 (float) : Sampling interval at the plane upstream [um].
        dx (float) : Sampling interval at the plane downstream [um].
        z (float) : Propagation distance [um].
        wl (float) : Incoming beam's wavelength [um].
        axis (int) : Axis of `wft` array along which the calculation is performed.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') library
            for the FFT  implementation.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Propagated wavefront.

    Raises:
        RuntimeError : If 'numpy' backend exits with eror during the calculation.
        ValueError : If `backend` option is invalid.

    Notes:
        The Rayleigh-Sommerfeld diffraction integral transform is defined as:

        .. math::
            u^{\prime}(x^{\prime}) = \frac{z}{j \sqrt{\lambda}} \int_{-\infty}^{+\infty}
            u(x) \mathrm{exp} \left[-j k r(x, x^{\prime}) \right] dx
        
        with

        .. math::
            r(x, x^{\prime}) = \left[ (x - x^{\prime})^2 + z^2 \right]^{1 / 2}

    References:
        .. [RSC] V. Nascov and P. C. Logofătu, "Fast computation algorithm
                for the Rayleigh-Sommerfeld diffraction formula using
                a type of scaled convolution," Appl. Opt. 48, 4310-4319
                (2009).
    """
    wft = check_array(wft, np.NPY_COMPLEX128)

    cdef int fail = 0
    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp *dims = wft.shape
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    with nogil:
        if backend == 'fftw':
            fail = rsc_fftw(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = rsc_np(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def fraunhofer_wp(np.ndarray wft not None, double dx0, double dx, double z,
                  double wl, int axis=-1, str backend='numpy', unsigned num_threads=1):
    r"""Fraunhofer diffraction propagator. Propagates a wavefront `wft` by `z`
    distance downstream. You can choose between 'fftw' and 'numpy' backends for
    FFT calculations.

    Args: 
        wft (numpy.ndarray) : Initial wavefront.
        dx0 (float) : Sampling interval at the plane upstream [um].
        dx (float) : Sampling interval at the plane downstream [um].
        z (float) : Propagation distance [um].
        wl (float) : Incoming beam's wavelength [um].
        axis (int) : Axis of `wft` array along which the calculation is performed.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') library
            for the FFT  implementation.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Propagated wavefront.

    Raises:
        RuntimeError : If 'numpy' backend exits with eror during the calculation.
        ValueError : If `backend` option is invalid.

    Notes:
        The Fraunhofer integral transform is defined as:

        .. math::
            u^{\prime}(x^{\prime}) = \frac{e^{-j k z}}{j \sqrt{\lambda z}}
            e^{-\frac{j k}{2 z} x^{\prime 2}} \int_{-\infty}^{+\infty} u(x)
            e^{j\frac{2 \pi}{\lambda z} x x^{\prime}} dx
    """
    wft = check_array(wft, np.NPY_COMPLEX128)

    cdef int fail = 0
    cdef np.npy_intp isize = np.PyArray_SIZE(wft)
    cdef int ndim = wft.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp *dims = wft.shape
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef complex *_inp = <complex *>np.PyArray_DATA(wft)
    with nogil:
        if backend == 'fftw':
            fail = fraunhofer_fftw(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        elif backend == 'numpy':
            fail = fraunhofer_np(_out, _inp, ndim, _dims, axis, dx0, dx, z, wl, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_kernel(double sigma, unsigned order=0, double truncate=4.):
    """Discrete Gaussian kernel.
    
    Args:
        sigma (float) : Standard deviation for Gaussian kernel.
        order (int) : The order of the filter. An order of 0 corresponds to
            convolution with a Gaussian kernel. A positive order corresponds
            to convolution with that derivative of a Gaussian. Default is 0.
        truncate (float) : Truncate the filter at this many standard deviations.
            Default is 4.0.
    
    Returns:
        numpy.ndarray : Gaussian kernel.
    """
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0], 1)
    return out

def gaussian_filter(np.ndarray inp not None, object sigma not None, object order not None=0,
                    str mode='reflect', double cval=0., double truncate=4., str backend='numpy',
                    unsigned num_threads=1):
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Args:
        inp (numpy.ndarray) : The input array.
        sigma (Union[float, List[float]]): Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        order (Union[int, List[int]]): The order of the filter along each axis is given as a
            sequence of integers, or as a single number. An order of 0 corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the `cval`
              parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by replicating
              the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by reflecting
              about the center of the last pixel. This mode is also sometimes referred to as
              whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by reflecting
              about the edge of the last pixel. This mode is also sometimes referred to as
              half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping around
              to the opposite edge.

        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate (float) : Truncate the filter at this many standard deviations. Default is 4.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads (int) : Number of threads.
    
    Returns:
        numpy.ndarray : Returned array of the same shape as `input`.
    """

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
    cdef int _mode = extend_mode_to_code(mode)
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
    cdef int inp_tn = np.PyArray_TYPE(inp)
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
    return check_array(out, inp_tn)

def gaussian_gradient_magnitude(np.ndarray inp not None, object sigma not None, str mode='reflect',
                                double cval=0.0, double truncate=4.0, str backend='numpy',
                                unsigned num_threads=1):
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Args:
        inp (numpy.ndarray) : The input array.
        sigma (Union[float, List[float]]): Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the `cval`
              parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by replicating
              the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by reflecting
              about the center of the last pixel. This mode is also sometimes referred to as
              whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by reflecting
              about the edge of the last pixel. This mode is also sometimes referred to as
              half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping around
              to the opposite edge.

        cval (float) : Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
        truncate (float) : Truncate the filter at this many standard deviations. Default is 4.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads (int) : Number of threads.

    Returns:
        numpy.ndarray : Gaussian gradient magnitude array. The array is the same shape as `input`.
    """
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = inp.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)

    cdef void *_inp
    cdef int inp_tn = np.PyArray_TYPE(inp)
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
    return check_array(out, inp_tn)

def bar_positions(double x0, double x1, double b_dx, double rd, long seed):
    """Generate a coordinate array of randomized barcode's bar positions.

    Args:
        x0 (float) : Barcode's lower bound along the x axis [um].
        x1 (float) : Barcode's upper bound along the x axis [um].
        b_dx (float) : Average bar's size [um].
        rd (float) : Random deviation of barcode's bar positions (0.0 - 1.0).
        seed (int) : Seed used for pseudo random number generation.

    Returns:
        numpy.ndarray : Array of barcode's bar coordinates.
    """
    cdef np.npy_intp size = 2 * (<np.npy_intp>((x1 - x0) / 2 / b_dx) + 1) if x1 > x0 else 0
    cdef np.npy_intp *dims = [size,]
    cdef np.ndarray[double] bars = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_bars = <double *>np.PyArray_DATA(bars)
    if size:
        with nogil:
            barcode_bars(_bars, size, x0, b_dx, rd, seed)
    return bars

cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray layers, complex t0,
                                   complex t1, double sigma, unsigned num_threads):
    x_arr = check_array(x_arr, np.NPY_FLOAT64)
    layers = check_array(layers, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef int ndim = x_arr.ndim
    cdef np.npy_intp *dims = x_arr.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_COMPLEX128)

    cdef np.npy_intp isize = np.PyArray_SIZE(x_arr)
    cdef np.npy_intp lsize = np.PyArray_SIZE(layers)
    cdef complex *_out = <complex *>np.PyArray_DATA(out)
    cdef double *_x = <double *>np.PyArray_DATA(x_arr)
    cdef double *_lyrs = <double *>np.PyArray_DATA(layers)
    with nogil:
        fail = ml_profile(_out, _x, isize, _lyrs, lsize, t0, t1, sigma, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def barcode_profile(np.ndarray x_arr not None, np.ndarray bars not None, double bulk_atn,
                    double bar_atn, double bar_sigma, unsigned num_threads=1):
    r"""Return an array of barcode's transmission profile calculated
    at `x_arr` coordinates.

    Args:
        x_arr (numpy.ndarray) : Array of the coordinates, where the transmission
            coefficients are calculated [um].    
        bars (numpy.ndarray) : Coordinates of barcode's bar positions [um].
        bulk_atn (float) : Barcode's bulk attenuation coefficient (0.0 - 1.0).
        bar_atn (float) : Barcode's bar attenuation coefficient (0.0 - 1.0).
        bar_sigma (float) : Inter-diffusion length [um].
        num_threads (int) : Number of threads used in the calculations.
    
    Returns:
        numpy.ndarray : Array of barcode's transmission profiles.

    Notes:
        Barcode's transmission profile is given by:
        
        .. math::
            t_b(x) = t_{air} + \frac{t_1 - t_{air}}{2}
            \left( \tanh\left(\frac{x - x^b_1}{\sigma_b} \right) +
            \tanh\left(\frac{x^b_N - x}{\sigma_b}\right) \right)\\
            - \frac{t_2 - t_1}{2}\sum_{n = 1}^{(N - 1) / 2} 
            \left( \tanh\left(\frac{x - x^b_{2 n}}{\sigma_b}\right) + 
            \tanh\left(\frac{x^b_{2 n + 1} - x}{\sigma_b}\right) \right)
        
        where :math:`t_1 = \sqrt{1 - A_{bulk}}`, :math:`t_2 = \sqrt{1 - A_{bulk} - A_{bar}}`
        are the transmission coefficients for the bulk and the bars of the sample,
        :math:`x^b_n` is a set of bars coordinates, and :math:`\sigma_b` is the
        inter-diffusion length.
    """
    cdef complex t0 = sqrt(1.0 - bulk_atn)
    cdef complex t1 = sqrt(1.0 - bulk_atn - bar_atn)
    return ml_profile_wrapper(x_arr, bars, t0, t1, bar_sigma, num_threads)

def mll_profile(np.ndarray x_arr not None, np.ndarray layers not None, complex t0,
                complex t1, double sigma, unsigned num_threads=1):
    r"""Return an array of MLL's transmission profile calculated
    at `x_arr` coordinates.

    Args:
        x_arr (numpy.ndarray) : Array of the coordinates, where the transmission
            coefficients are calculated [um].    
        layers (numpy.ndarray) : Coordinates of MLL's layers positions [um].
        t0 (complex) : Fresnel transmission coefficient for the first material of
            MLL's bilayer.
        t1 (complex) : Fresnel transmission coefficient for the first material of
            MLL's bilayer.
        sigma (float) : Inter-diffusion length [um].
        num_threads (int) : Number of threads used in the calculations.
    
    Returns:
        numpy.ndarray : Array of barcode's transmission profiles.

    Notes:
        MLL's transmission profile is given by:
        
        .. math::
            t_{MLL}(x) = t_{air} + \frac{t_1 - t_{air}}{2}
            \left( \tanh\left(\frac{x - x^b_1}{\sigma_b} \right) +
            \tanh\left(\frac{x^b_N - x}{\sigma_b}\right) \right)\\
            + \frac{t_2 - t_1}{2}\sum_{n = 1}^{(N - 1) / 2} 
            \left( \tanh\left(\frac{x - x^b_{2 n}}{\sigma_b}\right) + 
            \tanh\left(\frac{x^b_{2 n + 1} - x}{\sigma_b}\right) \right)
        
        where :math:`t_1`, :math:`t_2` are the transmission coefficients of the
        MLL's bilayers, :math:`x^b_n` is a set of bilayer coordinates,
        and :math:`\sigma_b` is the inter-diffusion length.
    """
    return ml_profile_wrapper(x_arr, layers, t0, t1, sigma, num_threads)

def make_frames(np.ndarray pfx not None, np.ndarray pfy not None, double dx, double dy,
                tuple shape, long seed, unsigned num_threads=1):
    """Generate intensity frames from one-dimensional intensity profiles (`pfx`,
    `pfy`) and white-field profiles (`wfx`, `wfy`). Intensity profiles resized into
    the shape of a frame. Poisson noise is applied if `seed` is non-negative.

    Args:
        pfx (numpy.ndarray) : Intensity profile along the x axis.
        pfy (numpy.ndarray) : Intensity profile along the y axis.
        dx (float) : Sampling interval along the x axis [um].
        dy (float) : Sampling interval along the y axis [um].
        shape (Tuple[int, int]) : Shape of the detector array.
        seed (int) : Seed for pseudo-random number generation.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Intensity frames.
    """
    pfx = check_array(pfx, np.NPY_FLOAT64)
    pfy = check_array(pfy, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef np.npy_intp *oshape = [pfx.shape[0], <np.npy_intp>(shape[0]), <np.npy_intp>(shape[1])]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(3, oshape, np.NPY_FLOAT64)
    cdef unsigned long *_ishape = [<unsigned long>(pfx.shape[0]), <unsigned long>(pfy.shape[0]),
                                   <unsigned long>(pfx.shape[1])]
    cdef unsigned long *_oshape = [<unsigned long>(oshape[0]), <unsigned long>(oshape[1]), <unsigned long>(oshape[2])]
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_pfx = <double *>np.PyArray_DATA(pfx)
    cdef double *_pfy = <double *>np.PyArray_DATA(pfy)
    with nogil:
        fail = frames(_out, _pfx, _pfy, dx, dy, _ishape, _oshape, seed, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def median(np.ndarray data not None, np.ndarray mask=None, int axis=0, unsigned num_threads=1):
    """Calculate a median along the `axis`.

    Args:
        data (numpy.ndarray) : Intensity frames.
        mask (numpy.ndarray) : Bad pixel mask.
        axis (int) : Array axis along which median values are calculated.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Array of medians along the given axis.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, data.shape, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)
        if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
            raise ValueError('mask and data arrays must have identical shapes')

    cdef unsigned long *_dims = <unsigned long *>data.shape

    cdef np.npy_intp *odims = <np.npy_intp *>malloc((ndim - 1) * sizeof(np.npy_intp))
    if odims is NULL:
        raise MemoryError('not enough memory')
    cdef int i
    for i in range(axis):
        odims[i] = data.shape[i]
    for i in range(axis + 1, ndim):
        odims[i - 1] = data.shape[i]

    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim - 1, odims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_ulong, num_threads)
        else:
            raise TypeError(f'data argument has incompatible type: {str(data.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    free(odims)
    return out

def median_filter(np.ndarray data not None, object size not None, np.ndarray mask=None,
                  str mode='reflect', double cval=0.0, unsigned num_threads=1):
    """Calculate a median along the `axis`.

    Args:
        data (numpy.ndarray) : Intensity frames.
        size (numpy.ndarray) : Gives the shape that is taken from the input array, at every
            element position, to define the input to the filter function. We adjust size to
            the number of dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
        mask (Optional[numpy.ndarray]) : Bad pixel mask.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the `cval`
              parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by replicating
              the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by reflecting
              about the center of the last pixel. This mode is also sometimes referred to as
              whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by reflecting
              about the edge of the last pixel. This mode is also sometimes referred to as
              half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping around
              to the opposite edge.
        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Filtered array. Has the same shape as `input`.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    cdef np.npy_intp *dims = data.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    cdef np.ndarray fsize = normalize_sequence(size, ndim, np.NPY_INTP)
    cdef unsigned long *_fsize = <unsigned long *>np.PyArray_DATA(fsize)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _mode, _cval, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _mode, _cval, compare_ulong, num_threads)
        else:
            raise TypeError(f'data argument has incompatible type: {str(data.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out