cimport numpy as np
import numpy as np
import cython
from libc.string cimport memcmp
from libc.math cimport log
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
    if not np.PyArray_IS_C_CONTIGUOUS(array):
        array = np.PyArray_GETCONTIGUOUS(array)
    cdef int tn = np.PyArray_TYPE(array)
    if tn != type_num:
        array = np.PyArray_Cast(array, type_num)
    return array

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
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

def next_fast_len(target: cython.uint, backend: str='numpy') -> cython.uint:
    r"""Find the next fast size of input data to fft, for zero-padding, etc.
    FFT algorithms gain their speed by a recursive divide and conquer strategy.
    This relies on efficient functions for small prime factors of the input length.
    Thus, the transforms are fastest when using composites of the prime factors handled
    by the fft implementation. If there are efficient functions for all radices <= n,
    then the result will be a number x >= target with only prime factors < n. (Also
    known as n-smooth numbers)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    backend : {'fftw', 'numpy'}, optional
        Find n-smooth number for the FFT implementation from the specified
        library.

    Returns
    -------
    n : int
        The smallest fast length greater than or equal to `target`.
    """
    if target < 0:
        raise ValueError('Target length must be positive')
    if backend == 'fftw':
        return next_fast_len_fftw(target)
    elif backend == 'numpy':
        return good_size(target)
    else:
        raise ValueError('{:s} is invalid backend'.format(backend))

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: cython.int=-1,
                 mode: str='constant', cval: cython.double=0.0, backend: str='numpy',
                 num_threads: cython.uint=1) -> np.ndarray:
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    kernel : numpy.ndarray
        Kernel array.
    axis : int, optional
        Array axis along which convolution is performed.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'constant'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    out : numpy.ndarray
        A multi-dimensional array containing the discrete linear
        convolution of `array` with `kernel`.
    """
    array = check_array(array, np.NPY_FLOAT64)
    kernel = check_array(kernel, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = array.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(array)
    cdef double *_krn = <double *>np.PyArray_DATA(kernel)
    with nogil:
        if backend == 'fftw':
            fail = fft_convolve_fftw(_out, _inp, ndim, _dims, _krn, ksize, axis, _mode, cval, num_threads)
        elif backend == 'numpy':
            fail = fft_convolve_np(_out, _inp, ndim, _dims, _krn, ksize, axis, _mode, cval, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def rsc_wp(wft: np.ndarray, dx0: cython.double, dx: cython.double, z: cython.double,
           wl: cython.double, axis: cython.int=-1, backend: str='numpy',
           num_threads: cython.uint=1) -> np.ndarray:
    r"""Wavefront propagator based on Rayleigh-Sommerfeld convolution
    method [RSC]_. Propagates a wavefront `wft` by `z` distance
    downstream. You can choose between 'fftw' and 'numpy' backends for FFT
    calculations. 'fftw' backend supports multiprocessing.

    Parameters
    ----------
    wft : numpy.ndarray
        Initial wavefront.
    dx0 : float
        Sampling interval at the plane upstream [um].
    dx : float
        Sampling interval at the plane downstream [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].
    axis : int, optional
        Axis of `wft` array along which the calculation is
        performed.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads: int, optional
        Number of threads used in calculation. Only 'fftw' backend
        supports it.

    Returns
    -------
    out : numpy.ndarray
        Propagated wavefront.

    Raises
    ------
    RuntimeError
        If 'numpy' backend exits with eror during the calculation.
    ValueError
        If `backend` option is invalid.

    Notes
    -----
    The Rayleigh–Sommerfeld diffraction integral transform is defined as:

    .. math::
        u^{\prime}(x^{\prime}) = \frac{z}{j \sqrt{\lambda}} \int_{-\infty}^{+\infty}
        u(x) \mathrm{exp} \left[-j k r(x, x^{\prime}) \right] dx
    
    with

    .. math::
        r(x, x^{\prime}) = \left[ (x - x^{\prime})^2 + z^2 \right]^{1 / 2}

    References
    ----------
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

def fraunhofer_wp(wft: np.ndarray, dx0: cython.double, dx: cython.double,
                  z: cython.double, wl: cython.double, axis: cython.int=-1,
                  backend: str='numpy', num_threads: cython.uint=1) -> np.ndarray:
    r"""Fraunhofer diffraction propagator. Propagates a wavefront `wft` by
    `z` distance downstream. You can choose between 'fftw' and 'numpy'
    backends for FFT calculations. 'fftw' backend supports multiprocessing.

    Parameters
    ----------
    wft : numpy.ndarray
        Initial wavefront.
    dx0 : float
        Sampling interval at the plane upstream [um].
    dx : float
        Sampling interval at the plane downstream [um].
    z : float
        Propagation distance [um].
    wl : float
        Incoming beam's wavelength [um].
    axis : int, optional
        Axis of `wft` array along which the calculation is
        performed.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads: int, optional
        Number of threads used in calculation. Only 'fftw' backend
        supports it.

    Returns
    -------
    out : numpy.ndarray
        Propagated wavefront.

    Raises
    ------
    RuntimeError
        If 'numpy' backend exits with eror during the calculation.
    ValueError
        If `backend` option is invalid.

    Notes
    -----
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

def gaussian_kernel(sigma: double, order: cython.uint=0, truncate: cython.double=4.) -> np.ndarray:
    """Discrete Gaussian kernel.
    
    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian kernel.
    order : int, optional
        The order of the filter. An order of 0 corresponds to convolution with a
        Gaussian kernel. A positive order corresponds to convolution with that
        derivative of a Gaussian. Default is 0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    
    Returns
    -------
    krn : np.ndarray
        Gaussian kernel.
    """
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0])
    return out

def gaussian_filter(inp: np.ndarray, sigma: object, order: object=0, mode: str='reflect',
                    cval: cython.double=0., truncate: cython.double=4., backend: str='numpy',
                    num_threads: cython.uint=1) -> np.ndarray:
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Parameters
    ----------
    inp : np.ndarray
        The input array.
    sigma : float or list of floats
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian
        filter are given for each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    order : int or list of ints, optional
        The order of the filter along each axis is given as a sequence of integers, or as
        a single number. An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a Gaussian.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'reflect'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.
    
    Returns
    -------
    out : np.ndarray
        Returned array of same shape as `input`.
    """
    inp = check_array(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.ndarray orders = normalize_sequence(order, ndim, np.NPY_UINT32)

    cdef int fail = 0
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            fail = gauss_filter(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate, num_threads, fft_convolve_fftw)
        elif backend == 'numpy':
            fail = gauss_filter(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate, num_threads, fft_convolve_np)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_gradient_magnitude(inp: np.ndarray, sigma: object, mode: str='reflect',
                                cval: cython.double=0., truncate: cython.double=4.,
                                backend: str='numpy', num_threads: cython.uint=1) -> np.ndarray:
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Parameters
    ----------
    inp : np.ndarray
        The input array.
    sigma : float or list of floats
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'reflect'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.
    """
    inp = check_array(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    
    cdef int fail = 0
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            fail = gauss_grad_mag(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate, num_threads, fft_convolve_fftw)
        elif backend == 'numpy':
            fail = gauss_grad_mag(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate, num_threads, fft_convolve_np)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def bar_positions(x0: cython.double, x1: cython.double, b_dx: cython.double,
                  rd: cython.double, seed: cython.ulong) -> np.ndarray:
    """Generate a coordinate array of randomized barcode's bar positions.

    Parameters
    ----------
    x0 : float
        Barcode's lower bound along the x axis [um].
    x1 : float
        Barcode's upper bound along the x axis [um].
    b_dx : float
        Average bar's size [um].
    rd : float
        Random deviation of barcode's bar positions (0.0 - 1.0).
    seed : int
        Seed used for pseudo random number generation.

    Returns
    -------
    bx_arr : numpy.ndarray
        Array of barcode's bar coordinates.
    """
    cdef np.npy_intp size = 2 * (<np.npy_intp>((x1 - x0) / 2 / b_dx) + 1) if x1 > x0 else 0
    cdef np.npy_intp *dims = [size,]
    cdef np.ndarray[double] bars = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_bars = <double *>np.PyArray_DATA(bars)
    if size:
        with nogil:
            barcode_bars(_bars, size, x0, b_dx, rd, seed)
    return bars

cdef np.ndarray ml_profile_wrapper(np.ndarray x_arr, np.ndarray layers, complex mt0,
                                   complex mt1, complex mt2, double sigma, unsigned num_threads):
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
        fail = ml_profile(_out, _x, isize, _lyrs, lsize, mt0, mt1, mt2, sigma, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def barcode_profile(x_arr: np.ndarray, bars: np.ndarray, bulk_atn: cython.double,
                    bar_atn: cython.double, bar_sigma: cython.double,
                    num_threads: cython.uint) -> np.ndarray:
    r"""Return an array of barcode's transmission profile calculated
    at `x_arr` coordinates.

    Parameters
    ----------
    x_arr : numpy.ndarray
        Array of the coordinates, where the transmission coefficients
        are calculated [um].    
    bars : numpy.ndarray
        Coordinates of barcode's bar positions [um].
    bulk_atn : float
        Barcode's bulk attenuation coefficient (0.0 - 1.0).
    bar_atn : float
        Barcode's bar attenuation coefficient (0.0 - 1.0).
    bar_sigma : float
        Bar's blurriness width [um].
    num_threads : int, optional
        Number of threads.
    
    Returns
    -------
    bar_profile : numpy.ndarray
        Array of barcode's transmission profiles.

    Notes
    -----
    Barcode's transmission profile is simulated with a set
    of error functions:
    
    .. math::
        \begin{multline}
            T_{b}(x) = 1 - \frac{T_{bulk}}{2} \left\{
            \mathrm{erf}\left[ \frac{x - x_{bar}[0]}{\sqrt{2} \sigma} \right] +
            \mathrm{erf}\left[ \frac{x_{bar}[n - 1] - x}{\sqrt{2} \sigma} \right]
            \right\} -\\
            \frac{T_{bar}}{4} \sum_{i = 1}^{n - 2} \left\{
            2 \mathrm{erf}\left[ \frac{x - x_{bar}[i]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{bar}[i - 1]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{bar}[i + 1]}{\sqrt{2} \sigma} \right]
            \right\}
        \end{multline}
    
    where :math:`x_{bar}` is an array of bar coordinates.
    """
    cdef complex mt0 = -1j * log(1 - bulk_atn)
    cdef complex mt1 = -1j * log(1 - bar_atn)
    return ml_profile_wrapper(x_arr, bars, mt0, mt1, 0., bar_sigma, num_threads)

def mll_profile(x_arr: np.ndarray, layers: np.ndarray, complex mt0,
                complex mt1, sigma: cython.double, num_threads: cython.uint) -> np.ndarray:
    r"""Return an array of MLL's transmission profile calculated
    at `x_arr` coordinates.

    Parameters
    ----------
    x_arr : numpy.ndarray
        Array of the coordinates, where the transmission coefficients
        are calculated [um].    
    layers : numpy.ndarray
        Coordinates of MLL's layers positions [um].
    mt0 : complex
        Fresnel transmission coefficient for the first material of MLL's
        bilayer.
    mt1 : complex
        Fresnel transmission coefficient for the first material of MLL's
        bilayer.
    sigma : float
        Interdiffusion length [um].
    num_threads : int, optional
        Number of threads.
    
    Returns
    -------
    bar_profile : numpy.ndarray
        Array of barcode's transmission profiles.

    Notes
    -----
    MLL's transmission profile is simulated with a set
    of error functions:
    
    .. math::
        \begin{multline}
            T_{b}(x) = 1 - \frac{T_{bulk}}{2} \left\{
            \mathrm{erf}\left[ \frac{x - x_{lyr}[0]}{\sqrt{2} \sigma} \right] +
            \mathrm{erf}\left[ \frac{x_{lyr}[n - 1] - x}{\sqrt{2} \sigma} \right]
            \right\} -\\
            \frac{T_{bar}}{4} \sum_{i = 1}^{n - 2} \left\{
            2 \mathrm{erf}\left[ \frac{x - x_{lyr}[i]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{lyr}[i - 1]}{\sqrt{2} \sigma} \right] -
            \mathrm{erf}\left[ \frac{x - x_{lyr}[i + 1]}{\sqrt{2} \sigma} \right]
            \right\}
        \end{multline}
    
    where :math:`x_{lyr}` is an array of MLL's layer coordinates.
    """
    return ml_profile_wrapper(x_arr, layers, 0., mt0, mt1, sigma, num_threads)

def make_frames(pfx: np.ndarray, pfy: np.ndarray, dx: cython.double, dy: cython.double,
                shape: tuple, seed: cython.long, num_threads: cython.uint) -> np.ndarray:
    """Generate intensity frames from one-dimensional intensity profiles (`pfx`,
    `pfy`) and whitefield profiles (`wfx`, `wfy`). Intensity profiles resized into
    the shape of a frame. Poisson noise is applied if `seed` is non-negative.

    Parameters
    ----------
    pfx : numpy.ndarray
        Intensity profile along the x axis.
    pfy : numpy.ndarray
        Intensity profile along the y axis.
    dx : float
        Sampling interval along the x axis [um].
    dy : float
        Sampling interval along the y axis [um].
    shape : tuple
        Shape of the detector array.
    seed : int, optional
        Seed for pseudo-random number generation.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    frames : numpy.ndarray
        Intensity frames.
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

def median(data: np.ndarray, mask: np.ndarray, axis: cython.int=0,
           num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    mask : numpy.ndarray
        Bad pixel mask.
    axis : int, optional
        Array axis along which median values are calculated.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    wfield : numpy.ndarray
        Whitefield.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)
    mask = check_array(mask, np.NPY_BOOL)

    cdef int ndim = data.ndim
    if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
        raise ValueError('mask and data arrays must have identical shapes')
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
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
    cdef void *_out = np.PyArray_DATA(out)
    cdef void *_data = np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_float, num_threads)
        elif type_num == np.NPY_INT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    free(odims)
    return out

def median_filter(data: np.ndarray, mask: np.ndarray, size: cython.uint=3, axis: cython.int=0,
                  mode: str='reflect', cval: cython.double=0., num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    mask : numpy.ndarray
        Bad pixel mask.
    size : int, optional
        `size` gives the shape that is taken from the input array, at every element position,
        to define the input to the filter function. Default is 3.
    axis : int, optional
        Array axis along which median values are calculated.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'reflect'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    wfield : numpy.ndarray
        Whitefield.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)
    mask = check_array(mask, np.NPY_BOOL)

    cdef int ndim = data.ndim
    if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
        raise ValueError('mask and data arrays must have identical shapes')
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp *dims = data.shape
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
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, axis, size, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, axis, size, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, axis, size, _mode, _cval, compare_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    return out
