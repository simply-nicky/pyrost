from typing import List, Optional, Tuple, Union
import numpy as np

def next_fast_len(target: int, backend: str='numpy') -> int:
    r"""Find the next fast size of input data to fft, for zero-padding, etc. FFT algorithms
    gain their speed by a recursive divide and conquer strategy. This relies on efficient
    functions for small prime factors of the input length. Thus, the transforms are fastest
    when using composites of the prime factors handled by the fft implementation. If there
    are efficient functions for all radices <= n, then the result will be a number x >= target
    with only prime factors < n. (Also known as n-smooth numbers)

    Args:
        target : Length to start searching from. Must be a positive integer.
        backend : Find n-smooth number for the FFT implementation from the numpy ('numpy') or
            FFTW ('fftw') library.

    Raises:
        ValueError : If `backend` is invalid.
        ValueError : If `target` is negative.

    Returns:
        The smallest fast length greater than or equal to `target`.
    """
    ...

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: int=-1,
                 mode: str='constant', cval: float=0.0, backend: str='numpy',
                 num_threads: int=1) -> np.ndarray:
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Args:
        array : Input array.
        kernel : Kernel array.
        axis : Array axis along which convolution is performed.
        mode : The mode parameter determines how the input array is extended
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

        cval :  Value to fill past edges of input if mode is 'constant'. Default
            is 0.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') library for the FFT
            implementation.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        A multi-dimensional array containing the discrete linear convolution of `array`
        with `kernel`.
    """
    ...

def gaussian_kernel(sigma: float, order: int=0, truncate: float=4.0) -> np.ndarray:
    """Discrete Gaussian kernel.

    Args:
        sigma : Standard deviation for Gaussian kernel.
        order : The order of the filter. An order of 0 corresponds to convolution with
            a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian. Default is 0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        Gaussian kernel.
    """
    ...

def gaussian_filter(inp: np.ndarray, sigma: Union[float, List[float]], order: Union[int, List[int]]=0,
                    mode: str='reflect', cval: float=0.0, truncate: float=4.0, backend: str='numpy',
                    num_threads: int=1) -> np.ndarray:
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Args:
        inp : The input array.
        sigma : Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        order : The order of the filter along each axis is given as a
            sequence of integers, or as a single number. An order of 0 corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian.
        mode : The mode parameter determines how the input array is extended when the
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

        cval : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads : Number of threads.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        Returned array of the same shape as `inp`.
    """
    ...

def gaussian_gradient_magnitude(inp: np.ndarray, sigma: Union[float, List[float]], mode: str='reflect',
                                cval: float=0.0, truncate: float=4.0, backend: str='numpy',
                                num_threads: int=1) -> np.ndarray:
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Args:
        inp : The input array.
        sigma : Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        mode : The mode parameter determines how the input array is extended when the
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

        cval : Value to fill past edges of input if mode is `constant`. Default is 0.0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads : Number of threads.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        Gaussian gradient magnitude array. The array is the same shape as `inp`.
    """
    ...

def median(inp: np.ndarray, mask: Optional[np.ndarray]=None,
           axis: Union[int, List[int], Tuple[int]]=0, num_threads: int=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        axis : Array axes along which median values are calculated.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `mask` and `inp` have different shapes.
        TypeError : If `inp` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Array of medians along the given axis.
    """
    ...

def median_filter(inp: np.ndarray, size: Optional[Union[int, Tuple[int, ...]]]=None,
                  footprint: Optional[np.ndarray]=None, mask: Optional[np.ndarray]=None,
                  inp_mask: Optional[np.ndarray]=None, mode: str='reflect', cval: float=0.0,
                  num_threads: int=1) -> np.ndarray:
    """Calculate a multidimensional median filter.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        size : See footprint, below. Ignored if footprint is given.
        footprint :  Either size or footprint must be defined. size gives the shape that is taken
            from the input array, at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly) a shape, but also
            which of the elements within this shape will get passed to the filter function. Thus
            size=(n, m) is equivalent to footprint=np.ones((n, m)). We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape (10, 10, 10), and
            size is 2, then the actual size used is (2, 2, 2). When footprint is given, size is
            ignored.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        inp_mask : Input mask. Median takes into account only the `inp` values, where `inp_mask`
            is True. `inp_mask` is equal to `mask` by default.
        mode : The mode parameter determines how the input array is extended when the
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
        cval : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : When neither `size` nor `footprint` are provided.
        TypeError : If `data` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Filtered array. Has the same shape as `inp`.
    """
    ...

def robust_mean(inp: np.ndarray, axis: Union[int, List[int], Tuple[int]]=0, r0: float=0.0,
                r1: float=0.5, n_iter: int=12, lm: float=9.0, num_threads: int=1) -> np.ndarray:
    """Calculate a mean along the `axis` by robustly fitting a Gaussian to input vector [RFG]_.
    The algorithm performs `n_iter` times the fast least kth order statistics (FLkOS [FLKOS]_)
    algorithm to fit a gaussian to data.
    
    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis : Array axes along which median values are calculated.
        r0 : A lower bound guess of ratio of inliers. We'd like to make a sample out of worst
            inliers from data points that are between `r0` and `r1` of sorted residuals.
        r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as high as you are
            sure the ratio of data is inlier.
        n_iter : Number of iterations of fitting a gaussian with the FLkOS algorithm.
        lm : How far (normalized by STD of the Gaussian) from the mean of the Gaussian, data is
            considered inlier.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `mask` and `inp` have different shapes.
        TypeError : If `inp` has incompatible type.
        RuntimeError : If C backend exited with error.

    References:
        .. [RFG] A. Sadri et al., "Automatic bad-pixel mask maker for X-ray pixel detectors with
                application to serial crystallography", J. Appl. Cryst. 55, 1549-1561 (2022).

        .. [FLKOS] A. Bab-Hadiashar and R. Hoseinnezhad, "Bridging Parameter and Data Spaces for
                  Fast Robust Estimation in Computer Vision," Digital Image Computing: Techniques
                  and Applications, pp. 1-8 (2008).

    Returns:
        Array of robust mean.
    """
    ...
