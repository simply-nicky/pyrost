from typing import List, Optional, Tuple, Union
import numpy as np

IntArray = Union[int, List[int], Tuple[int]]
FloatArray = Union[float, List[float], Tuple[float]]

def next_fast_len(target: int) -> int:
    r"""Find the next fast size of input data to fft, for zero-padding, etc. FFT algorithms
    gain their speed by a recursive divide and conquer strategy. This relies on efficient
    functions for small prime factors of the input length. Thus, the transforms are fastest
    when using composites of the prime factors handled by the fft implementation. If there
    are efficient functions for all radices <= n, then the result will be a number x >= target
    with only prime factors < n. (Also known as n-smooth numbers)

    Args:
        target : Length to start searching from. Must be a positive integer.

    Raises:
        ValueError : If `backend` is invalid.
        ValueError : If `target` is negative.

    Returns:
        The smallest fast length greater than or equal to `target`.
    """
    ...

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: Optional[IntArray] = None,
                 num_threads: int=1) -> np.ndarray:
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Args:
        array : Input array.
        kernel : Kernel array.
        axis : Array axis along which convolution is performed.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        A multi-dimensional array containing the discrete linear convolution of `array`
        with `kernel`.
    """
    ...

def gaussian_kernel(sigma: FloatArray, order: IntArray=0, truncate: float=4.0) -> np.ndarray:
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
                    mode: str='reflect', cval: float=0.0, truncate: float=4.0,
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
