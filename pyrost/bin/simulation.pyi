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

def rsc_wp(wft: np.ndarray, dx0: float, dx: float, z: float, wl: float, axis: int=-1,
           backend: str='numpy', num_threads: int=1) -> np.ndarray:
    r"""Wavefront propagator based on Rayleigh-Sommerfeld convolution
    method [RSC]_. Propagates a wavefront `wft` by `z` distance downstream.
    You can choose between 'fftw' and 'numpy' backends for FFT calculations.

    Args:
        wft : Initial wavefront.
        dx0 : Sampling interval at the plane upstream [um].
        dx : Sampling interval at the plane downstream [um].
        z : Propagation distance [um].
        wl : Incoming beam's wavelength [um].
        axis : Axis of `wft` array along which the calculation is performed.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') library
            for the FFT  implementation.
        num_threads : Number of threads used in the calculations.

    Returns:
        Propagated wavefront.

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
    ...

def fraunhofer_wp(wft: np.ndarray, dx0: float, dx: float, z: float, wl: float, axis: int=-1,
                  backend: str='numpy', num_threads: int=1) -> np.ndarray:
    r"""Fraunhofer diffraction propagator. Propagates a wavefront `wft` by `z`
    distance downstream. You can choose between 'fftw' and 'numpy' backends for
    FFT calculations.

    Args: 
        wft : Initial wavefront.
        dx0 : Sampling interval at the plane upstream [um].
        dx : Sampling interval at the plane downstream [um].
        z : Propagation distance [um].
        wl : Incoming beam's wavelength [um].
        axis : Axis of `wft` array along which the calculation is performed.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') library
            for the FFT  implementation.
        num_threads : Number of threads used in the calculations.

    Returns:
        Propagated wavefront.

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

def bar_positions(x0: float, x1: float, b_dx: float, rd: float, seed: int) -> np.ndarray:
    """Generate a coordinate array of randomized barcode's bar positions.

    Args:
        x0 : Barcode's lower bound along the x axis [um].
        x1 : Barcode's upper bound along the x axis [um].
        b_dx : Average bar's size [um].
        rd : Random deviation of barcode's bar positions (0.0 - 1.0).
        seed : Seed used for pseudo random number generation.

    Returns:
        Array of barcode's bar coordinates.
    """
    ...

def barcode_profile(x_arr: np.ndarray, bars: np.ndarray, bulk_atn: float,
                    bar_atn: float, bar_sigma: float, num_threads: int=1) -> np.ndarray:
    r"""Return an array of barcode's transmission profile calculated
    at `x_arr` coordinates.

    Args:
        x_arr : Array of the coordinates, where the transmission coefficients
            are calculated [um].
        bars : Coordinates of barcode's bar positions [um].
        bulk_atn : Barcode's bulk attenuation coefficient (0.0 - 1.0).
        bar_atn : Barcode's bar attenuation coefficient (0.0 - 1.0).
        bar_sigma : Inter-diffusion length [um].
        num_threads : Number of threads used in the calculations.

    Returns:
        Array of barcode's transmission profiles.

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
    ...

def mll_profile(x_arr: np.ndarray, layers: np.ndarray, t0: complex,
                t1: complex, sigma: float, num_threads: int=1) -> np.ndarray:
    r"""Return an array of MLL's transmission profile calculated
    at `x_arr` coordinates.

    Args:
        x_arr : Array of the coordinates, where the transmission coefficients
            are calculated [um].
        layers : Coordinates of MLL's layers positions [um].
        t0 : Fresnel transmission coefficient for the first material of MLL's bilayer.
        t1 : Fresnel transmission coefficient for the first material of MLL's bilayer.
        sigma : Inter-diffusion length [um].
        num_threads : Number of threads used in the calculations.

    Returns:
        Array of barcode's transmission profiles.

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
    ...

def make_frames(pfx: np.ndarray, pfy: np.ndarray, dx: float, dy: float,
                shape: Tuple[int, int], seed: int, num_threads: int=1) -> np.ndarray:
    """Generate intensity frames from one-dimensional intensity profiles (`pfx`,
    `pfy`) and white-field profiles (`wfx`, `wfy`). Intensity profiles resized into
    the shape of a frame. Poisson noise is applied if `seed` is non-negative.

    Args:
        pfx : Intensity profile along the x axis.
        pfy : Intensity profile along the y axis.
        dx : Sampling interval along the x axis [um].
        dy : Sampling interval along the y axis [um].
        shape : Shape of the detector array.
        seed : Seed for pseudo-random number generation.
        num_threads : Number of threads used in the calculations.

    Returns:
        Intensity frames.
    """
    ...

def median(inp: np.ndarray, mask: Optional[np.ndarray]=None, axis: int=0, num_threads: int=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        axis : Array axis along which median values are calculated.
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
