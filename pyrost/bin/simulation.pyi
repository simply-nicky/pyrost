from typing import Tuple
import numpy as np

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
