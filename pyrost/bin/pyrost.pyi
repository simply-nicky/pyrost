from typing import Tuple, Sequence
import numpy as np

def KR_reference(I_n: np.ndarray, W: np.ndarray, u: np.ndarray, di: np.ndarray,
                 dj: np.ndarray, ds_y: float, ds_x: float, hval: float,
                 return_nm0: bool=True, num_threads: int=1) -> Tuple[np.ndarray, int, int]:
    r"""Generate an unabberated reference image of the sample based on the pixel
    mapping `u` and the measured data `I_n` using the Kernel regression.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        u : The discrete geometrical mapping of the detector
            plane to the reference image.
        di : Initial sample's translations along the vertical
            detector axis in pixels.
        dj : Initial sample's translations along the horizontal
            detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x : Sampling interval of reference image in pixels along the
            horizontal axis.
        hval : Gaussian kernel bandwidth in pixels.
        return_nm0 : If True, also returns the lower bounds (`n0`, `m0`)
            of the reference image in pixels.
        num_threads : Number of threads.

    Returns:
        A tuple of three elements (`I0`, `n0`, `m0`). The elements are the following:

        * `I0` : Reference image array.
        * `n0` : The lower bounds of the vertical detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.
        * `m0` : The lower bounds of the horizontal detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.

    Notes:
        The pixel mapping `u` maps the intensity measurements from the detector
        plane to the reference plane as follows:

        .. math::
            i_{ref}[n, i, j] = u[0, i, j] + di[n], \; j_{ref}[n, i, j] = u[1, i, j] + dj[n]

        The reference image profile :math:`I_{ref}[ii, jj]` is obtained with the
        kernel regression extimator as follows:

        .. math::

            I_{ref}[i, j] = \frac{\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n,
            i^{\prime}, j^{\prime}], j - j_{ref}[n, i^{\prime}, j^{\prime}], h]
            \; W[i^{\prime}, j^{\prime}] I[n, i^{\prime}, j^{\prime}]}
            {\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n, i^{\prime}, j^{\prime}], 
            j - j_{ref}[n, i^{\prime}, j^{\prime}], h] \; W^2[i^{\prime}, j^{\prime}]}

        where :math:`K[i, j, h] = \frac{1}{\sqrt{2 \pi}} \exp(-\frac{i^2 + j^2}{h})`
        is the Gaussian kernel.
    """
    ...

def LOWESS_reference(I_n: np.ndarray, W: np.ndarray, u: np.ndarray, di: np.ndarray,
                     dj: np.ndarray, ds_y: float, ds_x: float, hval: float,
                     return_nm0: bool=True, num_threads: int=1) -> Tuple[np.ndarray, int, int]:
    r"""Generate an unabberated reference image of the sample based on the
    pixel mapping `u` and the measured data `I_n` using the Local Weighted
    Linear Regression (LOWESS).

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        u : The discrete geometrical mapping of the detector
            plane to the reference image.
        di : Initial sample's translations along the vertical
            detector axis in pixels.
        dj : Initial sample's translations along the horizontal
            detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x : Sampling interval of reference image in pixels along the
            horizontal axis.
        hval : Gaussian kernel bandwidth in pixels.
        return_nm0 : If True, also returns the lower bounds (`n0`, `m0`)
            of the reference image in pixels.
        num_threads : Number of threads.

    Returns:
        A tuple of three elements (`I0`, `n0`, `m0`). The elements are the following:

        * `I0` : Reference image array.
        * `n0` : The lower bounds of the vertical detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.
        * `m0` : The lower bounds of the horizontal detector axis of the reference
          image at the reference frame in pixels. Only provided if `return_nm0` is
          True.

    Notes:
        The pixel mapping `u` maps the intensity measurements from the
        detector plane to the reference plane as follows:

        .. math::
            i_{ref}[n, i, j] = u[0, i, j] + di[n], \; j_{ref}[n, i, j] = u[1, i, j] + dj[n]

        The reference image profile :math:`I_{ref}[ii, jj]` is obtained
        with the LOWESS regression extimator as follows:

        .. math::

            I_{ref}[i, j] = \frac{\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n,
            i^{\prime}, j^{\prime}], j - j_{ref}[n, i^{\prime}, j^{\prime}], h]
            \; r_{IW}[n, a^{IW}, b^{IW}_i, b^{IW}_j]}
            {\sum_{n, i^{\prime}, j^{\prime}} K[i - i_{ref}[n, i^{\prime}, j^{\prime}], 
            j - j_{ref}[n, i^{\prime}, j^{\prime}], h] \;
            r_{WW}[n, a^{WW}, b^{WW}_i, b^{WW}_j]}

        where :math:`r_{IW}[n, a, b_i, b_j]` and :math:`r_{WW}[n, a, b_i, b_j]`
        are the residuals at the pixel :math:`i, j` with linear coefficients
        defined with the least squares approach:

        .. math::

            r_{IW}[n, a, b_i, b_j] = I[n, i^{\prime}, j^{\prime}] W[i^{\prime}, j^{\prime}]
            - a - b_i (i - i_{ref}[n, i^{\prime}, j^{\prime}) - b_j
            (j - j_{ref}[n, i^{\prime}, j^{\prime})

        .. math::

            r_{WW}[n, a, b_i, b_j] = W^2[i^{\prime}, j^{\prime}]
            - a - b_i (i - i_{ref}[n, i^{\prime}, j^{\prime}) - b_j
            (j - j_{ref}[n, i^{\prime}, j^{\prime})

        and :math:`K[i, j, h] = \frac{1}{\sqrt{2 \pi}} \exp(-\frac{i^2 + j^2}{h})`
        is the Gaussian kernel.
    """
    ...

def pm_gsearch(I_n: np.ndarray, W: np.ndarray, I0: np.ndarray, sigma: np.ndarray, u0: np.ndarray,
               di: np.ndarray, dj: np.ndarray, search_window: Sequence[float], grid_size: Sequence[int],
               ds_y: float, ds_x: float, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Update the pixel mapping by minimizing mean-squared-error
    (MSE). Perform a grid search within the search window of `sw_y`,
    `sw_x` size along the vertical and fast axes accordingly in order to
    minimize the MSE at each point of the detector grid separately.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        I0 : Reference image of the sample.
        sigma : The standard deviation of `I_n`.
        u : The discrete geometrical mapping of the detector
            plane to the reference image.
        di : Initial sample's translations along the vertical
            detector axis in pixels.
        dj : Initial sample's translations along the horizontal
            detector axis in pixels.
        search_window : Search window size in pixels along the vertical detector
            axis.
        grid_size :  Grid size along one of the detector axes. The grid
            shape is then (grid_size, grid_size).
        ds_y : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x : Sampling interval of reference image in pixels along the
            horizontal axis.
        num_threads : Number of threads.

    Returns:
        A tuple of two elements (`u`, `derr`). The elements are the following:

        * `u` : Updated pixel mapping array.
        * `derr` : Error decrease for each pixel in the detector grid.

    Notes:
        The error metric as a function of pixel mapping displacements
        is given by:

        .. math::

            \varepsilon_{pm}[i, j, i^{\prime}, j^{\prime}] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] + i^{\prime} - di[n],
            u[1, i, j] + j^{\prime} - dj[n]]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def pm_rsearch(I_n: np.ndarray, W: np.ndarray, sigma: np.ndarray, I0: np.ndarray,
               u0: np.ndarray, di: np.ndarray, dj: np.ndarray, search_window: Sequence[float],
               n_trials: int, seed: int, ds_y: float, ds_x: float,
               num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Update the pixel mapping by minimizing mean-squared-error (MSE).
    Perform a random search within the search window of `sw_y`, `sw_x` size
    along the vertical and fast axes accordingly in order to minimize the MSE
    at each point of the detector grid separately.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Initial sample's translations along the vertical detector axis in pixels.
        dj : Initial sample's translations along the horizontal detector axis in pixels.
        search_window : Search window size in pixels along the vertical detector axis.
        n_trials : Number of points generated at each pixel of the detector grid.
        seed : Specify seed for the random number generation.
        ds_y : Sampling interval of reference image in pixels along the
            vertical axis.
        ds_x : Sampling interval of reference image in pixels along the
            horizontal axis.
        num_threads : Number of threads.

    Returns:
        A tuple of two elements (`u`, `derr`). The elements are the following:

        * `u` : Updated pixel mapping array.
        * `derr` : Error decrease for each pixel in the detector grid.

    Notes:
        The error metric as a function of pixel mapping displacements
        is given by:

        .. math::

            \varepsilon_{pm}[i, j, i^{\prime}, j^{\prime}] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] + i^{\prime} - di[n],
            u[1, i, j] + j^{\prime} - dj[n]]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def pm_devolution(I_n: np.ndarray, W: np.ndarray, sigma: np.ndarray, I0: np.ndarray,
                  u0: np.ndarray, di: np.ndarray, dj: np.ndarray, search_window: Sequence[float],
                  pop_size: int, n_iter: int, seed: int, ds_y: float, ds_x: float,
                  F: float=0.75, CR: float=0.7, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Update the pixel mapping by minimizing mean-squared-error (MSE). Perform
    a differential evolution within the search window of `sw_y`, `sw_x` size along
    the vertical and fast axes accordingly in order to minimize the MSE at each
    point of the detector grid separately.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Initial sample's translations along the vertical detector axis in pixels.
        dj : Initial sample's translations along the horizontal detector axis in pixels.
        search_window : Search window size in pixels along the vertical detector axis.
        pop_size : The total population size. Must be greater or equal to 4.
        n_iter : The maximum number of generations over which the entire population is evolved.
        seed : Specify seed for the random number generation.
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        F : The mutation constant. In the literature this is also known as
            differential weight. If specified as a float it should be in the
            range [0, 2].
        CR : The recombination constant, should be in the range [0, 1]. In
            the literature this is also known as the crossover probability.
        num_threads : Number of threads.

    Returns:
        A tuple of two elements (`u`, `derr`). The elements are the following:

        * `u` : Updated pixel mapping array.
        * `sgm`: Updated scaling map.
        * `derr` : Error decrease for each pixel in the detector grid.

    Notes:
        The error metric as a function of pixel mapping displacements
        is given by:

        .. math::

            \varepsilon_{pm}[i, j, i^{\prime}, j^{\prime}] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] + i^{\prime} - di[n],
            u[1, i, j] + j^{\prime} - dj[n]]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def tr_gsearch(I_n: np.ndarray, W: np.ndarray, sigma: np.ndarray, I0: np.ndarray,
               u: np.ndarray, di: np.ndarray, dj: np.ndarray, sw_y: float, sw_x: float,
               grid_size: int, ds_y: float, ds_x: float, loss: str='Huber',
               num_threads: int=1) -> np.ndarray:
    r"""Update the sample pixel translations by minimizing total mean-squared-error
    (:math:$MSE_{total}$). Perform a grid search within the search window of
    `sw_y` size in pixels for sample translations along the vertical axis, and
    of `sw_x` size in pixels for sample translations along the horizontal axis in
    order to minimize the total MSE.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Initial sample's translations along the vertical detector axis in pixels.
        dj : Initial sample's translations along the horizontal detector axis in pixels.
        sw_y : Search window size in pixels along the vertical detector axis.
        sw_x : Search window size in pixels along the horizontal detector axis.
        grid_size : Grid size along one of the detector axes. The grid shape is then
            (grid_size, grid_size).
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        num_threads : Number of threads.

    Returns:
        Updated sample pixel translations.

    Notes:
        The error metric as a function of sample shifts is given by:

        .. math::

            \varepsilon_{tr}[n, di^{\prime}, dj^{\prime}] = \frac{1}{Y X}
            \sum_{i = 0}^Y \sum_{j = 0}^Y f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] - di[n] - di^{\prime},
            u[1, i, j] - dj[n] - dj^{\prime}]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def pm_errors(I_n: np.ndarray, W: np.ndarray, sigma: np.ndarray, I0: np.ndarray,
              u: np.ndarray, di: np.ndarray, dj: np.ndarray, ds_y: float, ds_x: float,
              num_threads: int=1) -> np.ndarray:
    r"""Return the residuals for the pixel mapping fit.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the
            reference image.
        di : Sample's translations along the vertical detector axis in pixels.
        dj : Sample's translations along the fast detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        num_threads: Number of threads.

    Returns:
        Residual profile of the pixel mapping fit.

    See Also:
        :func:`pyrost.bin.pm_gsearch` : Description of error metric which
        is being minimized.

    Notes:
        The error metric is given by:

        .. math::

            \varepsilon_{pm}[i, j] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] - di[n], u[1, i, j] - dj[n]]}{\sigma}
            \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def pm_total_error(I_n: np.ndarray, W: np.ndarray, sigma: np.ndarray, I0: np.ndarray,
                   u: np.ndarray, di: np.ndarray, dj: np.ndarray, ds_y: float, ds_x: float,
                   num_threads: int=1) -> float:
    r"""Return the mean residual for the pixel mapping fit.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Sample's translations along the vertical detector axis in pixels.
        dj : Sample's translations along the fast detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        num_threads : Number of threads.

    Returns:
        Mean residual value of the pixel mapping fit.

    See Also:
        :func:`pyrost.bin.pm_gsearch` : Description of error metric which
        is being minimized.

    Notes:
        The error metric is given by:

        .. math::

            \varepsilon_{pm}[i, j] = \frac{1}{N}
            \sum_{n = 0}^N f\left( \frac{I[n, i, j] - W[i, j]
            I_{ref}[u[0, i, j] - di[n], u[1, i, j] - dj[n]]}{\sigma}
            \right)

        where :math:`f(x)` is the Huber loss function.
    """
    ...

def ref_errors(I_n: np.ndarray, W: np.ndarray, I0: np.ndarray, u: np.ndarray,
               di: np.ndarray, dj: np.ndarray, ds_y: float,
               ds_x: float, hval: float, num_threads: int=1) -> np.ndarray:
    r"""Return the residuals for the reference image regression.

    Args:
        I_n : Measured intensity frames.
        W : Measured frames' white-field.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Sample's translations along the vertical detector axis in pixels.
        dj: Sample's translations along the fast detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        hval : Kernel bandwidth in pixels.
        num_threads : Number of threads.

    Returns:
        Residuals array of the reference image regression.

    Notes:
        The pixel mapping `u` maps the intensity measurements from the
        detector plane to the reference plane as follows:

        .. math::
            i_{ref}[n, i, j] = u[0, i, j] + di[n], \; j_{ref}[n, i, j] = u[1, i, j] + dj[n]

        The error metric is given by:

        .. math::

            \varepsilon_{ref}[i, j] = \sum_{n, i^{\prime}, j^{\prime}}
            K[i - i_{ref}[n, i^{\prime}, j^{\prime}], 
            j - j_{ref}[n, i^{\prime}, j^{\prime}], h] \;
            f\left( \frac{I_n[n, i^{\prime}, j^{\prime}] -
            W[i^{\prime}, j^{\prime}] I_{ref}[i, j]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function
        and :math:`K[i, j, h] = \frac{1}{\sqrt{2 \pi}} \exp(-\frac{i^2 + j^2}{h})`
        is the Gaussian kernel.

    See Also:
        :func:`pyrost.bin.KR_reference` : Description of the reference image
        estimator.
    """
    ...

def ref_total_error(I_n: np.ndarray, W: np.ndarray, I0: np.ndarray, u: np.ndarray,
                    di: np.ndarray, dj: np.ndarray, ds_y: float, ds_x: float, hval: float,
                    num_threads: int=1) -> float:
    r"""Return the mean residual for the reference image regression.

    Args:
        I_n :  Measured intensity frames.
        W : Measured frames' white-field.
        sigma : The standard deviation of `I_n`.
        I0 : Reference image of the sample.
        u : The discrete geometrical mapping of the detector plane to the reference image.
        di : Sample's translations along the vertical detector axis in pixels.
        dj: Sample's translations along the fast detector axis in pixels.
        ds_y : Sampling interval of reference image in pixels along the vertical axis.
        ds_x : Sampling interval of reference image in pixels along the horizontal axis.
        hval : Kernel bandwidth in pixels.
        num_threads : Number of threads.

    Returns:
        Mean residual value.

    Notes:
        The pixel mapping `u` maps the intensity measurements from the
        detector plane to the reference plane as follows:

        .. math::
            i_{ref}[n, i, j] = u[0, i, j] + di[n], \; j_{ref}[n, i, j] = u[1, i, j] + dj[n]

        The error metric is given by:

        .. math::

            \varepsilon_{ref}[i, j] = \sum_{n, i^{\prime}, j^{\prime}}
            K[i - i_{ref}[n, i^{\prime}, j^{\prime}], 
            j - j_{ref}[n, i^{\prime}, j^{\prime}], h] \;
            f\left( \frac{I_n[n, i^{\prime}, j^{\prime}] -
            W[i^{\prime}, j^{\prime}] I_{ref}[i, j]}{\sigma} \right)

        where :math:`f(x)` is the Huber loss function
        and :math:`K[i, j, h] = \frac{1}{\sqrt{2 \pi}} \exp(-\frac{i^2 + j^2}{h})`
        is the Gaussian kernel.

    See Also:
        :func:`pyrost.bin.KR_reference` : Description of the reference image
        estimator.
    """
    ...

def ct_integrate(sy_arr: np.ndarray, sx_arr: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Perform the Fourier Transform wavefront reconstruction [FTI]_ with
    antisymmetric derivative integration [ASDI]_.

    Args:
        sx_arr :  of gradient values along the horizontal axis.
        sy_arr : Array of gradient values along the vertical axis.
        num_threads : Number of threads.

    Returns:
        Reconstructed wavefront.

    References:
        .. [FTI] C. Kottler, C. David, F. Pfeiffer, and O. Bunk,
                "A two-directional approach for grating based
                differential phase contrast imaging using hard x-rays,"
                Opt. Express 15, 1175-1181 (2007).
        .. [ASDI] Pierre Bon, Serge Monneret, and Benoit Wattellier,
                "Noniterative boundary-artifact-free wavefront
                reconstruction from its derivatives," Appl. Opt. 51,
                5698-5704 (2012).
    """
    ...
