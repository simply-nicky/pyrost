"""Routines for model regression based on nonlinear
least-squares algorithm. :class:`pyrost.AberrationsFit` fit the
lens' aberrations profile with the polynomial function
using nonlinear least-squares algorithm.

Examples:
    Generate a :class:`pyrost.AberrationsFit` object from
    :class:`pyrost.STData` container object `data` as follows:

    .. code-block:: python

        >>> fit_obj = data.get_fit()
        >>> fit = fit_obj.fit(max_order=3)
        >>> print(fit)
        {'c_3': -0.04718488324311934, 'c_4':  0.,
        'fit': array([-9.03305155e-04,  2.14699128e+00, -1.17287983e+03]),
        'ph_fit': array([-9.81298119e-07,  3.49854945e-03, -3.82244504e+00,  1.26179239e+03]),
        'rel_err': array([0.02331385, 0.01966198, 0.01679612]),
        'r_sq': 0.9923840802879347}
"""
from __future__ import annotations
from weakref import ReferenceType
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np
from scipy.optimize import least_squares
from .data_container import DataContainer, dict_to_object

class LeastSquares:
    """Basic nonlinear least-squares fit class. Based on
    :func:`scipy.optimize.least_squares`.

    See Also:
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    @classmethod
    def model(cls, fit: np.ndarray, x: np.ndarray, roi: Iterable) -> np.ndarray:
        """Return values of polynomial model function.

        Args:
            fit : Array of fit coefficients.
            x : Array of x coordinates.
            roi : Region of interest in the detector plane.

        Returns:
            Array of polynomial function values.
        """
        return np.polyval(fit, x[roi[0]:roi[1]])

    @classmethod
    def errors(cls, fit: np.ndarray, x: np.ndarray, y: np.ndarray,
               roi: Iterable) -> np.ndarray:
        """Return an array of model residuals.

        Args:
            fit : Array of fit coefficients.
            x : Array of x coordinates.
            y : Array of y coordinates.
            roi : Region of interest in the detector plane.

        Returns:
            Array of model residuals.
        """
        return cls.model(fit, x, roi) - y[roi[0]:roi[1]]

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, max_order: int=2,
            xtol: float=1e-14, ftol: float=1e-14, loss: str='cauchy',
            roi: Optional[Iterable]=None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit `x`, `y` with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Args:
            x : Array of x coordinates.
            y : Array of y coordinates.
            max_order : Maximum order of the polynomial model function.
            xtol : Tolerance for termination by the change of the independent variables.
            ftol : Tolerance for termination by the change of the cost function.
            loss : Determines the loss function. The following keyword values are
                allowed:

                * `linear`: ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * `soft_l1` : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * `huber` : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * `cauchy` (default) : ``rho(z) = ln(1 + z)``. Severely weakens
                  outliers influence, but may cause difficulties in optimization
                  process.
                * `arctan` : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.

        Returns:
            A tuple of three elements ('fit', 'err', 'r_sq'). The elements are the
            following:

            * `fit` : Array of fit coefficients.
            * `err` : Vector of errors of the `fit` fit coefficients.
            * `r_sq` : ``R**2`` goodness of fit.
        """
        if roi is None:
            roi = (0, x.size)
        fit = least_squares(cls.errors, np.zeros(max_order + 1),
                            loss=loss, args=(x, y, roi), xtol=xtol, ftol=ftol)
        r_sq = 1 - np.sum(cls.errors(fit.x, x, y, roi)**2) / \
               np.sum((y[roi[0]:roi[1]].mean() - y[roi[0]:roi[1]])**2)
        if np.linalg.det(fit.jac.T.dot(fit.jac)):
            cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
            err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
        else:
            err = 0
        return fit.x, err, r_sq

class AberrationsFit(DataContainer):
    """Least squares optimizer for the lens aberrations' profiles.
    :class:`AberrationsFit` is capable of fitting lens' pixel
    aberrations, deviation  angles, and phase profile with polynomial
    function. Based on :func:`scipy.optimise.least_squares`.

    Attributes:
        kwargs : Necessary and optional attributes specified in the notes
            section.
        det_ap : Aperture angle of a single pixel in detector plane.
        ref_ap : Aperture angle of a single pixel in reference plane.

    Notes:
        **Necessary attributes**:

        * defocus : Defocus distance [m].
        * distance : Sample-to-detector distance [m].
        * pixels : Pixel coordinates [pixels].
        * pixel_aberrations : Pixel aberrations profile [pixels].
        * pixel_size : Pixel's size [m].
        * wavelength : Incoming beam's wavelength [m].

        **Optional attributes**:

        * roi : Region of interest.
        * thetas : Scattering angles [rad].
        * theta_ab : angular displacement profile [rad].
        * phase : aberrations phase profile [rad].

    See Also:
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    attr_set = {'defocus', 'distance', 'parent', 'pixels', 'pixel_aberrations',
                'pixel_size', 'wavelength'}
    init_set = {'det_ap', 'phase', 'ref_ap', 'roi', 'thetas', 'theta_ab'}
    x_lookup = {'defocus': 'defocus_x', 'pixel_size': 'x_pixel_size'}
    y_lookup = {'defocus': 'defocus_y', 'pixel_size': 'y_pixel_size'}

    # Necessary attributes
    parent              : ReferenceType
    defocus             : float
    distance            : float
    pixels              : np.ndarray
    pixel_aberrations   : np.ndarray
    pixel_size          : float
    wavelength          : float

    # Automatically generated attributes
    det_ap              : float
    phase               : np.ndarray
    ref_ap              : float
    roi                 : np.ndarray
    thetas              : np.ndarray
    theta_ab            : np.ndarray

    def __init__(self, parent: ReferenceType, **kwargs: Union[float, np.ndarray]) -> None:
        """
        Args:
            parent : The Speckle tracking data container, from which the object
                is derived.
            kwargs : Necessary and optional attributes specified in the notes
                section.

        Raises:
            ValueError : If any of the necessary attributes has not been provided.
        """
        super(AberrationsFit, self).__init__(parent=parent, **kwargs)

        self._init_functions(det_ap=lambda: self.pixel_size / self.distance,
                             ref_ap=lambda: np.abs(self.det_ap * self.defocus / self.distance),
                             roi=lambda: np.array([0, self.pixels.size]),
                             thetas=lambda: self.pixels * self.det_ap,
                             theta_ab=lambda: self.pixel_aberrations * self.ref_ap,
                             phase=lambda: self.wnumber * np.cumsum(self.theta_ab * self.pixel_size))

        self._init_attributes()
        self.phase -= self.phase.mean()

    @property
    def wnumber(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @dict_to_object
    def crop_data(self, roi: Iterable) -> AberrationsFit:
        """Return a new :class:`AberrationsFit` object with the updated `roi`.

        Args:
            roi : Region of interest in the detector plane.

        Returns:
            New :class:`AberrationsFit` object with the updated `roi`.
        """
        return {'roi': np.asarray(roi, dtype=int)}

    @dict_to_object
    def remove_linear_term(self, fit: Optional[np.ndarray]=None, xtol: float=1e-14,
                           ftol: float=1e-14, loss: str='cauchy') -> AberrationsFit:
        """Return a new :class:`AberrationsFit` object with the linear term
        removed from `pixel_aberrations` profile.

        Args:
            fit : Fit coefficients of a first order polynomial. Inferred from
                `pixel_aberrations` by fitting a line if None.
            xtol : Tolerance for termination by the change of the independent
                variables.
            ftol : Tolerance for termination by the change of the cost function.
            loss : Determines the loss function. The following keyword values
                are allowed:

                * `linear` : ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * `soft_l1` : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * `huber` : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * `cauchy` (default) : ``rho(z) = ln(1 + z)``. Severely weakens
                  outliers influence, but may cause difficulties in optimization
                  process.
                * `arctan` : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.

        Returns:
            New :class:`AberrationsFit` object with the updated `pixel_aberrations`
            and `phase`.
        """
        if fit is None:
            fit = LeastSquares.fit(x=self.pixels, y=self.pixel_aberrations,
                                   roi=self.roi, max_order=1, xtol=xtol,
                                   ftol=ftol, loss=loss)[0]
        pixel_aberrations = self.pixel_aberrations - self.model(fit)
        return {'pixel_aberrations': pixel_aberrations, 'theta_ab': None, 'phase': None}

    @dict_to_object
    def update_center(self, center: float) -> AberrationsFit:
        """Return a new :class:`AberrationsFit` object with the pixels
        centered around `center`.

        Args:
            center : Index of the zerro scattering angle or direct
                beam pixel.

        Returns:
            New :class:`AberrationsFit` object with the updated `pixels`,
            `phase`, and `pixel_aberrations`.
        """
        if center <= self.pixels[0]:
            pixels = self.pixels - center
            return {'pixels': pixels}
        elif center >= self.pixels[-1]:
            pixels = center - self.pixels
            idxs = np.argsort(self.pixels)
            return {'pixels': pixels[idxs], 'thetas': None,
                    'pixel_aberrations': -self.pixel_aberrations[idxs], 'theta_ab': None}
        else:
            raise ValueError('Origin must be outside of the region of interest')

    @dict_to_object
    def update_phase(self) -> AberrationsFit:
        """Return a new :class:`AberrationsFit` object with the updated `phase`.

        Returns:
            New :class:`AberrationsFit` object with the updated `phase`.
        """
        return {'theta_ab': None, 'phase': None}

    def get(self, attr: str, value: Optional[Any]=None) -> Any:
        """Return a dataset with `mask` and `roi` applied. Return `value`
        if the attribute is not found.

        Args:
            attr : Attribute to fetch.
            value : Return if `attr` is not found.

        Returns:
            `attr` dataset with `mask` and `roi` applied. `value` if
            `attr` is not found.
        """
        if attr in self:
            val = super(AberrationsFit, self).get(attr)
            if not val is None:
                if attr in ['phase', 'pixels', 'pixel_aberrations', 'theta_ab']:
                    val = val[self.roi[0]:self.roi[1]]
            return val

        return value

    def model(self, fit: np.ndarray) -> np.ndarray:
        """Return the polynomial function values of lens' deviation angles fit.

        Args:
            fit : Lens` pixel aberrations fit coefficients.

        Returns:
            Array of polynomial function values.
        """
        return LeastSquares.model(fit, self.pixels, [0, self.pixels.size])

    def pix_to_phase(self, fit: np.ndarray) -> np.ndarray:
        """Convert fit coefficients from pixel aberrations fit to aberrations
        phase fit.

        Args:
            fit : Lens' pixel aberrations fit coefficients.

        Returns:
            Lens` phase aberrations fit coefficients.
        """
        nfit = np.zeros(fit.size + 1)
        nfit[:-1] = self.wnumber * fit * self.ref_ap * self.pixel_size
        nfit[:-1] /= np.arange(1, fit.size + 1)[::-1]
        nfit[-1] = -self.model(nfit).mean()
        return nfit

    def phase_to_pix(self, ph_fit: np.ndarray) -> np.ndarray:
        """Convert fit coefficients from pixel aberrations fit to aberrations
        phase fit.

        Args:
            ph_fit : Lens` phase aberrations fit coefficients.

        Returns:
            Lens' pixel aberrations fit coefficients.
        """
        fit = ph_fit[:-1] * self.wavelength / (2.0 * np.pi * self.ref_ap * self.pixel_size)
        fit *= np.arange(1, ph_fit.size)[::-1]
        return fit

    def fit(self, max_order: int=2, xtol: float=1e-14, ftol: float=1e-14,
            loss: str='cauchy') -> Dict[str, Union[float, np.ndarray]]:
        """Fit lens' pixel aberrations with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Args:
            max_order : Maximum order of the polynomial model function.
            xtol : Tolerance for termination by the change of the independent
                variables.
            ftol : Tolerance for termination by the change of the cost function.
            loss : Determines the loss function. The following keyword values
                are allowed:

                * `linear` : ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * `soft_l1` : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * `huber` : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * `cauchy` (default) : ``rho(z) = ln(1 + z)``. Severely weakens
                  outliers influence, but may cause difficulties in optimization
                  process.
                * `arctan` : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.

        Returns:
            A dictionary with the fitting information. The following elements are
            defined inside:

            * `c_3` : Third order aberrations coefficient [rad / mrad^3].
            * `c_4` : Fourth order aberrations coefficient [rad / mrad^4].
            * `fit` : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * `ph_fit` : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * `rel_err` : Vector of relative errors of the fit coefficients.
            * `r_sq` : ``R**2`` goodness of fit.

        See Also:
            :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        fit, err, r_sq = LeastSquares.fit(x=self.pixels, y=self.pixel_aberrations,
                                          roi=self.roi, max_order=max_order,
                                          xtol=xtol, ftol=ftol, loss=loss)
        ph_fit = self.pix_to_phase(fit)
        c_3, c_4 = 0., 0.

        if ph_fit.size >= 4:
            c_3 = ph_fit[-4] * (self.distance / self.pixel_size)**3 * 1e-9

        if ph_fit.size >= 5:
            c_4 = ph_fit[-5] * (self.distance / self.pixel_size)**4 * 1e-12

        return {'c_3': c_3, 'c_4': c_4, 'fit': fit, 'ph_fit': ph_fit,
                'rel_err': np.abs(err / fit), 'r_sq': r_sq}

    def fit_phase(self, max_order: int=3, xtol: float=1e-14, ftol: float=1e-14,
                  loss: str='linear') -> Dict[str, Union[float, np.ndarray]]:
        """Fit lens' phase aberrations with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Args:
            max_order : Maximum order of the polynomial model function.
            xtol : Tolerance for termination by the change of the independent
                variables.
            ftol : Tolerance for termination by the change of the cost function.
            loss : Determines the loss function. The following keyword values
                are allowed:

                * `linear` : ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * `soft_l1` : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * `huber` : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * `cauchy` (default) : ``rho(z) = ln(1 + z)``. Severely weakens
                  outliers influence, but may cause difficulties in optimization
                  process.
                * `arctan` : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.

        Returns:
            A dictionary with the fitting information. The following elements are
            defined inside:

            * `c_3` : Third order aberrations coefficient [rad / mrad^3].
            * `c_4` : Fourth order aberrations coefficient [rad / mrad^4].
            * `fit` : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * `ph_fit` : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * `rel_err` : Vector of relative errors of the fit coefficients.
            * `r_sq` : ``R**2`` goodness of fit.

        See Also:
            :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        ph_fit, err, r_sq = LeastSquares.fit(x=self.pixels, y=self.phase, roi=self.roi,
                                             max_order=max_order, xtol=xtol, ftol=ftol,
                                             loss=loss)
        fit = self.phase_to_pix(ph_fit)
        c_3, c_4 = 0., 0.

        if ph_fit.size >= 4:
            c_3 = ph_fit[-4] * (self.distance / self.pixel_size)**3 * 1e-9

        if ph_fit.size >= 5:
            c_4 = ph_fit[-5] * (self.distance / self.pixel_size)**4 * 1e-12

        return {'c_3': c_3, 'c_4': c_4, 'fit': fit, 'ph_fit': ph_fit,
                'rel_err': np.abs(err / ph_fit)[:-1], 'r_sq': r_sq}
