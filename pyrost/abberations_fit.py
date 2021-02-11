"""Routines for model regression based on nonlinear
least-squares algorithm. :class:`pyrost.AbberationsFit` fit the
lens' abberations profile with the polynomial function
using nonlinear least-squares algorithm.

Examples
--------
Generate a :class:`pyrost.AbberationsFit` object from
:class:`pyrost.STData` container object `st_data` as follows:

>>> fit_obj = AbberationsFit.import_data(st_data)

Fit a pixel abberations profile with a third
order polynomial:

>>> fit = fit_obj.fit(max_order=3)
>>> print(fit)
{'alpha': -0.04718488324311934,
 'fit': array([-9.03305155e-04,  2.14699128e+00, -1.17287983e+03]),
 'ph_fit': array([-9.81298119e-07,  3.49854945e-03, -3.82244504e+00,  1.26179239e+03]),
 'rel_err': array([0.02331385, 0.01966198, 0.01679612]),
 'r_sq': 0.9923840802879347}
"""
import numpy as np
from scipy.optimize import least_squares
from .data_container import DataContainer

class LeastSquares:
    """Basic nonlinear least-squares fit class.
    Based on :func:`scipy.optimize.least_squares`.

    See Also
    --------
    :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    @classmethod
    def model(cls, fit, x, roi):
        """Return values of polynomial model function.

        Parameters
        ----------
        x : numpy.ndarray
            Array of x coordinates.

        Returns
        -------
        numpy.ndarray
            Array of polynomial function values.
        """
        return np.polyval(fit, x[roi[0]:roi[1]])

    @classmethod
    def errors(cls, fit, x, y, roi):
        """Return an array of model residuals.

        Parameters
        ----------
        fit : numpy.ndarray
            Array of fit coefficients.
        x : numpy.ndarray
            Array of x coordinates.
        y : numpy.ndarray
            Array of y coordinates.

        Returns
        -------
        numpy.ndarray
            Array of model residuals.
        """
        return cls.model(fit, x, roi) - y[roi[0]:roi[1]]

    @classmethod
    def fit(cls, x, y, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy', roi=None):
        """Fit `x`, `y` with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Parameters
        ----------
        x : numpy.ndarray
            Array of x coordinates.
        y : numpy.ndarray
            Array of y coordinates.
        max_order : int, optional
            Maximum order of the polynomial model function.
        xtol : float, optional
            Tolerance for termination by the change of the independent variables.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
            Determines the loss function. The following keyword values are
            allowed:

            * 'linear : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy'' (default) : ``rho(z) = ln(1 + z)``. Severely weakens
              outliers influence, but may cause difficulties in optimization
              process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        Returns
        -------
        fit : numpy.ndarray
            Array of fit coefficients.
        err : numpy.ndarray
            Vector of errors of the `fit` fit coefficients.
        """
        if roi is None:
            roi = (0, x.size)
        fit = least_squares(cls.errors, np.zeros(max_order + 1),
                            loss=loss, args=(x, y, roi), xtol=xtol, ftol=ftol)
        r_sq = 1 - np.sum(cls.errors(fit.x, x, y, roi)**2) / np.sum((y[roi[0]:roi[1]].mean() - y[roi[0]:roi[1]])**2)
        if np.linalg.det(fit.jac.T.dot(fit.jac)):
            cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
            err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
        else:
            err = 0
        return fit.x, err, r_sq

class AbberationsFit(DataContainer):
    """Lens' abberations profile model regression using
    nonlinear least-squares algorithm. :class:`AbberationsFit`
    is capable of fitting lens' pixel abberations, deviation 
    angles, and phase profile with polynomial function.
    Based on :func:`scipy.optimise.least_squares`.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of the attributes' data specified in `attr_set`
        and `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.

    Raises
    ------
    ValueError
        If an attribute specified in `attr_set` has not been provided.

    Notes
    -----
    Necessary attributes:

    * defocus : Defocus distance [m].
    * distance : Sample-to-detector distance [m].
    * phase : Abberations phase profile [rad].
    * pixel_abberations : Pixel abberations profile [pixels].
    * pixel_size : Pixel's size [m].
    * wavelength : Incoming beam's wavelength [m].

    See Also
    --------
    :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    attr_set = {'defocus', 'distance', 'phase', 'pixel_abberations',
                'pixel_size', 'roi', 'wavelength'}
    fs_lookup = {'defocus': 'defocus_fs', 'pixel_size': 'x_pixel_size'}
    ss_lookup = {'defocus': 'defocus_ss', 'pixel_size': 'y_pixel_size'}

    def __init__(self, **kwargs):
        super(AbberationsFit, self).__init__(**kwargs)
        self.pixels = np.arange(self.roi[0], self.roi[1])
        self.pix_ap = self.pixel_size / self.distance
        self.theta_abberations = self.pixel_abberations * self.pix_ap

    @classmethod
    def import_data(cls, st_data, center=0, axis=1):
        """Return a new :class:`AbberationsFit` object
        with all the necessary data attributes imported from
        the :class:`STData` container object `st_data`.

        Parameters
        ----------
        st_data : STData
            :class:`STData` container object.
        axis : int, optional
            Detector's axis (0 - slow axis, 1 - fast axis).

        Returns
        -------
        AbberationsFit
            A new :class:`AbberationsFit` object.
        """
        data_dict = {attr: st_data.get(attr) for attr in cls.attr_set}
        if axis == 0:
            data_dict.update({attr: st_data.get(data_attr) for attr, data_attr in cls.ss_lookup.items()})
        elif axis == 1:
            data_dict.update({attr: st_data.get(data_attr) for attr, data_attr in cls.fs_lookup.items()})
        else:
            raise ValueError('invalid axis value: {:d}'.format(axis))
        data_dict['pixel_abberations'] = data_dict['pixel_abberations'][axis].mean(axis=1 - axis)
        data_dict['phase'] = data_dict['phase'].mean(axis=1 - axis)
        data_dict['roi'] = data_dict['roi'][2 * axis:2 * (axis + 1)] - center
        return cls(**data_dict)

    def model(self, fit, roi=None):
        """Return the polynomial function values of
        lens' deviation angles fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens` pixel abberations fit coefficients.
        roi : iterable, optional
            Region of interest. Full region if `roi` is None.

        Returns
        -------
        numpy.ndarray
            Array of polynomial function values.
        """
        if roi is None:
            roi = (0, self.pixels.size)
        return LeastSquares.model(fit, self.pixels, roi)

    def pix_to_phase(self, fit, roi=None):
        """Convert fit coefficients from pixel
        abberations fit to abberations phase fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens' pixel abberations fit coefficients.
        roi : iterable, optional
            Region of interest. Full region if `roi` is None.

        Returns
        -------
        numpy.ndarray
            Lens` phase abberations fit coefficients.
        """
        if roi is None:
            roi = (0, self.pixels.size)
        nfit = np.zeros(fit.size + 1)
        nfit[:-1] = 2 * np.pi * fit * self.pix_ap**2 * self.defocus / self.wavelength
        nfit[:-1] /= np.arange(1, fit.size + 1)[::-1]
        nfit[-1] = -self.model(nfit, roi).mean()
        return nfit

    def phase_to_pix(self, ph_fit):
        """Convert fit coefficients from pixel
        abberations fit to abberations phase fit.

        Parameters
        ----------
        ph_fit : numpy.ndarray
            Lens` phase abberations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Lens' pixel abberations fit coefficients.
        """
        fit = self.wavelength * ph_fit[:-1] / (2 * np.pi * self.pix_ap**2 * self.defocus)
        fit *= np.arange(1, ph_fit.size)[::-1]
        return fit

    def fit(self, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy', roi=None):
        """Fit lens' pixel abberations with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Parameters
        ----------
        max_order : int, optional
            Maximum order of the polynomial model function.
        xtol : float, optional
            Tolerance for termination by the change of the independent
            variables.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
            Determines the loss function. The following keyword values
            are allowed:

            * 'linear' : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' (default) : ``rho(z) = ln(1 + z)``. Severely weakens
              outliers influence, but may cause difficulties in optimization
              process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.
        roi : iterable, optional
            Region of interest. Full region if `roi` is None.

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * alpha : Third order abberations ceofficient [rad/mrad^3].
            * fit : Array of the polynomial function coefficients of the
              pixel abberations fit.
            * ph_fit : Array of the polynomial function coefficients of
              the phase abberations fit.
            * rel_err : Vector of relative errors of the fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        fit, err, r_sq = LeastSquares.fit(x=self.pixels, y=self.pixel_abberations, roi=roi,
                                          max_order=max_order, xtol=xtol, ftol=ftol, loss=loss)
        ph_fit = self.pix_to_phase(fit, roi)
        if ph_fit.size >= 4:
            alpha = ph_fit[-4] / self.pix_ap**3 * 1e-9
        else:
            alpha = None
        return {'alpha': alpha, 'fit': fit, 'ph_fit': ph_fit, 'rel_err': np.abs(err / fit), 'r_sq': r_sq}

    def fit_phase(self, max_order=3, xtol=1e-14, ftol=1e-14, loss='linear', roi=None):
        """Fit lens' phase abberations with polynomial function using
        :func:`scipy.optimise.least_squares`.

        Parameters
        ----------
        max_order : int, optional
            Maximum order of the polynomial model function.
        xtol : float, optional
            Tolerance for termination by the change of the independent
            variables.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
            Determines the loss function. The following keyword values
            are allowed:

            * 'linear' : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' (default) : ``rho(z) = ln(1 + z)``. Severely weakens
              outliers influence, but may cause difficulties in optimization
              process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.
        roi : iterable, optional
            Region of interest. Full region if `roi` is None.

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * alpha : Third order abberations ceofficient [rad/mrad^3].
            * fit : Array of the polynomial function coefficients of the
              pixel abberations fit.
            * ph_fit : Array of the polynomial function coefficients of
              the phase abberations fit.
            * rel_err : Vector of relative errors of the fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        ph_fit, err, r_sq = LeastSquares.fit(x=self.pixels, y=self.phase, roi=roi,
                                             max_order=max_order, xtol=xtol, ftol=ftol, loss=loss)
        fit = self.phase_to_pix(ph_fit)
        if ph_fit.size >= 4:
            alpha = ph_fit[-4] / self.pix_ap**3 * 1e-9
        else:
            alpha = None
        return {'alpha': alpha, 'fit': fit, 'ph_fit': ph_fit, 'rel_err': np.abs(err / ph_fit)[:-1], 'r_sq': r_sq}
