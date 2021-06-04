"""Routines for model regression based on nonlinear
least-squares algorithm. :class:`pyrost.AberrationsFit` fit the
lens' aberrations profile with the polynomial function
using nonlinear least-squares algorithm.

Examples
--------
Generate a :class:`pyrost.AberrationsFit` object from
:class:`pyrost.STData` container object `st_data` as follows:

>>> fit_obj = AberrationsFit.import_data(st_data)

Fit a pixel aberrations profile with a third
order polynomial:

>>> fit = fit_obj.fit(max_order=3)
>>> print(fit)
{'c_3': -0.04718488324311934, 'c_4':  0.,
 'fit': array([-9.03305155e-04,  2.14699128e+00, -1.17287983e+03]),
 'ph_fit': array([-9.81298119e-07,  3.49854945e-03, -3.82244504e+00,  1.26179239e+03]),
 'rel_err': array([0.02331385, 0.01966198, 0.01679612]),
 'r_sq': 0.9923840802879347}
"""
import numpy as np
from scipy.optimize import least_squares
from .data_container import DataContainer, dict_to_object

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

class AberrationsFit(DataContainer):
    """Least squares optimizer for the lens aberrations' profiles.
    :class:`AberrationsFit` is capable of fitting lens' pixel
    aberrations, deviation  angles, and phase profile with polynomial
    function. Based on :func:`scipy.optimise.least_squares`.

    Parameters
    ----------
    st_data : STData
        The Speckle tracking data container, from which the object
        is derived.
    **kwargs : dict
        Necessary and optional attributes specified in the notes
        section.

    Attributes
    ----------
    **kwargs : dict
        Necessary and optional attributes specified in the notes
        section.
    det_ap : float
        Aperture angle of a single pixel in detector plane.
    ref_ap : float
        Aperture angle of a single pixel in reference plane.

    Raises
    ------
    ValueError
        If any of the necessary attributes has not been provided.

    Notes
    -----
    Necessary attributes:

    * defocus : Defocus distance [m].
    * distance : Sample-to-detector distance [m].
    * pixels : Pixel coordinates [pixels].
    * pixel_aberrations : Pixel aberrations profile [pixels].
    * pixel_size : Pixel's size [m].
    * wavelength : Incoming beam's wavelength [m].

    Optional attributes:

    * roi : Region of interest.
    * thetas : Scattering angles [rad].
    * theta_aberrations : angular displacement profile [rad].
    * phase : aberrations phase profile [rad].

    See Also
    --------
    :func:`scipy.optimize.least_squares` : Full nonlinear
        least-squares algorithm description.
    """
    attr_set = {'defocus', 'distance', 'pixels', 'pixel_aberrations',
                'pixel_size', 'wavelength'}
    init_set = {'roi', 'thetas', 'theta_aberrations', 'phase'}
    fs_lookup = {'defocus': 'defocus_fs', 'pixel_size': 'x_pixel_size'}
    ss_lookup = {'defocus': 'defocus_ss', 'pixel_size': 'y_pixel_size'}

    def __init__(self, st_data=None, **kwargs):
        self.__dict__['_reference'] = st_data
        super(AberrationsFit, self).__init__(**kwargs)
        self.det_ap = self.pixel_size / self.distance
        self.ref_ap = np.abs(self.det_ap * self.defocus / self.distance)
        if self.roi is None:
            self.roi = np.array([0, self.pixels.size])
        if self.thetas is None:
            self.thetas = self.pixels * self.det_ap
        if self.theta_aberrations is None:
            self.theta_aberrations = self.pixel_aberrations * self.ref_ap
        if self.phase is None:
            self.phase = np.cumsum(self.theta_aberrations * self.pixel_size)
            self.phase *= 2 * np.pi / self.wavelength
            self.phase -= self.phase.mean()
        if not self._reference is None:
            self._reference._ab_fits.append(self)

    @classmethod
    def import_data(cls, st_data, center=0, axis=1):
        """Return a new :class:`AberrationsFit` object
        with all the necessary data attributes imported from
        the :class:`STData` container object `st_data`.

        Parameters
        ----------
        st_data : STData
            :class:`STData` container object.
        center : int, optional
            Index of the zerro scattering angle or direct
            beam pixel.
        axis : int, optional
            Detector's axis (0 - slow axis, 1 - fast axis).

        Returns
        -------
        AberrationsFit
            A new :class:`AberrationsFit` object.
        """
        data_dict = {attr: st_data.get(attr) for attr in cls.attr_set}
        if axis == 0:
            data_dict.update({attr: st_data.get(data_attr)
                              for attr, data_attr in cls.ss_lookup.items()})
        elif axis == 1:
            data_dict.update({attr: st_data.get(data_attr)
                              for attr, data_attr in cls.fs_lookup.items()})
        else:
            raise ValueError('invalid axis value: {:d}'.format(axis))
        data_dict['defocus'] = np.abs(data_dict['defocus'])
        if center <= st_data.roi[2 * axis]:
            data_dict['pixels'] = np.arange(st_data.roi[2 * axis],
                                            st_data.roi[2 * axis + 1]) - center
            data_dict['pixel_aberrations'] = data_dict['pixel_aberrations'][axis].mean(axis=1 - axis)
        elif center >= st_data.roi[2 * axis - 1] - 1:
            data_dict['pixels'] = center - np.arange(st_data.roi[2 * axis],
                                                     st_data.roi[2 * axis + 1])
            idxs = np.argsort(data_dict['pixels'])
            data_dict['pixel_aberrations'] = -data_dict['pixel_aberrations'][axis].mean(axis=1 - axis)[idxs]
            data_dict['pixels'] = data_dict['pixels'][idxs]
        else:
            raise ValueError('Origin must be outside of the region of interest')
        return cls(st_data=st_data, **data_dict)

    @dict_to_object
    def crop_data(self, roi):
        """Return a new :class:`AberrationsFit` object with the updated `roi`.

        Parameters
        ----------
        roi : iterable
            Region of interest in the detector plane.

        Returns
        -------
        AberrationsFit
            New :class:`AberrationsFit` object with the updated `roi`.
        """
        return {'st_data': self._reference, 'roi': np.asarray(roi, dtype=int)}

    @dict_to_object
    def remove_linear_term(self, fit=None, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """Return a new :class:`AberrationsFit` object with the
        linear term removed from `pixel_aberrations` profile.

        Parameters
        ----------
        fit : numpy.ndarray
            Fit coefficients of a first order polynomial. Inferred from
            `pixel_aberrations` by fitting a line if None.
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

        Returns
        -------
        AberrationsFit
            New :class:`AberrationsFit` object with the updated
            `pixel_aberrations` and `phase`.
        """
        if fit is None:
            fit = LeastSquares.fit(x=self.pixels, y=self.pixel_aberrations,
                                   roi=self.roi, max_order=1, xtol=xtol,
                                   ftol=ftol, loss=loss)[0]
        pixel_aberrations = self.pixel_aberrations - self.model(fit)
        return {'st_data': self._reference, 'pixel_aberrations': pixel_aberrations,
                'theta_aberrations': None, 'phase': None}

    @dict_to_object
    def update_center(self, center):
        """Return a new :class:`AberrationsFit` object with the pixels
        centered around `center`.

        Parameters
        ----------
        center : float
            Index of the zerro scattering angle or direct
            beam pixel.

        Returns
        -------
        STData
            New :class:`AberrationsFit` object with the updated `pixels`,
            `phase`, and `pixel_aberrations`.
        """
        if center <= self.pixels[0]:
            pixels = self.pixels - center
            return {'st_data': self._reference, 'pixels': pixels}
        elif center >= self.pixels[-1]:
            pixels = center - self.pixels
            idxs = np.argsort(self.pixels)
            return {'st_data': self._reference, 'pixels': pixels[idxs], 'thetas': None,
                    'pixel_aberrations': -self.pixel_aberrations[idxs], 'theta_aberrations': None,
                    'thetas': None}
        else:
            raise ValueError('Origin must be outside of the region of interest')

    @dict_to_object
    def update_phase(self):
        """Return a new :class:`AberrationsFit` object with the updated
        `phase`.

        Returns
        -------
        AberrationsFit
            New :class:`AberrationsFit` object with the updated `phase`.
        """
        return {'st_data': self._reference, 'theta_aberrations': None,
                'phase': None}

    def get(self, attr, value=None):
        """Return a dataset with `mask` and `roi` applied.
        Return `value` if the attribute is not found.

        Parameters
        ----------
        attr : str
            Attribute to fetch.
        value : object, optional
            Return if `attr` is not found.

        Returns
        -------
        numpy.ndarray or object
            `attr` dataset with `mask` and `roi` applied.
            `value` if `attr` is not found.
        """
        if attr in self:
            val = super(AberrationsFit, self).get(attr)
            if not val is None:
                if attr in ['phase', 'pixels', 'pixel_aberrations', 'theta_aberrations']:
                    val = val[self.roi[0]:self.roi[1]]
            return val
        else:
            return value

    def model(self, fit):
        """Return the polynomial function values of
        lens' deviation angles fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens` pixel aberrations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Array of polynomial function values.
        """
        return LeastSquares.model(fit, self.pixels, [0, self.pixels.size])

    def pix_to_phase(self, fit):
        """Convert fit coefficients from pixel
        aberrations fit to aberrations phase fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens' pixel aberrations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Lens` phase aberrations fit coefficients.
        """
        nfit = np.zeros(fit.size + 1)
        nfit[:-1] = 2 * np.pi / self.wavelength * fit * self.ref_ap * self.pixel_size
        nfit[:-1] /= np.arange(1, fit.size + 1)[::-1]
        nfit[-1] = -self.model(nfit).mean()
        return nfit

    def phase_to_pix(self, ph_fit):
        """Convert fit coefficients from pixel
        aberrations fit to aberrations phase fit.

        Parameters
        ----------
        ph_fit : numpy.ndarray
            Lens` phase aberrations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Lens' pixel aberrations fit coefficients.
        """
        fit = ph_fit[:-1] * self.wavelength / (2 * np.pi * self.ref_ap * self.pixel_size)
        fit *= np.arange(1, ph_fit.size)[::-1]
        return fit

    def fit(self, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """Fit lens' pixel aberrations with polynomial function using
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

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * c_3 : Third order aberrations coefficient [rad / mrad^3].
            * c_4 : Fourth order aberrations coefficient [rad / mrad^4].
            * fit : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * ph_fit : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * rel_err : Vector of relative errors of the fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
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

    def fit_phase(self, max_order=3, xtol=1e-14, ftol=1e-14, loss='linear'):
        """Fit lens' phase aberrations with polynomial function using
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

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * c_3 : Third order aberrations coefficient [rad / mrad^3].
            * c_4 : Fourth order aberrations coefficient [rad / mrad^4].
            * fit : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * ph_fit : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * rel_err : Vector of relative errors of the fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
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
