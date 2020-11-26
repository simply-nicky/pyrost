"""Routines for model regression based on nonlinear
least-squares algorithm. :class:`AbberationsFit` fit the
lens' abberations profile with the polynomial function
using nonlinear least-squares algorithm.

Examples
--------
Generate a :class:`AbberationsFit` object from
:class:`STData` container object `st_data` as follows:

>>> fit_ab = AbberationsFit.import_data(st_data)

Fit a pixel abberations profile `pixel_ab` with third
order polynomial:

>>> fit_res = fit_ab.fit_pixel_abberations(pixel_ab, max_order=3)
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from .data_container import DataContainer

class LeastSquares:
    """Basic nonlinear least-squares fit class.
    Based on :func:`scipy.optimize.least_squares`.

    See Also
    --------
    :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    @staticmethod
    def bounds(x, max_order=2):
        """Return the bounds of the regression problem.

        Parameters
        ----------
        max_order : int, optional
            Maximum order of the polynomial model function.

        Returns
        -------
        lb : numpy.ndarray
            Lower bounds.
        ub : numpy.ndarray
            Upper bounds.
        """
        lb = -np.inf * np.ones(max_order + 2)
        ub = np.inf * np.ones(max_order + 2)
        lb[-1] = 0; ub[-1] = x.shape[0]
        return (lb, ub)

    @staticmethod
    def model(fit, x):
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
        return np.polyval(fit[:-1], x - fit[-1])

    @staticmethod
    def init_x(x, y, max_order=2):
        """Return initial fit coefficients.

        Parameters
        ----------
        x : numpy.ndarray
            Array of x coordinates.
        y : numpy.ndarray
            Array of y coordinates.
        max_order : int, optional
            Maximum order of the polynomial model function.

        Returns
        -------
        numpy.ndarray
            Array of initial fit coefficients.
        """
        x0 = np.zeros(max_order + 2)
        u0 = gaussian_filter(y, y.shape[0] // 10)
        if np.median(np.gradient(np.gradient(u0))) > 0:
            idx = np.argmin(u0)
        else:
            idx = np.argmax(u0)
        x0[-1] = x[idx]
        return x0

    @classmethod
    def errors(cls, fit, x, y):
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
        return cls.model(fit, x) - y

    @classmethod
    def weighted_errors(cls, fit, x, y, w):
        """Return an array of weighted model
        residuals.

        Parameters
        ----------
        fit : numpy.ndarray
            Array of fit coefficients.
        x : numpy.ndarray
            Array of x coordinates.
        y : numpy.ndarray
            Array of y coordinates.
        w : numpy.ndarray
            Array of weight values.

        Returns
        -------
        numpy.ndarray
            Array of weighted model residuals.
        """
        return w * cls.errors(fit, x, y)

    @classmethod
    def fit(cls, x, y, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
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
        fit = least_squares(cls.errors, cls.init_x(x, y, max_order),
                            bounds=cls.bounds(x, max_order), loss=loss,
                            args=(x, y), xtol=xtol, ftol=ftol)
        if np.linalg.det(fit.jac.T.dot(fit.jac)):
            cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
            err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
        else:
            err = 0
        return fit.x, err

class AbberationsFit(DataContainer, LeastSquares):
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
    * pixel_size : Pixel's size [m].
    * wavelength : Incoming beam's wavelength [m].

    See Also
    --------
    :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
        algorithm description.
    """
    attr_set = {'defocus', 'distance', 'wavelength', 'pixel_size'}
    fs_lookup = {'defocus': 'defocus_fs', 'pixel_size': 'x_pixel_size'}
    ss_lookup = {'defocus': 'defocus_ss', 'pixel_size': 'y_pixel_size'}

    def __init__(self, **kwargs):
        super(AbberationsFit, self).__init__(**kwargs)

        self.pix_ap = self.pixel_size / self.distance

    @classmethod
    def import_data(cls, st_data, axis=1):
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
        return cls(**data_dict)

    def angles_model(self, ang_fit, pixels):
        """Return the polynomial function values of
        lens' deviation angles fit.

        Parameters
        ----------
        ang_fit : numpy.ndarray
            Lens` deviation angles fit coefficients.
        pixels: numpy.ndarray
            Array of x coordinates in pixels.

        Returns
        -------
        numpy.ndarray
            Array of polynomial function values.
        """
        return self.model(ang_fit, pixels * self.pix_ap)

    def phase_model(self, ph_fit, pixels):
        """Return the polynomial function values of the
        fit of lens' abberation phase profile.

        Parameters
        ----------
        ph_fit : numpy.ndarray
            Lens' abberations phase fit coefficients.
        pixels: numpy.ndarray
            Array of x coordinates in pixels.

        Returns
        -------
        numpy.ndarray
            Array of polynomial function values.
        """
        return self.model(ph_fit, pixels * self.pix_ap)

    def to_ang_fit(self, fit):
        """Convert fit coefficients from pixel
        abberations fit to deviation angles fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens' pixel abberations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Lens` deviation angles fit coefficients.
        """
        ang_fit, max_order = fit * self.pix_ap, fit.size - 2
        ang_fit[:-1] /= np.geomspace(self.pix_ap**max_order, 1, max_order + 1)
        return ang_fit

    def to_ph_fit(self, fit, pixels):
        """Convert fit coefficients from pixel
        abberations fit to abberations phase fit.

        Parameters
        ----------
        fit : numpy.ndarray
            Lens' pixel abberations fit coefficients.

        Returns
        -------
        numpy.ndarray
            Lens` abberations phase fit coefficients.
        """
        ph_fit, max_order = np.zeros(fit.size + 1), fit.size - 2
        ang_fit = self.to_ang_fit(fit)
        ph_fit[:-2] = ang_fit[:-1] * 2 * np.pi / self.wavelength * self.defocus / \
                      np.linspace(max_order + 1, 1, max_order + 1)
        ph_fit[-1] = ang_fit[-1]
        ph_fit[-2] -= self.model(ph_fit, pixels * self.pix_ap).mean()
        return ph_fit

    def fit_pixel_abberations(self, pixel_ab, pixels=None, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """Fit lens' pixel abberations `pixel_ab` with polynomial
        function using :func:`scipy.optimise.least_squares`.

        Parameters
        ----------
        pixel_ab : numpy.ndarray
            Lens' pixel abberations profile.
        pixels: numpy.ndarray
            Array of x coordinates in pixels.
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
        dict
            :class:`dict` with the following fields defined:

            * pixels : Array of x coordinates in pixels.
            * pix_fit : Array of the polynomial function coefficients of
              the `pixel_abberations` fit.
            * ang_fit : Array of the polynomial function coefficients of
              the deviation angles fit.
            * ph_fit : Array of the polynomial function coefficients of
              the `phase` fit.
            * pix_err : Vector of errors of the `pix_fit` fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        if pixels is None:
            pixels = np.arange(pixel_ab.shape[0])
        fit, err = self.fit(pixels, pixel_ab, max_order=max_order,
                            xtol=xtol, ftol=ftol, loss=loss)
        r_sq = 1 - np.sum(self.errors(fit, pixels, pixel_ab)**2) / \
               np.sum((pixel_ab.mean() - pixel_ab)**2)
        return {'pixels': pixels, 'pix_fit': fit, 'ang_fit': self.to_ang_fit(fit),
                'pix_err': err, 'ph_fit': self.to_ph_fit(fit, pixels), 'r_sq': r_sq}
