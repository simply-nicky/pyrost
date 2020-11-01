"""
data_processing.py - Robust Speckle Tracking data processing algorithm
"""
from __future__ import division

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from .bin import make_whitefield_st, make_reference, upm_search, upm_newton_1d, total_mse

class dict_to_object:
    def __init__(self, finstance):
        self.finstance = finstance

    def __get__(self, instance, cls):
        return return_obj_method(self.finstance.__get__(instance, cls), cls, instance)

class return_obj_method:
    def __init__(self, dict_func, cls, instance):
        self.dict_func, self.cls, self.instance = dict_func, cls, instance

    def __call__(self, *args, **kwargs):
        dct = {}
        for key, val in self.instance.__dict__.items():
            if isinstance(val, np.ndarray):
                dct[key] = np.copy(val)
            else:
                dct[key] = val
        dct.update(self.dict_func(*args, **kwargs))
        return self.cls(**dct)

    def inplace_update(self, *args, **kwargs):
        self.instance.__dict__.update(self.dict_func(*args, **kwargs))

class STData:
    """
    Speckle Tracking scan data container class
    Contains all the necessary data for the Speckle Tracking algorithm
    Contains the list of all the SpeckleTracking1D objects and the AbberationsFit object derived from it

    Necessary attributes:
    basis_vectors - detector basis vectors
    data - measured data
    defocus - defocus distance
    distance - sample-to-detector distance
    translations - sample translations
    wavelength - incoming beam wavelength
    x_pixel_size - slow axis pixel size
    y_pixel_size - fast axis pixel size

    Optional attributes:
    good_frames - good frames array
    mask - bad pixels mask
    pixel_translations - sample translations in the detector plane in pixels
    roi - region of interest
    whitefield - whitefield
    """
    attr_dict = {'basis_vectors', 'data', 'defocus', 'distance', 'translations',
                 'wavelength', 'x_pixel_size', 'y_pixel_size'}
    init_dict = {'deviation_angles', 'good_frames', 'mask', 'phase', 'pixel_map',
                 'pixel_translations', 'reference_image', 'roi', 'whitefield'}

    def __init__(self, protocol, **kwargs):
        self.protocol = protocol
        self._init_dict(**kwargs)

    def _init_dict(self, **kwargs):
        # Initialize configuration attributes
        for attr in self.attr_dict:
            if kwargs.get(attr) is None:
                raise ValueError('Attribute {:s} has not been provided'.format(attr))
            else:
                self.__dict__[attr] = kwargs.get(attr)
        for attr in self.init_dict:
            self.__dict__[attr] = kwargs.get(attr)

        # Set a good frames array and mask
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None or self.whitefield is None:
            self.mask = np.ones(self.data.shape[1:],
                                dtype=self.protocol.get_dtype('mask'))
            self.whitefield = make_whitefield_st(data=self.data[self.good_frames], mask=self.mask)

        # Set a pixel translations
        if self.pixel_translations is None:
            self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
            self.pixel_translations /= (self.basis_vectors**2).sum(axis=-1) * self.defocus / self.distance
            self.pixel_translations -= self.pixel_translations.mean(axis=0)

        # Set a pixel map, deviation angles, and phase
        if self.pixel_map is None:
            self.pixel_map = np.indices(self.whitefield.shape,
                                        dtype=self.protocol.get_dtype('pixel_map'))
            self.deviation_angles = np.zeros(self.pixel_map.shape,
                                             dtype=self.protocol.get_dtype('deviation_angles'))
            self.phase = np.zeros(self.whitefield.shape, dtype=self.protocol.get_dtype('phase'))

        # Initialize a list of SpeckleTracking1D objects
        self._st_objects = []
        SpeckleTracking1D.import_data(self)

        # Initialize an AbberationsFit object
        self._ab_fit = AbberationsFit.import_data(self)

    def _update_pixel_map_1d(self, fs_map):
        fs_idx = np.indices(self.whitefield.shape, dtype=self.protocol.get_dtype('pixel_map'))
        fs_idx = fs_idx[1, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        fs_map = np.tile(fs_map, (self.roi[1] - self.roi[0], 1))
        dev_ang = (fs_map - fs_idx) * self.x_pixel_size / self.distance
        phase = np.cumsum(dev_ang, axis=1) * self.x_pixel_size * self.defocus / self.distance * 2 * np.pi / self.wavelength

        self.pixel_map[1, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = fs_map
        self.deviation_angles[1, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = dev_ang
        self.phase[self.roi[0]: self.roi[1], self.roi[2]:self.roi[3]] = phase

    @dict_to_object
    def crop_data(self, roi):
        """
        Return a new object with the updated ROI
        """
        return {'roi': np.asarray(roi, dtype=np.int)}

    @dict_to_object
    def mask_frames(self, good_frames):
        """
        Return a new object with the updated good frames mask
        """
        return {'good_frames': np.asarray(good_frames, dtype=np.int),
                'whitefield': make_whitefield_st(data=self.data[good_frames], mask=self.mask)}

    @dict_to_object
    def make_whitefield(self):
        """
        Return a new object with the updated whitefield
        """
        return {'whitefield': make_whitefield_st(data=self.data[self.good_frames], mask=self.mask)}

    @dict_to_object
    def make_pixel_map(self):
        """
        Return a new object with the default pixel map
        """
        return {'pixel_map': None}

    @dict_to_object
    def update_defocus(self, defocus=None):
        """
        Return a new object with the updated defocus distance and sample's pixel translations
        """
        if defocus is None:
            defocus = self.defocus
        return {'defocus': defocus, 'pixel_translations': None}

    def get(self, attr, value=None):
        """
        Return the masked data of the given attribute
        """
        if attr in self.attr_dict | self.init_dict:
            data = self.__dict__[attr]
            if attr in ['data', 'deviation_angles', 'mask', 'phase', 'pixel_map', 'whitefield']:
                data = data[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            if attr in ['basis_vectors', 'data', 'pixel_translations', 'translations']:
                data = data[self.good_frames]
            return np.ascontiguousarray(data)
        else:
            return value

    def get_st_objects(self):
        """
        Return the list of bound SpeckleTracking1D objects
        """
        return self._st_objects

    def get_last_st(self):
        """
        Return the last SpeckleTracking1D object
        """
        return self._st_objects[-1]

    def get_ab_fit(self):
        """
        Return the AbberationsFit object
        """
        return self._ab_fit

    def write_cxi(self, cxi_file):
        """
        Write the data to a cxi file
        """
        for attr in self.attr_dict | self.init_dict:
            self.protocol.write_cxi(attr, self.__dict__[attr], cxi_file)

class SpeckleTracking1D:
    """
    One-dimensional Robust Speckle Tracking Algorithm

    roi - Region Of Interest
    good_frames - array with good frames indexes
    data - measured frames
    whitefield - the data whitefield
    dss_pix, dfs_pix - the sample translations along the slow and fast axes in pixels
    pixel_map - the pixel mapping between the data at the detector's plane
                and the reference image at the reference plane
    reference_image - the reference image
    n0, m0 - the lower bounds of fast and slow axes of the reference image at the reference frame [pixels]
    """
    attr_dict = {'data', 'dref', 'defocus', 'dfs_pix', 'dss_pix', 'pixel_map', 'whitefield'}
    init_dict = {'n0', 'm0', 'reference_image'}
    MIN_WFS = 10

    def __init__(self, **kwargs):
        for attr in self.attr_dict:
            if kwargs.get(attr) is None:
                raise ValueError('Attribute {:s} has not been provided'.format(attr))
            else:
                self.__dict__[attr] = kwargs.get(attr)
        for attr in self.init_dict:
            self.__dict__[attr] = kwargs.get(attr)

        if self.reference_image is None:
            self.update_reference.inplace_update()

        self.dref._st_objects.append(self)

    @classmethod
    def import_data(cls, st_data):
        """
        Return a new SpeckleTracking1D object from an STData object
        """
        data = (st_data.get('mask') * st_data.get('data')).sum(axis=1)[:, None].astype(np.float64)
        whitefield = st_data.get('whitefield').astype(np.float64)
        pixel_map = st_data.get('pixel_map')[:, [0]]
        dij_pix = np.ascontiguousarray(np.swapaxes(st_data.get('pixel_translations'), 0, 1))
        return cls(data=data, dref=st_data, defocus=st_data.defocus, dfs_pix=dij_pix[1],
                   dss_pix=dij_pix[0], pixel_map=pixel_map, whitefield=whitefield)

    @dict_to_object
    def update_reference(self, l_scale=10., sw_max=0):
        """
        Return new object with the updated reference image

        l_scale - length scale in pixels
        sw_max - search window size in pixels
        """
        I0, n0, m0 = make_reference(I_n=self.data, W=self.whitefield,
                                    u=self.pixel_map, di=self.dss_pix,
                                    dj=self.dfs_pix, ls=l_scale, wfs=sw_max)
        return {'reference_image': I0, 'n0': n0, 'm0': m0}

    @dict_to_object
    def update_pixel_map(self, sw_max, l_scale=3., method='newton'):
        """
        Return new object with the updated pixel map
        Available pixel map update methods are:
        - 'newton' - iterative Newton's method based on finite difference gradient approximation
        - 'search' - brute-force search along the search window

        sw_max - search window size in pixels
        l_scale - length scale in pixels
        """
        if method == 'search':
            pixel_map = upm_search(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                   u0=self.pixel_map, di=self.dss_pix - self.n0, dj=self.dfs_pix - self.m0,
                                   wss=1, wfs=sw_max, ls=l_scale)
        elif method == 'newton':
            pixel_map = upm_newton_1d(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                      u0=self.pixel_map, di=self.dss_pix - self.n0, dj=self.dfs_pix - self.m0,
                                      wss=1, wfs=sw_max, ls=l_scale)
        else:
            raise ValueError('Wrong method argument: {:s}'.format(method))
        return {'pixel_map': pixel_map}

    def update_data(self):
        """
        Update bound STData object
        """
        self.dref._update_pixel_map_1d(self.pixel_map[1])
        self.dref.reference_image = self.reference_image

    def defocus_sweep(self, df_arr, l_scale=120):
        """
        Calculate reference images for an array of defocus distances df_arr with the given length scale
        The length scale should be large enough in order to supress high frequency noise
        Return a new object with optimal defocus distance and a defocus sweep scan
        """
        sweep_scan = []
        for defocus in df_arr:
            dss_pix, dfs_pix = self.dss_pix * self.defocus / defocus
            reference_image = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                             di=dss_pix, dj=dfs_pix, ls=l_scale)[0]
            sweep_scan.append(np.mean(np.gradient(reference_image[0])**2))
        return np.array(sweep_scan)

    def iter_update(self, sw_max, ls_pm=3., ls_ri=10., n_iter=5, verbose=True, method='newton'):
        """
        Update the reference image and the pixel map iteratively
        Available pixel map update methods are:
        - 'newton' - iterative Newton's method based on finite difference gradient approximation
        - 'search' - brute-force search along the search window

        sw_max - search window size in pixels
        ls_pm, ls_ri - length scale in pixels for the pixel map and the reference image accordingly
        n_iter - number of iteration
        verbose - turn on verbosity
        """
        obj = self.update_reference(l_scale=ls_ri, sw_max=sw_max)
        errors = [obj.total_mse()]
        if verbose:
            print('Iteration No. 0: Total MSE = {:.3f}'.format(errors[0]))
        for it in range(1, n_iter + 1):
            pm = obj.update_pixel_map.dict_func(sw_max=sw_max, l_scale=ls_pm, method=method)
            ri = obj.update_reference.dict_func(l_scale=ls_ri)

            roi = slice(obj.m0 - ri['m0'], ri['reference_image'].shape[1] + obj.m0 - ri['m0'])
            obj.reference_image[:, roi] += gaussian_filter(ri['reference_image'] - obj.reference_image[:, roi],
                                                           (0, ls_ri))
            obj.pixel_map += gaussian_filter(pm['pixel_map'] - obj.pixel_map, (0, 0, ls_pm))
            
            errors.append(obj.total_mse())
            if verbose:
                print('Iteration No. {:d}: Total MSE = {:.3f}'.format(it, errors[-1]))
            if abs(errors[-1] - errors[-2]) <= 1e-3:
                break
        return obj, errors

    def total_mse(self, l_scale=3.):
        """
        Return total mean-squared-error (MSE)

        l_scale - length scale in pixels
        """
        return total_mse(I_n=self.data, W=self.whitefield, I0=self.reference_image, u=self.pixel_map,
                         di=self.dss_pix - self.n0, dj=self.dfs_pix - self.m0, ls=l_scale)

    def var_pixel_map(self, l_scale=20):
        """
        Return the pixel map variance

        l_scale - length scale in pixels
        """
        var_psn = np.mean(self.data)
        reference_image = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                         di=self.dss_pix, dj=self.dfs_pix, ls=l_scale)[0]
        dref_avg = np.mean(np.gradient(reference_image[0])**2)
        N, K = self.data.shape[0], self.data.shape[-1] / (self.dfs_pix[0] - self.dfs_pix[1])
        return var_psn * (1 / N + 1 / N / K) / dref_avg / np.mean(self.whitefield**2)

class LeastSquares:
    """
    Basic Least Squares fit class
    """
    @staticmethod
    def bounds(x, max_order=2):
        """
        Return the bounds of the regression problem

        max_order - maximum order of the polynomial fit
        """
        lb = -np.inf * np.ones(max_order + 2)
        ub = np.inf * np.ones(max_order + 2)
        lb[-1] = 0; ub[-1] = x.shape[0]
        return (lb, ub)

    @staticmethod
    def model(fit, x):
        """
        Return the model values of pixel abberations
        """
        return np.polyval(fit[:-1], x - fit[-1])

    @staticmethod
    def init_x(x, y, max_order=2):
        """
        Return initial fit coefficients

        max_order - maximum order of the polynomial fit
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
        """
        Return the model errors of the fit
        """
        return cls.model(fit, x) - y

    @classmethod
    def weighted_errors(cls, fit, x, y, w):
        """
        Return the weighted model errors of the given fit and weights array
        """
        return w * cls.errors(fit, x, y)

    @classmethod
    def fit(cls, x, y, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """
        Fit the data comprised of points x, y along the x and y axes based on nonlinear least-squares algorithm
        using the polynomial model function
        Use scipy.optimise.least_squares function

        max_order - maximum order of the polynomial fit
        xtol - tolerance for termination by the change of the independent variables
        ftol - tolerance for termination by the change of the cost function
        loss - determines the loss function, ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan' are allowed)

        Return the attained solution and the vector of residuals
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

class AbberationsFit(LeastSquares):
    """
    Lens abberations model fitting class

    defocus - defocus distance
    distance - sample-to-detector distance
    x_pixel_size - slow axis pixel size
    wavelength - incoming beam wavelength
    """
    attr_dict = {'defocus', 'distance', 'wavelength', 'x_pixel_size'}

    def __init__(self, **kwargs):
        for attr in self.attr_dict:
            if kwargs.get(attr) is None:
                raise ValueError('Attribute {:s} has not been provided'.format(attr))
            else:
                self.__dict__[attr] = float(kwargs.get(attr))

        self.pix_ap = self.x_pixel_size / self.distance

    @classmethod
    def import_data(cls, st_data):
        """
        Return a new AbberationsFit object from an STData object
        """
        data_dict = {attr: st_data.get(attr) for attr in cls.attr_dict}
        return cls(**data_dict)

    def angles_model(self, fit, pixels):
        """
        Return the model values of abberation angles
        """
        return self.model(self.to_ang_fit(fit), pixels * self.pix_ap)

    def phase_model(self, fit, pixels, phase):
        """
        Return the model values of lens' phase
        """
        return self.model(self.to_ph_fit(fit, pixels, phase), pixels * self.pix_ap)

    def to_ang_fit(self, fit):
        """
        Convert pixel abberations fit to angle abberations fit
        """
        ang_fit, max_order = fit * self.pix_ap, fit.size - 2
        ang_fit[:-1] /= np.geomspace(self.pix_ap**max_order, 1, max_order + 1)
        return ang_fit

    def to_ph_fit(self, fit, pixels, phase):
        """
        Convert pixel abberations fit to phase fit
        """
        ph_fit, max_order = np.zeros(fit.size + 1), fit.size - 2
        ang_fit = self.to_ang_fit(fit)
        ph_fit[:-2] = ang_fit[:-1] * 2 * np.pi / self.wavelength * self.defocus / \
                      np.linspace(max_order + 1, 1, max_order + 1)
        ph_fit[-1] = ang_fit[-1]
        ph_fit[-2] = np.mean(phase - self.model(ph_fit, pixels * self.pix_ap))
        return ph_fit

    def fit_pixel_ab(self, pixels, pixel_ab, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """
        Fit the lens' pixel abberations profile based on nonlinear least-squares algorithm
        using the polynomial model function
        Use scipy.optimise.least_squares function

        pixels, pixel_ab - lens' pixel abberations profile in pixels
        max_order - maximum order of the polynomial fit
        xtol - tolerance for termination by the change of the independent variables
        ftol - tolerance for termination by the change of the cost function
        loss - determines the loss function, ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan' are allowed)

        Return a dictionary of the pixel abberations fit ('fit'), the vector of residuals ('error'), 
        $R^2$ goodness of fit value ('r_sq'), and the fitting data ('pixels', 'pixels_ab') 
        """
        fit, err = self.fit(pixels, pixel_ab, max_order=max_order, xtol=xtol, ftol=ftol, loss=loss)
        r_sq = 1 - np.sum(self.errors(fit, pixels, pixel_ab)**2) / np.sum((pixel_ab.mean() - pixel_ab)**2)
        return {'fit': fit, 'error': err, 'r_sq': r_sq, 'pixel_ab': pixel_ab, 'pixels': pixels}