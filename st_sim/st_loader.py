from __future__ import division

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from .st_wrapper import cxi_protocol, INIParser, ROOT_PATH
from .bin import make_whitefield_st, make_reference, update_pixel_map, total_mse

class STLoader(INIParser):
    """
    Speckle Tracking scan loader class
    """
    attr_dict = {'paths': ('ALL',)}
    fmt_dict = {'paths': 'str'}

    def __init__(self, protocol=cxi_protocol(), **kwargs):
        super(STLoader, self).__init__(**kwargs)
        self.protocol = protocol

    def find_path(self, attr, cxi_file):
        """
        Find attribute path in a cxi file

        attr - the attribute to be found
        cxi_file - h5py File object
        """
        if self.protocol.default_paths[attr] in cxi_file:
            return self.protocol.default_paths[attr]
        elif attr in self.paths:
            for path in self.paths[attr]:
                if path in cxi_file:
                    return path
        else:
            return None

    def load(self, path, defocus=None, roi=None):
        """
        Load a cxi file and return a object
        """
        data_dict = {}
        if roi:
            data_dict['roi'] = np.asarray(roi, dtype=self.protocol.dtypes['roi'])
        if defocus:
            data_dict['defocus'] = np.asarray(defocus, dtype=self.protocol.dtypes['defocus'])
        with h5py.File(path, 'r') as cxi_file:
            for attr in self.protocol.default_paths:
                cxi_path = self.find_path(attr, cxi_file)
                if cxi_path:
                    data_dict[attr] = cxi_file[cxi_path][...].astype(self.protocol.dtypes[attr])
        return STData(self.protocol, **data_dict)

def loader():
    """
    Return the default cxi loader
    """
    return STLoader.import_ini(os.path.join(ROOT_PATH, 'st_protocol.ini'))

class dict_to_object:
    def __init__(self, finstance):
        self.finstance = finstance

    def __get__(self, instance, cls):
        return return_obj_method(self.finstance.__get__(instance, cls), cls, instance)

class return_obj_method:
    def __init__(self, dict_func, cls, instance):
        self.dict_func, self.cls, self.instance = dict_func, cls, instance

    def __call__(self, *args, **kwargs):
        dct = self.instance.__dict__.copy()
        dct.update(self.dict_func(*args, **kwargs))
        return self.cls(**dct)

    def inplace_update(self, *args, **kwargs):
        self.instance.__dict__.update(self.dict_func(*args, **kwargs))

class STData:
    """
    Speckle Tracking scan data class

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

    def __init__(self, protocol=cxi_protocol(), **kwargs):
        self.protocol = protocol
        self._init_dict(**kwargs)

    def _init_dict(self, **kwargs):
        # Initialize configuration attributes
        for attr in self.attr_dict | self.init_dict:
            if attr in kwargs:
                self.__dict__[attr] = np.asarray(kwargs[attr], dtype=self.protocol.dtypes[attr])
            elif attr in self.init_dict:
                self.__dict__[attr] = None
            else:
                raise ValueError('{0:s} has not been provided'.format(attr))

        # Set good frames array and mask
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None or self.whitefield is None:
            self.mask = np.ones(self.data.shape[1:],
                                dtype=self.protocol.dtypes['mask'])
            self.whitefield = make_whitefield_st(data=self.data[self.good_frames], mask=self.mask)

        # Set pixel translations
        if self.pixel_translations is None:
            self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
            self.pixel_translations /= (self.basis_vectors**2).sum(axis=-1) * self.defocus / self.distance
            self.pixel_translations -= self.pixel_translations.mean(axis=0)

        # Set pixel map, deviation angles, and phase
        if self.pixel_map is None:
            self.pixel_map = np.indices(self.whitefield.shape,
                                        dtype=self.protocol.dtypes['pixel_map'])
            self.deviation_angles = np.zeros(self.pixel_map.shape,
                                             dtype=self.protocol.dtypes['deviation_angles'])
            self.phase = np.zeros(self.whitefield.shape, dtype=self.protocol.dtypes['phase'])

        # Init list of SpeckleTracking1D objects
        self.st_objects = []
        SpeckleTracking1D.import_data(self)

    def _update_pixel_map_1d(self, fs_map):
        fs_idx = np.indices(self.whitefield.shape, dtype=self.protocol.dtypes['pixel_map'])
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
    def change_defocus(self, defocus):
        """
        Return a new object with the updated defocus distance
        """
        return {'defocus': defocus}

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
        return self.st_objects

    def get_last_st(self):
        """
        Return the last SpeckleTracking1D object
        """
        return self.st_objects[-1]

class SpeckleTracking1D:
    """
    One-dimensional Robust Speckle Tracking Algorithm

    roi - Region Of Interest
    good_frames - array with good frames indexes
    data - measured frames
    whitefield - whitefield
    dss_pix, dfs_pix - sample translations along the slow adn fast axes in pixels
    dss_avg, dfs_avg - average sample translation along the slow and fast axes in pixels
    pixel_map - pixel map
    reference_image - reference image
    """
    attr_dict = {'data', 'dref', 'defocus', 'dfs_pix', 'dss_pix', 'pixel_map', 'whitefield'}
    init_dict = {'dfs_avg', 'dss_avg', 'reference_image'}
    MIN_WFS = 10
    MAX_DFS = 40

    def __init__(self, **kwargs):
        for attr in self.attr_dict | self.init_dict:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]
            elif attr in self.init_dict:
                self.__dict__[attr] = None
            else:
                raise ValueError('{0:s} has not been provided'.format(attr))

        if self.dss_avg is None or self.dfs_avg is None:
            self.dfs_avg = min(np.mean(np.abs(self.dfs_pix[1:] - self.dfs_pix[:-1])), self.MAX_DFS)
            self.dss_avg = np.mean(np.abs(self.dss_pix[1:] - self.dss_pix[:-1]))
        if self.reference_image is None:
            self.update_reference.inplace_update()

        self.dref.st_objects.append(self)

    @classmethod
    def import_data(cls, st_data):
        """
        Return a new object from a data dictionary with the given items:

        roi - Region Of Interest
        good_frames - array with good frames indexes
        mask - bad pixels mask
        data - measured frames
        whitefield - whitefield
        pixel_translations - sample translations in pixels
        """
        data = (st_data.get('mask') * st_data.get('data')).sum(axis=1)[:, None].astype(np.float64)
        whitefield = st_data.get('whitefield').astype(np.float64)
        pixel_map = st_data.get('pixel_map')[:, [0]]
        dij_pix = np.ascontiguousarray(np.swapaxes(st_data.get('pixel_translations'), 0, 1))
        return cls(data=data, dref=st_data, defocus=st_data.defocus, dfs_pix=dij_pix[1],
                   dss_pix=dij_pix[0], pixel_map=pixel_map, whitefield=whitefield)

    @dict_to_object
    def update_reference(self, l_scale=2.5):
        """
        Return new object with the updated reference image

        l_scale - length scale in pixels
        """
        reference_image, dss_pix, dfs_pix = \
        make_reference(I_n=self.data, W=self.whitefield,
                       u=self.pixel_map, di=self.dss_pix,
                       dj=self.dfs_pix, ls=l_scale)
        return {'reference_image': reference_image, 'dfs_pix': dfs_pix, 'dss_pix': dss_pix}

    @dict_to_object
    def update_pixel_map(self, wfs, l_scale=2.5):
        """
        Return new object with the updated pixel map

        wfs - search window size in pixels
        l_scale - length scale in pixels
        """
        pixel_map = update_pixel_map(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                     u0=self.pixel_map, di=self.dss_pix, dj=self.dfs_pix,
                                     dss=self.dss_avg, dfs=self.dfs_avg, wss=1, wfs=wfs // 2)
        pixel_map = gaussian_filter(pixel_map, (0, 0, l_scale))
        return {'pixel_map': pixel_map}

    def update_data(self):
        """
        Update bound STData object
        """
        self.dref._update_pixel_map_1d(self.pixel_map[1])
        self.dref.reference_image = self.reference_image

    def defocus_sweep(self, df_arr, l_scale=120):
        """
        Calculate reference images for an array of defocus distances df_arr
        Return a new object with optimal defocus distance and a defocus sweep scan
        """
        sweep_scan = []
        for defocus in df_arr:
            dss_pix, dfs_pix = self.dss_pix * self.defocus / defocus
            reference_image = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                             di=dss_pix, dj=dfs_pix, ls=l_scale)[0]
            sweep_scan.append(np.mean(np.gradient(reference_image[0])**2))
        return np.array(sweep_scan)

    def iter_update(self, wfs, l_scale=2.5, n_iter=5, verbose=True):
        """
        Update the reference image and the pixel map iteratively

        wfs - search window size in pixels
        l_scale - length scale in pixels
        n_iter - number of iteration
        verbose - verbosity
        """
        obj = self.update_reference(l_scale=l_scale)
        errors = []
        for it in range(1, n_iter + 1):
            errors.append(obj.mse())
            if verbose:
                print('Iteration No. {:d}: MSE = {:.5f}'.format(it, errors[-1]))
            u0 = obj.pixel_map
            obj.update_pixel_map.inplace_update(wfs=wfs, l_scale=l_scale)
            wfs = max(int(np.abs(obj.pixel_map - u0).max()), self.MIN_WFS)
            if verbose:
                print('Iteration No. {:d}: Search window size = {:d}'.format(it, wfs))
            obj.update_reference.inplace_update(l_scale=l_scale)
        return obj, errors

    def mse(self):
        """
        Return mean-squared-error (MSE)
        """
        return total_mse(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                         u=self.pixel_map, di=self.dss_pix, dj=self.dfs_pix)

    def var_pixel_map(self, l_scale=20):
        """
        Return the pixel map variance
        """
        var_psn = np.mean(self.data)
        reference_image = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                         di=self.dss_pix, dj=self.dfs_pix, ls=l_scale)[0]
        dref_avg = np.mean(np.gradient(reference_image[0])**2)
        N, K = self.data.shape[0], self.data.shape[-1] / (self.dfs_pix[0] - self.dfs_pix[1])
        return var_psn * (1 / N + 1 / N / K) / dref_avg / np.mean(self.whitefield**2)

class AbberationsFit:
    """
    Lens abberations model fitting class

    max_order - maximum polinomial order of the model function
    """
    def __init__(self, max_order=2):
        self.max_order = max_order

    def bounds(self, pixels):
        """
        Return the bounds of the regression problem
        """
        lb = -np.inf * np.ones(self.max_order + 2)
        ub = np.inf * np.ones(self.max_order + 2)
        lb[-1] = 0; ub[-1] = pixels.shape[0]
        return (lb, ub)

    def errors(self, fit, pixels, pixel_ab):
        """
        Return the model errors
        """
        return self.model(fit, pixels) - pixel_ab

    def init_x(self, pixels, pixel_ab):
        """
        Return initial fit coefficients
        """
        x0 = np.zeros(self.max_order + 2)
        u0 = gaussian_filter(pixel_ab, pixel_ab.shape[0] // 10)
        if np.median(np.gradient(np.gradient(u0))) > 0:
            idx = np.argmin(u0)
        else:
            idx = np.argmax(u0)
        x0[-1] = pixels[idx]
        return x0

    def model(self, fit, pixels):
        """
        Return the model values of pixel abberations
        """
        return np.polyval(fit[:-1], pixels - fit[-1])

    def angles_model(self, fit, pixels, st_data):
        """
        Return the model values of abberation angles
        """
        pix_ap = st_data.x_pixel_size / st_data.distance
        return self.model(self.to_ang_fit(fit, st_data), pixels * pix_ap)

    def phase_model(self, fit, pixels, st_data):
        """
        Return the model values of lens' phase
        """
        pix_ap = st_data.x_pixel_size / st_data.distance
        return self.model(self.to_ph_fit(fit, pixels, st_data), pixels * pix_ap)

    def weighted_errors(self, fit, pixels, pixel_ab, weights):
        """
        Return the weighted model errors
        """
        return weights * self.errors(fit, pixels, pixel_ab)

    def to_ang_fit(self, fit, st_data):
        """
        Convert pixel abberations fit to angle abberations fit
        """
        pix_ap = st_data.x_pixel_size / st_data.distance
        ang_fit = fit * pix_ap
        ang_fit[:-1] /= np.geomspace(pix_ap**self.max_order, 1, self.max_order + 1)
        return ang_fit

    def to_ph_fit(self, fit, pixels, st_data):
        """
        Convert pixel abberations fit to phase fit
        """
        pix_ap = st_data.x_pixel_size / st_data.distance
        ph_fit = np.zeros(self.max_order + 3)
        ang_fit = self.to_ang_fit(fit, st_data)
        ph_fit[:-2] = ang_fit[:-1] * 2 * np.pi / st_data.wavelength * st_data.defocus / \
                      np.linspace(self.max_order + 1, 1, self.max_order + 1)
        ph_fit[-1] = ang_fit[-1]
        ph_fit[-2] = np.mean(st_data.get('phase') - self.model(ph_fit, pixels * pix_ap))
        return ph_fit

    def _fit(self, pixels, pixel_ab, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        fit = least_squares(self.errors, self.init_x(pixels, pixel_ab),
                            bounds=self.bounds(pixels), loss=loss,
                            args=(pixels, pixel_ab), xtol=xtol, ftol=ftol)
        if np.linalg.det(fit.jac.T.dot(fit.jac)):
            cov = np.linalg.inv(fit.jac.T.dot(fit.jac))
            err = np.sqrt(np.sum(fit.fun**2) / (fit.fun.size - fit.x.size) * np.abs(np.diag(cov)))
        else:
            err = 0
        return fit.x, err

    def fit(self, st_data, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """
        Return pixel abberations fit for the STData object
        """
        pixels = np.arange(st_data.roi[3] - st_data.roi[2])
        pixel_ab = st_data.get('deviation_angles')[1, 0] / st_data.x_pixel_size * st_data.distance
        fit, err = self._fit(pixels, pixel_ab, xtol=xtol, ftol=ftol, loss=loss)
        r_sq = 1 - np.sum(self.errors(fit, pixels, pixel_ab)**2) / np.sum((pixel_ab.mean() - pixel_ab)**2)
        return {'fit': fit, 'error': err, 'r_sq': r_sq, 'pixel_ab': pixel_ab, 'pixels': pixels}
