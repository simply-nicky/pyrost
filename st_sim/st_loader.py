import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from .st_wrapper import cxi_protocol, INIParser, ROOT_PATH
from .bin import make_whitefield_st, make_reference, update_pixel_map_search, total_mse

class STLoader(INIParser):
    """
    Speckle Tracking scan loader class
    """
    attr_dict = {'paths': ('ALL',)}
    fmt_dict = {'paths': 'list'}

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
        Load a cxi file and return a STData object
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
    attr_dict = {'basis_vectors', 'data', 'defocus', 'distance',
                 'translations', 'wavelength', 'x_pixel_size',
                 'y_pixel_size'}
    init_dict = {'good_frames', 'mask', 'pixel_translations',
                 'roi', 'whitefield'}

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

        # Set pixel translations
        if self.pixel_translations is None:
            self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
            self.pixel_translations /= (self.basis_vectors**2).sum(axis=-1) * self.defocus / self.distance
            self.pixel_translations -= self.pixel_translations.mean(axis=0)

        # Set good frames array and mask
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None or self.whitefield is None:
            self.mask = np.ones(self.data.shape[1:],
                                dtype=self.protocol.dtypes['mask'])
            self.whitefield = make_whitefield_st(data=self.data[self.good_frames], mask=self.mask)

    @dict_to_object
    def crop_data(self, roi):
        """
        Return new STData object with new ROI
        """
        return {'roi': np.asarray(roi, dtype=np.int)}

    @dict_to_object
    def mask_frames(self, good_frames):
        """
        Return new STData object with new good frames mask
        """
        return {'good_frames': np.asarray(good_frames, dtype=np.int),
                'whitefield': make_whitefield_st(data=self.data[good_frames], mask=self.mask)}

    @dict_to_object
    def change_defocus(self, defocus):
        """
        Return new STData with new defocus distance
        """
        return {'defocus': defocus, 'pixel_translations': \
                self.pixel_translations * self.defocus / defocus}

    def st_process(self):
        """
        Return a SpeckleTracking1D object
        """
        return SpeckleTracking1D.import_dict(**self.__dict__)

class SpeckleTracking1D:
    """
    One-dimensional Robust Speckle Tracking Algorithm

    roi - Region Of Interest
    good_frames - array with good frames indexes
    data - measured frames
    whitefield - whitefield
    dss_pix, dfs_pix - sample translations along the slow adn fast axes in pixels
    dss_avg, dfs_avg - average sample translation along the slow and fast axes in pixels
    pix_map - pixel map
    ref_img - reference image
    """
    attr_dict = {'roi', 'good_frames', 'data', 'whitefield', 'dss_pix', 'dfs_pix'}
    init_dict = {'pix_map', 'ref_img', 'dss_avg', 'dfs_avg'}

    def __init__(self, **kwargs):
        for attr in self.attr_dict | self.init_dict:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]
            elif attr in self.init_dict:
                self.__dict__[attr] = None
            else:
                raise ValueError('{0:s} has not been provided'.format(attr))
        
        if self.dss_avg is None or self.dfs_avg is None:
            self.dfs_avg = np.mean(self.dfs_pix[1:] - self.dfs_pix[:-1])
            self.dss_avg = np.mean(self.dss_pix[1:] - self.dss_pix[:-1])
        if self.pix_map is None:
            self.pix_map = np.indices(self.whitefield.shape, dtype=np.float64)
        if self.ref_img is None:
            self.ref_img, self.dss_pix, self.dfs_pix = \
            make_reference(I_n=self.data, W=self.whitefield,
                           u=self.pix_map, di=self.dss_pix,
                           dj=self.dfs_pix)

    @classmethod
    def import_dict(cls, **kwargs):
        """
        Return a SpeckleTracking object from a data dictionary with the given items:

        roi - Region Of Interest
        good_frames - array with good frames indexes
        mask - bad pixels mask
        data - measured frames
        whitefield - whitefield
        pixel_translations - sample translations in pixels
        """
        # fetch the data
        roi, good_frames = kwargs['roi'], kwargs['good_frames']
        dct = {'roi': roi, 'good_frames': good_frames}
        mask = kwargs['mask'][roi[0]:roi[1], roi[2]:roi[3]]
        data = np.ascontiguousarray(kwargs['data'][good_frames, roi[0]:roi[1], roi[2]:roi[3]])
        dct['data'] = (mask * data).sum(axis=1)[:, None].astype(np.float64)
        dct['whitefield'] = kwargs['whitefield'][roi[0]:roi[1], roi[2]:roi[3]].astype(np.float64)
        dct['dss_pix'] = kwargs['pixel_translations'][good_frames, 0]
        dct['dfs_pix'] = kwargs['pixel_translations'][good_frames, 1]
        return cls(**dct)

    @dict_to_object
    def crop_data(self, roi):
        """
        Return new object with the data cropped according to the given ROI
        """
        dct = {'roi': np.array([roi[0] + self.roi[0], roi[1] + self.roi[0],
                                roi[2] + self.roi[2], roi[3] + self.roi[2]])}
        dct['data'] = np.ascontiguousarray(self.data[:, roi[0]:roi[1], roi[2]:roi[3]])
        dct['pix_map'] = np.ascontiguousarray(self.pix_map[:, roi[0]:roi[1], roi[2]:roi[3]])
        dct['pix_map'][0] -= roi[0]; dct['pix_map'][1] -= roi[2]
        dct['whitefield'] = self.whitefield[roi[0]:roi[1], roi[2]:roi[3]]
        dct['ref_img'] = self.ref_img[roi[0]:roi[1], roi[2]:roi[3]]
        return dct

    @dict_to_object
    def mask_frames(self, good_frames):
        """
        Return new object with the frames masked according to the given good frames array
        """
        return {'good_frames': self.good_frames[good_frames],
                'data': self.data[good_frames],
                'dss_pix': self.dss_pix[good_frames],
                'dfs_pix': self.dfs_pix[good_frames]}

    @dict_to_object
    def update_reference(self, l_scale=2.5):
        """
        Return new object with the updated reference image

        l_scale - length scale in pixels
        """
        dct = {}
        dct['ref_img'], dct['dss_pix'], dct['dfs_pix'] = \
        make_reference(I_n=self.data, W=self.whitefield,
                       u=self.pix_map, di=self.dss_pix,
                       dj=self.dfs_pix, ls=l_scale)
        return dct

    @dict_to_object
    def update_pixel_map(self, wfs, l_scale=2.5):
        """
        Return new object with the updated pixel map

        wfs - search window size in pixels
        l_scale - length scale in pixels
        """
        pix_map = update_pixel_map_search(I_n=self.data, W=self.whitefield, I0=self.ref_img,
                                          u0=self.pix_map, di=self.dss_pix, dj=self.dfs_pix,
                                          dss=self.dss_avg, dfs=self.dfs_avg, wss=1, wfs=wfs // 2)
        return {'pix_map': gaussian_filter(pix_map, (0, 0, l_scale))}

    def mse(self):
        """
        Return mean-squared-error (MSE)
        """
        return total_mse(I_n=self.data, W=self.whitefield, I0=self.ref_img,
                         u=self.pix_map, di=self.dss_pix, dj=self.dfs_pix)

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
            obj.update_pixel_map.inplace_update(wfs=wfs, l_scale=l_scale)
            wfs = int(np.max(np.abs(obj.pix_map - np.indices(obj.whitefield.shape))))
            if verbose:
                print('Iteration No. {:d}: Search window size = {:d}'.format(it, wfs))
            obj.update_reference.inplace_update(l_scale=l_scale)
        return obj, errors
   