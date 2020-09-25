import os
import h5py
import numpy as np
from .st_wrapper import cxi_protocol, INIParser, ROOT_PATH
from .bin import make_whitefield_st, make_reference

class STLoader(INIParser):
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
        Load a cxi file
        """
        data_dict = {'protocol': self.protocol}
        if roi:
            data_dict['roi'] = np.asarray(roi, dtype=self.protocol.dtypes['roi'])
        if defocus:
            data_dict['defocus'] = np.asarray(defocus, dtype=self.protocol.dtypes['defocus'])
        with h5py.File(path, 'r') as cxi_file:
            for attr in self.protocol.default_paths:
                cxi_path = self.find_path(attr, cxi_file)
                if cxi_path:
                    data_dict[attr] = cxi_file[cxi_path][...].astype(self.protocol.dtypes[attr])
        return data_dict

def loader():
    """
    Return the default cxi loader
    """
    return STLoader.import_ini(os.path.join(ROOT_PATH, 'st_protocol.ini'))

class dict_to_object:
    def __init__(self, finstance):
        self.finstance = finstance

    def __get__(self, instance, cls):
        return return_obj_method(self.finstance.__get__(instance, cls), cls)

class return_obj_method:
    def __init__(self, dict_func, cls):
        self.dict_func, self.cls = dict_func, cls

    def __call__(self, *args, **kwargs):
        return self.cls(**self.dict_func(*args, **kwargs))

class STData:
    attr_dict = {'basis_vectors', 'data', 'defocus', 'distance',
                 'translations', 'wavelength', 'x_pixel_size',
                 'y_pixel_size'}
    init_dict = {'good_frames', 'm0', 'mask', 'n0', 'pixel_translations',
                 'reference_image', 'roi', 'whitefield'}

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
            self.pixel_translations /= np.sqrt((self.basis_vectors**2).sum(axis=-1)) * self.defocus
            self.pixel_translations -= self.pixel_translations.mean(axis=0)

        # Set good frames array and mask
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None or self.whitefield is None:
            self.mask = np.ones(self.data.shape[1:],
                                dtype=self.protocol.dtypes['mask'])
            self.whitefield = make_whitefield_st(data=self.data,
                                                 mask=self.mask.astype(np.uint8))

        # Apply ROI
        if self.roi is None:
            self.roi = np.array([0, self.data.shape[1], 0, self.data.shape[2]],
                                dtype=self.protocol.dtypes['roi'])
        self.data_st = self.data[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]].sum(axis=1)[:, None]
        self.mask_st = np.any(self.mask[self.roi[0]:self.roi[1]], axis=0)[None, self.roi[2]:self.roi[3]]
        self.w_st = make_whitefield_st(data=self.data_st, mask=self.mask_st.astype(np.uint8))
        self.pixel_map = np.indices(self.w_st.shape)
        self.n0 = -np.min(self.pixel_map[0]) + np.max(self.pixel_translations[:, 0])
        self.m0 = -np.min(self.pixel_map[1]) + np.max(self.pixel_translations[:, 1])

    @dict_to_object
    def change_roi(self, roi):
        """
        Return new STData object with new ROI
        """
        dct = self.__dict__.copy()
        dct['roi'] = roi
        return dct

    @dict_to_object
    def change_defocus(self, defocus):
        """
        Return new STData object with new ROI
        """
        dct = self.__dict__.copy()
        dct['defocus'] = defocus
        dct['pixel_translations'] *= self.defocus / defocus
        return dct

    # @dict_to_object
    # def make_reference(self, l_scale=2.5):
    #     """
    #     Generate the reference image

    #     l_scale - kernel length scale in pixels
    #     """
    #     dct = self.__dict__.copy()
    #     dct['reference_image'] = make_reference()