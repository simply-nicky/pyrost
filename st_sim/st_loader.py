import os
import h5py
import numpy as np

class STLoader():
    defaults = {'data_path': '/entry_1/data_1/data',
                'whitefield_path': '/speckle_tracking/whitefield',
                'mask_path': '/speckle_tracking/mask',
                'roi_path': '/speckle_tracking/roi',
                'defocus_path': '/speckle_tracking/defocus',
                'translations_path': '/entry_1/sample_1/geometry/translations',
                'good_frames_path': '/frame_selector/good_frames',
                'y_pixel_size_path': '/entry_1/instrument_1/detector_1/y_pixel_size',
                'x_pixel_size_path': '/entry_1/instrument_1/detector_1/x_pixel_size',
                'distance_path': '/entry_1/instrument_1/detector_1/distance',
                'wavelength_path': '/entry_1/instrument_1/source_1/wavelength',
                'basis_vectors_path': '/entry_1/instrument_1/detector_1/basis_vectors'}

    fmt = {'data_path': np.float32,
                'whitefield_path': np.float32,
                'mask_path': bool,
                'translations_path': np.float32,
                'good_frames_path': '/frame_selector/good_frames',
                'y_pixel_size_path': '/entry_1/instrument_1/detector_1/y_pixel_size',
                'x_pixel_size_path': '/entry_1/instrument_1/detector_1/x_pixel_size',
                'distance_path': '/entry_1/instrument_1/detector_1/distance',
                'wavelength_path': '/entry_1/instrument_1/source_1/wavelength',
                'basis_vectors_path': '/entry_1/instrument_1/detector_1/basis_vectors'}

    attr_set = {'data_path', 'translations_path', 'y_pixel_size_path',
                'x_pixel_size_path', 'distance_path', 'basis_vectors_path',
                'wavelength_path'}

    dtype = np.float32

    def __init__(self, **kwargs):
        for attr in self.defaults:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]
            else:
                self.__dict__[attr] = self.defaults[attr]                

    def load(self, path, roi=None):
        data_dict = {'path': path}
        with h5py.File(path, 'r') as hvf:
            if self.roi_path in hvf:
                data_dict[self.roi_path] = hvf[self.roi_path]
            elif roi:
                data_dict[self.roi_path] = roi
            else:
                data_dict[self.roi_path] = hvf[self.data_path].shape
            