"""
st_wrapper.py - Speckle Tracking Simulation Python wrapper module
"""
from __future__ import division
from __future__ import absolute_import

import os
import configparser
import argparse
import h5py
import numpy as np
from scipy import constants
from .bin import lens_wf, aperture_wf, fraunhofer_1d, barcode_steps
from .bin import barcode, fraunhofer_2d, make_frames, make_whitefield

ROOT_PATH = os.path.dirname(__file__)

class INIParser():
    """
    INI files parser class
    """
    err_txt = "Wrong format key '{0:s}' of option '{1:s}'"
    attr_dict = {}

    def __init__(self, **kwargs):
        for section in self.attr_dict:
            for option, _ in self.attr_dict[section]:
                self.__dict__[option] = kwargs[option]

    @classmethod
    def read_ini(cls, ini_file):
        """
        Read ini file
        """
        if not os.path.isfile(ini_file):
            raise ValueError("File {:s} doesn't exist".format(ini_file))
        ini_parser = configparser.ConfigParser()
        ini_parser.read(ini_file)
        return ini_parser

    @classmethod
    def import_ini(cls, ini_file):
        """
        Initialize an object class with the ini file

        ini_file - ini file path
        """
        ini_parser = cls.read_ini(ini_file)
        param_dict = {}
        for section in cls.attr_dict:
            for option, fmt in cls.attr_dict[section]:
                if fmt == 'float':
                    param_dict[option] = ini_parser.getfloat(section, option)
                elif fmt == 'int':
                    param_dict[option] = ini_parser.getint(section, option)
                else:
                    raise ValueError(cls.err_txt.format(fmt, option))
        return param_dict

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]

    def save_ini(self, filename):
        """
        Save experiment settings to an ini file
        """
        ini_parser = self.ini_parser()
        for section in self.attr_dict:
            ini_parser[section] = {option: self.__dict__[option]
                                   for option, fmt in self.attr_dict[section]}
        with open(filename, 'w') as ini_file:
            ini_parser.write(ini_file)

class STSim(INIParser):
    """
    Speckle Tracking Scan simulation (STSim)

    [exp_geom - experimental geometry parameters]
    defoc - lens defocus distance [um]
    det_dist - distance between the barcode and the detector [um]
    step_size - scan step size [um]
    n_frames - number of frames
    fs_size - fast axis frames size in pixels
    ss_size - slow axis frames size in pixels

    [source - source parameters]
    i0 - Source intensity [cnt]
    wl - wavelength [um]

    [lens - lens parameters]
    ap_x - lens size along the x axis [um]
    ap_y - lens size along the y axis [um]
    focus - focal distance [um]
    alpha - third order abberations [rad/mrad^3]

    [barcode - barcode parameters]
    bar_size - average bar size [um]
    bar_sigma - bar haziness width [um]
    attenuation - bar attenuation
    random_dev - bar random deviation
    """
    attr_dict = {'exp_geom': {('defoc', 'float'), ('det_dist', 'float'),
                              ('step_size', 'float'), ('n_frames', 'int'),
                              ('fs_size', 'int'), ('ss_size', 'int')},
                 'source':   {('i0', 'float'), ('wl', 'float')},
                 'lens':     {('ap_x', 'float'), ('ap_y', 'float'),
                              ('focus', 'float'), ('alpha', 'float')},
                 'barcode':  {('bar_size', 'float'), ('bar_sigma', 'float'),
                              ('attenuation', 'float'), ('random_dev', 'float')}}

    def __init__(self, **kwargs):
        super(STSim, self).__init__(**kwargs)
        self._init_data()

    def _init_data(self):

        self.x_arr = np.linspace(-0.8 * self.ap_x / self.focus * self.defoc,
                                 0.8 * self.ap_x / self.focus * self.defoc,
                                 int(1.6**2 * self.ap_x**2 * self.defoc / self.focus**2 / self.wl))
        self.xx_arr = np.linspace(-0.8 * self.ap_x / self.focus * self.det_dist,
                                  0.8 * self.ap_x / self.focus * self.det_dist,
                                  self.fs_size)
        self.y_arr = np.linspace(-0.6 * self.ap_y, 0.6 * self.ap_y, self.ss_size)
        self.wf0_x = lens_wf(x_arr=self.x_arr, defoc=self.defoc, f=self.focus,
                             wl=self.wl, alpha=self.alpha, ap=self.ap_x)
        self.wf0_y = aperture_wf(x_arr=self.y_arr, z=self.focus + self.defoc,
                                 wl=self.wl, ap=self.ap_y)
        self.wf1_y = fraunhofer_1d(wf0=self.wf0_y, x_arr=self.y_arr, xx_arr=self.y_arr,
                                   dist=self.det_dist, wl=self.wl)
        self.bsteps = barcode_steps(beam_dx=self.x_arr[-1] - self.x_arr[0],
                                    bar_size=self.bar_size, rnd_div=self.random_dev,
                                    step_size=self.step_size, n_frames=self.n_frames)
        self.bs_t = barcode(x_arr=self.x_arr, bsteps=self.bsteps, b_sigma=self.bar_sigma,
                            atn=self.attenuation, step_size=self.step_size, n_frames=self.n_frames)
        self.wf1_x = fraunhofer_2d(wf0=self.wf0_x * self.bs_t, x_arr=self.x_arr, xx_arr=self.xx_arr,
                                   dist=self.det_dist, wl=self.wl)
        self.smp_c = np.sqrt(self.i0) / self.wl / (self.focus + self.defoc)
        self.det_c = self.smp_c / self.wl / self.det_dist

    def sample_wavefronts(self):
        """
        Return wavefront profiles along the x axis at the sample plane
        """
        return self.wf0_x * self.bs_t * self.smp_c * self.wf0_y.max()

    def sample_frames(self):
        """
        Return intensity frames at the sample plane
        """
        return make_frames(wf_x=self.wf0_x * self.bs_t * self.smp_c, wf_y=self.wf0_y)

    def det_wavefronts(self):
        """
        Return wavefront profiles along the x axis at the detector plane
        """
        return self.wf1_x * self.det_c * self.wf1_y.max()

    def det_frames(self):
        """
        Return intensity frames at the detector plane
        """
        return make_frames(wf_x=self.wf1_x * self.det_c, wf_y=self.wf1_y)

    def ptychograph(self):
        """
        Return a ptychograph
        """
        return np.abs(self.det_c * self.wf1_x * self.wf1_y.max())**2

    def cxi_converter(self):
        """
        Return CXI Converter
        """
        return STConverter(st_sim=self)


class STConverter():
    """
    Converter class to Andrew's speckle-tracking software data format

    for more info, open the url:
    https://github.com/andyofmelbourne/speckle-tracking
    """
    unit_vector_fs = np.array([0, -1, 0])
    unit_vector_ss = np.array([-1, 0, 0])
    templates_dir = os.path.join(ROOT_PATH, 'ini_templates')

    defaults = {'data_path': '/entry_1/data_1/data',
                'whitefield_path': '/speckle_tracking/whitefield',
                'mask_path': '/speckle_tracking/mask', 'roi': (490, 510, 400, 1600),
                'roi_path': '/speckle_tracking/roi',
                'defocus_path': '/speckle_tracking/defocus',
                'translations_path': '/entry_1/sample_1/geometry/translations',
                'good_frames_path': '/frame_selector/good_frames',
                'y_pixel_size_path': '/entry_1/instrument_1/detector_1/y_pixel_size',
                'x_pixel_size_path': '/entry_1/instrument_1/detector_1/x_pixel_size',
                'distance_path': '/entry_1/instrument_1/detector_1/distance',
                'energy_path': '/entry_1/instrument_1/source_1/energy',
                'wavelength_path': '/entry_1/instrument_1/source_1/wavelength',
                'basis_vectors_path': '/entry_1/instrument_1/detector_1/basis_vectors'}

    attr_set = {'data_path', 'whitefield_path', 'mask_path', 'translations_path',
                'good_frames_path', 'y_pixel_size_path', 'x_pixel_size_path',
                'distance_path', 'energy_path', 'wavelength_path', 'basis_vectors_path',
                'roi', 'roi_path', 'defocus_path'}

    def __init__(self, st_sim, **kwargs):
        for attr in self.attr_set:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]
            elif attr in self.defaults:
                self.__dict__[attr] = self.defaults[attr]
            else:
                raise ValueError("Parameter '{:s}' has not been provided".format(attr))
        self._init_params(st_sim)
        self._init_parsers()
        self._init_data(st_sim)

    def _init_params(self, st_sim):
        self.step_size, self.n_frames = st_sim.step_size * 1e-6, st_sim.n_frames
        self.dist, self.defoc = st_sim.det_dist * 1e-6, st_sim.defoc * 1e-6
        self.wavelength = st_sim.wl * 1e-6
        self.energy = constants.h * constants.c / self.wavelength / constants.e
        self.pixel_vector = 1e-6 * np.array([(st_sim.xx_arr[-1] - st_sim.xx_arr[0]) / st_sim.fs_size,
                                             (st_sim.y_arr[-1] - st_sim.y_arr[0]) / st_sim.ss_size, 0])

    def _init_data(self, st_sim):
        self.data = st_sim.det_frames()
        self.mask = np.ones((self.data.shape[1], self.data.shape[2]), dtype=np.uint8)
        self.whitefield = make_whitefield(data=self.data, mask=self.mask)

    def _init_parsers(self):
        self.templates = []
        for filename in os.listdir(self.templates_dir):
            path = os.path.join(self.templates_dir, filename)
            if os.path.isfile(path) and filename.endswith('.ini'):
                template = os.path.splitext(filename)[0]
                self.__dict__[template] = self.parser_from_template(path)
                self.templates.append(template)
            else:
                raise RuntimeError("Wrong template file: {:s}".format(path))

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]

    def parser_from_template(self, template):
        """
        Return parser object using an ini file template

        template - path to the template file
        """
        ini_template = configparser.ConfigParser()
        ini_template.read(template)
        parser = configparser.ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(self)
                               for option in ini_template[section]}
        return parser

    def basis_vectors(self):
        """
        Return detector fast and slow axis basis vectors
        """
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.n_frames, 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.n_frames, 1))
        return np.stack((_vec_ss, _vec_fs), axis=1)

    def translations(self):
        """
        Translate detector coordinates
        """
        t_arr = np.zeros((self.n_frames, 3), dtype=np.float64)
        t_arr[:, 0] = np.arange(0, self.n_frames) * self.step_size
        return t_arr

    def save(self, dir_path):
        """
        Save the data at the given folder
        """
        os.makedirs(dir_path, exist_ok=True)
        for template in self.templates:
            ini_path = os.path.join(dir_path, template + '.ini')
            with open(ini_path, 'w') as ini_file:
                self.__getattribute__(template).write(ini_file)
        with h5py.File(os.path.join(dir_path, 'data.cxi'), 'w') as cxi_file:
            cxi_file.create_dataset(self.basis_vectors_path, data=self.basis_vectors())
            cxi_file.create_dataset(self.distance_path, data=self.dist)
            cxi_file.create_dataset(self.roi_path, data=np.array(self.roi))
            cxi_file.create_dataset(self.defocus_path, data=self.defoc)
            cxi_file.create_dataset(self.x_pixel_size_path, data=self.pixel_vector[0])
            cxi_file.create_dataset(self.y_pixel_size_path, data=self.pixel_vector[1])
            cxi_file.create_dataset(self.energy_path, data=self.energy)
            cxi_file.create_dataset(self.wavelength_path, data=self.wavelength)
            cxi_file.create_dataset(self.translations_path, data=self.translations())
            cxi_file.create_dataset(self.good_frames_path, data=np.arange(self.n_frames))
            cxi_file.create_dataset(self.mask_path, data=self.mask.astype(bool))
            cxi_file.create_dataset(self.whitefield_path, data=self.whitefield.astype(np.float32))
            cxi_file.create_dataset(self.data_path, data=self.data.astype(np.float32))

def main():
    """
    Main fuction to run Speckle Tracking simulation and save the data to a cxi file
    """
    parser = argparse.ArgumentParser(description="Run Speckle Tracking simulation")
    parser.add_argument('out_path', type=str, help="Output folder path")
    parser.add_argument('-f', '--ini_file', type=str,
                        help="Path to an INI file to fetch all of the simulation parameters")
    parser.add_argument('--defoc', type=float, help="Lens defocus distance, [um]")
    parser.add_argument('--det_dist', type=float, help="Distance between the barcode and the detector [um]")
    parser.add_argument('--step_size', type=float, help="Scan step size [um]")
    parser.add_argument('--n_frames', type=int, help="Number of frames")
    parser.add_argument('--fs_size', type=int, help="Fast axis frames size in pixels")
    parser.add_argument('--ss_size', type=int, help="Slow axis frames size in pixels")
    parser.add_argument('--i0', type=float, help="Source intensity [cnt]")
    parser.add_argument('--wl', type=float, help="Wavelength [um]")
    parser.add_argument('--ap_x', type=float, help="Lens size along the x axis [um]")
    parser.add_argument('--ap_y', type=float, help="Lens size along the y axis [um]")
    parser.add_argument('--focus', type=float, help="Focal distance [um]")
    parser.add_argument('--alpha', type=float, help="Third order abberations [rad/mrad^3]")
    parser.add_argument('--bar_size', type=float, help="Average bar size [um]")
    parser.add_argument('--bar_sigma', type=float, help="Bar haziness width [um]")
    parser.add_argument('--attenuation', type=float, help="Bar attenuation")
    parser.add_argument('--random_dev', type=float, help="Bar random deviation")

    default_path = os.path.join(ROOT_PATH, 'default.ini')
    if os.path.isfile(default_path):
        params = STSim.import_ini(default_path)
        args_dict = vars(parser.parse_args())
        for param in args_dict:
            if args_dict[param] is not None:
                params[param] = args_dict[param]
        if 'ini_file' in params:
            params.update(STSim.import_ini(params['ini_file']))

        st_sim = STSim(**params)
        st_sim.cxi_converter().save(params['out_path'])
    else:
        raise RuntimeError("Default ini file doesn't exist")
