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
from .bin import *

class INIParser():
    """
    INI files parser class
    """
    err_txt = "Wrong format key '{0:s}' of option '{1:s}'"
    attr_dict = {}

    def __init__(self, **kwargs):
        self.param_dict = {}
        for section in self.attr_dict:
            for option, _ in self.attr_dict[section]:
                self.param_dict[option] = kwargs[option]

    @classmethod
    def ini_parser(cls):
        """
        Return config parser
        """
        return configparser.ConfigParser()

    @classmethod
    def read_ini(cls, ini_file):
        """
        Read ini file
        """
        if not os.path.isfile(ini_file):
            raise ValueError("File {:s} doesn't exist".format(ini_file))
        ini_parser = cls.ini_parser()
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
        if attr in self.param_dict:
            return self.param_dict[attr]

    def save_ini(self, filename):
        """
        Save experiment settings to an ini file
        """
        ini_parser = self.ini_parser()
        for section in self.attr_dict:
            ini_parser[section] = {option: self.param_dict[option]
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
    frame_size - frames size in pixels

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
                            ('frame_size', 'int')},
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
                                 self.frame_size)
        self.xx_arr = np.linspace(-0.8 * self.ap_x / self.focus * self.det_dist,
                                  0.8 * self.ap_x / self.focus * self.det_dist,
                                  self.frame_size)
        self.y_arr = np.linspace(-0.6 * self.ap_y, 0.6 * self.ap_y, self.frame_size)
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
    
    def cxi_converter(self):
        """
        Return CXI Converter
        """
        return CXIConverter(self)

class CXIConverter():
    """
    CXI converter class

    st_sim - STSim class object
    """
    unit_vector_fs = np.array([0, -1, 0])
    unit_vector_ss = np.array([-1, 0, 0])

    def __init__(self, st_sim):
        self._init_params(st_sim)
        self._init_data(st_sim)
    
    def _init_params(self, st_sim):
        self.step_size, self.n_frames = st_sim.step_size, st_sim.n_frames
        self.dist, self.wl = st_sim.det_dist, st_sim.wl
        self.energy = constants.c * constants.h / st_sim.wl / constants.e
        self.pixel_vector = np.array([(st_sim.xx_arr[-1] - st_sim.xx_arr[0]) / st_sim.frame_size,
                                      (st_sim.y_arr[-1] - st_sim.y_arr[0]) / st_sim.frame_size, 0])

    def _init_data(self, st_sim):
        self.data = st_sim.det_frames()
        self.mask = np.ones((st_sim.frame_size, st_sim.frame_size), dtype=np.uint8)
        self.whitefield = make_whitefield(data=self.data, mask=self.mask)

    def basis_vectors(self):
        """
        Return detector fast and slow axis basis vectors
        """
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.n_frames, 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.n_frames, 1))
        return np.stack((_vec_ss, _vec_fs), axis=1)

    def translation(self):
        """
        Translate detector coordinates
        """
        t_arr = np.zeros((self.n_frames, 3), dtype=np.float64)
        t_arr[:, 0] = np.arange(0, self.n_frames) * self.step_size
        return t_arr

    def save(self, filename):
        """
        Save data to a cxi file with the given filename
        """
        with h5py.File(filename, 'w') as cxi_file:
            detector_1 = cxi_file.create_group('entry_1/instrument_1/detector_1')
            detector_1.create_dataset('basis_vectors', data=self.basis_vectors())
            detector_1.create_dataset('distance', data=self.dist)
            detector_1.create_dataset('x_pixel_size', data=self.pixel_vector[0])
            detector_1.create_dataset('y_pixel_size', data=self.pixel_vector[1])
            source_1 = cxi_file.create_group('entry_1/instrument_1/source_1')
            source_1.create_dataset('energy', data=self.energy)
            source_1.create_dataset('wavelength', data=self.wl)
            cxi_file.create_dataset('entry_1/sample_3/geometry/translation', data=self.translation())
            cxi_file.create_dataset('frame_selector/good_frames', data=np.ones(self.n_frames, dtype=np.uint8))
            cxi_file.create_dataset('mask_maker/mask', data=self.mask)
            cxi_file.create_dataset('make_whitefield/whitefield', data=self.whitefield)
            cxi_file.create_dataset('entry_1/data_1/data', data=self.data)

def main():
    """
    Main fuction to run Speckle Tracking simulation and save the data to a cxi file
    """
    parser = argparse.ArgumentParser(description="Run Speckle Tracking simulation")
    parser.add_argument('out_path', type=str, help="Outpu t cxi file path")
    parser.add_argument('-f', '--ini_file', type=str, help="Path to an INI file to fetch all of the simulation parameters")
    parser.add_argument('--defoc', type=float, default=1e2, help="Lens defocus distance, [um]")
    parser.add_argument('--det_dist', type=float, default=1.5e6, help="Distance between the barcode and the detector [um]")
    parser.add_argument('--step_size', type=float, default=5e-2, help="Scan step size [um]")
    parser.add_argument('--n_frames', type=int, default=300, help="Number of frames")
    parser.add_argument('--frame_size', type=int, default=2000, help="Frames size in pixels")
    parser.add_argument('--i0', type=float, default=1e4, help="Source intensity [cnt]")
    parser.add_argument('--wl', type=float, default=7.29e-5, help="Wavelength [um]")
    parser.add_argument('--ap_x', type=float, default=30, help="Lens size along the x axis [um]")
    parser.add_argument('--ap_y', type=float, default=60, help="Lens size along the y axis [um]")
    parser.add_argument('--focus', type=float, default=2e3, help="Focal distance [um]")
    parser.add_argument('--alpha', type=float, default=0.1, help="Third order abberations [rad/mrad^3]")
    parser.add_argument('--bar_size', type=float, default=3, help="Average bar size [um]")
    parser.add_argument('--bar_sigma', type=float, default=1e-1, help="Bar haziness width [um]")
    parser.add_argument('--attenuation', type=float, default=0.3, help="Bar attenuation")
    parser.add_argument('--random_dev', type=float, default=0.3, help="Bar random deviation")

    params = vars(parser.parse_args())
    if params['ini_file']:
        params.update(STSim.import_ini(params['ini_file']))

    st_sim = STSim(**params)
    st_sim.cxi_converter().save(params['out_path'])