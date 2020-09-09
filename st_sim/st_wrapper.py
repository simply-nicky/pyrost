"""
st_wrapper.py - Speckle Tracking Simulation Python wrapper module
"""
from __future__ import division
from __future__ import absolute_import

import os
import configparser
import logging
import datetime
import argparse
from sys import stdout
import h5py
import numpy as np
from scipy import constants
from .bin import aperture, barcode_steps, barcode, lens
from .bin import make_frames, make_whitefield
from .bin import fraunhofer_1d, fraunhofer_2d

ROOT_PATH = os.path.dirname(__file__)

class hybridmethod:
    """
    Hybrid method descriptor supporting two distinct methods bound to class and instance

    fclass - class bound method
    finstance - instance bound method
    doc - documentation
    """
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass, self.finstance = fclass, finstance
        self.__doc__ = doc or fclass.__doc__
        self.__isabstractmethod__ = bool(getattr(fclass, '__isabstractmethod__', False))

    def classmethod(self, fclass):
        """
        Class method decorator

        fclass - class bound method
        """
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        """
        Instance method decorator

        finstance - instance bound method
        """
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)

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
                elif fmt == 'bool':
                    param_dict[option] = ini_parser.getboolean(section, option)
                elif fmt == 'str':
                    param_dict[option] = ini_parser.get(section, option)
                else:
                    raise ValueError(cls.err_txt.format(fmt, option))
        return param_dict

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]

    @hybridmethod
    def export_ini(cls, **kwargs):
        """
        Return ini parser

        kwargs - extra parameters to save
        """
        ini_parser = configparser.ConfigParser()
        for section in cls.attr_dict:
            ini_parser[section] = {option: kwargs[option]
                                   for option, fmt in cls.attr_dict[section]}
        return ini_parser

    @export_ini.instancemethod
    def export_ini(self):
        """
        Return ini parser

        kwargs - extra parameters to save
        """
        return type(self).export_ini(**self.__dict__)

class STSim(INIParser):
    """
    Speckle Tracking Scan simulation (STSim)

    [exp_geom - experimental geometry parameters]
    defoc - lens defocus distance [um]
    det_dist - distance between the barcode and the detector [um]
    step_size - scan step size [um]
    n_frames - number of frames

    [detector - detector parameters]
    fs_size - fast axis frames size in pixels
    ss_size - slow axis frames size in pixels
    pix_size - detector pixel size [um]

    [source - source parameters]
    p0 - Source beam flux [cnt / s]
    wl - wavelength [um]
    th_s - source rocking curve width [rad]

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
    log_dir = os.path.join(ROOT_PATH, '../logs')
    attr_dict = {'exp_geom': {('defoc', 'float'), ('det_dist', 'float'),
                              ('step_size', 'float'), ('n_frames', 'int')},
                 'detector': {('fs_size', 'int'), ('ss_size', 'int'),
                              ('pix_size', 'float')},
                 'source':   {('p0', 'float'), ('wl', 'float'), ('th_s', 'float')},
                 'lens':     {('ap_x', 'float'), ('ap_y', 'float'),
                              ('focus', 'float'), ('alpha', 'float')},
                 'barcode':  {('bar_size', 'float'), ('bar_sigma', 'float'),
                              ('attenuation', 'float'), ('random_dev', 'float')},
                 'system':   {('verbose', 'bool')}}

    def __init__(self, **kwargs):
        super(STSim, self).__init__(**kwargs)
        self._init_logging()
        self._init_sample_data()
        self._init_det_data()

    def _init_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.level = logging.INFO
        filename = os.path.join(self.log_dir, datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S.log'))
        self.logger.addHandler(logging.FileHandler(filename))
        if self.verbose:
            self.logger.addHandler(logging.StreamHandler(stdout))
        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.info('Initializing')
        self.logger.info('Current parameters')
        for section in self.attr_dict:
            self.logger.info('[{:s}]'.format(section))
            for option, fmt in self.attr_dict[section]:
                log_txt = '\t{0:s}: {1:s}, [{2:s}]'.format(option, str(self.__dict__[option]), fmt)
                self.logger.info(log_txt)

    def _init_sample_data(self):
        x_span = self.det_beam_span(self.defoc + self.det_dist)
        det_beam_dx = max(x_span[1] - x_span[0], self.pix_size * self.fs_size)
        n_x = int(1.6 * self.ap_x / self.focus * self.defoc * det_beam_dx / self.wl / self.det_dist)
        n_y = int(1.2 * self.ap_y * self.ss_size * self.pix_size / self.wl / self.det_dist)
        self.logger.info("Initializing coordinate arrays at the sample's plane")
        self.logger.info('Number of points in x axis: {:d}'.format(n_x))
        self.logger.info('Number of points in y axis: {:d}'.format(n_y))
        self.x_arr = np.linspace(-0.8 * self.ap_x / self.focus * self.defoc,
                                 0.8 * self.ap_x / self.focus * self.defoc, n_x)
        self.y_arr = np.linspace(-0.6 * self.ap_y, 0.6 * self.ap_y, n_y)
        self.xx_arr = self.pix_size * np.arange(-self.fs_size // 2, self.fs_size // 2)
        self.yy_arr = self.pix_size * np.arange(-self.ss_size // 2, self.ss_size // 2)
        self.logger.info("Generating wavefields at the sample's plane")
        self.wf0_x = lens(x_arr=self.x_arr, wl=self.wl, ap=self.ap_x, focus=self.focus,
                          defoc=self.defoc, alpha=self.alpha)
        self.wf0_y = aperture(x_arr=self.y_arr, z=self.focus + self.defoc,
                              wl=self.wl, ap=self.ap_y)
        self.i0, self.smp_c = self.p0 / self.ap_x / self.ap_y, 1 / self.wl / (self.focus + self.defoc)
        self.logger.info("Wavefields generated")

    def _init_det_data(self):
        self.logger.info("Generating wavefields at the detector's plane")
        self.wf1_y = fraunhofer_1d(wf0=self.wf0_y, x_arr=self.y_arr, xx_arr=self.yy_arr,
                                   dist=self.det_dist, wl=self.wl)
        self.bsteps = barcode_steps(beam_dx=self.x_arr[-1] - self.x_arr[0],
                                    bar_size=self.bar_size, rnd_div=self.random_dev,
                                    step_size=self.step_size, n_frames=self.n_frames)
        self.bs_t = barcode(x_arr=self.x_arr, bsteps=self.bsteps, b_sigma=self.bar_sigma,
                            atn=self.attenuation, step_size=self.step_size, n_frames=self.n_frames)
        self.wf1_x = fraunhofer_2d(wf0=self.wf0_x * self.bs_t, x_arr=self.x_arr, xx_arr=self.xx_arr,
                                   dist=self.det_dist, wl=self.wl)
        self.det_c = self.smp_c / self.wl / self.det_dist
        self.logger.info("Wavefields generated")

    def source_curve(self, dist, dx):
        """
        Return source rocking curve array at the given distance from the lens

        dist - distance from the lens
        dx - sampling period
        """
        sigma = self.th_s * dist
        x_arr = dx * np.arange(-np.ceil(4 * sigma / dx), np.ceil(4 * sigma / dx) + 1)
        s_arr = np.exp(-x_arr**2 / 2 / sigma**2)
        return s_arr / s_arr.sum()

    def det_beam_span(self, dist):
        """
        Return beam span in x coordinate at the detector's plane at the given distance from the focal plane
        """
        fx_lim = np.array([-0.5 * self.ap_x / self.focus, 0.5 * self.ap_x / self.focus])
        th_lim = fx_lim + self.wl / 2 / np.pi * self.alpha * 3e9 * fx_lim**2 / dist
        return np.tan(th_lim) * dist

    def lens_phase(self):
        """
        Return lens wavefront at the sample plane, defocus, and abberations coefficient
        """
        beam_span = np.clip(self.det_beam_span(self.defoc), self.x_arr[0], self.x_arr[-1])
        roi = ((beam_span - self.x_arr[0]) // (self.x_arr[1] - self.x_arr[0])).astype(np.int)
        phase = np.unwrap(np.angle(self.wf0_x))
        ph_fit = np.polyfit(self.x_arr[roi[0]:roi[1]], phase[roi[0]:roi[1]], 3)
        defoc = np.pi / self.wl / ph_fit[1]
        res = {'phase': phase, 'defocus': defoc, 'alpha': -ph_fit[0] * defoc**3 * 1e-9}
        return res

    def snr(self):
        """
        Return SNR of intensity frames at the deetector plane
        """
        pdet = self.p0 * (1 - self.attenuation / 2)
        beam_span = np.clip(self.det_beam_span(self.defoc + self.det_dist), self.xx_arr[0], self.xx_arr[-1])
        ppix = pdet * self.pix_size / (beam_span[1] - beam_span[0])
        sigma_s = self.attenuation / 2 * np.sqrt((1 - 4 * self.bar_sigma / self.bar_size * np.tanh(self.bar_size / 4 / self.bar_sigma)))
        return sigma_s * np.sqrt(ppix)

    def sample_wavefronts(self):
        """
        Return wavefront profiles along the x axis at the sample plane
        """
        return self.wf0_x * self.bs_t * self.smp_c * self.wf0_y.max()

    def det_wavefronts(self):
        """
        Return wavefront profiles along the x axis at the detector plane
        """
        return self.wf1_x * self.det_c * self.wf1_y.max()

    def frames(self):
        """
        Return intensity frames at the detector plane
        """
        self.logger.info("Making frames at the detector's plane...")
        self.logger.info("Source blur size: {:f} um".format(self.th_s * self.det_dist))
        s_arr = self.source_curve(self.det_dist, self.pix_size)
        frames = make_frames(i_x=self.i0 * np.abs(self.wf1_x * self.det_c)**2,
                             i_y=np.abs(self.wf1_y)**2, sc_x=s_arr, sc_y=s_arr,
                             pix_size=self.pix_size)
        self.logger.info("The frames are generated, data shape: {:s}".format(str(frames.shape)))
        return frames

    def ptychograph(self):
        """
        Return a ptychograph
        """
        self.logger.info("Making ptychograph...")
        self.logger.info("Source blur size: {:f} um".format(self.th_s * self.det_dist))
        s_arr = self.source_curve(self.det_dist, self.pix_size)
        ptych = make_frames(i_x=self.i0 * np.abs(self.wf1_x * self.det_c)**2,
                            i_y=(np.abs(self.wf1_y)**2).sum() * np.ones(1),
                            sc_x=s_arr, sc_y=np.ones(1), pix_size=self.pix_size)
        self.logger.info("The ptychograph is generated, data shape: {:s}".format(str(ptych.shape)))
        return ptych

    def frames_cxi(self):
        """
        Return CXI Converter
        """
        return STConverter(st_sim=self, data=self.frames())

    def ptych_cxi(self):
        """
        Return CXI Converter
        """
        return STConverter(st_sim=self, data=self.ptychograph())

    def close(self):
        """
        Close logging handlers
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

class STConverter():
    """
    Converter class to Andrew's speckle-tracking software data format

    for more info, open the url:
    https://github.com/andyofmelbourne/speckle-tracking
    """
    unit_vector_fs = np.array([1, 0, 0])
    unit_vector_ss = np.array([0, -1, 0])
    templates_dir = os.path.join(ROOT_PATH, 'ini_templates')

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
                'energy_path': '/entry_1/instrument_1/source_1/energy',
                'wavelength_path': '/entry_1/instrument_1/source_1/wavelength',
                'basis_vectors_path': '/entry_1/instrument_1/detector_1/basis_vectors'}

    def __init__(self, st_sim, data, **kwargs):
        for attr in self.defaults:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]
            else:
                self.__dict__[attr] = self.defaults[attr]
        self._init_params(st_sim)
        self._init_parsers(st_sim)
        self._init_data(st_sim, data)

    def _init_params(self, st_sim):
        self.logger = st_sim.logger
        self.logger.info('Initializing a cxi file')
        self.logger.info("The file's data tree:")
        for attr in self.attr_set:
            self.logger.info("{0:s}: '{1:s}'".format(attr, str(self.__dict__[attr])))
        self.step_size, self.n_frames = st_sim.step_size * 1e-6, st_sim.n_frames
        self.dist, self.defoc = st_sim.det_dist * 1e-6, st_sim.defoc * 1e-6
        self.wavelength = st_sim.wl * 1e-6
        self.energy = constants.h * constants.c / self.wavelength / constants.e
        self.pixel_vector = 1e-6 * np.array([st_sim.pix_size, st_sim.pix_size, 0])

    def _init_parsers(self, st_sim):
        self.logger.info("Initializing speckle_tracking gui ini files")
        self.templates = []
        for filename in os.listdir(self.templates_dir):
            path = os.path.join(self.templates_dir, filename)
            if os.path.isfile(path) and filename.endswith('.ini'):
                template = os.path.splitext(filename)[0]
                self.__dict__[template] = self.parser_from_template(path)
                self.templates.append(template)
            else:
                raise RuntimeError("Wrong template file: {:s}".format(path))
        self.__dict__['parameters'] = st_sim.export_ini()
        self.templates.append('parameters')
        self.logger.info("The ini file parsers are initialized: {:s}".format(str(self.templates)))

    def _init_data(self, st_sim, data):
        self.logger.info("Making speckle tracking data...")
        self.data = data
        self.mask = np.ones((self.data.shape[1], self.data.shape[2]), dtype=np.uint8)
        self.whitefield = make_whitefield(data=self.data, mask=self.mask)
        beam_span = st_sim.det_beam_span(st_sim.defoc + st_sim.det_dist)
        x_roi = np.clip((beam_span - st_sim.xx_arr.min()) // st_sim.pix_size,
                        0, st_sim.fs_size)
        self.roi = (data.shape[1] // 2, data.shape[1] // 2 + 1, x_roi[0], x_roi[1])
        log_txt = "The data is generated, mask shape: {0:s}, whitefield shape: {1:s}".format(str(self.mask.shape),
                                                                                             str(self.whitefield.shape))
        self.logger.info(log_txt)

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
        t_arr[:, 0] = -np.arange(0, self.n_frames) * self.step_size
        return t_arr

    def save(self, dir_path):
        """
        Save the data at the given folder
        """
        self.logger.info("Saving data in the directory: {:s}".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)
        for template in self.templates:
            ini_path = os.path.join(dir_path, template + '.ini')
            with open(ini_path, 'w') as ini_file:
                self.__getattribute__(template).write(ini_file)
            self.logger.info('{:s} saved'.format(ini_path))
        self.logger.info("Making a cxi file...")
        with h5py.File(os.path.join(dir_path, 'data.cxi'), 'w') as cxi_file:
            cxi_file.create_dataset(self.basis_vectors_path, data=self.basis_vectors())
            cxi_file.create_dataset(self.distance_path, data=self.dist)
            cxi_file.create_dataset(self.roi_path, data=np.array(self.roi, dtype=np.int64))
            cxi_file.create_dataset(self.defocus_path, data=self.defoc)
            cxi_file.create_dataset(self.x_pixel_size_path, data=self.pixel_vector[1])
            cxi_file.create_dataset(self.y_pixel_size_path, data=self.pixel_vector[0])
            cxi_file.create_dataset(self.energy_path, data=self.energy)
            cxi_file.create_dataset(self.wavelength_path, data=self.wavelength)
            cxi_file.create_dataset(self.translations_path, data=self.translations())
            cxi_file.create_dataset(self.good_frames_path, data=np.arange(self.n_frames))
            cxi_file.create_dataset(self.mask_path, data=self.mask.astype(bool))
            cxi_file.create_dataset(self.whitefield_path, data=self.whitefield.astype(np.float32))
            cxi_file.create_dataset(self.data_path, data=self.data.astype(np.float32))
        self.logger.info("{:s} saved".format(os.path.join(dir_path, 'data.cxi')))

def defaults():
    """
    Return default parameters
    """
    return STSim.import_ini(os.path.join(ROOT_PATH, 'default.ini'))

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
    parser.add_argument('--p0', type=float, help="Source beam flux [cnt / s]")
    parser.add_argument('--wl', type=float, help="Wavelength [um]")
    parser.add_argument('--th_s', type=float, help="Source rocking curve width [rad]")
    parser.add_argument('--ap_x', type=float, help="Lens size along the x axis [um]")
    parser.add_argument('--ap_y', type=float, help="Lens size along the y axis [um]")
    parser.add_argument('--focus', type=float, help="Focal distance [um]")
    parser.add_argument('--alpha', type=float, help="Third order abberations [rad/mrad^3]")
    parser.add_argument('--bar_size', type=float, help="Average bar size [um]")
    parser.add_argument('--bar_sigma', type=float, help="Bar haziness width [um]")
    parser.add_argument('--attenuation', type=float, help="Bar attenuation")
    parser.add_argument('--random_dev', type=float, help="Bar random deviation")
    parser.add_argument('-v', '--verbose', action='store_true', help="Turn on verbosity")
    parser.add_argument('-p', '--ptych', action='store_true', help="Generate ptychograph data")

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
        if params['ptych']:
            st_sim.ptych_cxi().save(params['out_path'])
        else:
            st_sim.frames_cxi().save(params['out_path'])
    else:
        raise RuntimeError("Default ini file doesn't exist")
