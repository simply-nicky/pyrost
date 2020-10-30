"""
st_sim.py - Speckle Tracking Simulation Python wrapper module
"""
from __future__ import division
from __future__ import absolute_import

import os
import logging
import datetime
import argparse
from sys import stdout
import h5py
import numpy as np
from ..protocol import cxi_protocol, ROOT_PATH
from .parameters import STParams, parameters
from ..bin import aperture, barcode_steps, barcode_2d, lens
from ..bin import make_frames, make_whitefield
from ..bin import fraunhofer_1d, fraunhofer_2d

class STSim():
    """
    Speckle Tracking simulation class (STSim)

    st_params - STParams class object
    bsteps - coordinates of each sample's bar
    """
    log_dir = os.path.join(ROOT_PATH, '../logs')

    def __init__(self, st_params, bsteps=None):
        self.parameters = st_params
        self.__dict__.update(**self.parameters.export_dict())
        self._init_logging()
        self._init_coord()
        self._init_lens()
        self._init_barcode(bsteps)
        self._init_detector()

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

    def _init_coord(self):
        # Initializing coordinate parameters
        xx_span = self.fs_size * self.pix_size
        yy_span = self.ss_size * self.pix_size

        x_span = 1.6 * self.ap_x / self.focus * self.defoc
        y_span = 1.2 * self.ap_y
        n_x = int(x_span * xx_span / self.wl / self.det_dist)
        n_y = int(y_span * yy_span / self.wl / self.det_dist)

        # Initializing coordinate arrays
        self.logger.info("Initializing coordinate arrays at the sample's plane")
        self.logger.info('Number of points in x axis: {:d}'.format(n_x))
        self.logger.info('Number of points in y axis: {:d}'.format(n_y))

        self.x_arr = np.linspace(-x_span / 2, x_span / 2, n_x)
        self.y_arr = np.linspace(-y_span / 2, y_span / 2, n_y)
        self.xx_arr = np.linspace(-xx_span / 2, xx_span / 2, self.fs_size, endpoint=False)
        self.yy_arr = np.linspace(-yy_span / 2, yy_span / 2, self.ss_size, endpoint=False)

    def _init_lens(self):
        #Initializing wavefields at the sample's plane
        self.logger.info("Generating wavefields at the sample's plane")
        self.wf0_x = lens(x_arr=self.x_arr, wl=self.wl, ap=self.ap_x,
                          focus=self.focus, defoc=self.defoc, alpha=self.alpha,
                          x0=(self.x0 - 0.5) * self.ap_x)
        self.wf0_y = aperture(x_arr=self.y_arr, z=self.focus + self.defoc,
                              wl=self.wl, ap=self.ap_y)
        self.i0 = self.p0 / self.ap_x / self.ap_y
        self.smp_c = 1 / self.wl / (self.focus + self.defoc)
        self.logger.info("The wavefields have been generated")

    def _init_barcode(self, bsteps):
        self.logger.info("Generating barcode's transmission coefficients")
        if bsteps is None:
            bsteps = barcode_steps(x0=self.x_arr[0] + self.offset, x1=self.x_arr[-1] + \
                                   self.n_frames * self.step_size - self.offset,
                                   br_dx=self.bar_size, rd=self.random_dev)
        self.bsteps = bsteps
        self.bs_t = barcode_2d(x_arr=self.x_arr, bx_arr=self.bsteps, sgm=self.bar_sigma,
                               atn0=self.bulk_atn, atn=self.bar_atn, ss=self.step_size,
                               nf=self.n_frames)
        self.logger.info("The coefficients have been generated")

    def _init_detector(self):
        self.logger.info("Generating wavefields at the detector's plane")
        self.wf1_y = fraunhofer_1d(wf0=self.wf0_y, x_arr=self.y_arr, xx_arr=self.yy_arr,
                                   dist=self.det_dist, wl=self.wl)
        self.wf1_x = fraunhofer_2d(wf0=self.wf0_x * self.bs_t, x_arr=self.x_arr,
                                   xx_arr=self.xx_arr, dist=self.det_dist, wl=self.wl)
        self.det_c = self.smp_c / self.wl / self.det_dist
        self.logger.info("The wavefields have been generated")

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
    e_to_wl = 1.2398419843320026e-06

    write_attrs = {'basis_vectors', 'data', 'defocus', 'distance',
                   'energy', 'good_frames', 'mask', 'roi', 'translations',
                   'wavelength', 'whitefield', 'x_pixel_size', 'y_pixel_size'}

    def __init__(self, protocol=cxi_protocol(), coord_ratio=1e-6):
        self.protocol, self.crd_rat = protocol, coord_ratio

    def ini_parsers(self, st_params):
        """
        Return ini parsers based on the cxi protocol

        st_params - STParams class object
        """
        ini_parsers = {}
        for filename in os.listdir(self.templates_dir):
            path = os.path.join(self.templates_dir, filename)
            if os.path.isfile(path) and filename.endswith('.ini'):
                template = os.path.splitext(filename)[0]
                ini_parsers[template] = self.protocol.parser_from_template(path)
            else:
                raise RuntimeError("Wrong template file: {:s}".format(path))
        ini_parsers['parameters'] = st_params.export_ini()
        ini_parsers['protocol'] = self.protocol.export_ini()
        return ini_parsers

    def _pixel_vector(self, st_params):
        return self.crd_rat * np.array([st_params.pix_size, st_params.pix_size, 0])

    def _basis_vectors(self, st_params):
        pix_vec = self._pixel_vector(st_params)
        _vec_fs = np.tile(pix_vec * self.unit_vector_fs, (st_params.n_frames, 1))
        _vec_ss = np.tile(pix_vec * self.unit_vector_ss, (st_params.n_frames, 1))
        return np.stack((_vec_ss, _vec_fs), axis=1)

    def _defocus(self, st_params):
        return self.crd_rat * st_params.defoc

    def _distance(self, st_params):
        return self.crd_rat * st_params.det_dist

    def _energy(self, st_params):
        return self.e_to_wl / self._wavelength(st_params)

    def _translations(self, st_params):
        t_arr = np.zeros((st_params.n_frames, 3), dtype=np.float64)
        t_arr[:, 0] = -np.arange(0, st_params.n_frames) * st_params.step_size
        return self.crd_rat * t_arr

    def _wavelength(self, st_params):
        return self.crd_rat * st_params.wl

    def _x_pixel_size(self, st_params):
        return self._pixel_vector(st_params)[0]

    def _y_pixel_size(self, st_params):
        return self._pixel_vector(st_params)[1]

    def save_sim(self, data, st_sim, dir_path):
        """
        Save STSim object at the given folder
        """
        self.save(data=data, st_params=st_sim.parameters,
                  dir_path=dir_path, logger=st_sim.logger)

    def save(self, data, st_params, dir_path, logger=None, roi=None):
        """
        Save speckle tracking data at the given folder

        data - data to be saved
        st_params - STParams object
        dir_path - output path
        logger - logger object
        roi - data region of interest
        """
        if logger:
            logger.info("Saving data in the directory: {:s}".format(dir_path))
            logger.info("Making ini files...")
        os.makedirs(dir_path, exist_ok=True)
        for name, parser in self.ini_parsers(st_params).items():
            ini_path = os.path.join(dir_path, name + '.ini')
            with open(ini_path, 'w') as ini_file:
                parser.write(ini_file)
            if logger:
                logger.info('{:s} saved'.format(ini_path))
        if logger:
            logger.info("Making a cxi file...")
            logger.info("Using the following cxi protocol:")
            self.protocol.log(logger)
        data_dict = {'data': data, 'mask': np.ones(data.shape[1:], dtype=np.uint8),
                     'good_frames': np.arange(data.shape[0])}
        data_dict['whitefield'] = make_whitefield(mask=data_dict['mask'], data=data)
        if roi is None:
            roi = st_params.roi(data.shape)
        data_dict['roi'] = np.asarray(roi)
        with h5py.File(os.path.join(dir_path, 'data.cxi'), 'w') as cxi_file:
            for attr in self.write_attrs:
                if attr in data_dict:
                    self.protocol.write_cxi(attr, data_dict[attr], cxi_file)
                else:
                    dset = self.__getattribute__('_' + attr)(st_params)
                    self.protocol.write_cxi(attr, dset, cxi_file)
        if logger:
            logger.info("{:s} saved".format(os.path.join(dir_path, 'data.cxi')))

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
    parser.add_argument('--x0', type=float, help="Lens' abberations center point [0.0 - 1.0]")
    parser.add_argument('--bar_size', type=float, help="Average bar size [um]")
    parser.add_argument('--bar_sigma', type=float, help="Bar haziness width [um]")
    parser.add_argument('--bar_atn', type=float, help="Bar attenuation")
    parser.add_argument('--bulk_atn', type=float, help="Bulk attenuation")
    parser.add_argument('--random_dev', type=float, help="Bar random deviation")
    parser.add_argument('--offset', type=float, help="sample's offset at the beginning and the end of the scan [um]")
    parser.add_argument('-v', '--verbose', action='store_true', help="Turn on verbosity")
    parser.add_argument('-p', '--ptych', action='store_true', help="Generate ptychograph data")

    params = parameters().export_dict()
    args_dict = vars(parser.parse_args())
    for param in args_dict:
        if not args_dict[param] is None:
            params[param] = args_dict[param]
    if 'ini_file' in params:
        st_params = STParams.import_ini(args_dict['ini_file'])
    else:
        st_params = STParams.import_dict(**params)

    st_sim = STSim(st_params)
    st_converter = STConverter()
    if params['ptych']:
        data = st_sim.ptychograph()
    else:
        data = st_sim.frames()
    st_converter.save_sim(data, st_sim, params['out_path'])
    st_sim.close()
