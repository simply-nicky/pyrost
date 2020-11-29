"""One-dimensional Speckle Tracking scan simulation.
Generate intensity frames based on Fresnel diffraction
theory. :class:`STSim` does the heavy lifting of calculating
the wavefront propagation to the detector plane.
:class:`STConverter` exports simulated data to a `CXI`_ format
file accordingly to the provided :class:`Protocol` object and
saves the protocol and experimental parameters to the same folder.

Logs of the simulation process are saved in `logs` folder.

Examples
--------

Perform the simulation for a given :class:`STParams` object.

>>> import pyrost.simulation as st_sim
>>> st_params = st_sim.parameters()
>>> sim_obj = st_sim.STSim(st_params)
2020-11-20 12:21:21,574 - STSim - INFO - Initializing
2020-11-20 12:21:21,575 - STSim - INFO - Current parameters
2020-11-20 12:21:21,576 - STSim - INFO - Initializing coordinate arrays at the sample's plane
2020-11-20 12:21:21,576 - STSim - INFO - Number of points in x axis: 12876
2020-11-20 12:21:21,577 - STSim - INFO - Number of points in y axis: 905
2020-11-20 12:21:21,579 - STSim - INFO - Generating wavefields at the sample's plane
2020-11-20 12:21:50,557 - STSim - INFO - The wavefields have been generated
2020-11-20 12:21:50,559 - STSim - INFO - Generating barcode's transmission coefficients
2020-11-20 12:21:50,776 - STSim - INFO - The coefficients have been generated
2020-11-20 12:21:50,777 - STSim - INFO - Generating wavefields at the detector's plane
2020-11-20 12:22:26,552 - STSim - INFO - The wavefields have been generated

Return an array of intensity frames at the detector's plane.

>>> data = sim_obj.frames()
2020-11-20 12:32:33,024 - STSim - INFO - Making frames at the detector's plane...
2020-11-20 12:32:33,025 - STSim - INFO - Source blur size: 400.000000 um
2020-11-20 12:32:35,434 - STSim - INFO - The frames are generated, data shape: (300, 1000, 2000)

Save the simulated data to a `CXI`_ file using the default protocol.

>>> st_conv = st_sim.STConverter()
>>> st_conv.save_sim(data, sim_obj, 'test')
2020-11-20 12:33:40,385 - STSim - INFO - Saving data in the directory: test
2020-11-20 12:33:40,387 - STSim - INFO - Making ini files...
2020-11-20 12:33:40,397 - STSim - INFO - test/update_pixel_map.ini saved
2020-11-20 12:33:40,398 - STSim - INFO - test/zernike.ini saved
2020-11-20 12:33:40,399 - STSim - INFO - test/calculate_phase.ini saved
2020-11-20 12:33:40,401 - STSim - INFO - test/make_reference.ini saved
2020-11-20 12:33:40,402 - STSim - INFO - test/calc_error.ini saved
2020-11-20 12:33:40,403 - STSim - INFO - test/speckle_gui.ini saved
2020-11-20 12:33:40,405 - STSim - INFO - test/generate_pixel_map.ini saved
2020-11-20 12:33:40,406 - STSim - INFO - test/update_translations.ini saved
2020-11-20 12:33:40,408 - STSim - INFO - test/parameters.ini saved
2020-11-20 12:33:40,409 - STSim - INFO - test/protocol.ini saved
2020-11-20 12:33:40,410 - STSim - INFO - Making a cxi file...
2020-11-20 12:33:40,411 - STSim - INFO - Using the following cxi protocol:
2020-11-20 12:33:40,412 - STSim - INFO - basis_vectors [float]: '/entry_1/instrument_1/detector_1/basis_vectors' 
2020-11-20 12:33:40,413 - STSim - INFO - data [float]: '/entry_1/data_1/data' 
2020-11-20 12:33:40,413 - STSim - INFO - defocus [float]: '/speckle_tracking/defocus' 
2020-11-20 12:33:40,414 - STSim - INFO - defocus_fs [float]: '/speckle_tracking/dfs' 
2020-11-20 12:33:40,415 - STSim - INFO - defocus_ss [float]: '/speckle_tracking/dss' 
2020-11-20 12:33:40,416 - STSim - INFO - distance [float]: '/entry_1/instrument_1/detector_1/distance' 
2020-11-20 12:33:40,417 - STSim - INFO - energy [float]: '/entry_1/instrument_1/source_1/energy' 
2020-11-20 12:33:40,418 - STSim - INFO - good_frames [int]: '/frame_selector/good_frames' 
2020-11-20 12:33:40,419 - STSim - INFO - m0 [int]: '/speckle_tracking/m0' 
2020-11-20 12:33:40,419 - STSim - INFO - mask [bool]: '/speckle_tracking/mask' 
2020-11-20 12:33:40,420 - STSim - INFO - n0 [int]: '/speckle_tracking/n0' 
2020-11-20 12:33:40,421 - STSim - INFO - phase [float]: '/speckle_tracking/phase' 
2020-11-20 12:33:40,422 - STSim - INFO - pixel_map [float]: '/speckle_tracking/pixel_map' 
2020-11-20 12:33:40,422 - STSim - INFO - pixel_abberations [float]: '/speckle_tracking/pixel_abberations' 
2020-11-20 12:33:40,423 - STSim - INFO - pixel_translations [float]: '/speckle_tracking/pixel_translations' 
2020-11-20 12:33:40,424 - STSim - INFO - reference_image [float]: '/speckle_tracking/reference_image' 
2020-11-20 12:33:40,425 - STSim - INFO - roi [int]: '/speckle_tracking/roi' 
2020-11-20 12:33:40,425 - STSim - INFO - translations [float]: '/entry_1/sample_1/geometry/translations' 
2020-11-20 12:33:40,426 - STSim - INFO - wavelength [float]: '/entry_1/instrument_1/source_1/wavelength' 
2020-11-20 12:33:40,427 - STSim - INFO - whitefield [float]: '/speckle_tracking/whitefield' 
2020-11-20 12:33:40,428 - STSim - INFO - x_pixel_size [float]: '/entry_1/instrument_1/detector_1/x_pixel_size' 
2020-11-20 12:33:40,429 - STSim - INFO - y_pixel_size [float]: '/entry_1/instrument_1/detector_1/y_pixel_size' 
2020-11-20 12:33:51,966 - STSim - INFO - test/data.cxi saved

.. _CXI: https://www.cxidb.org/cxi.html
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
from ..data_processing import STData
from .st_sim_param import STParams, parameters
from ..bin import aperture_wp, barcode_steps, barcode_profile, lens_wp
from ..bin import make_frames, make_whitefield
from ..bin import fraunhofer_1d, fraunhofer_1d_scan

class STSim:
    """One-dimensional Speckle Tracking scan simulation class.
    Generates barcode's transmission profile and lens' abberated
    wavefront, whereupon propagates the wavefront to the detector's
    plane. Logs all the steps to `logs` folder.

    Parameters
    ----------
    st_params : STParams
        Experimental parameters.
    bsteps : numpy.ndarray, optional
        Array of barcode's bar coordinates. Generates the array
        automatically if it's not provided.

    See Also
    --------
    st_sim_param : Full list of experimental parameters.
    """
    log_dir = os.path.join(ROOT_PATH, '../logs')

    def __init__(self, st_params, bsteps=None):
        self.parameters = st_params
        self._init_logging()
        self._init_coord()
        self._init_lens()
        self._init_barcode(bsteps)
        self._init_detector()

    def __getattr__(self, attr):
        if attr in self.parameters:
            return self.parameters.__getattr__(attr)

    def _init_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
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

        x_span = 1.6 * self.ap_x / self.focus * self.defocus
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
        self.wf0_x = lens_wp(x_arr=self.x_arr, wl=self.wl, ap=self.ap_x,
                             focus=self.focus, defoc=self.defocus, alpha=self.alpha,
                             xc=(self.x0 - 0.5) * self.ap_x)
        self.wf0_y = aperture_wp(x_arr=self.y_arr, z=self.focus + self.defocus,
                                 wl=self.wl, ap=self.ap_y)
        self.i0 = self.p0 / self.ap_x / self.ap_y
        self.smp_c = 1 / self.wl / (self.focus + self.defocus)
        self.logger.info("The wavefields have been generated")

    def _init_barcode(self, bsteps):
        self.logger.info("Generating barcode's transmission coefficients")
        if bsteps is None:
            bsteps = barcode_steps(x0=self.x_arr[0] + self.offset, x1=self.x_arr[-1] + \
                                   self.n_frames * self.step_size - self.offset,
                                   br_dx=self.bar_size, rd=self.rnd_dev)
        self.bsteps = bsteps
        self.bs_t = barcode_profile(x_arr=self.x_arr, bx_arr=self.bsteps, sgm=self.bar_sigma,
                                    atn0=self.bulk_atn, atn=self.bar_atn, ss=self.step_size,
                                    nf=self.n_frames)
        self.logger.info("The coefficients have been generated")

    def _init_detector(self):
        self.logger.info("Generating wavefields at the detector's plane")
        self.wf1_y = fraunhofer_1d(wf0=self.wf0_y, x_arr=self.y_arr, xx_arr=self.yy_arr,
                                   z=self.det_dist, wl=self.wl)
        self.wf1_x = fraunhofer_1d_scan(wf0=self.wf0_x * self.bs_t, x_arr=self.x_arr,
                                        xx_arr=self.xx_arr, z=self.det_dist, wl=self.wl)
        self.det_c = self.smp_c / self.wl / self.det_dist
        self.logger.info("The wavefields have been generated")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        if not exc_type is None:
            return False

    def source_curve(self, dist, dx):
        """Return source's rocking curve profile at `dist` distance from
        the lens.

        Parameters
        ----------
        dist : float
            Distance from the lens to the rocking curve profile[um].
        dx : float
            Sampling distance [um].

        Returns
        -------
        numpy.ndarray
            Source's rocking curve profile.
        """
        sigma = self.th_s * dist
        x_arr = dx * np.arange(-np.ceil(4 * sigma / dx), np.ceil(4 * sigma / dx) + 1)
        s_arr = np.exp(-x_arr**2 / 2 / sigma**2)
        return s_arr / s_arr.sum()

    def sample_wavefronts(self):
        """Return beam's wavefront profiles along the x axis at the
        sample's plane.

        Returns
        -------
        numpy.ndarray
            Beam's wavefront at the sample's plane.
        """
        return self.wf0_x * self.bs_t * self.smp_c * self.wf0_y.max()

    def det_wavefronts(self):
        """Return beam's wavefront profiles along the x axis at the
        detector's plane.

        Returns
        -------
        numpy.ndarray
            Beam's wavefront at the detector's plane.
        """
        return self.wf1_x * self.det_c * self.wf1_y.max()

    def frames(self):
        """Return intensity frames at the detector plane. Applies
        Poisson noise.

        Returns
        -------
        numpy.ndarray
            Intensity frames.
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
        """Return a ptychograph of intensity frames. Applies Poisson
        noise.

        Returns
        -------
        numpy.ndarray
            Ptychograph.
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
        """Close logging handlers.
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

class STConverter:
    """
    Converter class to export simulated data from :class:`STSim` to a CXI
    file. :class:`STConverter` also exports experimental parameters and the
    used protocol to INI files.

    Parameters
    ----------
    protocol : Protocol
        CXI protocol, which contains all the attribute's paths and data types.
    coord_ratio : float, optional
        Coordinates ratio between the simulated and saved data.

    Attributes
    ----------
    templates_dir : str
        Path to ini templates (for exporting `protocol` and :class:`STParams`).
    write_attrs : dict
        Dictionary with all the attributes which are saved in CXI file.
    protocol : Protocol
        CXI protocol, which contains all the attribute's paths and data types.
    coord_ratio : float, optional
        Coordinates ratio between the simulated and saved data.

    Notes
    -----
    List of the attributes saved in CXI file:

    * basis_vectors : Detector basis vectors.
    * data : Measured intensity frames.
    * defocus_fs : Defocus distance along the fast detector axis.
    * defocus_ss : Defocus distance along the slow detector axis.
    * distance : Sample-to-detector distance.
    * energy : Incoming beam photon energy [eV].
    * good_frames : An array of good frames' indices.
    * mask : Bad pixels mask.
    * roi : Region of interest in the detector's plane.
    * translations : Sample's translations.
    * wavelength : Incoming beam's wavelength.
    * whitefield : Measured frames' whitefield.
    * x_pixel_size : Pixel's size along the fast detector
      axis.
    * y_pixel_size : Pixel's size along the slow detector
      axis.
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

    def _ini_parsers(self, st_params):
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

    def export_dict(self, data, st_params):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and `st_params` to :class:`dict` object.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        st_params : STParams
            Experimental parameters.

        Returns
        -------
        data_dict : dict
            Dictionary with all the data from `data` and `st_params`.

        See Also
        --------
        STConverter - full list of the attributes stored in `data_dict`.
        """
        data_dict = {}

        # Initialize basis vectors
        pix_vec = self.crd_rat * np.array([st_params.pix_size, st_params.pix_size, 0])
        data_dict['x_pixel_size'] = pix_vec[0]
        data_dict['y_pixel_size'] = pix_vec[1]
        vec_fs = np.tile(pix_vec * self.unit_vector_fs, (st_params.n_frames, 1))
        vec_ss = np.tile(pix_vec * self.unit_vector_ss, (st_params.n_frames, 1))
        data_dict['basis_vectors'] = np.stack((vec_ss, vec_fs), axis=1)

        # Initialize data
        data_dict['data'] = data
        data_dict['good_frames'] = np.arange(data.shape[0],
                                             dtype=self.protocol.get_dtype('good_frames'))
        data_dict['mask'] = np.ones(data.shape[1:], dtype=self.protocol.get_dtype('mask'))
        data_dict['whitefield'] = make_whitefield(mask=data_dict['mask'], data=data)

        # Initialize defocus distances
        data_dict['defocus_fs'] = self.crd_rat * st_params.defocus
        data_dict['defocus_ss'] = self.crd_rat * st_params.defocus

        # Initialize sample-to-detector distance
        data_dict['distance'] = self.crd_rat * st_params.det_dist

        # Initialize beam's wavelength and energy
        data_dict['wavelength'] = self.crd_rat * st_params.wl
        data_dict['energy'] = self.e_to_wl / data_dict['wavelength']

        # Initialize region of interest
        fs_lb, fs_ub = st_params.fs_roi()
        data_dict['roi'] = np.array([0, data.shape[1], fs_lb, fs_ub])

        # Initialize sample translations
        t_arr = np.zeros((st_params.n_frames, 3), dtype=self.protocol.get_dtype('translations'))
        t_arr[:, 0] = -np.arange(0, st_params.n_frames) * st_params.step_size
        data_dict['translations'] = self.crd_rat * t_arr

        for attr in data_dict:
            data_dict[attr] = np.asarray(data_dict[attr], dtype=self.protocol.get_dtype(attr))
        return data_dict

    def export_data(self, data, st_params):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and `st_params` to a data container.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        st_params : STParams
            Experimental parameters.

        Returns
        -------
        STData
            Data container with all the data from `data` and `st_params`.

        See Also
        --------
        STConverter - full list of the attributes stored in `data_dict`.
        """
        return STData(protocol=self.protocol, **self.export_dict(data, st_params))

    def save_sim(self, data, st_sim, dir_path):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and a :class:`STSim` object
        `st_sim` to `dir_path` folder.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        st_sim : STSim
            Speckle Tracking simulation object.
        dir_path : str
            Path to the folder, where all the files are saved.

        Notes
        -----
        List of the files saved in `dir_path`:

        * 'data.cxi' : CXI file with all the data attributes.
        * 'protocol.ini' : CXI protocol.
        * 'parameters.ini' : experimental parameters.
        * {'calc_error.ini', 'calculate_phase.ini', 'generate_pixel_map.ini',
          'make_reference.ini', 'speckle_gui.ini', 'update_pixel_map.ini',
          'update_translations.ini', 'zernike.ini'} : INI files to work with
          Andrew's `speckle_tracking <https://github.com/andyofmelbourne/speckle-tracking>`_
          GUI.
        """
        self.save(data=data, st_params=st_sim.parameters,
                  dir_path=dir_path, logger=st_sim.logger)

    def save(self, data, st_params, dir_path, logger=None):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and `st_params` to `dir_path` folder.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        st_params : STParams
            Experimental parameters.
        dir_path : str
            Path to the folder, where all the files are saved.
        logger : logging.Logger, optional
            Logging interface.

        See Also
        --------
        STConverter.save_sim : Full list of the files saved in `dir_path`.
        """
        if logger:
            logger.info("Saving data in the directory: {:s}".format(dir_path))
            logger.info("Making ini files...")
        os.makedirs(dir_path, exist_ok=True)
        for name, parser in self._ini_parsers(st_params).items():
            ini_path = os.path.join(dir_path, name + '.ini')
            with open(ini_path, 'w') as ini_file:
                parser.write(ini_file)
            if logger:
                logger.info('{:s} saved'.format(ini_path))
        if logger:
            logger.info("Making a cxi file...")
            logger.info("Using the following cxi protocol:")
            self.protocol.log(logger)
        data_dict = self.export_dict(data, st_params)
        with h5py.File(os.path.join(dir_path, 'data.cxi'), 'w') as cxi_file:
            for attr in data_dict:
                self.protocol.write_cxi(attr, data_dict[attr], cxi_file)
        if logger:
            logger.info("{:s} saved".format(os.path.join(dir_path, 'data.cxi')))

def converter(coord_ratio=1e-6, float_precision='float64'):
    """Return the default simulation converter.

    Parameters
    ----------
    coord_ratio : float, optional
        Coordinates ratio between the simulated and saved data.
    float_precision: {'float32', 'float64'}, optional
        Floating point precision.

    Returns
    -------
    STConverter
        Default simulation data converter.

    See Also
    --------
    STConverter : Full converter class description.
    """
    return STConverter(protocol=cxi_protocol(float_precision),
                       coord_ratio=coord_ratio)

def main():
    """Main fuction to run Speckle Tracking simulation and save the results to a CXI file.
    """
    parser = argparse.ArgumentParser(description="Run Speckle Tracking simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_path', type=str, help="Output folder path")
    parser.add_argument('-f', '--ini_file', type=str,
                        help="Path to an INI file to fetch all of the simulation parameters")
    parser.add_argument('--defocus', type=float, help="Lens defocus distance, [um]")
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
    parser.add_argument('--rnd_dev', type=float, help="Bar random deviation")
    parser.add_argument('--offset', type=float,
                        help="sample's offset at the beginning and the end of the scan [um]")
    parser.add_argument('-v', '--verbose', action='store_true', help="Turn on verbosity")
    parser.add_argument('-p', '--ptych', action='store_true', help="Generate ptychograph data")
    parser.set_defaults(**parameters().export_dict())

    args = vars(parser.parse_args())
    if args['ini_file']:
        st_params = STParams.import_ini(args['ini_file'])
    else:
        st_params = STParams.import_dict(**args)

    st_converter = STConverter()
    with STSim(st_params) as sim_obj:
        if args['ptych']:
            data = sim_obj.ptychograph()
        else:
            data = sim_obj.frames()
        st_converter.save_sim(data, sim_obj, args['out_path'])
