"""One-dimensional Speckle Tracking scan simulation.
Generate intensity frames based on Fresnel diffraction
theory. :class:`pyrost.simulation.STSim` does the heavy lifting
of calculating the wavefront propagation to the detector plane.
:class:`pyrost.simulation.STConverter` exports simulated data to a
`CXI <https://www.cxidb.org/cxi.html>`_ format file accordingly to
the provided :class:`pyrost.CXIProtocol` object and saves the protocol
and experimental parameters to the same folder.

Examples
--------

Perform the simulation for a given :class:`pyrost.simulation.STParams` object
`params`:

>>> import pyrost.simulation as sim
>>> sim_obj = sim.STSim(params)

Return an array of intensity frames at the detector plane.

>>> data = sim_obj.frames()

Save the simulated data to a `CXI <https://www.cxidb.org/cxi.html>`_ file using
the default protocol.

>>> st_conv = sim.STConverter()
>>> st_conv.save_sim(data, sim_obj, 'test')
"""
import os
import argparse
import h5py
import numpy as np
from ..cxi_protocol import CXIProtocol, ROOT_PATH
from ..data_container import DataContainer, dict_to_object
from ..data_processing import STData
from .st_parameters import STParams
from ..bin import rsc_wp, fraunhofer_wp, fft_convolve
from ..bin import make_frames, median, gaussian_gradient_magnitude

class STSim(DataContainer):
    """One-dimensional Speckle Tracking scan simulation class.
    Generates barcode's transmission profile and lens' abberated
    wavefront, whereupon propagates the wavefront to the detector's
    plane.

    Parameters
    ----------
    params : STParams
        Experimental parameters.
    backend : {'fftw', 'numpy'}, optional
        Choose the backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads used in the calculations.
    **kwargs : dict, optional
        Attributes specified in `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.
    init_set : set
        Set of optional data attributes.

    Notes
    -----
    Necessary attributes:

    * backend : Choose the backend library for the FFT implementation.
    * num_threads : Number of threads used in the calculations.
    * params : Experimental parameters.

    Optional attributes:

    * bars : Barcode's bar positions [um].
    * det_wfx : Wavefront at the detector plane along the x
      axis.
    * det_wfy : Wavefront at the detector plane along the y
      axis.
    * det_iy : Intensity profile at the detector plane along
      the x axis.
    * det_iy : Intensity profile at the detector plane along
      the y axis.
    * lens_wfx : Wavefront at the lens plane along the x
      axis.
    * lens_wfy : Wavefront at the lens plane along the y
      axis.
    * n_x : Wavefront size along the x axis.
    * n_y : Wavefront size along the y axis.
    * roi : Region of interest in detector plane.
    * smp_pos : Sample translations along the x axis.
    * smp_profile : Barcode's transmission profile.
    * smp_wfx : Wavefront at the sample plane along the x
      axis.
    * smp_wfy : Wavefront at the sample plane along the y
      axis.

    See Also
    --------
    st_sim_param : Full list of experimental parameters.
    """
    backends = {'numpy', 'fftw'}
    attr_set = {'backend', 'params'}
    init_set = {'bars', 'det_wfx', 'det_wfy', 'det_ix', 'det_iy', 'lens_wfx',
                'lens_wfy', 'roi', 'smp_pos', 'smp_profile', 'smp_wfx', 'smp_wfy'}

    def __init__(self, params, backend='numpy', **kwargs):
        if not backend in self.backends:
            err_msg = f'backend must be one of the following: {str(self.backends):s}'
            raise ValueError(err_msg)
        super(STSim, self).__init__(backend=backend, params=params, **kwargs)
        self._init_dict()

    def _init_dict(self):
        # Initialize barcode's bar positions
        if self.bars is None:
            self.bars = self.params.bar_positions(dist=self.params.defocus)

        # Initialize wavefronts at the lens plane
        if self.lens_wfx is None or self.lens_wfy is None:
            self.lens_wfx, self.lens_wfy = self.params.lens_wavefronts()

        # Initialize wavefronts at the sample plane
        if self.smp_wfx is None:
            dx0 = 2 * self.params.ap_x / self.x_size
            dx1 = np.abs(dx0 * self.params.defocus / self.params.focus)
            z01 = self.params.focus + self.params.defocus
            self.smp_wfx = rsc_wp(wft=self.lens_wfx, dx0=dx0, dx=dx1, z=z01, wl=self.params.wl,
                                  backend=self.backend, num_threads=self.params.num_threads)
        if self.smp_wfy is None:
            dy0 = 2 * self.params.ap_y / self.y_size
            z01 = self.params.focus + self.params.defocus
            self.smp_wfy = rsc_wp(wft=self.lens_wfy, dx0=dy0, dx=dy0, z=z01, wl=self.params.wl,
                                  backend=self.backend, num_threads=self.params.num_threads)

        # Initialize sample's translations
        if self.smp_pos is None:
            self.smp_pos = self.params.sample_positions()

        # Initialize sample's transmission profile
        if self.smp_profile is None:
            dx1 = np.abs(2 * self.params.ap_x * self.params.defocus \
                         / self.params.focus / self.x_size)
            x1_arr = dx1 * np.arange(-self.x_size // 2, self.x_size // 2) + self.smp_pos[:, None]
            self.smp_profile = self.params.barcode_profile(x_arr=x1_arr, bars=self.bars,
                                                           num_threads=self.params.num_threads)

        # Initialize wavefronts at the detector plane
        if self.det_wfx is None:
            dx1 = np.abs(2 * self.params.ap_x * self.params.defocus \
                         / self.params.focus / self.x_size)
            dx2 = self.params.fs_size * self.params.pix_size / self.x_size
            wft = self.smp_wfx * self.smp_profile
            self.det_wfx = fraunhofer_wp(wft=wft, dx0=dx1, dx=dx2, z=self.params.det_dist,
                                         wl=self.params.wl, backend=self.backend,
                                         num_threads=self.params.num_threads)
            self.det_wx = np.abs(fraunhofer_wp(wft=self.smp_wfx, dx0=dx1, dx=dx2,
                                               z=self.params.det_dist, wl=self.params.wl,
                                               backend=self.backend, num_threads=1))

        if self.det_wfy is None:
            dy1 = 2 * self.params.ap_y / self.y_size
            dy2 = self.params.ss_size * self.params.pix_size / self.y_size
            self.det_wfy = fraunhofer_wp(wft=self.smp_wfy, dx0=dy1, dx=dy2, z=self.params.det_dist,
                                         wl=self.params.wl, backend=self.backend,
                                         num_threads=1)
            self.det_wy = np.abs(fraunhofer_wp(wft=self.smp_wfy, dx0=dy1, dx=dy2,
                                               z=self.params.det_dist, wl=self.params.wl,
                                               backend=self.backend, num_threads=1))

        # Initialize intensity profiles at the detector plane
        if self.det_ix is None:
            dx = self.params.fs_size * self.params.pix_size / self.x_size
            sc_x = self.params.source_curve(dist=self.params.defocus + self.params.det_dist, step=dx)
            det_ix = np.sqrt(self.params.p0) / self.params.ap_x * np.abs(self.det_wfx)**2
            self.det_ix = fft_convolve(array=det_ix, kernel=sc_x, backend=self.backend,
                                       num_threads=self.params.num_threads)
        if self.det_iy is None:
            dy = self.params.ss_size * self.params.pix_size / self.y_size
            sc_y = self.params.source_curve(dist=self.params.defocus + self.params.det_dist, step=dy)
            det_iy = np.sqrt(self.params.p0) / self.params.ap_y * np.abs(self.det_wfy)**2
            self.det_iy = fft_convolve(array=det_iy, kernel=sc_y, backend=self.backend,
                                       num_threads=self.params.num_threads)

        # Initialize region of interest
        if self.roi is None:
            x0, x1 = self.params.beam_span(self.params.det_dist)
            if (x1 - x0) < self.params.fs_size * self.params.pix_size:
                dx = self.params.fs_size * self.params.pix_size / self.x_size
                cnt_x, cnt_y = self.x_size // 2 + int((x0 + x1) / 2 // dx), self.y_size // 2
                grad_x = gaussian_gradient_magnitude(self.det_wx, self.x_size // 100,
                                                     mode='nearest',
                                                     num_threads=self.params.num_threads)
                grad_y = gaussian_gradient_magnitude(self.det_wy, self.y_size // 100,
                                                     mode='nearest',
                                                     num_threads=self.params.num_threads)
                fs0 = (np.argmax(grad_x[:cnt_x]) * self.params.fs_size) // self.x_size
                fs1 = ((cnt_x + np.argmax(grad_x[cnt_x:])) * self.params.fs_size) // self.x_size
            else:
                fs0, fs1 = 0, self.params.fs_size
            ss0 = (np.argmax(grad_y[:cnt_y]) * self.params.ss_size) // self.y_size
            ss1 = ((cnt_y + np.argmax(grad_y[cnt_y:])) * self.params.ss_size) // self.y_size
            self.roi = np.array([ss0, ss1, fs0, fs1])

    @property
    def x_size(self):
        return self.lens_wfx.size

    @property
    def y_size(self):
        return self.lens_wfy.size

    @dict_to_object
    def update_bars(self, bars):
        """Return a new :class:`STSim` object with the updated
        `bars`.

        Parameters
        ----------
        bars : numpy.ndarray
            Array of barcode's bar positions.

        Returns
        -------
        STSim
            A new :class:`STSim` object with the updated
            `bars`.
        """
        return {'bars': bars, 'smp_profile': None, 'det_wfx': None, 'det_ix': None}

    @dict_to_object
    def update_roi(self, roi):
        """Return a new :class:`STSim` object with the updated
        region of interest.

        Parameters
        ----------
        roi : numpy.ndarray
            Region of interest in detector plane. The values are
            given in pixels as following : [`x0`, `x1`, `y0`, `y1`].

        Returns
        -------
        STSim
            A new :class:`STSim` object with the updated
            `roi`.
        """
        return {'roi': roi}

    def frames(self, wfieldx=None, wfieldy=None, apply_noise=True):
        """Return intensity frames at the detector plane. Applies
        Poisson noise if `apply_noise` is True.

        Parameters
        ----------
        wfieldx : np.ndarray, optional
            Whitefield profile along the x axis.
        wfieldy : np.ndarray, optional
            whitefield profile along the y aixs.
        apply_noise : bool, optional
            Apply Poisson noise if it's True.

        Returns
        -------
        numpy.ndarray
            Intensity frames.
        """
        dx = self.params.fs_size * self.params.pix_size / self.x_size
        dy = self.params.ss_size * self.params.pix_size / self.y_size
        seed = self.params.seed if apply_noise else -1
        frames = make_frames(pfx=self.det_ix, pfy=self.det_iy, dx=dx, dy=dy,
                             shape=(self.params.ss_size, self.params.fs_size),
                             seed=seed, num_threads=self.params.num_threads)
        if not wfieldx is None:
            frames *= (wfieldx / wfieldx.mean())
        if not wfieldy is None:
            frames *= (wfieldy / wfieldy.mean())[:, None]
        return frames

    def ptychograph(self, wfieldx=None, wfieldy=None, apply_noise=True):
        """Return a ptychograph of intensity frames. Applies Poisson
        noise if `apply_noise` is True.

        Parameters
        ----------
        wfieldx : np.ndarray, optional
            Whitefield profile along the x axis.
        wfieldy : np.ndarray, optional
            whitefield profile along the y aixs.
        apply_noise : bool, optional
            Apply Poisson noise if it's True.

        Returns
        -------
        numpy.ndarray
            Ptychograph.
        """
        data = self.frames(wfieldx=wfieldx, wfieldy=wfieldy, apply_noise=apply_noise)
        return data.sum(axis=1)[:, None]

class STConverter:
    """
    Converter class to export simulated data from :class:`STSim` to a CXI
    file. :class:`STConverter` also exports experimental parameters and the
    used protocol to INI files.

    Parameters
    ----------
    protocol : CXIProtocol
        CXI protocol, which contains all the attribute's paths and data types.
    coord_ratio : float, optional
        Coordinates ratio between the simulated and saved data.

    Attributes
    ----------
    templates_dir : str
        Path to ini templates (for exporting `protocol` and :class:`STParams`).
    write_attrs : dict
        Dictionary with all the attributes which are saved in CXI file.
    protocol : CXIProtocol
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
    * roi : Region of interest in the detector plane.
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
    e_to_wl = 1.2398419843320026e-06 # [eV * m]

    write_attrs = {'basis_vectors', 'data', 'defocus', 'distance',
                   'energy', 'good_frames', 'mask', 'roi', 'translations',
                   'wavelength', 'whitefield', 'x_pixel_size', 'y_pixel_size'}

    def __init__(self, protocol=None, coord_ratio=1e-6):
        if protocol is None:
            protocol = CXIProtocol()
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

    def export_dict(self, data, roi, smp_pos, st_params):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and `st_params` to :class:`dict` object.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        roi : numpy.ndarray
            Region of interest in detector plane. The values are
            given in pixels as following : [`x0`, `x1`, `y0`, `y1`].
        smp_pos : numpy.ndarray
            Sample translations.
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
        data_dict['mask'] = np.ones(data.shape, dtype=self.protocol.get_dtype('mask'))
        data_dict['whitefield'] = median(mask=data_dict['mask'], data=data, axis=0, num_threads=st_params.num_threads)

        # Initialize defocus distances
        data_dict['defocus_fs'] = self.crd_rat * st_params.defocus
        data_dict['defocus_ss'] = self.crd_rat * st_params.defocus

        # Initialize sample-to-detector distance
        data_dict['distance'] = self.crd_rat * st_params.det_dist

        # Initialize beam's wavelength and energy
        data_dict['wavelength'] = self.crd_rat * st_params.wl
        data_dict['energy'] = self.e_to_wl / data_dict['wavelength']

        # Initialize region of interest
        if data_dict['data'].shape[1] == 1:
            data_dict['roi'] = np.clip([0, 1, roi[2], roi[3]], 0, data_dict['data'].shape[2])
        else:
            data_dict['roi'] = np.clip(roi, 0, data_dict['data'].shape[2])

        # Initialize sample translations
        t_arr = np.zeros((st_params.n_frames, 3), dtype=self.protocol.get_dtype('translations'))
        t_arr[:, 0] = -smp_pos
        data_dict['translations'] = self.crd_rat * t_arr
        return data_dict

    def export_data(self, data, sim_obj):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and a :class:`STSim` object `sim_obj`
        to a data container.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        sim_obj : STSim
            Speckle Tracking simulation object.

        Returns
        -------
        STData
            Data container with all the data from `data` and `sim_obj`.

        See Also
        --------
        STConverter - full list of the attributes stored in `data_dict`.
        """
        return STData(protocol=self.protocol, **self.export_dict(data, roi=sim_obj.roi,
                      smp_pos=sim_obj.smp_pos, st_params=sim_obj.params))

    def save_sim(self, data, sim_obj, dir_path):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and a :class:`STSim` object
        `sim_obj` to `dir_path` folder.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        sim_obj : STSim
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
        self.save(data=data, roi=sim_obj.roi, smp_pos=sim_obj.smp_pos,
                  st_params=sim_obj.params, dir_path=dir_path)

    def save(self, data, roi, smp_pos, st_params, dir_path):
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`), `smp_pos`, and `st_params` to `dir_path`
        folder.

        Parameters
        ----------
        data : numpy.ndarray
            Simulated data.
        roi : numpy.ndarray
            Region of interest in detector plane. The values are
            given in pixels as following : [`x0`, `x1`, `y0`, `y1`].
        smp_pos : numpy.ndarray
            Sample translations.
        st_params : STParams
            Experimental parameters.
        dir_path : str
            Path to the folder, where all the files are saved.

        See Also
        --------
        STConverter.save_sim : Full list of the files saved in `dir_path`.
        """
        os.makedirs(dir_path, exist_ok=True)
        for name, parser in self._ini_parsers(st_params).items():
            ini_path = os.path.join(dir_path, name + '.ini')
            with open(ini_path, 'w') as ini_file:
                parser.write(ini_file)
        data_dict = self.export_dict(data, roi, smp_pos, st_params)
        with h5py.File(os.path.join(dir_path, 'data.cxi'), 'w') as cxi_file:
            for attr in data_dict:
                self.protocol.write_cxi(attr, data_dict[attr], cxi_file)

def main():
    """Main fuction to run Speckle Tracking simulation and save the results to a CXI file.
    """
    parser = argparse.ArgumentParser(description="Run Speckle Tracking simulation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('out_path', type=str, help="Output folder path")
    parser.add_argument('-f', '--ini_file', type=str,
                        help="Path to an INI file to fetch all of the simulation parameters")
    parser.add_argument('--defocus', type=float, help="Lens defocus distance, [um]")
    parser.add_argument('--det_dist', type=float,
                        help="Distance between the barcode and the detector [um]")
    parser.add_argument('--step_size', type=float, help="Scan step size [um]")
    parser.add_argument('--step_rnd', type=float,
                        help="Random deviation of sample translations [0.0 - 1.0]")
    parser.add_argument('--n_frames', type=int, help="Number of frames")
    parser.add_argument('--fs_size', type=int, help="Fast axis frames size in pixels")
    parser.add_argument('--ss_size', type=int, help="Slow axis frames size in pixels")
    parser.add_argument('--p0', type=float, help="Source beam flux [cnt / s]")
    parser.add_argument('--wl', type=float, help="Wavelength [um]")
    parser.add_argument('--th_s', type=float, help="Source rocking curve width [rad]")
    parser.add_argument('--ap_x', type=float, help="Lens size along the x axis [um]")
    parser.add_argument('--ap_y', type=float, help="Lens size along the y axis [um]")
    parser.add_argument('--focus', type=float, help="Focal distance [um]")
    parser.add_argument('--alpha', type=float, help="Third order aberrations [rad/mrad^3]")
    parser.add_argument('--ab_cnt', type=float,
                        help="Lens' aberrations center point [0.0 - 1.0]")
    parser.add_argument('--bar_size', type=float, help="Average bar size [um]")
    parser.add_argument('--bar_sigma', type=float, help="Bar haziness width [um]")
    parser.add_argument('--bar_atn', type=float, help="Bar attenuation")
    parser.add_argument('--bulk_atn', type=float, help="Bulk attenuation")
    parser.add_argument('--bar_rnd', type=float, help="Bar random deviation")
    parser.add_argument('--offset', type=float,
                        help="sample's offset at the beginning and the end of the scan [um]")
    parser.add_argument('-p', '--ptych', action='store_true', help="Generate ptychograph data")
    parser.set_defaults(**STParams().export_dict())

    args = vars(parser.parse_args())
    if args['ini_file']:
        st_params = STParams.import_ini(args['ini_file'])
    else:
        st_params = STParams(**args)

    st_converter = STConverter()
    sim_obj = STSim(st_params)
    if args['ptych']:
        data = sim_obj.ptychograph()
    else:
        data = sim_obj.frames()
    st_converter.save_sim(data, sim_obj, args['out_path'])
    print(f"The simulation results have been saved to {os.path.abspath(args['out_path']):s}")
