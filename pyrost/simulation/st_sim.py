"""One-dimensional speckle tracking scan simulation. Generate intensity frames
based on Fresnel diffraction theory. :class:`pyrost.simulation.STSim` does the
heavy lifting of calculating the wavefront propagation to the detector plane.
:class:`pyrost.simulation.STConverter` exports simulated data to a
`CXI <https://www.cxidb.org/cxi.html>`_ format file accordingly to the provided
:class:`pyrost.CXIProtocol` object and saves the protocol and experimental parameters
to the same folder.

Examples:

    Perform the simulation for a given :class:`pyrost.simulation.STParams` object
    `params` with :class:`pyrost.simulation.STSim`. Then generate a stack frames with
    :func:`pyrost.simulation.STSim.frames`, and save the simulated data to a
    `CXI <https://www.cxidb.org/cxi.html>`_ file using the default protocol as follows:

    >>> import pyrost.simulation as st_sim
    >>> params = st_sim.STParams.import_default()
    >>> sim_obj = st_sim.STSim(params)
    >>> data = sim_obj.frames()
    >>> st_conv = st_sim.STConverter()
    >>> st_conv.save_sim(data, sim_obj, 'test') # doctest: +SKIP
"""
from __future__ import annotations
import os
import argparse
from typing import Dict, Optional, Union
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

    Attributes:
        attr_set : Set of attributes in the container which are necessary
            to initialize in the constructor.
        init_set : Set of optional data attributes.

    Notes:
        **Necessary attributes**:

        * backend : the backend library for the FFT implementation.
        * num_threads : Number of threads used in the calculations.
        * params : Experimental parameters.

        **Optional attributes**:

        * bars : Barcode's bar positions [um].
        * det_wfx : Wavefront at the detector plane along the x axis.
        * det_wfy : Wavefront at the detector plane along the y axis.
        * det_iy : Intensity profile at the detector plane along the
          x axis.
        * det_iy : Intensity profile at the detector plane along the
          y axis.
        * lens_wfx : Wavefront at the lens plane along the x axis.
        * lens_wfy : Wavefront at the lens plane along the y axis.
        * n_x : Wavefront size along the x axis.
        * n_y : Wavefront size along the y axis.
        * roi : Region of interest in detector plane.
        * smp_pos : Sample translations along the x axis.
        * smp_profile : Barcode's transmission profile.
        * smp_wfx : Wavefront at the sample plane along the x axis.
        * smp_wfy : Wavefront at the sample plane along the y axis.
    """
    backends = {'numpy', 'fftw'}
    attr_set = {'backend', 'params'}
    init_set = {'bars', 'det_wfx', 'det_wfy', 'det_ix', 'det_iy', 'lens_wfx',
                'lens_wfy', 'roi', 'smp_pos', 'smp_profile', 'smp_wfx', 'smp_wfy'}

    inits = {'bars'       : lambda obj: obj.params.bar_positions(dist=obj.params.defocus),
             'lens_wfx'   : lambda obj: obj.params.lens_x_wavefront(),
             'lens_wfy'   : lambda obj: obj.params.lens_y_wavefront(),
             'smp_wfx'    : lambda obj: obj._sample_x_wavefront(),
             'smp_wfy'    : lambda obj: obj._sample_y_wavefront(),
             'smp_pos'    : lambda obj: obj.params.sample_positions(),
             'smp_profile': lambda obj: obj._sample_profile(),
             'det_wfx'    : lambda obj: obj._detector_x_wavefront(),
             'det_wfy'    : lambda obj: obj._detector_y_wavefront(),
             'det_ix'     : lambda obj: obj._detector_x_intensity(),
             'det_iy'     : lambda obj: obj._detector_y_intensity(),
             'roi'        : lambda obj: obj.find_beam_roi()}

    def __init__(self, params: STParams, backend: str='numpy',
                 **kwargs: Union[int, np.ndarray]) -> None:
        """
        Args:
            params : Set of simulation parameters.
            backend : Choose between numpy ('numpy') or FFTW ('fftw') library for the FFT
                implementation.
            num_threads : Number of threads used in the calculations.
            kwargs : Any of the optional attributes specified in :class:`STSim` notes.

        Raises:
            ValueError : If the 'backend' keyword is invalid.
        """
        if not backend in self.backends:
            err_msg = f'backend must be one of the following: {str(self.backends):s}'
            raise ValueError(err_msg)
        super(STSim, self).__init__(backend=backend, params=params, **kwargs)

    def _sample_x_wavefront(self) -> np.ndarray:
        dx0 = 2.0 * self.params.ap_x / self.x_size
        dx1 = np.abs(dx0 * self.params.defocus / self.params.focus)
        z01 = self.params.focus + self.params.defocus
        return rsc_wp(wft=self.lens_wfx, dx0=dx0, dx=dx1, z=z01, wl=self.params.wl,
                        backend=self.backend, num_threads=self.params.num_threads)

    def _sample_y_wavefront(self) -> np.ndarray:
        dy0 = 2.0 * self.params.ap_y / self.y_size
        z01 = self.params.focus + self.params.defocus
        return rsc_wp(wft=self.lens_wfy, dx0=dy0, dx=dy0, z=z01, wl=self.params.wl,
                        backend=self.backend, num_threads=self.params.num_threads)


    def _sample_profile(self) -> np.ndarray:
        dx1 = np.abs(2.0 * self.params.ap_x * self.params.defocus / self.params.focus / self.x_size)
        x1_arr = dx1 * np.arange(-self.x_size // 2, self.x_size // 2) + self.smp_pos[:, None]
        return self.params.barcode_profile(x_arr=x1_arr, dx=dx1, bars=self.bars)

    def _detector_x_wavefront(self) -> np.ndarray:
        dx1 = np.abs(2.0 * self.params.ap_x * self.params.defocus / self.params.focus / self.x_size)
        dx2 = self.params.detx_size * self.params.pix_size / self.x_size
        wft = self.smp_wfx * self.smp_profile
        return fraunhofer_wp(wft=wft, dx0=dx1, dx=dx2, z=self.params.det_dist,
                             wl=self.params.wl, backend=self.backend,
                             num_threads=self.params.num_threads)

    def _detector_y_wavefront(self) -> np.ndarray:
        dy1 = 2.0 * self.params.ap_y / self.y_size
        dy2 = self.params.dety_size * self.params.pix_size / self.y_size
        return fraunhofer_wp(wft=self.smp_wfy, dx0=dy1, dx=dy2, z=self.params.det_dist,
                             wl=self.params.wl, backend=self.backend,
                             num_threads=1)

    def _detector_x_intensity(self) -> np.ndarray:
        dx = self.params.detx_size * self.params.pix_size / self.x_size
        sc_x = self.params.source_curve(dist=self.params.defocus + self.params.det_dist, step=dx)
        det_ix = np.sqrt(self.params.p0) / self.params.ap_x * np.abs(self.det_wfx)**2
        return fft_convolve(array=det_ix, kernel=sc_x, backend=self.backend,
                            num_threads=self.params.num_threads)

    def _detector_y_intensity(self) -> np.ndarray:
        dy = self.params.dety_size * self.params.pix_size / self.y_size
        sc_y = self.params.source_curve(dist=self.params.defocus + self.params.det_dist, step=dy)
        det_iy = np.sqrt(self.params.p0) / self.params.ap_y * np.abs(self.det_wfy)**2
        return fft_convolve(array=det_iy, kernel=sc_y, backend=self.backend,
                            num_threads=self.params.num_threads)

    def find_beam_roi(self) -> np.ndarray:
        """Calculate the beam's field of view at the detector grid as a
        list of four pixel coordinates.

        Returns:
            A set of four coordinates ('y0', 'y1', 'x0', 'x1'), where 'y0' and
            'y1' are the lower and upper bounds of the beam along the vertical
            detector axis and 'x0' and 'x1' are the lower and upper bounds of
            the beam along the horizontal axis.
        """
        dx1 = np.abs(2.0 * self.params.ap_x * self.params.defocus / self.params.focus / self.x_size)
        dx2 = self.params.detx_size * self.params.pix_size / self.x_size
        wfield_x = np.abs(fraunhofer_wp(wft=self.smp_wfx, dx0=dx1, dx=dx2,
                                        z=self.params.det_dist, wl=self.params.wl,
                                        backend=self.backend, num_threads=1))

        dy1 = 2.0 * self.params.ap_y / self.y_size
        dy2 = self.params.dety_size * self.params.pix_size / self.y_size
        wfield_y = np.abs(fraunhofer_wp(wft=self.smp_wfy, dx0=dy1, dx=dy2,
                                        z=self.params.det_dist, wl=self.params.wl,
                                        backend=self.backend, num_threads=1))

        x0, x1 = self.params.beam_span(self.params.det_dist)

        if (x1 - x0) < self.params.detx_size * self.params.pix_size:
            dx = self.params.detx_size * self.params.pix_size / self.x_size
            cnt_x, cnt_y = self.x_size // 2 + int(0.5 * (x0 + x1) // dx), self.y_size // 2
            grad_x = gaussian_gradient_magnitude(wfield_x, self.x_size // 100, mode='nearest',
                                                    num_threads=self.params.num_threads)
            grad_y = gaussian_gradient_magnitude(wfield_y, self.y_size // 100, mode='nearest',
                                                    num_threads=self.params.num_threads)
            x0 = (np.argmax(grad_x[:cnt_x]) * self.params.detx_size) // self.x_size
            x1 = ((cnt_x + np.argmax(grad_x[cnt_x:])) * self.params.detx_size) // self.x_size
        else:
            x0, x1 = 0, self.params.detx_size

        y0 = (np.argmax(grad_y[:cnt_y]) * self.params.dety_size) // self.y_size
        y1 = ((cnt_y + np.argmax(grad_y[cnt_y:])) * self.params.dety_size) // self.y_size
        return np.array([y0, y1, x0, x1])

    @property
    def x_size(self) -> int:
        return self.lens_wfx.size

    @property
    def y_size(self) -> int:
        return self.lens_wfy.size

    @dict_to_object
    def update_bars(self, bars: np.ndarray) -> STSim:
        """Return a new :class:`STSim` object with the updated
        `bars`.

        Args:
            bars : Array of barcode's bar positions.

        Returns:
            A new :class:`STSim` object with the updated `bars`.
        """
        return {'bars': bars, 'smp_profile': None, 'det_wfx': None, 'det_ix': None}

    @dict_to_object
    def update_roi(self, roi: np.ndarray) -> STSim:
        """Return a new :class:`STSim` object with the updated
        region of interest.

        Args:
            roi : Region of interest in detector plane. The values are
            given in pixels as following : [`x0`, `x1`, `y0`, `y1`].

        Returns:
            A new :class:`STSim` object with the updated `roi`.
        """
        return {'roi': roi}

    def frames(self, wfieldx: Optional[np.ndarray]=None, wfieldy: Optional[np.ndarray]=None,
               apply_noise: bool=True) -> np.ndarray:
        """Return intensity frames at the detector plane. Applies
        Poisson noise if `apply_noise` is True.

        Args:
            wfieldx : Whitefield profile along the x axis.
            wfieldy : Whitefield profile along the y aixs.
            apply_noise : Apply Poisson noise if it's True.

        Returns:
            Intensity frames.
        """
        dx = self.params.detx_size * self.params.pix_size / self.x_size
        dy = self.params.dety_size * self.params.pix_size / self.y_size
        seed = self.params.seed if apply_noise else -1
        frames = make_frames(pfx=self.det_ix, pfy=self.det_iy, dx=dx, dy=dy,
                             shape=(self.params.dety_size, self.params.detx_size),
                             seed=seed, num_threads=self.params.num_threads)
        if not wfieldx is None:
            frames *= (wfieldx / wfieldx.mean())
        if not wfieldy is None:
            frames *= (wfieldy / wfieldy.mean())[:, None]
        return frames

    def ptychograph(self, wfieldx: Optional[np.ndarray]=None, wfieldy: Optional[np.ndarray]=None,
                    apply_noise: bool=True) -> np.ndarray:
        """Return a ptychograph of intensity frames. Applies Poisson
        noise if `apply_noise` is True.

        Args:
            wfieldx : Whitefield profile along the x axis.
            wfieldy : Whitefield profile along the y aixs.
            apply_noise : Apply Poisson noise if it's True.

        Returns:
            Ptychograph.
        """
        data = self.frames(wfieldx=wfieldx, wfieldy=wfieldy, apply_noise=apply_noise)
        return data.sum(axis=1)[:, None]

class STConverter:
    """
    Converter class to export simulated data from :class:`STSim` to a CXI
    file. :class:`STConverter` also exports experimental parameters and the
    used protocol to INI files.

    Attributes:
        templates_dir : Path to ini templates (for exporting `protocol` and :class:`STParams`).
        write_attrs : Dictionary with all the attributes which are saved in CXI file.
        protocol : CXI protocol, which contains all the attribute's paths and data types.
        coord_ratio : Coordinates ratio between the simulated and saved data.

    Notes:
        List of the attributes saved in CXI file:

        * basis_vectors : Detector basis vectors.
        * data : Measured intensity frames.
        * defocus_x : Defocus distance along the horizontal detector axis.
        * defocus_y : Defocus distance along the vertical detector axis.
        * distance : Sample-to-detector distance.
        * energy : Incoming beam photon energy [eV].
        * good_frames : An array of good frames' indices.
        * mask : Bad pixels mask.
        * roi : Region of interest in the detector plane.
        * translations : Sample's translations.
        * wavelength : Incoming beam's wavelength.
        * whitefield : Measured frames' whitefield.
        * x_pixel_size : Pixel's size along the horizontal detector
          axis.
        * y_pixel_size : Pixel's size along the vertical detector
          axis.
    """
    unit_vector_fs = np.array([1, 0, 0])
    unit_vector_ss = np.array([0, -1, 0])
    templates_dir = os.path.join(ROOT_PATH, 'ini_templates')
    e_to_wl = 1.2398419843320026e-06 # [eV * m]

    write_attrs = {'basis_vectors', 'data', 'defocus', 'distance',
                   'energy', 'good_frames', 'mask', 'roi', 'translations',
                   'wavelength', 'whitefield', 'x_pixel_size', 'y_pixel_size'}

    def __init__(self, protocol: Optional[CXIProtocol]=CXIProtocol.import_default(),
                 coord_ratio: float=1e-6) -> None:
        """
        Args:
            protocol : CXI protocol, which contains all the attribute's paths and data types.
            coord_ratio : Coordinates ratio between the simulated and saved data.
        """
        self.protocol, self.crd_rat = protocol, coord_ratio

    def _ini_parsers(self, st_params: STParams) -> Dict[str, dict]:
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

    def export_dict(self, data: np.ndarray, roi: np.ndarray, smp_pos: np.ndarray,
                    st_params: STParams) -> Dict[str, Union[np.ndarray, float]]:
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and `st_params` to :class:`dict` object.

        Args:
            data : Simulated stack of frames.
            roi : Region of interest in detector plane. The values are
                given in pixels as following : [`x0`, `x1`, `y0`, `y1`].
            smp_pos : Sample translations.
            st_params : Set of simulation parameters.

        Returns:
            Dictionary with all the data from `data` and `st_params`.

        See Also:
            :class:`STConverter` : full list of the attributes stored in the output
            dictionary.
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
        data_dict['defocus_x'] = self.crd_rat * st_params.defocus
        data_dict['defocus_y'] = self.crd_rat * st_params.defocus

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

    def export_data(self, data: np.ndarray, sim_obj: STSim) -> STData:
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and a :class:`STSim` object `sim_obj`
        to a data container.

        Args:
            data : Simulated stack of frames.
            sim_obj : Speckle tracking simulation object.

        Returns:
            Data container :class:`pyrost.STData` with all the data from `data`
            and `sim_obj`.
        """
        return STData(protocol=self.protocol, **self.export_dict(data, roi=sim_obj.roi,
                      smp_pos=sim_obj.smp_pos, st_params=sim_obj.params))

    def save(self, data: np.ndarray, roi: np.ndarray, smp_pos: np.ndarray, st_params: STParams,
             dir_path: str) -> None:
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`), `smp_pos`, and `st_params` to `dir_path`
        folder.

        Args:
            data : Simulated stack of frames.
            roi : Region of interest in detector plane. The values are given in
                pixels as following : [`x0`, `x1`, `y0`, `y1`].
            smp_pos : Sample translations.
            st_params : Set of simulation parameters.
            dir_path : Path to the folder, where all the files are saved.

        Returns:
            None

        See Also:
            :func:`STConverter.save_sim` : Full list of the files saved in
            `dir_path`.
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

    def save_sim(self, data: np.ndarray, sim_obj: STSim, dir_path: str) -> None:
        """Export simulated data `data` (fetched from :func:`STSim.frames` or
        :func:`STSim.ptychograph`) and a :class:`STSim` object `sim_obj` to
        `dir_path` folder.

        Args:
            data : Simulated stack of frames.
            sim_obj : Speckle tracking simulation object.
            dir_path : Path to the folder, where all the files are saved.

        Returns:
            None

        Notes:
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
    parser.add_argument('--detx_size', type=int, help="horizontal axis frames size in pixels")
    parser.add_argument('--dety_size', type=int, help="vertical axis frames size in pixels")
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
                        help="Sample's offset at the beginning and the end of the scan [um]")
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
