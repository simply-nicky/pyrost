""":class:`pyrost.simulation.STSim` does the heavy lifting of calculating the wavefront
propagation to the detector plane. :class:`pyrost.simulation.STConverter` exports
simulated data to a `CXI <https://www.cxidb.org/cxi.html>`_ format file accordingly to
the provided :class:`pyrost.CXIProtocol` object and saves the protocol and experimental
parameters to the same folder.

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
from typing import Iterable, Optional, Tuple, Union
import numpy as np
from ..cxi_protocol import CXIProtocol, CXIStore
from ..data_container import DataContainer, dict_to_object
from ..data_processing import STData, Crop
from .st_parameters import STParams
from ..bin import rsc_wp, fraunhofer_wp, fft_convolve
from ..bin import make_frames, gaussian_gradient_magnitude

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
        * num_threads : Number of threads used in the calculations.
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

    # Necessary attributes
    backend     : str
    params      : STParams

    # Automatically generated attributes
    bars        : np.ndarray
    det_ix      : np.ndarray
    det_iy      : np.ndarray
    det_wfx     : np.ndarray
    det_wfy     : np.ndarray
    lens_wfx    : np.ndarray
    lens_wfy    : np.ndarray
    roi         : Tuple[int, int, int, int]
    smp_pos     : np.ndarray
    smp_profile : np.ndarray
    smp_wfx     : np.ndarray
    smp_wfy     : np.ndarray

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

        self._init_functions(bars=lambda: self.params.bar_positions(dist=self.params.defocus),
                             lens_wfx=self.params.lens_x_wavefront, lens_wfy=self.params.lens_y_wavefront,
                             smp_wfx=self._sample_x_wavefront, smp_wfy=self._sample_y_wavefront,
                             smp_pos=self.params.sample_positions, smp_profile=self._sample_profile,
                             det_wfx=self._detector_x_wavefront, det_wfy=self._detector_y_wavefront,
                             det_ix=self._detector_x_intensity, det_iy=self._detector_y_intensity,
                             roi=self.find_beam_roi)

        self._init_attributes()

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

    def find_beam_roi(self) -> Tuple[int, int, int, int]:
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
            cnt_x = self.x_size // 2 + int(0.5 * (x0 + x1) // dx)
            grad_x = gaussian_gradient_magnitude(wfield_x, self.x_size // 100, mode='nearest',
                                                 num_threads=self.params.num_threads)
            x0 = (np.argmax(grad_x[:cnt_x]) * self.params.detx_size) // self.x_size
            x1 = ((cnt_x + np.argmax(grad_x[cnt_x:])) * self.params.detx_size) // self.x_size
        else:
            x0, x1 = 0, self.params.detx_size

        cnt_y = self.y_size // 2
        grad_y = gaussian_gradient_magnitude(wfield_y, self.y_size // 100, mode='nearest',
                                             num_threads=self.params.num_threads)
        y0 = (np.argmax(grad_y[:cnt_y]) * self.params.dety_size) // self.y_size
        y1 = ((cnt_y + np.argmax(grad_y[cnt_y:])) * self.params.dety_size) // self.y_size
        return (y0, y1, x0, x1)

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

class STConverter(DataContainer):
    """
    Converter class to export simulated data from :class:`STSim` to a CXI
    file. Takes an instance of :class:`STSim` and a generated stack of frames
    and automatically generates all the attributes necessary for the robust
    speckle tracking reconstruction (see :class:`pyrost.STData`).

    Attributes:
        sim_obj : a :class:`STSim` instance.

    Notes:
        **Necessary attributes**:

        * data : Simulated intensity frames.

        **Automatically generated attributes**:

        * basis_vectors : Detector basis vectors.
        * defocus_x : Defocus distance along the horizontal detector axis.
        * defocus_y : Defocus distance along the vertical detector axis.
        * distance : Sample-to-detector distance.
        * translations : Sample's translations.
        * transform : a :class:`pyrost.Crop` transform, that crops the frames
          according to automatically generated region of interest.
        * wavelength : Incoming beam's wavelength.
        * x_pixel_size : Pixel's size along the horizontal detector axis.
        * y_pixel_size : Pixel's size along the vertical detector axis.
    """
    unit_vector_fs = np.array([1, 0, 0])
    unit_vector_ss = np.array([0, -1, 0])
    attr_set = {'sim_obj', 'data', 'crd_rat'}
    init_set = {'basis_vectors', 'defocus_x', 'defocus_y', 'distance', 'translations',
                'transform', 'wavelength', 'x_pixel_size', 'y_pixel_size'}

    # Necessary attributes
    sim_obj         : STSim
    data            : np.ndarray
    crd_rat         : float

    # Automatically generated attributes
    basis_vectors   : np.ndarray
    defocus_x       : float
    defocus_y       : float
    distance        : float
    transform       : Crop
    translations    : np.ndarray
    wavelength      : float
    x_pixel_size    : float
    y_pixel_size    : float

    def __init__(self, sim_obj: STSim, data: np.ndarray, crd_rat: float=1e-6) -> None:
        """
        Args:
            sim_obj : a :class:`STSim` instance.
            data : Simulated stack of frames from `sim_obj`. The data may be
                generated either by :func:`STSim.frames` or :func:`STSim.ptychograph`
                method.
            crd_rat : Coordinates ratio between the simulated and saved data.
        """
        super(STConverter, self).__init__(sim_obj=sim_obj, data=data, crd_rat=crd_rat)

        self._init_functions(defocus_x=lambda: self.crd_rat * self.sim_obj.params.defocus,
                             defocus_y=lambda: self.crd_rat * self.sim_obj.params.defocus,
                             distance=lambda: self.crd_rat * self.sim_obj.params.det_dist,
                             wavelength=lambda: self.crd_rat * self.sim_obj.params.wl,
                             x_pixel_size=lambda: self.crd_rat * self.sim_obj.params.pix_size,
                             y_pixel_size=lambda: self.crd_rat * self.sim_obj.params.pix_size,
                             basis_vectors=self._basis_vectors, transform=self._crop,
                             translations=self._translations)

        self._init_attributes()

    def _basis_vectors(self):
        pix_vec = np.array([self.y_pixel_size, self.x_pixel_size, 0])
        vec_fs = np.tile(pix_vec * self.unit_vector_fs, (self.sim_obj.params.n_frames, 1))
        vec_ss = np.tile(pix_vec * self.unit_vector_ss, (self.sim_obj.params.n_frames, 1))
        return np.stack((vec_ss, vec_fs), axis=1)

    def _crop(self):
        crop = Crop(self.sim_obj.roi)
        if self.data.shape[1] == 1:
            crop = crop.integrate(axis=0)
        return crop

    def _translations(self):
        t_arr = np.zeros((self.sim_obj.params.n_frames, 3))
        t_arr[:, 0] = -self.sim_obj.smp_pos
        return self.crd_rat * t_arr

    def export_data(self, out_path: str, apply_transform: bool=True,
                    protocol: CXIProtocol=CXIProtocol.import_default()) -> STData:
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`) and a :class:`STSim` object `sim_obj`
        to a data container.

        Args:
            out_path : Path to the folder, where all the files are saved.
            apply_transform : Apply :class:`pyrost.Crop` to crop in the data with
                the region of interest `roi`.
            protocol : CXI file protocol.

        Returns:
            Data container :class:`pyrost.STData` with all the necessary
            attributes generated.
        """
        files = CXIStore(out_path, out_path, protocol=protocol)
        data = self.data
        if apply_transform:
            data = self.transform.forward(data)
        data_dict = {attr: self.get(attr) for attr in self.init_set}
        return STData(files=files, data=data, **data_dict)

    def save(self, out_path: str, apply_transform: bool=True,
             protocol: CXIProtocol=CXIProtocol.import_default(),
             mode: str='append', idxs: Optional[Iterable[int]]=None) -> None:
        """Export simulated data `data` (fetched from :func:`STSim.frames`
        or :func:`STSim.ptychograph`), `smp_pos`, and `st_params` to `dir_path`
        folder.

        Args:
            out_path : Path to the folder, where all the files are saved.
            apply_transform : Apply :class:`pyrost.Crop` to crop in the data with
                the region of interest `roi`.
            protocol : CXI file protocol.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices `idxs`.
                * `overwrite` : Overwrite the existing dataset.

            idxs : A set of frame indices where the data is saved if `mode` is
                `insert`.

        Returns:
            None
        """
        data = self.export_data(out_path=out_path, protocol=protocol,
                                apply_transform=apply_transform)
        data.save(mode=mode, idxs=idxs)

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
    parser.set_defaults(**STParams.import_default())

    args = vars(parser.parse_args())
    out_path = args.pop('out_path')
    ini_file = args.pop('ini_file')
    is_ptych = args.pop('ptych')
    if ini_file:
        st_params = STParams.import_ini(ini_file)
    else:
        st_params = STParams.import_default(**args)

    sim_obj = STSim(st_params)
    if is_ptych:
        data = sim_obj.ptychograph()
    else:
        data = sim_obj.frames()
    STConverter(sim_obj, data).save(out_path, mode='overwrite')
    print(f"The simulation results have been saved to {os.path.abspath(out_path)}")
