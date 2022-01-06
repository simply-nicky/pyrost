"""Multislice beam propagation simulation. Generate a profile of
the beam that propagates through a bulky sample.
:class:`pyrost.multislice.MLL` generates a transmission profile of
a wedged Multilayer Laue lens (MLL). :class:`pyrost.multislice.MSPropagator`
calculates the beam propagation and spits out the wavefield profile of
the propagated beam together with the transmission profile of the sample
at each slice.

Examples:

    Initialize a MLL object with the :class:`pyrost.multislice.MSParams`
    parameters object `params` with :func:`pyrost.multislice.MLL.import_params`.
    Then you can initialize a multislice beam propagator
    :class:`pyrost.multislice.MSPropagator` with `params` and `mll` and perform
    the multislice beam propagation as follows:

    >>> import pyrost.multislice as ms_sim
    >>> params = ms_sim.MSParams.import_default()
    >>> mll = ms_sim.MLL.import_params(params)
    >>> ms_prgt = ms_sim.MSPropagator(params, mll)
    >>> ms_prgt.beam_propagate() # doctest: +SKIP

    All the results are saved into `ms_prgt.beam_profile` and
    `ms_prgt.smp_profile` attributes.
"""
from __future__ import annotations
from multiprocessing import cpu_count
from typing import Iterable, Optional, Tuple, Union
from tqdm.auto import tqdm
import numpy as np
from .ms_parameters import MSParams
from ..data_container import DataContainer, dict_to_object
from ..bin import mll_profile, FFTW, empty_aligned, rsc_wp

class MLL(DataContainer):
    """
    Multilayer Laue lens class.

    Parameters
    ----------
    layers : numpy.ndarray
        MLL's bilayers x coordinates [um].
    mat1_r, mat2_r : complex
        Fresnel transmission coefficients for each of the materials,
        that the MLL's bilayers are composed of. The coefficients are the
        ratio of a wavefield propagating through a single slice of the
        lens.
    sigma : float
        Bilayer's interdiffusion length [um].

    Attributes
    ----------
    attr_set : set
        Set of the attributes in the container which are necessary
        to initialize in the constructor.
    """
    attr_set = {'layers', 'mat1_r', 'mat2_r', 'sigma'}
    en_to_wl = 1.239841929761768 # h * c / e [eV * um]

    def __init__(self, mat1_r: complex, mat2_r: complex, sigma: float,
                 layers: np.ndarray) -> None:
        super(MLL, self).__init__(mat1_r=mat1_r, mat2_r=mat2_r, sigma=sigma,
                                  layers=layers)

    @classmethod
    def import_params(cls, params: MSParams) -> MLL:
        """Return a new :class:`MLL` object import from the
        :class:`MSParams` multislice parameters object.

        Parameters
        ----------
        params : MSParams
            Experimental parameters of the multislice beam
            propagation simulation.

        Returns
        -------
        mll : MLL
            New :class:`MLL` object.
        """
        n_arr = np.arange(params.n_min, params.n_max)
        z_coords = params.z_step * np.arange(params.mll_depth // params.z_step)
        layers = np.sqrt(n_arr * params.focus * params.mll_wl + \
                              n_arr**2 * params.mll_wl**2 / 4)
        layers = layers * (1 - z_coords[:, None] / (2 * params.focus))
        mat1_r = params.get_mat1_r(cls.en_to_wl / params.wl)
        mat2_r = params.get_mat2_r(cls.en_to_wl / params.wl)
        sigma = params.mll_sigma
        return cls(mat1_r=mat1_r, mat2_r=mat2_r, sigma=sigma,
                   layers=layers)

    @property
    def n_slices(self) -> int:
        "Total number of slices"
        return self.layers.shape[0]

    @dict_to_object
    def update_interdiffusion(self, sigma: float) -> MLL:
        """Return a new :class:`MLL` object with the updated `sigma`.

        Parameters
        ----------
        sigma : float
            Bilayer's interdiffusion length [um].

        Returns
        -------
        mll : MLL
            New :class:`MLL` object with the updated `sigma`.
        """
        return {'sigma': sigma}

    @dict_to_object
    def update_materials(self, mat1_r: complex, mat2_r: complex) -> MLL:
        """Return a new :class:`MLL` object with the updated materials
        `mat1_r` and `mat2_r`.

        Parameters
        ----------
        mat1_r, mat2_r : complex
            Fresnel transmission coefficients for each of the materials,
            that the MLL's bilayers are composed of. The coefficients are the
            ratio of a wavefield propagating through a single slice of the
            lens.

        Returns
        -------
        mll : MLL
            New :class:`MLL` object with the updated `mat1_r` and `mat2_r`.
        """
        return {'mat1_r': mat1_r, 'mat2_r': mat2_r}

    def get_span(self) -> Tuple[float, float]:
        """Return the pair of bounds (x_min, x_max) of the MLL
        along the x axis.

        Returns
        -------
        x_min, x_max : float
            MLL's bounds along the x axis [um].
        """
        return (self.layers.min(), self.layers.max())

    def get_profile(self, x_arr: np.ndarray, output: np.ndarray,
                    num_threads: int=1) -> Iterable:
        """Return a generator, that yields transmission profiles of
        the lens at each slice and writes to `output` array.

        Parameters
        ----------
        x_arr : numpy.ndarray
            Coordinates array [um].
        output: numpy.ndarray
            Output array.
        num_threads : int, optional
            Number of threads.

        Returns
        -------
        slices : iterable
            The generator, which yields lens' transmission profiles.
        """
        for idx, layer in enumerate(self.layers):
            output[idx, :] = mll_profile(x_arr=x_arr, layers=layer, mt0=self.mat1_r,
                                         mt1=self.mat2_r, sigma=self.sigma,
                                         num_threads=num_threads)
            yield output[idx]

class MSPropagator(DataContainer):
    """One-dimensional Multislice beam propagation class.
    Generates beam profile, that propagates through the sample
    using multislice approach.

    Parameters
    ----------
    params : MSParams
        Experimental parameters
    sample : MLL
        Sample class, that generates the sample's
        transmission profile.
    num_threads : int, optional
        Number of threads used in the calculations.
    **kwargs : dict
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

    * sample : Sample's transmission profile generator.
    * num_threads : Number of threads used in the calculations.
    * params : Experimental parameters.

    Optional attributes:

    * beam_profile : Beam profiles at each slice.
    * fx_arr : Spatial frequencies array [um^-1].
    * kernel : Diffraction kernel.
    * smp_profile : Sample's transmission profile at each slice.
    * x_arr : Coordinates array in the transverse plane [um].
    * wf_inc : Wavefront at the entry surface.
    * z_arr : Coordinates array along the propagation axis [um].
    """
    attr_set = {'params', 'sample'}
    init_set = {'beam_profile', 'fx_arr', 'kernel', 'num_threads', 'smp_profile', 'x_arr',
                'wf_inc', 'z_arr'}

    inits = {'num_threads'  : lambda obj: np.clip(1, 64, cpu_count()),
             'x_arr'        : lambda obj: obj.params.get_xcoords(),
             'z_arr'        : lambda obj: obj.params.get_zcoords(),
             'fx_arr'       : lambda obj: np.fft.fftfreq(obj.size, obj.params.x_step),
             'kernel'       : lambda obj: obj.params.get_kernel(obj.fx_arr) / obj.fx_arr.size,
             'wf_inc'       : lambda obj: obj._inc_wavefront()}

    def __init__(self, params: MSParams, sample: MLL, num_threads: Optional[int]=None,
                 **kwargs: np.ndarray) -> None:
        super(MSPropagator, self).__init__(params=params, sample=sample,
                                           num_threads=num_threads, **kwargs)

    def _inc_wavefront(self):
        wf_inc = np.ones(self.x_arr.shape, dtype=np.complex128)
        x_min, x_max = self.sample.get_span()
        wf_inc[(self.x_arr < x_min) | (self.x_arr > x_max)] = 0.0
        return wf_inc

    @property
    def size(self) -> int:
        "Number of points in a single slice."
        return self.x_arr.size

    @dict_to_object
    def update_inc_wavefront(self, wf_inc: np.ndarray) -> MSPropagator:
        """Return a new :class:`MSPropagator` object with the updated `wf_inc`.

        Parameters
        ----------
        wf_inc : numpy.ndarray
            Wavefront at the entry surface.

        Returns
        -------
        MSPropagator
            A new :class:`MSPropagator` object with the updated `wf_inc`.
        """
        if wf_inc.shape != self.x_arr.shape:
            raise ValueError(f'Wavefront shape must be equal to {self.x_arr.shape:s}')
        if wf_inc.dtype != np.complex128:
            raise ValueError("Wavefront datatype must be 'complex128'")
        return {'beam_profile': None, 'wf_inc': wf_inc}

    def generate_sample(self, verbose: bool=True) -> None:
        """Generate the transmission profile of the sample. The results are
        written to `smp_profile` attribute.

        Parameters
        ----------
        verbose : bool, optional
            Set verbosity of the computation process.
        """
        self.smp_profile = empty_aligned((self.sample.n_slices, self.x_arr.size),
                                         dtype='complex128')
        itor = self.sample.get_profile(x_arr=self.x_arr, output=self.smp_profile,
                                       num_threads=self.num_threads)

        if verbose:
            itor = tqdm(enumerate(itor), total=self.sample.n_slices,
                        bar_format='{desc} {percentage:3.0f}% {bar} Slice {n_fmt} / {total_fmt} '\
                                '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        for idx, _ in itor:
            if verbose:
                itor.set_description(f"z = {self.z_arr[idx]:.2f} um")

    def beam_propagate(self, verbose: bool=True) -> None:
        """Perform the multislice beam propagation. The results are
        written to `beam_profile` attribute. Generates `smp_profile`
        attribute if it wasn't initialized before.

        Parameters
        ----------
        verbose : bool, optional
            Set verbosity of the computation process.
        """
        self.beam_profile = empty_aligned((self.sample.n_slices + 1, self.x_arr.size),
                                          dtype='complex128')
        current_slice = empty_aligned(self.x_arr.size, dtype='complex128')
        self.beam_profile[0] = self.wf_inc
        current_slice[:] = self.wf_inc

        fft_obj = FFTW(current_slice, current_slice, flags=('FFTW_ESTIMATE',),
                       threads=self.num_threads)
        ifft_obj = FFTW(current_slice, self.beam_profile[1], direction='FFTW_BACKWARD',
                        flags=('FFTW_ESTIMATE',), threads=self.num_threads)

        if self.smp_profile is None:
            self.smp_profile = empty_aligned((self.sample.n_slices, self.x_arr.size),
                                             dtype='complex128')
            itor = self.sample.get_profile(x_arr=self.x_arr, output=self.smp_profile,
                                           num_threads=self.num_threads)
        else:
            itor = (layer for layer in self.smp_profile)

        miniters = max(self.sample.n_slices // 100, 1)
        itor = tqdm(itor, total=self.sample.n_slices, miniters=miniters, disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} Slice {n_fmt} / {total_fmt} '\
                               '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        for idx, layer in enumerate(itor):
            if verbose and not idx % miniters:
                itor.set_description(f"z = {self.z_arr[idx]:.2f} um")
            current_slice *= layer
            fft_obj.execute()
            current_slice *= self.kernel
            ifft_obj.update_arrays(current_slice, self.beam_profile[idx + 1])
            ifft_obj.execute()
            current_slice[:] = ifft_obj.output_array

    def beam_downstream(self, z_arr: np.ndarray, step: Optional[float]=None,
                        return_coords: bool=True, verbose: bool=True,
                        backend: str='fftw') -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Return the wavefront at distance `z_arr` downstream from
        the exit surface using Rayleigh-Sommerfeld convolution.

        Parameters
        ----------
        z_arr : numpy.ndarray
            Array of distances [um].
        step : float, optional
            Sampling interval of the downstream coordinate array [um].
            Equals to the sampling interval at the exit surface if it's
            None.
        return_coords : bool, optional
            Return the coordinates array of the downstream plane.
        verbose : bool, optional
            Set verbosity of the computation process.

        Returns
        -------
        wavefronts : numpy.ndarray
            Set of wavefronts calculated at the distance `z_arr`
            downstream from the exit surface.
        x_arr : numpy.ndarray
            Array of coordinates at the plane downstream [um].
            Only if `return_coords` is True.

        Raises
        ------
        AttributeError
            If `beam_profile` has not been generated. Call
            :func:`MSPropagator.beam_propagate` to generate.
        """
        if self.beam_profile is None:
            raise AttributeError('The beam profile has not been generated')
        if step is None:
            step = self.params.x_step
        size = int(self.x_arr.size**2 * max(self.params.x_step, step) * \
                   (step + self.params.x_step) / self.params.wl / np.min(z_arr))
        wf0 = self.beam_profile[-1]
        if size > self.x_arr.size:
            wf0 = np.pad(wf0, ((size - self.x_arr.size) // 2, (size - self.x_arr.size) // 2))
        else:
            size = self.x_arr.size

        itor = np.atleast_1d(z_arr)
        if verbose:
            itor = tqdm(itor, total=itor.size)

        wavefronts = []
        for dist in itor:
            if verbose:
                itor.set_description(f"z = {dist:.2f} um")
            wavefronts.append(rsc_wp(wft=wf0, dx0=self.params.x_step, dx=step,
                                     z=dist, wl=self.params.wl, backend=backend,
                                     num_threads=self.num_threads))
        if return_coords:
            x_arr = step * np.arange(-size // 2, size - size // 2) + np.mean(self.x_arr)
            return np.squeeze(np.stack(wavefronts, axis=1)), x_arr
        return np.squeeze(np.stack(wavefronts, axis=1))
