"""
Examples:

    :func:`pyrost.simulation.STParams.import_default` generates the speckle tracking
    experimental parameters, which could be later parsed to :class:`pyrost.simulation.STSim`
    in order to perform the simulation.

    >>> import pyrost.simulation as st_sim
    >>> st_params = st_sim.STParams.import_default()
    >>> print(st_params)
    {'exp_geom': {'defocus': 100.0, 'det_dist': 2000000.0, 'n_frames': 300, '...': '...'},
     'detector': {'detx_size': 2000, 'dety_size': 1000, 'pix_size': 55.0},
     'source': {'p0': 200000.0, 'th_s': 0.0002, 'wl': 7.29e-05}, '...': '...'}
"""
from __future__ import annotations
import os
from typing import Dict, Iterable, Iterator, Tuple, Union, Optional
from multiprocessing import cpu_count
import numpy as np
from ..ini_parser import INIParser, ROOT_PATH
from ..bin import bar_positions, barcode_profile, gaussian_kernel, gaussian_filter

ST_PARAMETERS = os.path.join(ROOT_PATH, 'config/st_parameters.ini')

class STParams(INIParser):
    """Container with the simulation parameters for the wavefront propagation.

    Attributes:
        kwargs : Experimental parameters enlisted in :ref:`st-parameters`.

    See Also:
        :ref:`st-parameters` : Full list of experimental parameters.
    """
    attr_dict = {'exp_geom': ('defocus', 'det_dist', 'n_frames',
                              'step_size', 'step_rnd'),
                 'detector': ('detx_size', 'dety_size', 'pix_size'),
                 'source':   ('p0', 'th_s', 'wl'),
                 'lens':     ('alpha', 'ap_x', 'ap_y', 'focus', 'ab_cnt'),
                 'barcode':  ('bar_atn', 'bar_rnd', 'bar_sigma', 'bar_size',
                              'bulk_atn', 'offset'),
                 'system':   ('num_threads', 'seed')}

    fmt_dict = {'exp_geom': 'float', 'exp_geom/n_frames': 'int',
                'detector': 'int', 'detector/pix_size': 'float',
                'source': 'float', 'lens': 'float', 'barcode': 'float',
                'system': 'int'}

    # exp_geom attributes
    defocus     : float
    det_dist    : float
    n_frames    : int
    step_size   : float
    step_rnd    : float

    # detector attributes
    detx_size   : int
    dety_size   : int
    pix_size    : float

    # source attributes
    p0          : float
    th_s        : float
    wl          : float

    # lens attributes
    alpha       : float
    ap_x        : float
    ap_y        : float
    focus       : float
    ab_cnt      : float

    # barcode attributes
    bar_atn     : float
    bar_rnd     : float
    bar_sigma   : float
    bar_size    : float
    bulk_atn    : float
    offset      : float

    # system attributes
    num_threads : int
    seed        : int

    def __init__(self, barcode: Dict[str, float], detector: Dict[str, Union[int, float]],
                 exp_geom: Dict[str, Union[int, float]], lens: Dict[str, float],
                 source: Dict[str, float], system: Dict[str, int]) -> None:
        """
        Args:
            barcode : A dictionary of barcode sample parameters. The following elements
                are accepted:

                * `bar_size` : Average bar's size [um].
                * `bar_sigma` : Bar bluriness width [um].
                * `bar_atn` : Bar's attenuation coefficient [0.0 - 1.0].
                * `bulk_atn` : Barcode's bulk attenuation coefficient [0.0 - 1.0].
                * `bar_rnd` : Bar's coordinates random deviation [0.0 - 1.0].
                * `offset` : Barcode's offset at the beginning and at the end
                  of the scan from the detector's bounds [um].

            detector : A dictionary of detector parameters. The following elements are 
                accepted:

                * `detx_size` : Detector's size along the horizontal axis in pixels.
                * `dety_size` : Detector's size along the vertical axis in pixels.
                * `pix_size` : Detector's pixel size [um].

            exp_geom : A dictionary of experimental geometry parameters. The following elements
                are accepted:

                * `defocus` : Lens' defocus distance [um].
                * `det_dist` : Distance between the barcode and the detector [um].
                * `step_size` : Scan step size [um].
                * `n_frames` : Number of frames.

            lens : A dictionary of lens parameters. The following elements are accepted:

                * `ap_x` : Lens' aperture size along the x axis [um].
                * `ap_y` : Lens' aperture size along the y axis [um].
                * `focus` : Focal distance [um].
                * `alpha` : Third order aberrations coefficient [rad / mrad^3].
                * `ab_cnt` : Lens' aberrations center point [0.0 - 1.0].

            source : A dictionary of X-ray source parameters. The following elements are
                accepted:

                * `p0` : Source beam flux [cnt / s].
                * `wl` : Source beam's wavelength [um].
                * `th_s` : Source rocking curve width [rad].

            system : A dictionary of calculation parameters. The following elements are
                accepted:

                * `seed` : Seed used in all the pseudo-random number generations.
                * `num_threads` : Number of threads used in the calculations.
        """
        super(STParams, self).__init__(barcode=barcode, detector=detector,
                                       exp_geom=exp_geom, lens=lens, source=source,
                                       system=system)
        if self.seed <= 0:
            self.update_seed()
        if self.num_threads <= 0:
            self.update_threads()

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __iter__(self) -> Iterator[str]:
        return self._lookup.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self._lookup

    def __repr__(self) -> str:
        return self._format(self.export_dict()).__repr__()

    def __str__(self) -> str:
        return self._format(self.export_dict()).__str__()

    def keys(self) -> Iterable[str]:
        return list(self)

    @classmethod
    def import_default(cls, **kwargs: Union[int, float]) -> STParams:
        """Return the default :class:`STParams`. Extra arguments
        override the default values if provided.

        Args:
            kwargs : Simulation parameters enlisted in :ref:`st-parameters`.

        Returns:
            An :class:`STParams` object with the default parameters.
        """
        return cls.import_ini(ST_PARAMETERS, **kwargs)

    @classmethod
    def import_ini(cls, ini_file: str, **kwargs: Union[int, float]) -> STParams:
        """Initialize a :class:`STParams` object with an
        ini file.

        Args:
            ini_file : Path to the ini file.
            kwargs : Experimental parameters enlisted in :ref:`st-parameters`.
                Override the parameters imported from the `ini_file`.

        Returns:
            A :class:`STParams` object with all the attributes imported
            from the ini file.
        """
        attr_dict = cls._import_ini(ini_file)
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(**attr_dict)

    def update_seed(self, seed: Optional[int]=None) -> None:
        """Update seed used in pseudo-random number generation.

        Args:
            seed : New seed value. Chosen randomly if None.
        """
        if seed is None or seed <= 0:
            seed = np.random.default_rng().integers(0, np.iinfo(np.int_).max, endpoint=False)
        self.seed = seed

    def update_threads(self, num_threads: Optional[int]=None) -> None:
        """Update number of threads used in calculcations.

        Args:
            num_threads : Number of threads used in the computations.
        """
        if num_threads is None or num_threads <= 0 or num_threads > 64:
            num_threads = np.clip(1, 64, cpu_count())
        self.num_threads = num_threads

    def x_wavefront_size(self) -> int:
        r"""Return wavefront array size along the x axis, that
        satisfies the sampling condition.

        Returns:
            Array size.

        Notes:
            The sampling condition when we propagate the wavefield from
            the lens plane to the sample plane is as follows:

            .. math::
                N_x >= \frac{\mathrm{max} \left( \Delta x_{lens}, \Delta x_{sample} \right)
                (\Delta x_{lens} + \Delta x_{sample})}{\lambda d} =
                \frac{4 a_x^2 \mathrm{max} \left( f, df \right)}{f^2 \lambda}

            Whereupon, the sampling condition when we propagate the wavefield to the detector
            plane is:

            .. math::
                N_x >= \frac{\Delta x_{sample} \Delta x_{det}}{\lambda d} =
                \frac{2 a_x n_{x} \Delta_{pix} df}{f \lambda d_{det}}
        """
        nx_ltos = int(4 * self.ap_x**2 * max(self.focus, np.abs(self.defocus)) \
                      / self.focus**2 / self.wl)
        nx_stod = int(2 * self.detx_size * self.pix_size * self.ap_x * np.abs(self.defocus) \
                      / self.focus / self.wl / self.det_dist)
        return max(nx_ltos, nx_stod)

    def y_wavefront_size(self) -> int:
        r"""Return wavefront array size along the y axis, that
        satisfies the sampling condition.

        Returns:
            Array size.

        Notes:
            The sampling condition when we propagate the wavefield from
            the lens plane to the sample plane is as follows:

            .. math::
                N_y >= \frac{\mathrm{max} \left( \Delta y_{lens}, \Delta y_{sample} \right)
                (\Delta y_{lens} + \Delta y_{sample})}{\lambda d} =
                \frac{8 a_y^2}{(f + df) \lambda}

            Whereupon, the sampling condition when we propagate the wavefield to the detector
            plane is:

            .. math::
                N_y >= \frac{\Delta y_{sample} \Delta y_{det}}{\lambda d} =
                \frac{2 a_y n_{y} \Delta_{pix}}{\lambda d_{det}}
        """
        ny_ltos = int(8 * self.ap_y**2 / (self.focus + self.defocus) / self.wl)
        ny_stod = int(2 * self.dety_size * self.pix_size * self.ap_y / self.wl / self.det_dist)
        return max(ny_ltos, ny_stod)

    def lens_x_wavefront(self, n_x: Optional[int]=None,
                         return_step: bool=False) -> Tuple[np.ndarray, float]:
        r"""Return wavefields at the lens plane along x axis.

        Args:
            n_x : Array size along the x axis. Equals to
                :func:`STParams.x_wavefront_size` if it's None.
            return_step : Return the step size along x axis if it's True.

        Returns:
            A tuple of two elements ('u0_x', 'dx'). The elements
            are the following:

            * `u0_x` : Wavefront along the x axis.
            * `dx` : Step size along the x axis [um]. Only if `return_step` is True.

        Notes:
            The exit-surface at the lens plane:

            .. math::
                U_0(x) = \Pi(a_x x) \exp
                \left[ -\frac{j \pi x^2}{\lambda f} + j \alpha
                \left( \frac{x - x_{ab_cnt}}{f} \right)^3 \right]
        """
        if n_x is None:
            n_x = self.x_wavefront_size()

        dx = 2.0 * self.ap_x / n_x
        x0 = dx * np.arange(-n_x // 2, n_x // 2)
        x_cnt = (self.ab_cnt - 0.5) * self.ap_x
        u0_x = np.exp(1j * np.pi * x0 * x0 / self.wl / self.focus - \
                      1e9j * self.alpha * ((x0 - x_cnt) / self.focus)**3)
        u0_x[np.abs(x0) > 0.5 * self.ap_x] = 0

        if return_step:
            return u0_x, dx
        return u0_x

    def lens_y_wavefront(self, n_y: Optional[int]=None,
                         return_step: bool=False) -> Tuple[np.ndarray, float]:
        r"""Return wavefields at the lens plane along y axis.

        Args:
            n_y : Array size along the y axis. Equals to
                :func:`STParams.y_wavefront_size` if it's None.
            return_step : Return the step size along y axis if it's True.

        Returns:
            A tuple of two elements ('u0_y', 'dy'). The elements
            are the following:

            * `u0_y` : Wavefront along the y axis.
            * `dy` : Step size along the y axis [um]. Only if `return_step` is True.

        Notes:
            The exit-surface at the lens plane:

            .. math::
                U_0(y) = \Pi(a_y y)
        """
        if n_y is None:
            n_y = self.y_wavefront_size()

        dy = 2.0 * self.ap_y / n_y
        y0 = dy * np.arange(-n_y // 2, n_y // 2)
        u0_y = np.ones(n_y, dtype=np.complex128)
        u0_y[np.abs(y0) > 0.5 * self.ap_y] = 0

        if return_step:
            return u0_y, dy
        return u0_y

    def beam_span(self, dist: float) -> Tuple[float, float]:
        """Return beam span along the x axis at distance `dist`
        from the focal plane.

        Args:
            dist : Distance from the focal plane [um].

        Returns:
            Tuple of two items ('th_lb', 'th_ub'). The elements are
            the following:

            * `th_lb` : Beam's lower bound [um].
            * `th_ub` : Beam's upper bound [um].
        """
        th_lb = -0.5 * self.ap_x / self.focus + self.wl / np.pi * self.alpha * \
                3.75e8 * (self.ap_x / self.focus)**2 / dist
        th_ub = 0.5 * self.ap_x / self.focus + self.wl / np.pi * self.alpha * \
                3.75e8 * (self.ap_x / self.focus)**2 / dist
        return np.tan(th_lb) * dist, np.tan(th_ub) * dist

    def bar_positions(self, dist: float, rnd_dev: bool=True) -> np.ndarray:
        """Generate a coordinate array of barcode's bar positions at distance
        `dist` from focal plane.

        Args:
            dist : Distance from the focal plane [um].
            rnd_dev : Randomize positions if it's True.

        Returns:
            Array of barcode's bar coordinates.

        See Also:
            :func:`pyrost.bin.bar_positions` : Full details of randomized barcode
            steps generation algorithm.
        """
        x0, x1 = self.beam_span(dist)
        seed = self.seed if rnd_dev else -1
        return bar_positions(x0=x0 + self.offset, b_dx=self.bar_size, rd=self.bar_rnd,
                             x1=x1 + self.step_size * self.n_frames - self.offset,
                             seed=seed)

    def sample_positions(self) -> np.ndarray:
        """Generate an array of sample's translations with random deviation.

        Returns:
            Array of sample translations [um].
        """
        rng = np.random.default_rng(self.seed)
        rnd_arr = 2 * self.step_rnd * (rng.random(self.n_frames) - 0.5)
        return self.step_size * (np.arange(self.n_frames) + rnd_arr)

    def barcode_profile(self, x_arr: np.ndarray, dx: float, bars: np.ndarray) -> np.ndarray:
        """Generate a barcode's transmission profile at `x_arr` coordinates.

        Args:
            x_arr : Array of the coordinates, where the transmission
                coefficients are calculated [um].
            dx : Sampling interval of the coordinate array [um].
            bars : Array of barcode's bar positions [um].

        Returns:
            Barcode's transmission profile.

        See Also:
            :func:`pyrost.bin.barcode_profile` : Full details of barcode's
            transmission profile generation algorithm.
        """
        return barcode_profile(x_arr=x_arr, bars=bars, bulk_atn=self.bulk_atn,
                               bar_atn=self.bar_atn, bar_sigma=self.bar_sigma,
                               num_threads=self.num_threads)

    def source_curve(self, dist: float, step: float) -> np.ndarray:
        """Return source's rocking curve profile at `dist` distance from
        the lens.

        Args:
            dist : Distance from the lens to the rocking curve profile [um].
            step : Sampling interval [um].

        Returns:
            Source's rocking curve profile.
        """
        return gaussian_kernel(sigma=dist * self.th_s / step)
