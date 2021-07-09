"""
Examples
--------

:func:`pyrost.st_parameters` generates the speckle tracking experimental
parameters, which could be later parsed to :class:`pyrost.simulation.STSim`
in order to perform the simulation.

>>> import pyrost.simulation as sim
>>> st_params = sim.STParams.import_ini()
>>> print(st_params)
{'defocus': 400.0, 'det_dist': 2000000.0, 'step_size': 0.1, 'n_frames': 300,
'fs_size': 2000, 'ss_size': 1000, 'pix_size': 55.0, '...': '...'}
"""
import os
import numpy as np
from multiprocessing import cpu_count
from ..ini_parser import INIParser, ROOT_PATH
from ..bin import bar_positions, barcode_profile, gaussian_kernel

ST_PARAMETERS = os.path.join(ROOT_PATH, 'config/st_parameters.ini')

class STParams(INIParser):
    """Container with the experimental parameters of
    one-dimensional Speckle Tracking scan. All the experimental
    parameters are enlisted in :mod:`st_parameters`.

    Parameters
    ----------
    barcode : dict, optional
        Barcode sample parameters. Default parameters are
        used if not provided.
    detector : dict, optional
        Detector parameters. Default parameters are used
        if not provided.
    exp_geom : dict, optional
        Experimental geometry parameters. Default parameters
        are used if not provided.
    lens : dict, optional
        Lens parameters. Default parameters are used if
        not provided.
    source : dict, optional
        X-ray source parameters. Default parameters are
        used if not provided.
    seed : int, optional
        Seed value for random number generation.

    Attributes
    ----------
    **kwargs : dict
        Experimental parameters enlisted in :mod:`st_parameters`.

    See Also
    --------
    st_parameters : Full list of experimental parameters.
    """
    attr_dict = {'exp_geom': ('defocus', 'det_dist', 'n_frames',
                              'step_size', 'step_rnd'),
                 'detector': ('fs_size', 'pix_size', 'ss_size'),
                 'source':   ('p0', 'th_s', 'wl'),
                 'lens':     ('alpha', 'ap_x', 'ap_y', 'focus', 'ab_cnt'),
                 'barcode':  ('bar_atn', 'bar_rnd', 'bar_sigma', 'bar_size',
                              'bulk_atn', 'offset'),
                 'system':   ('num_threads', 'seed')}

    fmt_dict = {'exp_geom': 'float', 'exp_geom/n_frames': 'int',
                'detector': 'int', 'detector/pix_size': 'float',
                'source': 'float', 'lens': 'float', 'barcode': 'float',
                'system': 'int'}
    FMT_LEN = 7

    def __init__(self, barcode=None, detector=None, exp_geom=None, lens=None, source=None, system=None):
        if barcode is None:
            barcode = self._import_ini(ST_PARAMETERS)['barcode']
        if detector is None:
            detector = self._import_ini(ST_PARAMETERS)['detector']
        if exp_geom is None:
            exp_geom = self._import_ini(ST_PARAMETERS)['exp_geom']
        if lens is None:
            lens = self._import_ini(ST_PARAMETERS)['lens']
        if source is None:
            source = self._import_ini(ST_PARAMETERS)['source']
        if system is None:
            system = self._import_ini(ST_PARAMETERS)['system']
        super(STParams, self).__init__(barcode=barcode, detector=detector,
                                       exp_geom=exp_geom, lens=lens, source=source,
                                       system=system)
        if self.seed <= 0:
            self.update_seed()
        if self.num_threads <= 0:
            self.update_threads()

    @classmethod
    def _lookup_dict(cls):
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __iter__(self):
        return self._lookup.__iter__()

    def __contains__(self, attr):
        return attr in self._lookup

    def __repr__(self):
        return self._format(self.export_dict()).__repr__()

    def __str__(self):
        return self._format(self.export_dict()).__str__()

    @classmethod
    def import_default(cls, **kwargs):
        """Return the default :class:`STParams`. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        **kwargs : dict
            Experimental parameters enlisted in :mod:`st_parameters`.

        Returns
        -------
        STParams
            An :class:`STParams` object with the default parameters.

        See Also
        --------
        st_parameters : Full list of the experimental parameters.
        """
        return cls.import_ini(ST_PARAMETERS, **kwargs)

    @classmethod
    def import_ini(cls, ini_file, **kwargs):
        """Initialize a :class:`STParams` object with an
        ini file.

        Parameters
        ----------
        ini_file : str, optional
            Path to the ini file. Load the default parameters if None.
        **kwargs : dict
            Experimental parameters enlisted in :mod:`st_parameters`.
            Initialized with `ini_file` if not provided.

        Returns
        -------
        st_params : STParams
            A :class:`STParams` object with all the attributes imported
            from the ini file.

        See Also
        --------
        st_parameters : Full list of the experimental parameters.
        """
        attr_dict = cls._import_ini(ini_file)
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(**attr_dict)

    def update_seed(self, seed=None):
        """Update seed used in pseudo-random number generation.

        Parameters
        ----------
        seed : int, optional
            New seed value. Chosen randomly if None.
        """
        if seed is None or seed <= 0:
            seed = np.random.default_rng().integers(0, np.iinfo(np.int_).max, endpoint=False)
        self.seed = seed

    def update_threads(self, num_threads=None):
        """Update number of threads used in calculcations.

        Parameters
        ----------
        num_threads : int, optional
            New seed value. Chosen randomly if None.
        """
        if num_threads is None or num_threads <= 0 or num_threads > 64:
            num_threads = np.clip(1, 64, cpu_count())
        self.num_threads = num_threads

    def x_wavefront_size(self):
        r"""Return wavefront array size along the x axis, that
        satisfies the sampling condition.

        Returns
        -------
        n_x : int
            Array size.

        Notes
        -----
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
            \frac{2 a_x n_{fs} \Delta_{pix} df}{f \lambda d_{det}}
        """
        nx_ltos = int(4 * self.ap_x**2 * max(self.focus, np.abs(self.defocus)) \
                      / self.focus**2 / self.wl)
        nx_stod = int(2 * self.fs_size * self.pix_size * self.ap_x * np.abs(self.defocus) \
                      / self.focus / self.wl / self.det_dist)
        return max(nx_ltos, nx_stod)

    def y_wavefront_size(self):
        r"""Return wavefront array size along the y axis, that
        satisfies the sampling condition.

        Returns
        -------
        n_y : int
            Array size.

        Notes
        -----
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
            \frac{2 a_y n_{ss} \Delta_{pix}}{\lambda d_{det}}
        """
        ny_ltos = int(8 * self.ap_y**2 / (self.focus + self.defocus) / self.wl)
        ny_stod = int(2 * self.ss_size * self.pix_size * self.ap_y / self.wl / self.det_dist)
        return max(ny_ltos, ny_stod)

    def lens_wavefronts(self, n_x=None, n_y=None, return_dxdy=False):
        r"""Return wavefields at the lens plane along x and y axes.

        Parameters
        ----------
        n_x : int, optional
            Array size along the x axis. Equals to
            :func:`STParams.x_wavefront_size` if it's None.
        n_y : int, optional
            Array size along the y axis. Equals to
            :func:`STParams.y_wavefront_size` if it's None.
        return_dxdy: bool, optional
            Return step sizes along x and y axes if it's True.

        Returns
        -------
        u0_x : numpy.ndarray
            Wavefront along the x axis.
        u0_y : numpy.ndarray
            Wavefront along the y axis.
        dx : float
            Step size along the x axis [um]. Only if
            `return_dxdy` is True.
        dy : float
            Step size along the y axis [um]. Only if
            `return_dxdy` is True.

        Notes
        -----
        The exit-surface at the lens plane:

        .. math::
            U_0(x) = \Pi(a_x x) \exp
            \left[ -\frac{j \pi x^2}{\lambda f} + j \alpha
            \left( \frac{x - x_{ab_cnt}}{f} \right)^3 \right]

        .. math::
            U_0(y) = \Pi(a_y y)
        """
        if n_x is None:
            n_x = self.x_wavefront_size()
        if n_y is None:
            n_y = self.y_wavefront_size()
        dx, dy = 2 * self.ap_x / n_x, 2 * self.ap_y / n_y
        x0 = dx * np.arange(-n_x // 2, n_x // 2)
        y0 = dy * np.arange(-n_y // 2, n_y // 2)
        x_cnt = (self.ab_cnt - 0.5) * self.ap_x
        u0_x = np.exp(1j * np.pi * x0**2 / self.wl / self.focus - \
                      1j * 1e9 * self.alpha * ((x0 - x_cnt) / self.focus)**3)
        u0_y = np.ones(n_y, dtype=np.complex128)
        u0_x[np.abs(x0) > self.ap_x / 2] = 0
        u0_y[np.abs(y0) > self.ap_y / 2] = 0
        if return_dxdy:
            return u0_x, u0_y, dx, dy
        else:
            return u0_x, u0_y

    def beam_span(self, dist):
        """Return beam span along the x axis at distance `dist`
        from the focal plane.

        Parameters
        ----------
        dist : float
            Distance from the focal plane [um].

        Returns
        -------
        th_lb : float
            Beam's lower bound [um].
        th_ub : float
            Beam's upper bound [um].
        """
        th_lb = -0.5 * self.ap_x / self.focus + self.wl / np.pi * self.alpha * \
                3.75e8 * (self.ap_x / self.focus)**2 / dist
        th_ub = 0.5 * self.ap_x / self.focus + self.wl / np.pi * self.alpha * \
                3.75e8 * (self.ap_x / self.focus)**2 / dist
        return np.tan(th_lb) * dist, np.tan(th_ub) * dist

    def bar_positions(self, dist, rnd_dev=True):
        """Generate a coordinate array of barcode's bar positions at
        distance `dist` from focal plane.

        Parameters
        ----------
        dist : float
            Distance from the focal plane [um].
        rnd_dev : bool, optional
            Randomize positions if it's True.

        Returns
        -------
        bar_pos : numpy.ndarray
            Array of barcode's bar coordinates.

        See Also
        --------
        bin.bar_positions : Full details of randomized barcode steps
            generation algorithm.
        """
        x0, x1 = self.beam_span(dist)
        seed = self.seed if rnd_dev else -1
        return bar_positions(x0=x0 + self.offset, b_dx=self.bar_size, rd=self.bar_rnd,
                             x1=x1 + self.step_size * self.n_frames - self.offset,
                             seed=seed)

    def sample_positions(self):
        """Generate an array of sample's translations with random deviation.

        Returns
        -------
        smp_pos : numpy.ndarray
            Array of sample translations [um].
        """
        rng = np.random.default_rng(self.seed)
        rnd_arr = 2 * self.step_rnd * (rng.random(self.n_frames) - 0.5)
        return self.step_size * (np.arange(self.n_frames) + rnd_arr)

    def barcode_profile(self, x_arr, bars, num_threads=1):
        """Generate a barcode's transmission profile at `x_arr`
        coordinates.

        Parameters
        ----------
        bar_pos : numpy.ndarray
            Array of barcode's bar positions [um].
        x_arr : numpy.ndarray
            Array of the coordinates, where the transmission
            coefficients are calculated [um].
        num_threads : int, optional
            Number of threads.

        Returns
        -------
        b_tr : numpy.ndarray
            Barcode's transmission profile.

        See Also
        --------
        bin.barcode_profile : Full details of barcode's transmission
            profile generation algorithm.
        """
        return barcode_profile(x_arr=x_arr, bars=bars, bulk_atn=self.bulk_atn,
                               bar_atn=self.bar_atn, bar_sigma=self.bar_sigma,
                               num_threads=num_threads)

    def source_curve(self, dist, step):
        """Return source's rocking curve profile at `dist` distance from
        the lens.

        Parameters
        ----------
        dist : float
            Distance from the lens to the rocking curve profile [um].
        step : float
            Sampling interval [um].

        Returns
        -------
        numpy.ndarray
            Source's rocking curve profile.
        """
        return gaussian_kernel(dist * self.th_s / step)
