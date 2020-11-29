"""Experimental parameters for the one-dimenional Speckle Tracking
scan. :class:`STParams` contains all the parameters and provides
tools to import and export parameters from an INI file. Invoke
:func:`parameters` to get the default experimental parameters.

Examples
--------

Generate parameters, which could be later parsed to :class:`STSim`
in order to perform the simulation.

>>> import pyrost.simulation as st_sim
>>> st_params = st_sim.parameters()
>>> print(st_params)
{'defocus': 400.0, 'det_dist': 2000000.0, 'step_size': 0.1,
'n_frames': 300, 'fs_size': 2000, 'ss_size': 1000, 'pix_size': 55.0,
'p0': 200000.0, 'wl': 7.29e-05, 'th_s': 0.0002, 'ap_x': 40.0,
'ap_y': 2.0, 'focus': 1500.0, 'alpha': -0.05, 'x0': 0.5,
'bar_size': 0.1, 'bar_sigma': 0.01, 'bar_atn': 0.3, 'bulk_atn': 0.0,
'rnd_dev': 0.6, 'offset': 0.0, 'verbose': True}

Notes
-----
List of experimental parameters:

* Experimental geometry parameters:

    * defocus : Lens' defocus distance [um].
    * det_dist : Distance between the barcode and the
      detector [um].
    * step_size : Scan step size [um].
    * n_frames : Number of frames.

* Detector parameters:

    * fs_size : Detector's size along the fast axis in
      pixels.
    * ss_size : Detector's size along the slow axis in
      pixels.
    * pix_size : Detector's pixel size [um].

* Source parameters:

    * p0 : Source beam flux [cnt / s].
    * wl : Incoming beam's wavelength [um].
    * th_s : Source rocking curve width [rad].

* Lens parameters:

    * ap_x : Lens' aperture size along the x axis [um].
    * ap_y : Lens' aperture size along the y axis [um].
    * focus : Focal distance [um].
    * alpha : Third order abberations ceofficient [rad/mrad^3].
    * x0 : Lens' abberations center point [0.0 - 1.0].

* Barcode sample parameters:

    * bar_size : Average bar's size [um].
    * bar_sigma : Bar bluriness width [um].
    * bar_atn : Bar's attenuation coefficient [0.0 - 1.0].
    * bulk_atn : Barcode's bulk attenuation coefficient [0.0 - 1.0].
    * rnd_dev : Bar's coordinates random deviation [0.0 - 1.0].
    * offset : Barcode's offset at the beginning and at the end
      of the scan from the detector's bounds [um].
"""
import os
import numpy as np
from ..protocol import INIParser, ROOT_PATH

PARAMETERS_FILE = os.path.join(ROOT_PATH, 'config/parameters.ini')

class STParams(INIParser):
    """Container with the experimental parameters of
    one-dimensional Speckle Tracking scan. All the experimental
    parameters are enlisted in `attr_dict`.

    Parameters
    ----------
    **kwargs : dict
        Values for the exerimental parameters specified
        in `attr_dict`.

    Attributes
    ----------
    attr_dict : dict
        Dictionary which contains all the experimental
        parameters.
    fmt_dict: dict
        Dictionary which specifies the data types of the
        parameters in `attr_dict`.

    See Also
    --------
    st_sim_param : Full list of experimental parameters.
    """
    attr_dict = {'exp_geom': ('defocus', 'det_dist', 'step_size',
                              'n_frames'),
                 'detector': ('fs_size', 'ss_size', 'pix_size'),
                 'source':   ('p0', 'wl', 'th_s'),
                 'lens':     ('ap_x', 'ap_y', 'focus', 'alpha', 'x0'),
                 'barcode':  ('bar_size', 'bar_sigma', 'bar_atn',
                              'bulk_atn', 'rnd_dev', 'offset'),
                 'system':   ('verbose',)}

    fmt_dict = {'exp_geom': 'float', 'exp_geom/n_frames': 'int',
                'detector': 'int', 'detector/pix_size': 'float',
                'source': 'float', 'lens': 'float', 'barcode': 'float',
                'system/verbose': 'bool'}

    @classmethod
    def lookup_dict(cls):
        """Look-up table between the sections and the parameters.

        Returns
        -------
        dict
            Look-up dictionary.
        """
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __init__(self, **kwargs):
        super(STParams, self).__init__(**kwargs)
        self.__dict__['_lookup'] = self.lookup_dict()

    def __iter__(self):
        return self._lookup.__iter__()

    def __contains__(self, attr):
        return attr in self._lookup

    def __getattr__(self, attr):
        if attr in self._lookup:
            return self.__dict__[self._lookup[attr]][attr]
        else:
            raise AttributeError(attr + " doesn't exist")

    def __setattr__(self, attr, value):
        if attr in self._lookup:
            fmt = self.get_format(self._lookup[attr], attr)
            self.__dict__[self._lookup[attr]][attr] = fmt(value)
        else:
            raise AttributeError(attr + ' not allowed')

    @classmethod
    def import_dict(cls, **kwargs):
        init_dict = {}
        for section in cls.attr_dict:
            init_dict[section] = {}
            for option in cls.attr_dict[section]:
                init_dict[section][option] = kwargs[option]
        return cls(**init_dict)

    def beam_span(self, dist):
        """Return beam span along the x axis at distance `dist`
        from the focal plane.

        Parameters
        ----------
        dist : float
            Distance from the focal plane.

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

    def fs_roi(self):
        """Return detector's region of interest along the fast axis.

        Returns
        -------
        fs_lb : int
            Beam's lower bound along the detector's fast axis.
        fs_ub : int
            Beam's upper bound along the detector's fast axis.
        """
        x_lb, x_ub = self.beam_span(self.det_dist)
        fs_lb = int(x_lb // self.pix_size) + self.fs_size // 2
        fs_ub = int(x_ub // self.pix_size) + self.fs_size // 2
        fs_lb = fs_lb if fs_lb > 0 else 0
        fs_ub = fs_ub if fs_ub < self.fs_size else self.fs_size
        return fs_lb, fs_ub

    def export_dict(self):
        """Export experimental parameters to :class:`dict`.

        Returns
        -------
        param_dict : dict
            Experimental parameters.
        """
        param_dict = {}
        for section in self.attr_dict:
            for option in self.attr_dict[section]:
                param_dict[option] = self.__dict__[section][option]
        return param_dict

    def log(self, logger):
        """Log all the experimental parameters with `logger`.

        Parameters
        ----------
        logger : logging.Logger
            Logging interface.
        """
        for section in self.attr_dict:
            logger.info('[{:s}]'.format(section))
            for option in self.attr_dict[section]:
                log_txt = '\t{0:s}: {1:s}, [{2:s}]'
                log_msg = log_txt.format(option, str(self.__dict__[section][option]),
                                         self.fmt_dict[os.path.join(section, option)])
                logger.info(log_msg)

def parameters(**kwargs):
    """Return the default :class:`STParams` object. Override any
    experimental parameters with `**kwargs`.

    Parameters
    ----------
    **kwargs : dict
        Dictionary which contains experimental
        parameters values.

    Returns
    -------
    st_params : STParams
        Default experimental parameters.

    See Also
    --------
    STParams : Full list of the experimental parameters.
    """
    st_params = STParams.import_ini(PARAMETERS_FILE).export_dict()
    st_params.update(**kwargs)
    return STParams.import_dict(**st_params)
