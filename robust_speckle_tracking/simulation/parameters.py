"""
parameters.py - Speckle Tracking simulation's parameters implementation
"""
import os
import numpy as np
from ..protocol import INIParser, ROOT_PATH

class STParams(INIParser):
    """
    Speckle Tracking parameters class (STParams)

    [exp_geom - experimental geometry parameters]
    defocus - lens defocus distance [um]
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
    x0 - lens' abberation center point [0.0 - 1.0]

    [barcode - barcode parameters]
    bar_size - average bar size [um]
    bar_sigma - bar haziness width [um]
    bar_atn - bar attenuation
    bulk_atn - bulk attenuation
    random_dev - bar random deviation
    offset - sample's offset at the beginning and the end of the scan [um]
    """
    attr_dict = {'exp_geom': ('defoc', 'det_dist', 'step_size', 'n_frames'),
                 'detector': ('fs_size', 'ss_size', 'pix_size'),
                 'source':   ('p0', 'wl', 'th_s'),
                 'lens':     ('ap_x', 'ap_y', 'focus', 'alpha', 'x0'),
                 'barcode':  ('bar_size', 'bar_sigma', 'bar_atn',
                              'bulk_atn', 'random_dev', 'offset'),
                 'system':   ('verbose',)}

    fmt_dict = {'exp_geom': 'float', 'exp_geom/n_frames': 'int',
                'detector': 'int', 'detector/pix_size': 'float',
                'source': 'float', 'lens': 'float', 'barcode': 'float',
                'system/verbose': 'bool'}

    @classmethod
    def lookup_dict(cls):
        """
        Return the look-up table
        """
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __init__(self, **kwargs):
        super(STParams, self).__init__(**kwargs)
        self.__dict__['_lookup'] = self.lookup_dict()

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
        """
        Return beam span in x coordinate at the given distance from the focal plane

        dist - distance from focal plane
        """
        fx_max = 0.5 * self.ap_x / self.focus
        fx_lim = np.array([-fx_max, fx_max])
        th_lim = fx_lim + self.wl / 2 / np.pi * self.alpha * 3e9 * fx_lim**2 / dist
        return np.tan(th_lim) * dist

    def roi(self, shape):
        """
        Return ROI for the given coordinate array based on incident beam span

        dist - distance from focal plane
        x_min, x_max - coordinate array extremities
        dx - coordinate array step
        """
        beam_span = np.clip(self.beam_span(self.det_dist),
                            -shape[-1] // 2 * self.pix_size,
                            (shape[-1] // 2 - 1) * self.pix_size)
        fs_roi = (beam_span // self.pix_size + shape[-1] // 2).astype(np.int)
        return np.array([0, shape[1], fs_roi[0], fs_roi[1]])

    def export_dict(self):
        """
        Return Speckle Tracking parameters dictionary
        """
        param_dict = {}
        for section in self.attr_dict:
            for option in self.attr_dict[section]:
                param_dict[option] = self.__dict__[section][option]
        return param_dict

    def log(self, logger):
        """
        Log Speckle Tracking parameters
        """
        for section in self.attr_dict:
            logger.info('[{:s}]'.format(section))
            for option in self.attr_dict[section]:
                log_txt = '\t{0:s}: {1:s}, [{2:s}]'
                log_msg = log_txt.format(option, str(self.__dict__[section][option]),
                                         self.fmt_dict[os.path.join(section, option)])
                logger.info(log_msg)

def parameters(**kwargs):
    """
    Return default parameters, if not superseded by parameters parsed in argument

    [exp_geom - experimental geometry parameters]
    defocus - lens defocus distance [um]
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
    x0 - lens' abberation center point [0.0 - 1.0]

    [barcode - barcode parameters]
    bar_size - average bar size [um]
    bar_sigma - bar haziness width [um]
    bar_atn - bar attenuation
    bulk_atn - bulk attenuation
    random_dev - bar random deviation
    offset - sample's offset at the beginning and the end of the scan [um]
    """
    st_params = STParams.import_ini(os.path.join(ROOT_PATH, 'config/parameters.ini')).export_dict()
    st_params.update(**kwargs)
    return STParams.import_dict(**st_params)
