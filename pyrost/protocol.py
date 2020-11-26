"""Speckle Tracking data protocol for `CXI`_ format files.
The :class:`Protocol` class provides paths to the necessary
data attribute in a `CXI`_ file and corresponding data types.
The :class:`STLoader` automatically loads all the necessary
data from a `CXI`_ file and returns an :class:`STData` data
container object.

.. _CXI: https://www.cxidb.org/cxi.html

Examples
--------
Generate the default CXI protocol.

>>> import pyrost as rst
>>> rst.cxi_protocol()
<pyrost.protocol.Protocol at 0x7fd7c0c965d0>

Or generate the default CXI loader.

>>> rst.loader()
<pyrost.protocol.STLoader at 0x7fd7e0d0f590>

Notes
-----
Data attributes necessary for the Speckle Tracking
algorithm:

* basis_vectors : Detector basis vectors [m].
* data : Measured intensity frames.
* defocus : Defocus distance (supersedes defocus_ss
  and defocus_fs values) [m].
* defocus_ss : Defocus distance for the slow detector
  axis [m].
* defocus_fs : Defocus distance for the fast detector
  axis [m].
* distance : Sample-to-detector distance [m].
* energy : Incoming beam photon energy [eV].
* good_frames : An array of good frames' indices.
* m0 : The lower bounds of the fast detector axis of
  the reference image at the reference frame in pixels.
* mask : Bad pixels mask.
* n0 : The lower bounds of the slow detector axis of
  the reference image at the reference frame in pixels.
* phase : Phase profile of lens' abberations.
* pixel_map : The pixel mapping between the data at
  the detector's plane and the reference image at
  the reference plane.
* pixel_abberations : Lens' abberations along
  the fast and slow axes in pixels.
* pixel_translations : Sample's translations in
  the detector plane in pixels.
* reference_image : The unabberated reference image
  of the sample.
* roi : Region of interest in the detector's plane.
* translations : Sample's translations [m].
* wavelength : Incoming beam's wavelength [m].
* whitefield : Measured frames' whitefield.
* x_pixel_size : Pixel's size along the fast detector
  axis [m].
* y_pixel_size : Pixel's size along the slow detector
  axis [m].

Configuration parameters:

* float_precision ('float32', 'float64') : Floating point
  precision.
"""
import os
import configparser
import re
import h5py
import numpy as np
from .data_processing import STData

ROOT_PATH = os.path.dirname(__file__)
PROTOCOL_FILE = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class hybridmethod:
    """Hybrid method descriptor supporting
    two distinct methodsbound to class and instance.

    Parameters
    ----------
    fclass : method
        Class bound method.
    finstance : method, optional
        Instance bound method.
    doc : str, optional
        Method's docstring.

    Attributes
    ----------
    fclass : method
        Class bound method.
    finstance : method
        Instance bound method.
    doc : str
        Method's dosctring.
    """
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass, self.finstance = fclass, finstance
        self.__doc__ = doc or fclass.__doc__
        self.__isabstractmethod__ = bool(getattr(fclass, '__isabstractmethod__', False))

    def classmethod(self, fclass):
        """Class method decorator

        Parameters
        ----------
        fclass : method
            Class bound method.

        Returns
        -------
        hybridmethod
            A new instance with the class bound method added to the object
        """
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        """Instance method decorator

        Parameters
        ----------
        finstance : method
            Instance bound method.

        Returns
        -------
        hybridmethod
            A new instance with the instance bound method added to the object
        """
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)

class INIParser:
    """INI files parser class with the methods for importing and exporting ini files
    and Python dictionaries.

    Parameters
    ----------
    **kwargs : dict
        Attributes specified in `attr_dict`.

    Attributes
    ----------
    err_txt : str
        Error text.
    known_types : dict
        Look-up dictionary of supported types for formatting.
    attr_dict : dict
        Dictionary of provided attributes.
    fmt_dict : dict
        Dictionary of attributes' types used for formatting.

    Raises
    ------
    AttributeError
        If the attribute specified in `attr_dict`
        has not been provided in keyword arguments ``**kwargs``.
    """
    err_txt = "Wrong format key '{0:s}' of option '{1:s}'"
    known_types = {'int': int, 'bool': bool, 'float': float, 'str': str}
    attr_dict, fmt_dict = {}, {}
    LIST_SPLITTER = r'\s*,\s*'
    LIST_MATCHER = r'^\[([\s\S]*)\]$'

    def __init__(self, **kwargs):
        for section in self.attr_dict:
            self.__dict__[section] = {}
            if 'ALL' in self.attr_dict[section]:
                if section in kwargs:
                    for option in kwargs[section]:
                        self._init_value(section, option, kwargs)
            else:
                for option in self.attr_dict[section]:
                    self._init_value(section, option, kwargs)

    def _init_value(self, section, option, kwargs):
        fmt = self.get_format(section, option)
        if option in kwargs[section]:
            if isinstance(kwargs[section][option], list):
                self.__dict__[section][option] = [fmt(part) for part in kwargs[section][option]]
            else:
                self.__dict__[section][option] = fmt(kwargs[section][option])
        else:
            raise AttributeError("The '{:s}' option has not been provided".format(option))

    @classmethod
    def read_ini(cls, protocol_file):
        """Read the `protocol_file` and return an instance of
        :class:`configparser.ConfigParser` class.

        Parameters
        ----------
        protocol_file : str
            Path to the file.

        Returns
        -------
        configparser.ConfigParser
            Parser object with all the data contained in the
            INI file.

        Raises
        ------
        ValueError
            If the file doesn't exist.
        """
        if not os.path.isfile(protocol_file):
            raise ValueError("File {:s} doesn't exist".format(protocol_file))
        ini_parser = configparser.ConfigParser()
        ini_parser.read(protocol_file)
        return ini_parser

    @classmethod
    def get_format(cls, section, option):
        """Return the attribute's format specified by `fmt_dict`.

        Parameters
        ----------
        section : str
            Attribute's section.

        option : str
            Attribute's name.

        Returns
        -------
        type
            Type of the attribute.
        """
        fmt = cls.fmt_dict.get(os.path.join(section, option))
        if not fmt:
            fmt = cls.fmt_dict.get(section)
        return cls.known_types[fmt]

    @classmethod
    def get_value(cls, ini_parser, section, option):
        """Return an attribute from an INI file's parser object `ini_parser`.

        Parameters
        ----------
        ini_parser : configparser.ConfigParser
            A parser object of an INI file.

        section : str
            Attribute's section.

        option : str
            Attribute's option.

        Returns
        -------
            Attribute's value imported from the INI file.
        """
        fmt = cls.get_format(section, option)
        string = ini_parser.get(section, option)
        is_list = re.search(cls.LIST_MATCHER, string)
        if is_list:
            return [fmt(part.strip())
                    for part in re.split(cls.LIST_SPLITTER, is_list.group(1))]
        else:
            return fmt(string.strip())

    @classmethod
    def import_ini(cls, protocol_file):
        """Initialize an :class:`INIParser` object class with an
        ini file.

        Parameters
        ----------
        protocol_file : str
            Path to the file.

        Returns
        -------
        INIParser
            An :class:`INIParser` object with all the attributes imported
            from the INI file.
        """
        ini_parser = cls.read_ini(protocol_file)
        param_dict = {}
        for section in cls.attr_dict:
            param_dict[section] = {}
            if 'ALL' in cls.attr_dict[section]:
                if section in ini_parser:
                    for option in ini_parser[section]:
                        val = cls.get_value(ini_parser, section, option)
                        param_dict[section][option] = val
            else:
                for option in cls.attr_dict[section]:
                    val = cls.get_value(ini_parser, section, option)
                    param_dict[section][option] = val
        return cls(**param_dict)

    def __str__(self):
        return self.export_dict().__str__()

    def export_dict(self):
        """Return a :class:`dict` object with all the attributes.

        Returns
        -------
        dict
            Dictionary with all the attributes conntained in the object.
        """
        return {section: self.__dict__[section] for section in self.attr_dict}

    @hybridmethod
    def export_ini(cls, **kwargs):
        """Return a :class:`configparser.ConfigParser` object
        with all the attributes exported the class.

        Parameters
        ----------
        **kwargs : dict
            Extra parameters to export to the
            :class:`configparser.ConfigParser` object.

        Returns
        ------
        configparser.ConfigParser
            A parser object with all the attributes contained in
            `attr_dict`.
        """
        ini_parser = configparser.ConfigParser()
        for section in cls.attr_dict:
            if 'ALL' in cls.attr_dict[section]:
                ini_parser[section] = kwargs[section]
            else:
                ini_parser[section] = {option: kwargs[section][option]
                                       for option in cls.attr_dict[section]}
        return ini_parser

    @export_ini.instancemethod
    def export_ini(self):
        """Return a :class:`configparser.ConfigParser` object
        with all the attributes exported from the object.

        Returns
        -------
        configparser.ConfigParser
            A parser object with all the attributes contained in
            the object.
        """
        return type(self).export_ini(**self.__dict__)

class Protocol(INIParser):
    """CXI protocol class. Contains a CXI file tree path with
    the paths written to all the data attributes necessary
    for the Speckle Tracking algorithm (enlisted in `datatypes_lookup`),
    corresponding attributes' data types, and global configuration
    parameters.

    Parameters
    ----------
    **kwargs : dict
        Values for the attributes specified in
        `datatypes_lookup` and configuration parameters.

    Attributes
    ----------
    datatypes_lookup : dict
        Dictionary which enlists all the data attributes
        necessary for the Speckle Tracking algorithm and
        their corresponding data types.

    See Also
    --------
    protocol : Full list of data attributes and configuration 
        parameters.
    """
    datatypes_lookup = {'basis_vectors': 'float', 'data': 'float', 'defocus': 'float',
                        'defocus_fs': 'float', 'defocus_ss': 'float', 'distance': 'float',
                        'energy': 'float', 'good_frames': 'int', 'm0': 'int', 'mask': 'bool',
                        'n0': 'int', 'phase': 'float', 'pixel_map': 'float',
                        'pixel_abberations': 'float', 'pixel_translations': 'float',
                        'reference_image': 'float', 'roi': 'int', 'translations': 'float',
                        'wavelength': 'float', 'whitefield': 'float',
                        'x_pixel_size': 'float', 'y_pixel_size': 'float'}
    attr_dict = {'default_paths': list(datatypes_lookup), 'config': ('float_precision', )}
    fmt_dict = {'default_paths': 'str', 'config': 'str'}
    log_txt = "{attr:s} [{fmt:s}]: '{path:s}' "

    def __init__(self, **kwargs):
        super(Protocol, self).__init__(**kwargs)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        elif self.config['float_precision'] == 'float64':
            self.known_types['float'] = np.float64
        else:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    def parser_from_template(self, path):
        """Return a :class:`configparser.ConfigParser` object using
        an ini file template.

        Parameters
        ----------
        path : str
            Path to the ini file template.

        Returns
        -------
        configparser.ConfigParser
            Parser object with the attributes populated according
            to the protocol.
        """
        ini_template = configparser.ConfigParser()
        ini_template.read(path)
        parser = configparser.ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(**self.default_paths)
                               for option in ini_template[section]}
        return parser

    def __iter__(self):
        return self.default_paths.__iter__()

    def __contains__(self, attr):
        return attr in self.default_paths

    def log(self, logger):
        """Log the protocol with `logger`. Log all the attributes,
        their datatypes, and their paths.

        Parameters
        ----------
        logger : logging.Logger
            Logging interface.
        """
        for attr in self.default_paths:
            logger.info(self.log_txt.format(attr=attr, fmt=self.datatypes_lookup[attr],
                                            path=self.default_paths[attr]))

    def get_path(self, attr, value=None):
        """Return the atrribute's path in the cxi file.
        Return `value` if `attr` is not found.

        Parameters
        ----------
        attr : str
            The attribute to look for.

        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        str or None
            Attribute's cxi file path.
        """
        return self.default_paths.get(attr, value)

    def get_dtype(self, attr, value=None):
        """Return the attribute's data type.
        Return `value` if `attr` is not found.

        Parameters
        ----------
        attr : str
            The data attribute.

        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        type or None
            Attribute's data type.
        """
        return self.known_types.get(self.datatypes_lookup.get(attr), value)

    def read_cxi(self, attr, cxi_file, cxi_path=None, dtype=None):
        """Read `attr` from the CXI file `cxi_file` at the path
        defined by the protocol. If `cxi_path` or `dtype` argument
        are provided, it will override the protocol.

        Parameters
        ----------
        attr : str
            Data attribute.
        cxi_file : h5py.File
            h5py File object of the CXI file.
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.
        dtype : type, optional
            Data type of the attribute. If `dtype` is None,
            the type will be inferred according to the protocol.

        Returns
        -------
        numpy.ndarray or None
            The value of the attribute extracted from the CXI file.
        """
        if cxi_path is None:
            cxi_path = self.get_path(attr, cxi_path)
        if cxi_path in cxi_file:
            return cxi_file[cxi_path][...].astype(self.get_dtype(attr, dtype))
        else:
            return None

    def write_cxi(self, attr, data, cxi_file, overwrite=True, cxi_path=None, dtype=None):
        """Write data to the CXI file `cxi_file` under the path
        specified by the protocol. If `cxi_path` or `dtype` argument
        are provided, it will override the protocol.

        Parameters
        ----------
        attr : str
            Data attribute.
        data : numpy.ndarray
            Data which is bound to be saved.
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.
        overwrite : bool, optional
            Overwrite the content of `cxi_file` if it's True.
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.
        dtype : type, optional
            Data type of the attribute. If `dtype` is None,
            the type will be inferred according to the protocol.

        Raises
        ------
        ValueError
            If `overwrite` is False and the data is already present
            at the given location in `cxi_file`.
        """
        if data is None:
            pass
        else:
            if cxi_path is None:
                cxi_path = self.get_path(attr, cxi_path)
            if cxi_path in cxi_file:
                if overwrite:
                    del cxi_file[cxi_path]
                else:
                    raise ValueError('{:s} is already present in {:s}'.format(attr,
                                                                              cxi_file.filename))
            data = np.asarray(data, dtype=self.get_dtype(attr, dtype))
            cxi_file.create_dataset(cxi_path, data=data)

def cxi_protocol(float_precision='float64'):
    """Return the default CXI :class:`Protocol` object, with
    the floating point precision specified by `float_precision`.

    Parameters
    ----------
    float_precision : {'float32', 'float64'}, optional
        Floating point precision.

    Returns
    -------
    Protocol
        Default CXI protocol.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    """
    protocol = Protocol.import_ini(PROTOCOL_FILE).export_dict()
    protocol.update(config={'float_precision': float_precision})
    return Protocol(**protocol)

class STLoader(INIParser):
    """Speckle Tracking data loader class.
    Looks for all the necessary data attributes
    in a cxi file and returns an :class:`STData` object.
    Search data in the paths provided by the `protocol`
    and the paths parsed to the constructor with `**kwargs`.

    Parameters
    ----------
    protocol : Protocol
        Protocol object.
    **kwargs : dict
        Extra paths to the data attributes in a CXI file,
        which override `protocol`.

    Attributes
    ----------
    protocol : Protocol
        Protocol object.
    **kwargs : dict
        Extra paths to the data attributes in a CXI file,
        which override `protocol`.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    STData : Data container with all the data  necessary for
        Speckle Tracking.
    """
    attr_dict = {'paths': ('ALL',)}
    fmt_dict = {'paths': 'str'}

    def __init__(self, protocol=cxi_protocol(), **kwargs):
        super(STLoader, self).__init__(**kwargs)
        self.protocol = protocol

    def find_path(self, attr, cxi_file):
        """Find attribute's path in a CXI file `cxi_file`.

        Parameters
        ----------
        attr : str
            Data attribute.
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.

        Returns
        -------
        str or None
            Atrribute's path in the CXI file,
            return None if the attribute is not found.
        """
        if attr in self.paths:
            for path in self.paths[attr]:
                if path in cxi_file:
                    return path
        else:
            return None

    def load_dict(self, path, **kwargs):
        """Load a CXI file and return a :class:`dict` with
        all the data fetched from the file.

        Parameters
        ----------
        path : str
            Path to the cxi file.
        **kwargs : dict
            Dictionary of attribute values,
            which will be parsed to the `STData` object instead.

        Returns
        -------
        dict
            Dictionary with all the data fetched from the CXI file.
        """
        data_dict = {}
        with h5py.File(path, 'r') as cxi_file:
            for attr in self.protocol:
                cxi_path = self.find_path(attr, cxi_file)
                if attr in kwargs and not kwargs[attr] is None:
                    data_dict[attr] = np.asarray(kwargs[attr], dtype=self.protocol.get_dtype(attr))
                else:
                    data_dict[attr] = self.protocol.read_cxi(attr, cxi_file, cxi_path=cxi_path)
        if not data_dict['defocus'] is None:
            data_dict['defocus_ss'] = data_dict['defocus']
            data_dict['defocus_fs'] = data_dict['defocus']
        return data_dict

    def load(self, path, **kwargs):
        """Load a CXI file and return an :class:`STData` class object.

        Parameters
        ----------
        path : str
            Path to the cxi file.
        **kwargs : dict
            Dictionary of attribute values,
            which will be parsed to the `STData` object instead.

        Returns
        -------
        STData
            Data container object with all the necessary data
            for the Speckle Tracking algorithm.
        """
        return STData(self.protocol, **self.load_dict(path, **kwargs))

def loader(float_precision='float64'):
    """Return the default CXI loader.

    Parameters
    ----------
    float_precision : {'float32', 'float64'}, optional
        Floating point precision.

    Returns
    -------
    STLoader
        Default CXI loader.

    See Also
    --------
    STLoader : Full loader class description.
    """
    protocol = cxi_protocol(float_precision)
    kwargs = STLoader.import_ini(PROTOCOL_FILE).export_dict()
    return STLoader(protocol, **kwargs)
