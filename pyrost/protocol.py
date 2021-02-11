"""
Examples
--------
Generate the default CXI protocol.

>>> import pyrost as rst
>>> rst.cxi_protocol()
{'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors':
'float', 'data': 'float', 'defocus': 'float', '...': '...'}, 'default_paths':
{'basis_vectors': '/entry_1/instrument_1/detector_1/basis_vectors',
'data': '/entry_1/data_1/data', 'defocus': '/speckle_tracking/defocus', '...': '...'}}

Or generate the default CXI loader.

>>> rst.loader()
{'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors':
'float', 'data': 'float', 'defocus': 'float', '...': '...'}, 'default_paths':
{'basis_vectors': '/entry_1/instrument_1/detector_1/basis_vectors', 'data':
'/entry_1/data_1/data', 'defocus': '/speckle_tracking/defocus', '...': '...'},
'load_paths': {'good_frames': ['/speckle_tracking/good_frames',
'/frame_selector/good_frames', '/process_3/good_frames', '...'], 'mask':
['/speckle_tracking/mask', '/mask_maker/mask', '/entry_1/instrument_1/detector_1/mask',
'...'], 'translations': ['/entry_1/sample_1/geometry/translations',
'/entry_1/sample_1/geometry/translation', '/pos_refine/translation', '...'],
'...': '...'}, 'policy': {'basis_vectors': 'True', 'data': 'True', 'defocus':
'True', '...': '...'}}
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
    FMT_LEN = 3

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
    def _import_ini(cls, protocol_file):
        ini_parser = cls.read_ini(protocol_file)
        kwargs = {}
        for section in cls.attr_dict:
            kwargs[section] = {}
            if 'ALL' in cls.attr_dict[section]:
                if section in ini_parser:
                    for option in ini_parser[section]:
                        val = cls.get_value(ini_parser, section, option)
                        kwargs[section][option] = val
            else:
                for option in cls.attr_dict[section]:
                    val = cls.get_value(ini_parser, section, option)
                    kwargs[section][option] = val
        return kwargs

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
        return cls(**cls._import_ini(protocol_file))

    @classmethod
    def _format(cls, obj):
        crop_obj = {}
        for key, val in list(obj.items())[:cls.FMT_LEN]:
            if isinstance(val, dict):
                val = cls._format(val)
            elif isinstance(val, list):
                val = val[:cls.FMT_LEN] + ['...']
            crop_obj[key] = val
        if len(obj) > cls.FMT_LEN:
            crop_obj['...'] = '...'
        return crop_obj

    def __repr__(self):
        crop_dict = {key: self._format(val) for key, val in self.export_dict().items()}
        return crop_dict.__repr__()

    def __str__(self):
        crop_dict = {key: self._format(val) for key, val in self.export_dict().items()}
        return crop_dict.__str__()

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
    the paths written to all the data attributes necessary for
    the Speckle Tracking algorithm, corresponding attributes'
    data types, and floating point precision.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict, optional
        Dictionary with attributes' CXI default file paths.
    float_precision : {'float32', 'float64'}, optional
        Floating point precision.

    Attributes
    ----------
    config : dict
        Protocol configuration.
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict
        Dictionary with attributes' CXI default file paths.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    """
    attr_dict = {'config': ('float_precision', ), 'datatypes': ('ALL', ), 'default_paths': ('ALL', )}
    fmt_dict = {'config': 'str','datatypes': 'str', 'default_paths': 'str'}

    def __init__(self, datatypes=None, default_paths=None, float_precision='float64'):
        if datatypes is None:
            datatypes = {}
        if default_paths is None:
            default_paths = {}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in default_paths}
        default_paths = {attr: val for attr, val in default_paths.items() if attr in datatypes}
        super(Protocol, self).__init__(config={'float_precision': float_precision},
                                       datatypes=datatypes, default_paths=default_paths)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        elif self.config['float_precision'] == 'float64':
            self.known_types['float'] = np.float64
        else:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    @classmethod
    def import_ini(cls, protocol_file):
        """Initialize an :class:`Protocol` object class with an
        ini file.

        Parameters
        ----------
        protocol_file : str
            Path to the file.

        Returns
        -------
        Protocol
            An :class:`Protocol` object with all the attributes imported
            from the INI file.
        """
        kwargs = cls._import_ini(protocol_file)
        return cls(datatypes=kwargs['datatypes'], default_paths=kwargs['default_paths'],
                   float_precision=kwargs['config']['float_precision'])

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

    def get_default_path(self, attr, value=None):
        """Return the atrribute's default path in the CXI file.
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
        return self.known_types.get(self.datatypes.get(attr), value)

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
            cxi_path = self.get_default_path(attr, cxi_path)
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
                cxi_path = self.get_default_path(attr, cxi_path)
            if cxi_path in cxi_file:
                if overwrite:
                    del cxi_file[cxi_path]
                else:
                    raise ValueError('{:s} is already present in {:s}'.format(attr,
                                                                              cxi_file.filename))
            data = np.asarray(data, dtype=self.get_dtype(attr, dtype))
            cxi_file.create_dataset(cxi_path, data=data)

def cxi_protocol(datatypes=None, default_paths=None, float_precision=None):
    """Return the default CXI porotocol.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict, optional
        Dictionary with attributes' CXI default file paths.
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
    if datatypes is None:
        datatypes = {}
    if default_paths is None:
        default_paths = {}
    kwargs = Protocol.import_ini(PROTOCOL_FILE).export_dict()
    kwargs['datatypes'].update(**datatypes)
    kwargs['default_paths'].update(**default_paths)
    if float_precision is None:
        float_precision = kwargs['config']['float_precision']
    return Protocol(datatypes=kwargs['datatypes'], default_paths=kwargs['default_paths'],
                    float_precision=float_precision)

class STLoader(Protocol):
    """Speckle Tracking data loader class. Loads data from a
    CXI file and returns a :class:`STData` container or a
    :class:`dict` with the data. Search data in the paths
    provided by `protocol` and `load_paths`.

    Parameters
    ----------
    protocol : Protocol
        Protocol object.
    load_paths : dict, optional
        Extra paths to the data attributes in a CXI file,
        which override `protocol`. Accepts only the attributes
        enlisted in `protocol`.
    policy : dict, optional
        A dictionary with loading policy. Contains all the
        attributes that are available in `protocol` and the
        corresponding flags. If a flag is True, the attribute
        will be loaded from a file. By default only the attributes
        necessary for a :class:`STData` container will be loaded.

    Attributes
    ----------
    config : dict
        Protocol configuration.
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict
        Dictionary with attributes' CXI default file paths.
    load_paths : dict
        Extra set of paths to the attributes enlisted in `datatypes`.
    policy: dict
        Loading policy.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    STData : Data container with all the data  necessary for
        Speckle Tracking.
    """
    attr_dict = {'config': ('float_precision', ), 'datatypes': ('ALL', ), 'default_paths': ('ALL', ),
                 'load_paths': ('ALL',), 'policy': ('ALL', )}
    fmt_dict = {'config': 'str','datatypes': 'str', 'default_paths': 'str', 'load_paths': 'str', 'policy': 'str'}

    def __init__(self, protocol, load_paths=None, policy=None):
        if load_paths is None:
            load_paths = {}
        else:
            load_paths = {attr: paths for attr, paths in load_paths.items()
                          if attr in protocol.default_paths}
        if policy is None:
            policy = {}
        else:
            policy = {attr: paths for attr, paths in policy.items()
                          if attr in protocol.default_paths}
        super(Protocol, self).__init__(config=protocol.config, datatypes=protocol.datatypes,
                                       default_paths=protocol.default_paths, load_paths=load_paths, policy=policy)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        elif self.config['float_precision'] == 'float64':
            self.known_types['float'] = np.float64
        else:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    @classmethod
    def import_ini(cls, protocol_file):
        """Initialize an :class:`STLoader` object class with an
        ini file.

        Parameters
        ----------
        protocol_file : str
            Path to the file.

        Returns
        -------
        STLoader
            An :class:`STLoader` object with all the attributes imported
            from the INI file.
        """
        kwargs = cls._import_ini(protocol_file)
        protocol = Protocol.import_ini(protocol_file)
        return cls(protocol=protocol, load_paths=kwargs['load_paths'], policy=kwargs['policy'])

    def get_load_paths(self, attr, value=None):
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
        list
            Set of attribute's paths.
        """
        paths = [super(STLoader, self).get_default_path(attr, value)]
        if attr in self.load_paths:
            paths.extend(self.load_paths[attr])
        return paths

    def get_policy(self, attr, value=None):
        policy = self.policy.get(attr, value)
        if isinstance(policy, str):
            return policy == 'True'
        else:
            return bool(policy)

    def get_protocol(self):
        """Return a CXI protocol from the loader.

        Returns
        -------
        Protocol
            CXI protocol.
        """
        return Protocol(datatypes=self.datatypes, default_paths=self.default_paths,
                        float_precision=self.config['float_precision'])

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
        paths = self.get_load_paths(attr)
        for path in paths:
            if path is None or path in cxi_file:
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
        **kwargs : dict, optional
            Dictionary of attribute values,
            which will be parsed to the `STData` object instead.

        Returns
        -------
        dict
            Dictionary with all the data fetched from the CXI file.
        """
        data_dict = {}
        with h5py.File(path, 'r') as cxi_file:
            for attr in self:
                if self.get_policy(attr, False):
                    cxi_path = self.find_path(attr, cxi_file)
                    if attr in kwargs and not kwargs[attr] is None:
                        data_dict[attr] = np.asarray(kwargs[attr], dtype=self.get_dtype(attr))
                    else:
                        data_dict[attr] = self.read_cxi(attr, cxi_file, cxi_path=cxi_path)
                else:
                    data_dict[attr] = None
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
        return STData(self.get_protocol(), **self.load_dict(path, **kwargs))

def loader(protocol=None, load_paths=None, policy=None):
    """Return the default CXI loader.

    Parameters
    ----------
    float_precision : {'float32', 'float64'}, optional
        Floating point precision.
    load_paths : dict, optional
        Extra paths to the data attributes in a CXI file,
        which override the :func:`cxi_protocol`. Accepts
        only the attributes enlisted in :func:`cxi_protocol`.
    policy : dict, optional
        A dictionary with loading policy. Contains all the
        attributes that are available in :func:`cxi_protocol`
        and the corresponding flags. If a flag is True, the
        attribute will be loaded from a file. By default only
        the attributes necessary for a :class:`STData` container
        will be loaded.

    Returns
    -------
    STLoader
        Default CXI loader.

    See Also
    --------
    cxi_protocol : Default CXI protocol.
    STLoader : Full loader class description.
    STData : Data container with all the data  necessary for
        Speckle Tracking.
    """
    kwargs = STLoader.import_ini(PROTOCOL_FILE).export_dict()
    if protocol is None:
        protocol = Protocol(datatypes=kwargs['datatypes'],
                            default_paths=kwargs['default_paths'],
                            float_precision=kwargs['config']['float_precision'])
    if load_paths is None:
        load_paths = {}
    if policy is None:
        policy = {}
    kwargs['load_paths'].update(**load_paths)
    kwargs['policy'].update(**policy)
    return STLoader(protocol, load_paths=kwargs['load_paths'],
                    policy=kwargs['policy'])
