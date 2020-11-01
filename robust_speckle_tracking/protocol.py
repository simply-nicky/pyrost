"""
protocol.py - Speckle Tracking data file's protocol implementation
"""
import os
import configparser
import re
import h5py
import numpy as np
from .data_processing import STData

ROOT_PATH = os.path.dirname(__file__)

class hybridmethod:
    """
    Hybrid method descriptor supporting two distinct methods bound to class and instance

    fclass - class bound method
    finstance - instance bound method
    doc - documentation
    """
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass, self.finstance = fclass, finstance
        self.__doc__ = doc or fclass.__doc__
        self.__isabstractmethod__ = bool(getattr(fclass, '__isabstractmethod__', False))

    def classmethod(self, fclass):
        """
        Class method decorator

        fclass - class bound method
        """
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        """
        Instance method decorator

        finstance - instance bound method
        """
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)

class INIParser():
    """
    INI files parser class
    """
    err_txt = "Wrong format key '{0:s}' of option '{1:s}'"
    known_types = {'int': int, 'float': float, 'bool': bool, 'str': str}
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
        if isinstance(kwargs[section][option], list):
            self.__dict__[section][option] = [fmt(part) for part in kwargs[section][option]]
        else:
            self.__dict__[section][option] = fmt(kwargs[section][option])

    @classmethod
    def read_ini(cls, ini_file):
        """
        Read the ini_file
        """
        if not os.path.isfile(ini_file):
            raise ValueError("File {:s} doesn't exist".format(ini_file))
        ini_parser = configparser.ConfigParser()
        ini_parser.read(ini_file)
        return ini_parser

    @classmethod
    def get_format(cls, section, option):
        """
        Return the attribute's format
        """
        fmt = cls.fmt_dict.get(os.path.join(section, option))
        if not fmt:
            fmt = cls.fmt_dict.get(section)
        return cls.known_types[fmt]

    @classmethod
    def get_value(cls, ini_parser, section, option):
        """
        Return an attribute from the ini_parser
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
    def import_ini(cls, ini_file):
        """
        Initialize an object class with the ini file

        ini_file - ini file path
        """
        ini_parser = cls.read_ini(ini_file)
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
        """
        Return dict object
        """
        return {section: self.__dict__[section] for section in self.attr_dict}

    @hybridmethod
    def export_ini(cls, **kwargs):
        """
        Return ini parser

        kwargs - extra parameters to save
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
        """
        Return ini parser

        kwargs - extra parameters to save
        """
        return type(self).export_ini(**self.__dict__)

class Protocol(INIParser):
    """
    CXI protocol class
    Contains a cxi file hierarchy and corresponding data types
    """
    attr_dict = {'default_paths': ('ALL',), 'dtypes': ('ALL',)}
    fmt_dict = {'default_paths': 'str', 'dtypes': 'str'}
    log_txt = "{attr:s} [{fmt:s}]: '{path:s}' "

    def parser_from_template(self, path):
        """
        Return parser object using an ini file template

        path - path to the template file
        """
        ini_template = configparser.ConfigParser()
        ini_template.read(path)
        parser = configparser.ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(**self.default_paths)
                               for option in ini_template[section]}
        return parser

    def __iter__(self):
        return self.dtypes.__iter__()

    def __contains__(self, attr):
        return attr in self.dtypes

    def log(self, logger):
        """
        Log the protocol

        logger - a logger object
        """
        for attr in self.default_paths:
            logger.info(self.log_txt.format(attr=attr, fmt=self.dtypes[attr],
                                            path=self.default_paths[attr]))

    def get_path(self, attr, value=None):
        """
        Return the atrribute's path in the cxi file, return value if not found
        """
        return self.default_paths.get(attr, value)

    def get_dtype(self, attr, value=None):
        """
        Return the attribute's data-type, return value if not found
        """
        return self.dtypes.get(attr, value)

    def read_cxi(self, attr, cxi_file, cxi_path=None, dtype=None):
        """
        Read the attribute from the cxi file at the path defined by the protocol
        If cxi_path argument is provided, it will override the protocol

        attr - the attribute to read
        cxi_file - an h5py File object
        """
        if cxi_path is None:
            cxi_path = self.get_path(attr, cxi_path)
        if cxi_path in cxi_file:
            return cxi_file[cxi_path][...].astype(self.get_dtype(attr, dtype))
        else:
            return None

    def write_cxi(self, attr, data, cxi_file, overwrite=True, cxi_path=None, dtype=None):
        """
        Write data to the cxi file as specified by the protocol
        If cxi_path or dtype argument are provided, it will override the protocol

        attr - the attribute to be written
        data - the attribute's data
        cxi_file - an h5py File object
        overwrite - overwrite the cxi file
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

def cxi_protocol():
    """
    Return default cxi protocol
    """
    return Protocol.import_ini(os.path.join(ROOT_PATH, 'config/cxi_protocol.ini'))

class STLoader(INIParser):
    """
    Speckle Tracking scan loader class
    Looks for all the requisite data attributes in a cxi file and returns an STData object.
    Search data in the paths provided by the protocol and the paths parsed to the constructor.

    protocol - Protocol object
    paths - dictionary of attributes' paths, where the loader looks for the data
    """
    attr_dict = {'paths': ('ALL',)}
    fmt_dict = {'paths': 'str'}

    def __init__(self, protocol=cxi_protocol(), **kwargs):
        super(STLoader, self).__init__(**kwargs)
        self.protocol = protocol

    def find_path(self, attr, cxi_file):
        """
        Find attribute path in a cxi file

        attr - the attribute to be found
        cxi_file - h5py File object
        """
        if attr in self.paths:
            for path in self.paths[attr]:
                if path in cxi_file:
                    return path
        else:
            return None

    def _load(self, path, **kwargs):
        data_dict = {}
        with h5py.File(path, 'r') as cxi_file:
            for attr in self.protocol:
                cxi_path = self.find_path(attr, cxi_file)
                if attr in kwargs and kwargs[attr]:
                    data_dict[attr] = np.asarray(kwargs[attr], dtype=self.protocol.get_dtype(attr))
                else:
                    data_dict[attr] = self.protocol.read_cxi(attr, cxi_file, cxi_path=cxi_path)
        return data_dict

    def load(self, path, **kwargs):
        """
        Load a cxi file and return an STData class object

        path - path to the cxi file
        kwargs - a dictionary of attributes to override
        """
        data_dict = self._load(path, **kwargs)
        return STData(self.protocol, **data_dict)

def loader():
    """
    Return the default cxi loader
    """
    return STLoader.import_ini(os.path.join(ROOT_PATH, 'config/cxi_protocol.ini'))
