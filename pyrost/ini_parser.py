""":class:`INIParser` ini parser implementation.
"""
import os
import configparser
import re

ROOT_PATH = os.path.dirname(__file__)

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
        self.__dict__['_lookup'] = {}
        self.__dict__['ini_dict'] = {section: {} for section in self.attr_dict}
        for section in self.attr_dict:
            if 'ALL' in self.attr_dict[section] and section in kwargs:
                value = {option: self._get_value(section, option, kwargs)
                         for option in kwargs[section]}
            else:
                value = {option: self._get_value(section, option, kwargs)
                         for option in self.attr_dict[section]}
            self.__setattr__(section, value)
        self._lookup = self._lookup_dict()

    def _get_value(self, section, option, kwargs):
        if not option in kwargs[section]:
            raise AttributeError("The '{:s}' option has not been provided".format(option))
        fmt = self.get_format(section, option)
        if isinstance(kwargs[section][option], list):
            return [fmt(part) for part in kwargs[section][option]]
        return fmt(kwargs[section][option])

    @classmethod
    def _lookup_dict(cls):
        """Look-up table between the sections and the parameters.

        Returns
        -------
        dict
            Look-up dictionary.
        """
        return {}

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
        return cls.known_types.get(fmt, str)

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
            return [fmt(part.strip('\'\"'))
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
    def _format(cls, obj):
        crop_obj = {}
        for key, val in list(obj.items())[:cls.FMT_LEN]:
            if isinstance(val, dict):
                val = cls._format(val)
            elif isinstance(val, list) and len(val) > cls.FMT_LEN:
                val = val[:cls.FMT_LEN] + ['...']
            crop_obj[key] = val
        if len(obj) > cls.FMT_LEN:
            crop_obj['...'] = '...'
        return crop_obj

    def __getattr__(self, attr):
        if attr in self._lookup:
            return self.ini_dict[self._lookup[attr]][attr]
        elif attr in self.ini_dict:
            return self.ini_dict[attr]
        else:
            raise AttributeError(attr + " doesn't exist")

    def __setattr__(self, attr, value):
        if attr in self._lookup:
            self.ini_dict[self._lookup[attr]][attr] = value
        elif attr in self.ini_dict:
            self.ini_dict[attr] = value
        else:
            super(INIParser, self).__setattr__(attr, value)

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
        return {section: self.__getattr__(section) for section in self.attr_dict}

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
        return type(self).export_ini(**self.ini_dict)
