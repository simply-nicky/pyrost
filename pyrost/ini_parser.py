""":class:`INIParser` ini parser implementation.
"""
from __future__ import annotations
import os
from configparser import ConfigParser
import re
from typing import Any, Dict, ItemsView, KeysView, List, TypeVar, Type, Union, ValuesView
import numpy as np

T = TypeVar('T')            # Object type
Desc = TypeVar('Desc')      # Descriptor type
ROOT_PATH = os.path.dirname(__file__)

class hybridmethod:
    """Hybrid method descriptor supporting
    two distinct methodsbound to class and instance.

    Attributes:
        fclass : Class bound method.
        finstance : Instance bound method.
        doc : Method's dosctring.
    """
    def __init__(self, fclass: Desc, finstance: Desc=None, doc: str=None) -> None:
        """Args:
            fclass : Class bound method.
            finstance : Instance bound method.
            doc : Method's docstring.
        """
        self.fclass, self.finstance = fclass, finstance
        self.__doc__ = doc or fclass.__doc__
        self.__isabstractmethod__ = bool(getattr(fclass, '__isabstractmethod__', False))

    def classmethod(self, fclass: Desc) -> hybridmethod:
        """Class method decorator

        Args:
            fclass : Class bound method.

        Returns:
            A new instance with the class bound method added
            to the object.
        """
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance: Desc) -> hybridmethod:
        """Instance method decorator

        Args:
            finstance : Instance bound method.

        Returns:
            A new instance with the instance bound method added
            to the object.
        """
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance: T, cls: Type[T]) -> Any:
        if instance is None or self.finstance is None:
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)

class INIParser:
    """INI files parser class with the methods for importing and exporting
    ini files and Python dictionaries.

    Attributes:
        err_txt : Error text.
        known_types : Look-up dictionary of supported types for formatting.
        attr_dict : Dictionary of provided attributes.
        fmt_dict : Dictionary of attributes' types used for formatting.
    """
    err_txt = "Wrong format key '{0:s}' of option '{1:s}'"
    known_types = {'int': int, 'bool': bool, 'float': float, 'str': str}
    attr_dict, fmt_dict = {}, {}
    LIST_SPLITTER = r'\s*,\s*'
    LIST_MATCHER = r'^\[([\s\S]*)\]$'
    FMT_LEN = 3

    def __init__(self, **kwargs: Any) -> None:
        """Args:
            kwargs : Attributes specified in `attr_dict`.

        Raises:
            AttributeError : If the attribute specified in `attr_dict`
                has not been provided in keyword arguments ``**kwargs``.
        """
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

    def _get_value(self, section: str, option: str, kwargs: Dict) -> Any:
        if not option in kwargs[section]:
            raise AttributeError("The '{:s}' option has not been provided".format(option))
        fmt = self.get_format(section, option)

        if isinstance(kwargs[section][option], (list, tuple)):
            return [fmt(part) for part in kwargs[section][option]]

        if isinstance(kwargs[section][option], np.ndarray):
            if kwargs[section][option].ndim > 1:
                raise ValueError(f'{kwargs[section][option]:s} must be one-dimensional')
            return [fmt(part) for part in kwargs[section][option]]

        return fmt(kwargs[section][option])

    @staticmethod
    def str_to_list(strings: Union[str, List[str]]) -> List[str]:
        """Convert `strings` to a list of strings.

        Args:
            strings : String or a list of strings

        Returns:
            List of strings.
        """
        if isinstance(strings, (str, list)):
            if isinstance(strings, str):
                return [strings,]
            return strings

        raise ValueError('strings must be a string or a list of strings')

    @classmethod
    def _lookup_dict(cls) -> Dict:
        """Look-up table between the sections and the parameters.

        Returns:
            Look-up dictionary.
        """
        return {}

    @classmethod
    def read_ini(cls, protocol_file: str) -> ConfigParser:
        """Read the `protocol_file` and return an instance of
        :class:`configparser.ConfigParser` class.

        Args:
            protocol_file : Path to the file.

        Returns:
            Parser object with all the data contained in the
            INI file.

        Raises:
            ValueError : If the file doesn't exist.
        """
        if not os.path.isfile(protocol_file):
            raise ValueError(f"File {protocol_file} doesn't exist")
        ini_parser = ConfigParser()
        ini_parser.read(protocol_file)
        return ini_parser

    @classmethod
    def get_format(cls, section: str, option: str) -> Type:
        """Return the attribute's format specified by `fmt_dict`.

        Args:
            section : Attribute's section.
            option : Attribute's name.

        Returns:
            Type of the attribute.
        """
        fmt = cls.fmt_dict.get(os.path.join(section, option))
        if not fmt:
            fmt = cls.fmt_dict.get(section)
        return cls.known_types.get(fmt, str)

    @classmethod
    def get_value(cls, ini_parser: ConfigParser, section: str, option: str) -> Any:
        """Return an attribute from an INI file's parser object `ini_parser`.

        Args:
            ini_parser : A parser object of an INI file.
            section : Attribute's section.
            option : Attribute's option.

        Returns:
            Attribute's value imported from the INI file.
        """
        fmt = cls.get_format(section, option)
        string = ini_parser.get(section, option)
        is_list = re.search(cls.LIST_MATCHER, string)
        if is_list:
            return [fmt(part.strip('\'\"'))
                    for part in re.split(cls.LIST_SPLITTER, is_list.group(1))]

        return fmt(string.strip())

    @classmethod
    def _import_ini(cls, protocol_file: str) -> Dict[str, Dict]:
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
    def _format(cls, obj: Dict) -> Dict:
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

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__.get('_lookup', {}):
            return self.ini_dict[self.__dict__['_lookup'][attr]][attr]
        if attr in self.__dict__.get('ini_dict', {}):
            return self.__dict__['ini_dict'][attr]
        raise AttributeError(attr + " doesn't exist")

    def __getitem__(self, attr: str) -> Any:
        return self.__getattr__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._lookup:
            self.ini_dict[self._lookup[attr]][attr] = value
        elif attr in self.ini_dict:
            self.ini_dict[attr] = value
        else:
            super(INIParser, self).__setattr__(attr, value)

    def __repr__(self) -> str:
        crop_dict = {key: self._format(val) for key, val in self.export_dict().items()}
        return crop_dict.__repr__()

    def __str__(self) -> str:
        crop_dict = {key: self._format(val) for key, val in self.export_dict().items()}
        return crop_dict.__str__()

    def keys(self) -> KeysView:
        return self.attr_dict.keys()

    def items(self) -> ItemsView:
        """Return (key, value) pairs of the datasets stored in the container.

        Returns:
            (key, value) pairs of the datasets stored in the container.
        """
        return dict(self).items()

    def values(self) -> ValuesView:
        """Return the attributes' data stored in the container.

        Returns:
            List of data stored in the container.
        """
        return dict(self).values()

    def export_dict(self) -> Dict[str, Any]:
        """Return a :class:`dict` object with all the attributes.

        Returns:
            Dictionary with all the attributes conntained in the object.
        """
        return dict(self)

    @hybridmethod
    def export_ini(self, **kwargs: Dict) -> ConfigParser:
        """Return a :class:`configparser.ConfigParser` object
        with all the attributes exported from the class.

        Args:
            kwargs : Extra parameters to export to the
                :class:`configparser.ConfigParser` object.

        Returns:
            A parser object with all the parsing specifications
            contained in class.
        """
        ini_parser = ConfigParser()
        for section in self.attr_dict:
            if 'ALL' in self.attr_dict[section]:
                ini_parser[section] = kwargs[section]
            else:
                ini_parser[section] = {option: kwargs[section][option]
                                       for option in self.attr_dict[section]}
        return ini_parser

    @export_ini.instancemethod
    def export_ini(self) -> ConfigParser:
        """Return a :class:`configparser.ConfigParser` object
        with all the attributes exported from the object.

        Returns:
            A parser object with all the parsing specifications
            contained in the object.
        """
        return type(self).export_ini(**self.ini_dict)
