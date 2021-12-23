"""Examples
--------
Generate the default built-in CXI protocol as follows:

>>> import pyrost as rst
>>> rst.CXIProtocol.import_default()
{'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors': 'float',
'data': 'uint', 'defocus': 'float', '...': '...'}, 'default_paths': {'basis_vectors':
'/speckle_tracking/basis_vectors', 'data': '/entry/data/data', 'defocus':
'/speckle_tracking/defocus', '...': '...'}, 'is_data': {'basis_vectors': 'False',
'data': 'True', 'defocus': 'False', '...': '...'}}

Or generate the default CXI loader:

>>> rst.CXILoader()
{'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors': 'float',
'data': 'uint', 'defocus': 'float', '...': '...'}, 'default_paths': {'basis_vectors':
'/speckle_tracking/basis_vectors', 'data': '/entry/data/data', 'defocus':
'/speckle_tracking/defocus', '...': '...'}, 'load_paths': {'basis_vectors':
['/entry_1/instrument_1/detector_1/basis_vectors', '/entry/instrument/detector/basis_vectors'],
'data': ['/entry/instrument/detector/data', '/entry_1/instrument_1/detector_1/data',
'/entry_1/data_1/data'], 'distance': ['/entry_1/instrument_1/detector_1/distance', ''],
'...': '...'}, 'is_data': {'basis_vectors': 'False', 'data': 'True', 'defocus': 'False',
'...': '...'}, 'policy': {'basis_vectors': 'True', 'data': 'True', 'defocus': 'True',
'...': '...'}}
"""
from __future__ import annotations
import os
from configparser import ConfigParser
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import h5py
import numpy as np
from .ini_parser import ROOT_PATH, INIParser

CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class CXIProtocol(INIParser):
    """CXI protocol class. Contains a CXI file tree path with
    the paths written to all the data attributes necessary for
    the Speckle Tracking algorithm, corresponding attributes'
    data types, and floating point precision.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', 'uint',
        or 'bool' are allowed.
    default_paths : dict, optional
        Dictionary with attributes path in the CXI file.
    is_data : dict, optional
        Dictionary with the flags if the attribute is of data type.
        Data type is 2- or 3-dimensional and has the same data shape
        as `data`.
    float_precision : {'float32', 'float64'}, optional
        Floating point precision.

    Attributes
    ----------
    config : dict
        Protocol configuration.
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', 'uint',
        or 'bool' are allowed.
    is_data : dict
        Dictionary with the flags if the attribute is of data type.
        Data type is 2- or 3-dimensional and has the same data shape
        as `data`.
    default_paths : dict
        Dictionary with attributes' CXI default file paths.

    See Also
    --------
    cxi_protocol : Full list of data attributes and configuration
        parameters.
    """
    known_types = {'int': np.int64, 'bool': np.bool, 'float': np.float64, 'str': str,
                   'uint': np.uint32}
    attr_dict = {'config': ('float_precision', ), 'datatypes': ('ALL', ),
                 'default_paths': ('ALL', ), 'is_data': ('ALL', )}
    fmt_dict = {'config': 'str', 'datatypes': 'str', 'default_paths': 'str',
                'is_data': 'str'}

    def __init__(self, datatypes: Dict[str, str], default_paths: Dict[str, str], is_data: Dict[str, str],
                 float_precision: str='float64') -> None:
        default_paths = {attr: val for attr, val in default_paths.items() if attr in datatypes}
        is_data = {attr: val for attr, val in is_data.items() if attr in datatypes}
        super(CXIProtocol, self).__init__(config={'float_precision': float_precision},
                                          datatypes=datatypes, default_paths=default_paths,
                                          is_data=is_data)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        if not self.config['float_precision'] in ['float32', 'float64']:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    @classmethod
    def import_default(cls, datatypes: Optional[Dict[str, str]]=None, default_paths: Optional[Dict[str, str]]=None,
                       is_data: Optional[Dict[str, str]]=None, float_precision: str='float64') -> CXIProtocol:
        """Return the default :class:`CXIProtocol` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', 'uint',
            or 'bool' are allowed.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file.
        is_data : dict, optional
            Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`.
        float_precision : {'float32', 'float64'}, optional
            Floating point precision.

        Returns
        -------
        CXIProtocol
            A :class:`CXIProtocol` object with the default parameters.

        See Also
        --------
        cxi_protocol : more details about the default CXI protocol.
        """
        return cls.import_ini(CXI_PROTOCOL, datatypes, default_paths, is_data, float_precision)

    @classmethod
    def import_ini(cls, ini_file: str, datatypes: Optional[Dict[str, str]]=None,
                   default_paths: Optional[Dict[str, str]]=None, is_data: Optional[Dict[str, str]]=None,
                   float_precision: str='float64') -> CXIProtocol:
        """Initialize a :class:`CXIProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default CXI protocol if None.
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', 'uint',
            or 'bool' are allowed. Initialized with `ini_file` if None.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file. Initialized with
            `ini_file` if None.
        is_data : dict, optional
            Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`. Initialized with `ini_file` if None.
        float_precision : {'float32', 'float64'}, optional
            Floating point precision. Initialized with `ini_file` if None.

        Returns
        -------
        CXIProtocol
            A :class:`CXIProtocol` object with all the attributes imported
            from the ini file.

        See Also
        --------
        cxi_protocol : more details about the default CXI protocol.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not default_paths is None:
            kwargs['default_paths'].update(**default_paths)
        if not is_data is None:
            kwargs['is_data'].update(**is_data)
        if float_precision is None:
            float_precision = kwargs['config']['float_precision']
        return cls(datatypes=kwargs['datatypes'], default_paths=kwargs['default_paths'],
                   is_data=kwargs['is_data'], float_precision=float_precision)

    def parser_from_template(self, path: str) -> ConfigParser:
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
        ini_template = ConfigParser()
        ini_template.read(path)
        parser = ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(**self.default_paths)
                               for option in ini_template[section]}
        return parser

    def __iter__(self) -> Iterator:
        return self.default_paths.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.default_paths

    def get_default_path(self, attr: str, value: Optional[str]=None) -> str:
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

    def get_dtype(self, attr: str, dtype: Optional[str]='float') -> type:
        """Return the attribute's data type. Return `dtype` if the
        attribute's data type is not found.

        Parameters
        ----------
        attr : str
            The data attribute.
        dtype : str, optional
            Data type which is returned if the attribute's data type
            is not found.

        Returns
        -------
        type
            Attribute's data type.
        """
        return self.known_types.get(self.datatypes.get(attr, dtype))

    def get_is_data(self, attr: str, value: bool=False) -> bool:
        """Return if the attribute is of data type. Data type is
        2- or 3-dimensional and has the same data shape as `data`.

        Parameters
        ----------
        attr : str
            The data attribute.
        value : bool, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        bool
            True if `attr` is of data type.
        """
        is_data = self.is_data.get(attr, value)
        if isinstance(is_data, str):
            return is_data in ['True', 'true', '1', 'y', 'yes']
        else:
            return bool(is_data)

    @staticmethod
    def get_dataset_shapes(cxi_path: str, cxi_obj: h5py.Group) -> List[Union[str, Tuple]]:
        """Visit recursively all the underlying datasets and return
        their names and shapes.

        Parameters
        ----------
        cxi_path : str
            Path of the HDF5 group.
        cxi_obj: h5py.Group
            Group object.

        Returns
        -------
        shapes : list
            List of all the datasets and their shapes inside `cxi_obj`.
        """
        shapes = []

        def caller(sub_path, obj):
            if isinstance(obj, h5py.Dataset):
                shapes.append((os.path.join(cxi_path, sub_path), obj.shape))

        cxi_obj.visititems(caller)
        return shapes

    def read_shape(self, cxi_file: h5py.File, cxi_path: str) -> List[Union[str, Tuple]]:
        """Read data shapes from the CXI file `cxi_file` at
        the `cxi_path` path inside the CXI file recursively.

        Parameters
        ----------
        cxi_file : h5py.File
            h5py File object of the CXI file.
        cxi_path : str
            Path to the data attribute.

        Returns
        -------
        list of tuples
            The attribute's data shapes extracted from the CXI file.
            Returns an empty list if no datasets has been found.
        """
        shapes = []
        if cxi_path is not None and cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                shapes.append((cxi_path, cxi_obj.shape))
            elif isinstance(cxi_obj, h5py.Group):
                shapes.extend(self.get_dataset_shapes(cxi_path, cxi_obj))
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")

        return shapes

    def read_cxi(self, attr: str, cxi_file: h5py.File,
                 cxi_path: Optional[str]=None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read `attr` from the CXI file `cxi_file` at the `cxi_path`
        path. If `cxi_path` is not provided the default path for the
        given attribute is used.

        Parameters
        ----------
        attr : str
            Data attribute.
        cxi_file : h5py.File
            h5py File object of the CXI file.
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.

        Returns
        -------
        numpy.ndarray or dict
            Returns data array if :code:`cxi_file[cxi_path]` is a
            dataset or a dictionary if :code:`cxi_file[cxi_path]` is
            a group.
        """
        if cxi_path is None:
            cxi_path = self.get_default_path(attr)
        shapes = self.read_shape(cxi_file, cxi_path)

        data_dict = {}
        if shapes:
            prefix = os.path.commonpath(list([cxi_path for cxi_path, _ in shapes]))
            for cxi_path, _ in shapes:
                key = os.path.relpath(cxi_path, prefix)
                if key.isnumeric():
                    key = int(key)
                data_dict[key] = cxi_file[cxi_path][()]

            if '.' in data_dict:
                data = np.asarray(data_dict['.'], dtype=self.get_dtype(attr))
                if data.size == 1:
                    return data.item()
                return data

        return data_dict

    @staticmethod
    def _write_dset(cxi_file: h5py.File, cxi_path: str, data: np.ndarray, dtype: type,
                    **kwargs: Any) -> None:
        try:
            cxi_file[cxi_path][...] = data
        except TypeError:
            del cxi_file[cxi_path]
            cxi_file.create_dataset(cxi_path, data=data, dtype=dtype, **kwargs)
        except (KeyError, ValueError):
            cxi_file.create_dataset(cxi_path, data=data, dtype=dtype, **kwargs)

    def write_cxi(self, attr: str, data: np.ndarray, cxi_file: h5py.File,
                  cxi_path: Optional[str]=None) -> None:
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
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.

        Raises
        ------
        ValueError
            If `overwrite` is False and the data is already present
            at the given location in `cxi_file`.
        """
        if cxi_path is None:
            cxi_path = self.get_default_path(attr, cxi_path)

        if isinstance(data, dict):
            for key, val in data.items():
                self._write_dset(cxi_file, os.path.join(cxi_path, str(key)), data=val,
                                    dtype=self.get_dtype(attr))

        elif self.get_is_data(attr):
            self._write_dset(cxi_file, cxi_path, data=data, dtype=self.get_dtype(attr),
                                chunks=(1,) + data.shape[1:], maxshape=(None,) + data.shape[1:])

        else:
            self._write_dset(cxi_file, cxi_path, data=data, dtype=self.get_dtype(attr))
