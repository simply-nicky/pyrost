"""Log protocol (:class:`pyrost.LogProtocol`) is a helper class to
retrieve the data from the log files generated at the Sigray laboratory,
which contain the readouts from the motors and other instruments
during the speckle tracking scan.

Examples:
    Generate the default built-in log protocol:

    >>> import pyrost as rst
    >>> rst.LogProtocol()
    {'log_keys': {'det_dist': ['Session logged attributes', 'Z-LENSE-DOWN_det_dist'],
    'exposure': ['Type: Method', 'Exposure'], 'n_steps': ['Type: Scan', 'Points count'],
    '...': '...'}, 'datatypes': {'det_dist': 'float', 'exposure': 'float', 'n_steps':
    'int', '...': '...'}}

    Generate the default log file converter:

    >>> rst.KamzikConverter()
    {'fs_vec': array([-5.5e-05,  0.0e+00,  0.0e+00]), 'protocol': {'datatypes':
    {'lens_down_dist': 'float', 'lens_up_dist': 'float', 'exposure': 'float', '...': '...'},
    'log_keys': {'lens_down_dist': ['Z-LENSE-DOWN_det_dist'], 'lens_up_dist':
    ['Z-LENSE-UP_det_dist'], 'exposure': ['Exposure'], '...': '...'}, 'part_keys':
    {'lens_down_dist': 'Session logged attributes', 'lens_up_dist': 'Session logged attributes',
    'exposure': 'Type: Method', '...': '...'}}, 'ss_vec': array([ 0.0e+00, -5.5e-05,  0.0e+00])}
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from .data_container import DataContainer, dict_to_object
from .ini_parser import ROOT_PATH, INIParser

LOG_PROTOCOL = os.path.join(ROOT_PATH, 'config/log_protocol.ini')

class LogProtocol(INIParser):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Attributes:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int',
            'bool', or 'str' are allowed.
        log_keys : Dictionary with attributes' log file keys.
        part_keys : Dictionary with the part names inside the log file
            where the attributes are stored.
    """
    attr_dict   = {'datatypes': ('ALL',), 'log_keys': ('ALL',), 'part_keys': ('ALL',)}
    fmt_dict    = {'datatypes': 'str', 'log_keys': 'str', 'part_keys': 'str'}
    unit_dict   = {'percent': 1e-2, 'mm,mdeg': 1e-3, 'µm,um,udeg,µdeg': 1e-6,
                   'nm,ndeg': 1e-9, 'pm,pdeg': 1e-12}

    datatypes   : Dict[str, str]
    log_keys    : Dict[str, List[str]]
    part_keys   : Dict[str, str]

    def __init__(self, datatypes: Dict[str, str], log_keys: Dict[str, List[str]],
                 part_keys: Dict[str, str]) -> None:
        """
        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                'bool', or 'str' are allowed.
            log_keys : Dictionary with attributes' log file keys.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored.
        """
        log_keys = {attr: val for attr, val in log_keys.items() if attr in datatypes}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in log_keys}
        super(LogProtocol, self).__init__(datatypes=datatypes, log_keys=log_keys,
                                          part_keys=part_keys)

    @classmethod
    def import_default(cls, datatypes: Optional[Dict[str, str]]=None,
                       log_keys: Optional[Dict[str, List[str]]]=None,
                       part_keys: Optional[Dict[str, str]]=None) -> LogProtocol:
        """Return the default :class:`LogProtocol` object. Extra arguments
        override the default values if provided.

        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                or 'bool' are allowed.
            log_keys : Dictionary with attributes' log file keys.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored.

        Returns:
            A :class:`LogProtocol` object with the default parameters.
        """
        return cls.import_ini(LOG_PROTOCOL, datatypes, log_keys, part_keys)

    @classmethod
    def import_ini(cls, ini_file: str, datatypes: Optional[Dict[str, str]]=None,
                   log_keys: Optional[Dict[str, List[str]]]=None,
                   part_keys: Optional[Dict[str, str]]=None) -> LogProtocol:
        """Initialize a :class:`LogProtocol` object class with an
        ini file.

        Args:
            ini_file : Path to the ini file. Load the default log protocol if None.
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                or 'bool' are allowed. Initialized with `ini_file` if None.
            log_keys : Dictionary with attributes' log file keys. Initialized with
                `ini_file` if None.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored. Initialized with `ini_file`
                if None.

        Returns:
            A :class:`LogProtocol` object with all the attributes imported
            from the ini file.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not log_keys is None:
            kwargs['log_keys'].update(**log_keys)
        if not part_keys is None:
            kwargs['part_keys'].update(**part_keys)
        return cls(datatypes=kwargs['datatypes'], log_keys=kwargs['log_keys'],
                   part_keys=kwargs['part_keys'])

    @classmethod
    def _get_unit(cls, key: str) -> float:
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                if unit in key:
                    return cls.unit_dict[unit_key]
        return 1.0

    @classmethod
    def _has_unit(cls, key: str) -> bool:
        has_unit = False
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                has_unit |= (unit in key)
        return has_unit

    def load_attributes(self, path: str) -> Dict[str, Any]:
        """Return attributes' values from a log file at
        the given `path`.

        Args:
            path : Path to the log file.

        Returns:
            Dictionary with the attributes retrieved from
            the log file.
        """
        if not isinstance(path, str):
            raise ValueError('path must be a string')
        with open(path, 'r') as log_file:
            log_str = ''
            for line in log_file:
                if line.startswith('# '):
                    log_str += line.strip('# ')
                else:
                    break

        # List all the sector names
        part_keys = list(self.part_keys.values())

        # Divide log into sectors
        parts_list = [part for part in re.split('(' + '|'.join(part_keys) + \
                      '|--------------------------------)\n*', log_str) if part]

        # Rearange sectors into a dictionary
        parts = {}
        for idx, part in enumerate(parts_list):
            if part in part_keys:
                if part == 'Session logged attributes':
                    attr_keys, attr_vals = parts_list[idx + 1].strip('\n').split('\n')
                    parts['Session logged attributes'] = ''
                    for key, val in zip(attr_keys.split(';'), attr_vals.split(';')):
                        parts['Session logged attributes'] += key + ': ' + val + '\n'
                else:
                    val = parts_list[idx + 1]
                    match = re.search(r'Device:.*\n', val)
                    if match:
                        name = match[0].split(': ')[-1][:-1]
                        parts[part + ', ' + name] = val

        # Populate attributes dictionary
        attr_dict = {part_name: {} for part_name in parts}
        for part_name, part in parts.items():
            for attr, part_key in self.part_keys.items():
                if part_key in part_name:
                    for log_key in self.log_keys[attr]:
                        # Find the attribute's mention and divide it into a key and value pair
                        match = re.search(log_key + r'.*\n', part)
                        if match:
                            raw_str = match[0]
                            raw_val = raw_str.strip('\n').split(': ')[1]
                            # Extract a number string
                            val_num = re.search(r'[-]*\d+[.]*\d*', raw_val)
                            dtype = self.known_types[self.datatypes[attr]]
                            attr_dict[part_name][attr] = dtype(val_num[0] if val_num else raw_val)
                            # Apply unit conversion if needed
                            if np.issubdtype(dtype, np.floating):
                                attr_dict[part_name][attr] *= self._get_unit(raw_str)
        return attr_dict

    def load_data(self, path: str, idxs: Optional[Iterable[int]]=None,
                  return_idxs=False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Retrieve the main data array from the log file.

        Args:
            path : Path to the log file.
            idxs : Array of data indices to load. Loads info for all
                the frames if None.
            return_idxs : Return a set of indices loaded from the log file.

        Returns:
            Dictionary with data fields and their names retrieved from the
            log file.
        """
        if idxs is not None:
            idxs = np.asarray(idxs)
            idxs.sort()

        line_count = 0
        with open(path, 'r') as log_file:
            for line_idx, line in enumerate(log_file):
                if line.startswith('# '):
                    if 'WARNING' not in line:
                        keys_line = line.strip('# ')
                else:
                    data_line = line

                    if idxs is None:
                        skiprows = line_idx
                        max_rows = None
                        break

                    if idxs.size == 0:
                        skiprows = line_idx
                        max_rows = 0
                        break

                    if line_count == idxs[0]:
                        skiprows = line_idx
                    if line_count == idxs[-1]:
                        max_rows = line_idx - skiprows + 1
                        break

                    line_count += 1

        keys = keys_line.strip('\n').split(';')
        data_strings = data_line.strip('\n').split(';')

        dtypes = {'names': [], 'formats': []}
        converters = {}
        for idx, (key, val) in enumerate(zip(keys, data_strings)):
            dtypes['names'].append(key)
            unit = self._get_unit(key)
            if 'float' in key:
                dtypes['formats'].append(np.dtype(float))
                converters[idx] = lambda item, unit=unit: unit * float(item)
            elif 'int' in key:
                if self._has_unit(key):
                    converters[idx] = lambda item, unit=unit: unit * float(item)
                    dtypes['formats'].append(np.dtype(float))
                else:
                    dtypes['formats'].append(np.dtype(int))
            elif 'Array' in key:
                dtypes['formats'].append(np.ndarray)
                func = lambda part, unit=unit: unit * float(part)
                conv = lambda item, func=func: np.asarray(list(map(func, item.strip(b' []').split(b','))))
                converters[idx] = conv
            else:
                dtypes['formats'].append('<S' + str(len(val)))
                converters[idx] = lambda item: item.strip(b' []')

        txt_dict = {}
        txt_tuple = np.loadtxt(path, delimiter=';', converters=converters,
                               dtype=dtypes, unpack=True, skiprows=skiprows,
                               max_rows=max_rows)

        if idxs is None:
            txt_dict.update(zip(keys, txt_tuple))
            idxs = np.arange(txt_tuple[0].size)
        elif idxs.size == 0:
            txt_dict.update(zip(keys, txt_tuple))
        else:
            txt_dict.update({key: np.atleast_1d(data)[idxs - np.min(idxs)]
                             for key, data in zip(keys, txt_tuple)})

        if return_idxs:
            return txt_dict, idxs
        return txt_dict

class KamzikConverter(DataContainer):
    """A converter class, that generates CXI datasets aceeptable by :class:`pyrost.STData`
    from Kamzik log files.

    Attributes:
        protocol : Log file protocol.
        fs_vec : Fast (horizontal) scan detector axis.
        ss_vec : Slow (vertical) scan detector axis.
        idxs : Frame indices read from a log file.
        log_attr : Dictionary of log attributes.
        log_data : Dictionary of log datasets.
    """
    attr_set = {'protocol', 'fs_vec', 'ss_vec'}
    init_set = {'idxs', 'log_attr', 'log_data'}

    cxi_attrs = {'basis_vectors': 'basis_vectors', 'dist_down': 'distance', 'dist_up': 'distance',
                 'sim_translations': 'translations', 'log_translations': 'translations',
                 'x_pixel_size': 'x_pixel_size', 'y_pixel_size': 'y_pixel_size'}

    protocol:   LogProtocol
    fs_vec:     np.ndarray
    ss_vec:     np.ndarray

    idxs:       Optional[np.ndarray]
    log_attr:   Optional[Dict[str, Any]]
    log_data:   Optional[Dict[str, Any]]

    def __init__(self, protocol: LogProtocol=LogProtocol.import_default(),
                 fs_vec: np.ndarray=np.array([-55e-6, 0., 0.]),
                 ss_vec: np.ndarray=np.array([0., -55e-6, 0.]),
                 idxs: Optional[np.ndarray]=None, log_attr: Optional[Dict[str, Any]]=None,
                 log_data: Optional[Dict[str, Any]]=None) -> None:
        """
        Args:
            protocol : Log file protocol.
            fs_vec : Fast (horizontal) scan detector axis.
            ss_vec : Slow (vertical) scan detector axis.
            idxs : Frame indices read from a log file.
            log_attr : Dictionary of log attributes read from a log file.
            log_data : Dictionary of log datasets read from a log file.
        """
        super(KamzikConverter, self).__init__(protocol=protocol, fs_vec=fs_vec, ss_vec=ss_vec,
                                              idxs=idxs, log_attr=log_attr, log_data=log_data)

    @property
    def n_frames(self):
        return None if self.idxs is None else self.idxs.size

    @property
    def x_pixel_size(self):
        return np.sqrt((self.fs_vec * self.fs_vec).sum())

    @property
    def y_pixel_size(self):
        return np.sqrt((self.ss_vec * self.ss_vec).sum())

    @dict_to_object
    def read_logs(self, log_path: str, idxs: Optional[Iterable[int]]=None) -> KamzikConverter:
        """Read a log file under the path `log_path`. Read out only the frame indices
        defined by `idxs`. If `idxs` is None, read the whole log file.

        Args:
            log_path : Path to the log file.
            idxs : List of indices to read. Read the whole log file if None.

        Returns:
            A new :class:`KamzikConverter` object with `log_attr`, `log_data`, and `idxs`
            updated.
        """
        log_attr = self.protocol.load_attributes(log_path)
        log_data, idxs = self.protocol.load_data(log_path, idxs=idxs, return_idxs=True)
        return {'log_attr': log_attr, 'log_data': log_data, 'idxs': idxs}

    def find_log_part_key(self, attr: str) -> Optional[str]:
        """Find a name of the log dictionary corresponding to an attribute
        name `attr`.

        Args:
            attr : A name of the attribute to find.

        Returns:
            A name of the log dictionary, corresponding to the given attribute
            name `attr`.
        """
        log_attr = self.get('log_attr', {})
        log_keys = self.protocol.log_keys.get(attr, [])
        for part in log_attr:
            for log_key in log_keys:
                if log_key in part:
                    return part
        return None

    def find_log_attribute(self, attr: str, part_key: Optional[str]=None) -> Optional[Any]:
        """Find a value in the log attributes corresponding to an
        attribute name `attr`.

        Args:
            attr : A name of the attribute to find.
            part_key : Search in the given part of the log dictionary if provided.

        Returns:
            Value of the log attribute. Returns None if nothing is found.
        """
        if part_key is None:
            part_key = self.protocol.part_keys.get(attr, '')
        log_attr = self.get('log_attr', {})
        part_dict = log_attr.get(part_key, {})
        value = part_dict.get(attr, None)
        return value

    def find_log_dataset(self, attr: str) -> Optional[np.ndarray]:
        """Find a dataset in the log data corresponding to an
        attribute name `attr`.

        Args:
            attr : A name of the attribute to find.

        Returns:
            Dataset for the given attribute. Returns None if nothing is found.
        """
        log_keys = self.protocol.log_keys.get(attr, [])
        log_data = self.get('log_data', {})
        for data_key, log_dset in log_data.items():
            for log_key in log_keys:
                if log_key in data_key:
                    return log_dset
        return None

    def _is_basis_vectors(self) -> bool:
        return self.n_frames is not None

    def _is_dist_down(self) -> bool:
        return self.find_log_attribute('lens_down_dist') is not None

    def _is_dist_up(self) -> bool:
        return self.find_log_attribute('lens_up_dist') is not None

    def _is_log_translations(self) -> bool:
        return (self.find_log_attribute('x_sample') is not None and
                self.find_log_attribute('y_sample') is not None and
                self.find_log_attribute('z_sample') is not None and
                (self.find_log_dataset('x_sample') is not None or
                 self.find_log_dataset('y_sample') is not None or
                 self.find_log_dataset('z_sample') is not None))

    def _is_sim_translations(self) -> bool:
        return (self.find_log_attribute('x_sample') is not None and
                self.find_log_attribute('y_sample') is not None and
                self.find_log_attribute('z_sample') is not None and
                (self.find_log_part_key('x_sample') is not None or
                 self.find_log_part_key('y_sample') is not None or
                 self.find_log_part_key('z_sample') is not None))

    def cxi_keys(self) -> List[str]:
        """Return a list of available CXI attributes.

        Returns:
            List of available CXI attributes.
        """
        cxi_dict = {'basis_vectors': self._is_basis_vectors,
                    'dist_down': self._is_dist_down,
                    'dist_up': self._is_dist_up,
                    'sim_translations': self._is_sim_translations,
                    'log_translations': self._is_log_translations}
        return [attr for attr, func in cxi_dict.items() if func()]

    def _get_basis_vectors(self) -> np.ndarray:
        return np.stack((np.tile(self.ss_vec, (self.n_frames, 1)),
                         np.tile(self.fs_vec, (self.n_frames, 1))), axis=1)

    def _get_dist_down(self) -> float:
        return self.find_log_attribute('lens_down_dist')

    def _get_dist_up(self) -> float:
        return self.find_log_attribute('lens_up_dist')

    def _get_sim_translations(self) -> np.ndarray:
        translations = np.tile((self.find_log_attribute('x_sample'),
                                self.find_log_attribute('y_sample'),
                                self.find_log_attribute('z_sample')), (self.n_frames, 1))
        translations = np.nan_to_num(translations)

        step_sizes, n_steps = [], []
        for scan_motor, unit_vec in zip(['x_sample', 'y_sample', 'z_sample'],
                                        [np.array([1., 0., 0.]),
                                         np.array([0., 1., 0.]),
                                         np.array([0., 0., 1.])]):
            part_key = self.find_log_part_key(scan_motor)
            if part_key is not None:
                step_sizes.append(self.log_attr[part_key].get('step_size') * unit_vec)
                n_steps.append(self.log_attr[part_key].get('n_points'))

        steps = np.tensordot(np.stack(np.mgrid[[slice(0, n) for n in n_steps]], axis=0),
                             np.stack(step_sizes, axis=0), (0, 0)).reshape(-1, 3)
        return translations + steps

    def _get_log_translations(self) -> np.ndarray:
        translations = np.tile((self.find_log_attribute('x_sample'),
                                self.find_log_attribute('y_sample'),
                                self.find_log_attribute('z_sample')), (self.n_frames, 1))
        translations = np.nan_to_num(translations)

        for idx, scan_motor in enumerate(['x_sample', 'y_sample', 'z_sample']):
            dset = self.find_log_dataset(scan_motor)
            if dset is not None:
                translations[:dset.size, idx] = dset
        return translations

    def cxi_get(self, attrs: Union[str, List[str]]) -> Dict[str, Any]:
        """Convert Kamzik log files data into CXI attributes, that are accepted by
        :class:`pyrost.STData` container. To see full list of available CXI
        attributes, use :func:`KamzikConverter.cxi_keys`.

        Args:
            attrs: List of CXI attributes to generate. The method will raise an error
                if any of the attributes is unavailable.

        Raises:
            ValueError : If any of attributes in `attrs` in unavailable.

        Returns:
            A dictionary of CXI attributes, that are accepted by :class:`pyrost.STData`
            container.
        """
        cxi_dict = {'basis_vectors': self._get_basis_vectors,
                    'dist_down': self._get_dist_down,
                    'dist_up': self._get_dist_up,
                    'sim_translations': self._get_sim_translations,
                    'log_translations': self._get_log_translations}
        data = {'x_pixel_size': self.x_pixel_size, 'y_pixel_size': self.y_pixel_size}
        cxi_keys = self.cxi_keys()
        for attr in attrs:
            if attr in cxi_keys:
                data[self.cxi_attrs[attr]] = cxi_dict[attr]()
            else:
                raise ValueError(f"CXI attribute '{attr}' is unavailable")

        return data
