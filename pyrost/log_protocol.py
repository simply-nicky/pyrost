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
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .data_processing import CXIStore, STData, Transform

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
    attr_dict = {'datatypes': ('ALL',), 'log_keys': ('ALL',), 'part_keys': ('ALL',)}
    fmt_dict = {'datatypes': 'str', 'log_keys': 'str', 'part_keys': 'str'}
    unit_dict = {'percent': 1e-2, 'mm,mdeg': 1e-3, 'µm,um,udeg,µdeg': 1e-6,
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

    def load_data(self, path: str, indices: Optional[Iterable[int]]=None,
                  return_indices=False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Retrieve the main data array from the log file.

        Args:
            path : Path to the log file.
            indices : Array of data indices to load. Loads info for all
                the frames by default.

        Returns:
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        if indices is not None:
            indices.sort()

        row_cnt = 0
        with open(path, 'r') as log_file:
            for line_idx, line in enumerate(log_file):
                if line.startswith('# '):
                    if 'WARNING' not in line:
                        keys_line = line.strip('# ')
                else:
                    data_line = line

                    if row_cnt == 0:
                        first_row = line_idx
                    if indices is not None and row_cnt == indices[0]:
                        skiprows = line_idx - first_row
                    if indices is not None and row_cnt == indices[-1]:
                        max_rows = line_idx - skiprows
                        break

                    row_cnt += 1
            else:
                if indices is None:
                    indices = np.arange(row_cnt)
                    skiprows = 0
                    max_rows = line_idx - skiprows
                else:
                    indices = indices[:np.searchsorted(indices, row_cnt)]
                    if not indices.size:
                        skiprows = line_idx
                    max_rows = line_idx - skiprows

        keys = keys_line.strip('\n').split(';')
        data_strings = data_line.strip('\n').split(';')

        dtypes = {'names': [], 'formats': []}
        converters = {}
        for idx, (key, val) in enumerate(zip(keys, data_strings)):
            dtypes['names'].append(key)
            unit = self._get_unit(key)
            if 'float' in key:
                dtypes['formats'].append(np.float_)
                converters[idx] = lambda item, unit=unit: unit * float(item)
            elif 'int' in key:
                if self._has_unit(key):
                    converters[idx] = lambda item, unit=unit: unit * float(item)
                    dtypes['formats'].append(np.float_)
                else:
                    dtypes['formats'].append(np.int)
            elif 'Array' in key:
                dtypes['formats'].append(np.ndarray)
                converters[idx] = lambda item, unit=unit: np.array([float(part.strip(b' []')) * unit
                                                                    for part in item.split(b',')])
            else:
                dtypes['formats'].append('<S' + str(len(val)))
                converters[idx] = lambda item: item.strip(b' []')

        data_tuple = np.loadtxt(path, delimiter=';', converters=converters,
                                dtype=dtypes, unpack=True, skiprows=skiprows,
                                max_rows=max_rows + 1)
        data_dict = {key: data[indices - skiprows] for key, data in zip(keys, data_tuple)}

        if return_indices:
            return data_dict, indices
        return data_dict

def cxi_converter_sigray(out_path: str, scan_num: int, dir_path: str='/gpfs/cfel/group/cxi/labs/MLL-Sigray',
                         target: str='Mo', distance: Optional[float]=None, lens: str='up',
                         indices: Optional[Iterable[int]]=None, transform: Optional[Transform]=None,
                         **attributes: Any) -> STData:
    """Convert measured frames and log files from the Sigray laboratory to a
    :class:`pyrost.STData` data container.

    Args:
        out_path : Path to the file, where all the results can be saved.
        scan_num : Scan number.
        dir_path : Path to the root directory, where the data is located.
        target : Sigray X-ray source target used. The following values are
            accepted:

            * `Mo` : Mollibdenum.
            * `Cu` : Cuprum.
            * `Rh` : Rhodium.

        distance : Detector distance in meters.
        lens : Specify if the lens mounted in the upper holder ('up') or in
            the lower holder ('down'). If specified, the lens-to-detector
            distance will be automatically parsed from the log file.
        indices : Array of data indices to load. Loads info for all the
            frames by default.
        transform : Frames transform object.
        attributes : Dictionary of attribute values, that override the loaded
            values in :class:`pyrost.STData`.

    Returns:
        :class:`pyrost.STData` data container with the extracted data.
    """
    wl_dict = {'Mo': 7.092917530503447e-11, 'Cu': 1.5498024804150033e-10,
               'Rh': 6.137831605603974e-11}


    ss_vec = np.array([0., -1., 0.])
    fs_vec = np.array([-1., 0., 0.])

    log_path = os.path.join(dir_path, f'scan-logs/Scan_{scan_num:d}.log')
    data_dir = os.path.join(dir_path, f'scan-frames/Scan_{scan_num:d}')
    h5_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir)
                       if path.endswith('Lambda.nxs')])

    files = CXIStore(input_files=h5_files, output_file=out_path)

    log_prt = LogProtocol.import_default()
    log_attrs = log_prt.load_attributes(log_path)
    log_data, indices = log_prt.load_data(log_path, indices=indices,
                                          return_indices=True)

    x_pixel_size = 55e-6
    y_pixel_size = 55e-6

    n_frames = indices.size
    pix_vec = np.tile(np.array([[x_pixel_size, y_pixel_size, 0]]), (n_frames, 1))
    basis_vectors = np.stack([pix_vec * ss_vec, pix_vec * fs_vec], axis=1)

    with np.load(os.path.join(ROOT_PATH, 'data/sigray_mask.npz')) as mask_file:
        mask = np.tile(mask_file['mask'][None], (n_frames, 1, 1))

    x_sample = log_attrs['Session logged attributes'].get('x_sample', 0.0)
    y_sample = log_attrs['Session logged attributes'].get('y_sample', 0.0)
    z_sample = log_attrs['Session logged attributes'].get('z_sample', 0.0)
    translations = np.nan_to_num(np.tile([[x_sample, y_sample, z_sample]], (n_frames, 1)))
    for data_key, log_dset in log_data.items():
        for log_key in log_prt.log_keys['x_sample']:
            if log_key in data_key:
                translations[:log_dset.size, 0] = log_dset
        for log_key in log_prt.log_keys['y_sample']:
            if log_key in data_key:
                translations[:log_dset.size, 1] = log_dset
        for log_key in log_prt.log_keys['z_sample']:
            if log_key in data_key:
                translations[:log_dset.size, 2] = log_dset

    if distance is None:
        if lens == 'up':
            distance = log_attrs['Session logged attributes']['lens_up_dist']
        elif lens == 'down':
            distance = log_attrs['Session logged attributes']['lens_down_dist']
        else:
            raise ValueError(f'lens keyword is invalid: {lens:s}')

    data = STData(files, basis_vectors=basis_vectors, mask=mask, translations=translations,
                  distance=distance, x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size,
                  wavelength=wl_dict[target], **attributes)
    if transform:
        data = data.update_transform(transform)
    data = data.load('data', indices=indices)
    return data
