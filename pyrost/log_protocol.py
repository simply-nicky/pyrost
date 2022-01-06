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
from typing import Any, Dict, Iterable, List, Optional
import h5py
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .data_processing import CXILoader, STData
from .bin import median

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

    def load_data(self, path: str, frame_indices: Optional[Iterable[int]]=None) -> Dict[str, np.ndarray]:
        """Retrieve the main data array from the log file.

        Args:
            path : Path to the log file.
            frame_indices : Array of data indices to load. Loads info for all
                the frames by default.

        Returns:
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        if frame_indices is not None:
            frame_indices.sort()

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
                    if frame_indices is not None and row_cnt == frame_indices[0]:
                        skiprows = line_idx - first_row
                    if frame_indices is not None and row_cnt == frame_indices[-1]:
                        max_rows = line_idx - skiprows
                        break

                    row_cnt += 1
            else:
                if frame_indices is None:
                    frame_indices = np.arange(row_cnt)
                    skiprows = 0
                    max_rows = line_idx - skiprows
                else:
                    frame_indices = frame_indices[:np.searchsorted(frame_indices, row_cnt)]
                    if not frame_indices.size:
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
        data_dict = {key: data[frame_indices - skiprows] for key, data in zip(keys, data_tuple)}
        data_dict['indices'] = frame_indices
        return data_dict

def cxi_converter_sigray(scan_num: int, dir_path: str='/gpfs/cfel/group/cxi/labs/MLL-Sigray',
                         target: str='Mo', distance: Optional[float]=None, lens: str='up',
                         frame_indices: Optional[Iterable[int]]=None, **attributes: Any) -> STData:
    """Convert measured frames and log files from the
    Sigray laboratory to a :class:`pyrost.STData` data
    container.

    Args:
        scan_num : Scan number.
        dir_path : Path to the root directory, where the data is located.
        target : Sigray X-ray source target used. The following values are
            accepted:

            * 'Mo' : Mollibdenum.
            * 'Cu' : Cuprum.
            * 'Rh' : Rhodium.

        distance : Detector distance in meters.
        lens : Specify if the lens mounted in the upper holder ('up') or in
            the lower holder ('down'). If specified, the lens-to-detector
            distance will be automatically parsed from the log file.
        frame_indices : Array of data indices to load. Loads info for all the
            frames by default.
        attributes : Dictionary of attribute values, that override the loaded
            values.

    Returns:
        Data container with the extracted data.
    """
    wl_dict = {'Mo': 7.092917530503447e-11, 'Cu': 1.5498024804150033e-10,
               'Rh': 6.137831605603974e-11}

    log_prt = LogProtocol.import_default()
    cxi_loader = CXILoader.import_default()

    ss_vec = np.array([0., -1., 0.])
    fs_vec = np.array([-1., 0., 0.])

    log_path = os.path.join(dir_path, f'scan-logs/Scan_{scan_num:d}.log')
    data_dir = os.path.join(dir_path, f'scan-frames/Scan_{scan_num:d}')
    h5_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir)
                       if path.endswith('Lambda.nxs')])

    log_attrs = log_prt.load_attributes(log_path)
    log_data = log_prt.load_data(log_path, frame_indices=frame_indices)

    data_dict = cxi_loader.load_to_dict(h5_files, frame_indices=log_data['indices'],
                                        wavelength=wl_dict[target], distance=distance,
                                        **attributes)
    data_dict['x_pixel_size'] *= 1e-6
    data_dict['y_pixel_size'] *= 1e-6

    pix_vec = np.tile(np.array([[data_dict['x_pixel_size'], data_dict['y_pixel_size'], 0]]),
                      (data_dict['data'].shape[0], 1))
    data_dict['basis_vectors'] = np.stack([pix_vec * ss_vec, pix_vec * fs_vec], axis=1)

    with np.load(os.path.join(ROOT_PATH, 'data/sigray_mask.npz')) as mask_file:
        if mask_file['mask'].shape == data_dict['data'].shape[1:]:
            data_dict['mask'] = np.tile(mask_file['mask'][None],
                                        (data_dict['data'].shape[0], 1, 1))

    x_sample = log_attrs['Session logged attributes'].get('x_sample', 0.0)
    y_sample = log_attrs['Session logged attributes'].get('y_sample', 0.0)
    z_sample = log_attrs['Session logged attributes'].get('z_sample', 0.0)
    data_dict['translations'] = np.tile([[x_sample, y_sample, z_sample]],
                                        (data_dict['data'].shape[0], 1))
    for data_key, log_dset in log_data.items():
        for log_key in log_prt.log_keys['x_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 0] = log_dset
        for log_key in log_prt.log_keys['y_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 1] = log_dset
        for log_key in log_prt.log_keys['z_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 2] = log_dset

    if data_dict.get('distance', None) is None:
        if lens == 'up':
            data_dict['distance'] = log_attrs['Session logged attributes']['lens_up_dist']
        elif lens == 'down':
            data_dict['distance'] = log_attrs['Session logged attributes']['lens_down_dist']
        else:
            raise ValueError(f'lens keyword is invalid: {lens:s}')

    return STData(**data_dict)

def tilt_converter_sigray(scan_num: int, out_path: str,
                          dir_path: str='/gpfs/cfel/group/cxi/labs/MLL-Sigray',
                          target: str='Mo', distance: float=2.,
                          frame_indices: Optional[Iterable[int]]=None) -> None:
    """Save measured frames and log files from a tilt
    scan to a h5 file.

    Args:
        scan_num : Scan number.
        out_path : Path of the output file
        dir_path : Path to the root directory, where the data is located.
        target : Sigray X-ray source target used. The following values are
            accepted:

            * 'Mo' : Mollibdenum.
            * 'Cu' : Cuprum.
            * 'Rh' : Rhodium.

        distance : Detector distance in meters.
        frame_indices : Array of data indices to load. Loads info for all the
            frames by default.
    """
    energy_dict = {'Mo': 17.48, 'Cu': 8.05, 'Rh': 20.2} # keV
    flip_dict={'Yaw-LENSE-UP': False, 'Pitch-LENSE-UP': False,
               'Yaw-LENSE-DOWN': True, 'Pitch-LENSE-DOWN': True}
    sum_axis = {'Yaw-LENSE-UP': 0, 'Pitch-LENSE-UP': 1,
                'Yaw-LENSE-DOWN': 0, 'Pitch-LENSE-DOWN': 1}

    log_prt = LogProtocol.import_default()
    cxi_loader = CXILoader.import_default()

    log_path = os.path.join(dir_path, f'scan-logs/Scan_{scan_num:d}.log')
    data_dir = os.path.join(dir_path, f'scan-frames/Scan_{scan_num:d}')
    h5_files = sorted([os.path.join(data_dir, path) for path in os.listdir(data_dir)
                       if path.endswith('Lambda.nxs')])

    log_data = log_prt.load_data(log_path, frame_indices=frame_indices)

    data_dict = cxi_loader.load_to_dict(h5_files, frame_indices=log_data['indices'])
    data = data_dict['data']
    with np.load(os.path.join(ROOT_PATH, 'data/sigray_mask.npz')) as mask_file:
        mask = np.tile(mask_file['mask'][None, ()], (data.shape[0], 1, 1))

    whitefield = median(data, mask, axis=0)
    db_coord = np.unravel_index(np.argmax(whitefield), whitefield.shape)

    for flip_key in flip_dict:
        if any(flip_key in data_type for data_type in log_data):
            scan_type = [data_type for data_type in log_data if flip_key in data_type][0]
            translations = log_data[scan_type]
            if sum_axis[flip_key]:
                data = np.sum(data[:, :, db_coord[1] - 10:db_coord[1] + 10], axis=2)
                theta = np.linspace(0, data.shape[1], data.shape[1]) - db_coord[0]
                theta *= 36e-5 * data_dict['x_pixel_size'] / (2 * np.pi * distance)
            else:
                data = np.sum(data[:, db_coord[0] - 10:db_coord[0] + 10], axis=1)
                theta = np.linspace(data.shape[1], 0, data.shape[1]) - db_coord[1]
                theta *= 36e-5 * data_dict['y_pixel_size'] / (2 * np.pi * distance)
            if flip_dict[flip_key]:
                data = np.flip(data, axis=0)
            break
    else:
        raise ValueError('The scan type is not supported')

    with h5py.File(out_path, 'w') as out_file:
        out_file.create_dataset("Data", data=data)
        out_file.create_dataset("Omega", data=translations * 1e9) # must be in ndeg for some reason
        out_file.create_dataset("2Theta", data=theta)
        out_file.create_dataset("Energy", data=energy_dict[target])
