"""Examples
--------
Generate the default built-in log protocol:

>>> import pyrost as rst
>>> rst.LogProtocol()
{'log_keys': {'det_dist': ['Session logged attributes', 'Z-LENSE-DOWN_det_dist'],
'exposure': ['Type: Method', 'Exposure'], 'n_steps': ['Type: Scan', 'Points count'],
'...': '...'}, 'datatypes': {'det_dist': 'float', 'exposure': 'float', 'n_steps':
'int', '...': '...'}}
"""
import os
import re
import h5py
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .cxi_protocol import CXIProtocol, CXILoader
from .data_processing import STData
from .bin import median

LOG_PROTOCOL = os.path.join(ROOT_PATH, 'config/log_protocol.ini')

class LogProtocol(INIParser):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', 'bool',
        or 'str' are allowed.
    log_keys : dict, optional
        Dictionary with attributes' log file keys.

    Attributes
    ----------
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', 'bool',
        or 'str' are allowed.
    log_keys : dict
        Dictionary with attributes' log file keys.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    """
    attr_dict = {'log_keys': ('ALL',), 'datatypes': ('ALL',)}
    fmt_dict = {'log_keys': 'str', 'datatypes': 'str'}
    unit_dict = {'percent': 1e-2, 'mm,mdeg': 1e-3, 'µm,um,udeg,µdeg': 1e-6,
                 'nm,ndeg': 1e-9, 'pm,pdeg': 1e-12}

    def __init__(self, log_keys=None, datatypes=None):
        if log_keys is None:
            log_keys = self._import_ini(LOG_PROTOCOL)['log_keys']
        if datatypes is None:
            datatypes = self._import_ini(LOG_PROTOCOL)['datatypes']
        log_keys = {attr: val for attr, val in log_keys.items() if attr in datatypes}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in log_keys}
        super(LogProtocol, self).__init__(log_keys=log_keys, datatypes=datatypes)

    @classmethod
    def import_default(cls, datatypes=None, log_keys=None):
        """Return the default :class:`LogProtocol` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        log_keys : dict, optional
            Dictionary with attributes' log file keys. Initialized with
            `ini_file` if None.

        Returns
        -------
        LogProtocol
            A :class:`LogProtocol` object with the default parameters.

        See Also
        --------
        log_protocol : more details about the default CXI protocol.
        """
        return cls.import_ini(LOG_PROTOCOL, datatypes, log_keys)

    @classmethod
    def import_ini(cls, ini_file, datatypes=None, log_keys=None):
        """Initialize a :class:`LogProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default log protocol if None.
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        log_keys : dict, optional
            Dictionary with attributes' log file keys. Initialized with
            `ini_file` if None.

        Returns
        -------
        LogProtocol
            A :class:`LogProtocol` object with all the attributes imported
            from the ini file.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not log_keys is None:
            kwargs['log_keys'].update(**log_keys)
        return cls(datatypes=kwargs['datatypes'], log_keys=kwargs['log_keys'])

    @classmethod
    def _get_unit(cls, key):
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                if unit in key:
                    return cls.unit_dict[unit_key]
        return 1.

    @classmethod
    def _has_unit(cls, key):
        has_unit = False
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                has_unit |= (unit in key)
        return has_unit

    def load_attributes(self, path):
        """Return attributes' values from a log file at
        the given `path`.

        Parameters
        ----------
        path : str
            Path to the log file.

        Returns
        -------
        attr_dict : dict
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

        # Divide log into sectors
        parts_list = [part for part in re.split('(' + \
                     '|'.join([key[0] for key in self.log_keys.values()]) + \
                     '|--------------------------------)\n*', log_str) if part]

        # List all the sector names
        part_keys = [part_key for part_key, _ in self.log_keys.values()]

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
            for attr, [part_key, log_key] in self.log_keys.items():
                if part_key in part_name:
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

    def load_data(self, path):
        """Retrieve the main data array from the log file.

        Parameters
        ----------
        path : str
            Path to the log file.

        Returns
        -------
        data : dict
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        with open(path, 'r') as log_file:
            for line in log_file:
                if line.startswith('# '):
                    keys_line = line.strip('# ')
                else:
                    data_line = line
                    break

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

        return dict(zip(keys, np.loadtxt(path, delimiter=';',
                                         converters=converters,
                                         dtype=dtypes, unpack=True)))

def cxi_converter_sigray(scan_num, target='Mo', distance=None, lens='up'):
    """Convert measured frames and log files from the
    Sigray laboratory to a :class:`pyrost.STData` data
    container.

    Parameters
    ----------
    scan_num : int
        Scan number.
    target : {'Mo', 'Cu', 'Rh'}, optional
        Sigray X-ray source target used.
    distance : float, optional
        Detector distance in meters.
    lens : {'up', 'down'}, optional
        Specify the lens mount. If specified, the lens-to-detector
        distance will be automatically parsed from the log file.

    Returns
    -------
    STData
        Data container with the extracted data.
    """
    wl_dict = {'Mo': 7.092917530503447e-11, 'Cu': 1.5498024804150033e-10,
               'Rh': 6.137831605603974e-11}

    h5_prt = CXIProtocol(default_paths={'data': 'entry/instrument/detector/data',
                                        'x_pixel_size': 'entry/instrument/detector/x_pixel_size',
                                        'y_pixel_size': 'entry/instrument/detector/y_pixel_size'},
                         datatypes={'data': 'float', 'x_pixel_size': 'float',
                                    'y_pixel_size': 'float'})
    cxi_prt = CXIProtocol()
    log_prt = LogProtocol()
    cxi_loader = CXILoader(h5_prt)

    ss_vec = np.array([0., -1., 0.])
    fs_vec = np.array([-1., 0., 0.])

    log_path = f'/gpfs/cfel/cxi/labs/MLL-Sigray/scan-logs/Scan_{scan_num:d}.log'
    dir_path = f'/gpfs/cfel/cxi/labs/MLL-Sigray/scan-frames/Scan_{scan_num:d}'
    h5_files = sorted([os.path.join(dir_path, path) for path in os.listdir(dir_path)
                       if path.endswith('Lambda.nxs')])

    data = np.concatenate(list(cxi_loader.load_data(h5_files).values()), axis=-3)
    attrs = cxi_loader.load_attributes(h5_files[0])
    log_attrs = log_prt.load_attributes(log_path)
    log_data = log_prt.load_data(log_path)

    n_steps = min(next(iter(log_data.values())).shape[0], data.shape[0])
    pix_vec = np.tile(np.array([[attrs['x_pixel_size'], attrs['y_pixel_size'], 0]]),
                      (n_steps, 1)) * 1e-6
    basis_vectors = np.stack([pix_vec * ss_vec, pix_vec * fs_vec], axis=1)

    with np.load(os.path.join(ROOT_PATH, 'data/sigray_mask.npz')) as mask_file:
        mask = np.tile(mask_file['mask'][None], (n_steps, 1, 1))

    translations = np.tile([[log_attrs['Session logged attributes']['x_sample'],
                             log_attrs['Session logged attributes']['y_sample'],
                             log_attrs['Session logged attributes']['z_sample']]],
                           (n_steps, 1))
    for data_key in log_data:
        if 'X-SAM' in data_key:
            translations[:, 0] = log_data[data_key][:n_steps]
        if 'Y-SAM' in data_key:
            translations[:, 1] = log_data[data_key][:n_steps]

    if distance is None:
        if lens == 'up':
            distance = log_attrs['Session logged attributes']['lens_up_dist']
        elif lens == 'down':
            distance = log_attrs['Session logged attributes']['lens_down_dist']
        else:
            raise ValueError(f'lens keyword is invalid: {lens:s}')

    return STData(basis_vectors=basis_vectors, data=data[:n_steps], distance=distance,
                  mask=mask, translations=translations, wavelength=wl_dict[target],
                  x_pixel_size=attrs['x_pixel_size'] * 1e-6,
                  y_pixel_size=attrs['y_pixel_size'] * 1e-6, protocol=cxi_prt)

def tilt_converter_sigray(scan_num, out_path, target='Mo', distance=2.):
    """Save measured frames and log files from a tilt
    scan to a h5 file.

    Parameters
    ----------
    scan_num : int
        Scan number.
    out_path : str
        Path of the output file
    target : {'Mo', 'Cu', 'Rh'}, optional
        Sigray X-ray source target used.
    distance : float, optional
        Detector distance in meters.
    """
    energy_dict = {'Mo': 17.48, 'Cu': 8.05, 'Rh': 20.2} # keV
    flip_dict={'Yaw-LENSE-UP': False, 'Pitch-LENSE-UP': False,
            'Yaw-LENSE-DOWN': True, 'Pitch-LENSE-DOWN': True}
    sum_axis = {'Yaw-LENSE-UP': 0, 'Pitch-LENSE-UP': 1,
                'Yaw-LENSE-DOWN': 0, 'Pitch-LENSE-DOWN': 1}

    h5_prt = CXIProtocol(default_paths={'data': 'entry/instrument/detector/data',
                                        'x_pixel_size': 'entry/instrument/detector/x_pixel_size',
                                        'y_pixel_size': 'entry/instrument/detector/y_pixel_size'},
                         datatypes={'data': 'float', 'x_pixel_size': 'float',
                                    'y_pixel_size': 'float'})
    log_prt = LogProtocol()
    cxi_loader = CXILoader(h5_prt)

    log_path = f'/gpfs/cfel/cxi/labs/MLL-Sigray/scan-logs/Scan_{scan_num:d}.log'
    dir_path = f'/gpfs/cfel/cxi/labs/MLL-Sigray/scan-frames/Scan_{scan_num:d}'
    h5_files = sorted([os.path.join(dir_path, path) for path in os.listdir(dir_path)
                       if path.endswith('Lambda.nxs')])

    data = np.concatenate(list(cxi_loader.load_data(h5_files).values()), axis=-3)
    attrs = cxi_loader.load_attributes(h5_files[0])
    log_data = log_prt.load_data(log_path)

    data = np.concatenate(list(cxi_loader.load_data(h5_files).values()), axis=-3)
    n_steps = min(next(iter(log_data.values())).shape[0], data.shape[0])
    data = data[:n_steps]

    with np.load(os.path.join(ROOT_PATH, 'data/sigray_mask.npz')) as mask_file:
        mask = np.tile(mask_file['mask'][None], (n_steps, 1, 1))

    whitefield = median(data, mask, axis=0)
    db_coord = np.unravel_index(np.argmax(whitefield), whitefield.shape)

    for flip_key in flip_dict:
        if any(flip_key in data_type for data_type in log_data):
            scan_type = [data_type for data_type in log_data if flip_key in data_type][0]
            translations = log_data[scan_type][:n_steps]
            if sum_axis[flip_key]:
                data = np.sum(data[:, :, db_coord[1] - 10:db_coord[1] + 10], axis=2)
                theta = np.linspace(0, data.shape[1], data.shape[1]) - db_coord[0]
                theta *= 36e-5 * attrs['x_pixel_size'] / (2 * np.pi * distance)
            else:
                data = np.sum(data[:, db_coord[0] - 10:db_coord[0] + 10], axis=1)
                theta = np.linspace(data.shape[1], 0, data.shape[1]) - db_coord[1]
                theta *= 36e-5 * attrs['y_pixel_size'] / (2 * np.pi * distance)
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
