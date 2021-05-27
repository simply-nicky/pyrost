import os
import re
import numpy as np
from .ini_parser import ROOT_PATH, INIParser

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

    def __init__(self, log_keys=None, datatypes=None):
        if log_keys is None:
            log_keys = {}
        if datatypes is None:
            datatypes = {}
        log_keys = {attr: val for attr, val in log_keys.items() if attr in datatypes}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in log_keys}
        super(LogProtocol, self).__init__(log_keys=log_keys, datatypes=datatypes)

    @classmethod
    def import_ini(cls, ini_file=None, datatypes=None, log_keys=None):
        """Initialize a :class:`LogProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str, optional
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
        if ini_file is None:
            ini_file = LOG_PROTOCOL
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not log_keys is None:
            kwargs['log_keys'].update(**log_keys)
        return cls(datatypes=kwargs['datatypes'], log_keys=kwargs['log_keys'])

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
        with open(path, 'r') as log_file:
            log_str = ''
            for line in log_file:
                if line.startswith('# '):
                    log_str += line.strip('# ')

        # Divide log into sectors
        parts = [part for part in re.split('(' + \
                 '|'.join([key[0] for key in self.log_keys.values()]) + \
                 '|--------------------------------)\n*', log_str) if part]

        # Rearrange logged attributes sector
        if 'Session logged attributes' in parts:
            idx = parts.index('Session logged attributes') + 1
            attr_keys, attr_vals = parts[idx].strip('\n').split('\n')
            parts[idx] = ''
            for key, val in zip(attr_keys.split(';'), attr_vals.split(';')):
                parts[idx] += key + ': ' + val + '\n'

        # Populate attributes dictionary
        attr_dict = {}
        for attr, [log_type, log_key] in self.log_keys.items():
            part = parts[parts.index(log_type) + 1]
            val_str = re.search(log_key + r'.*\n', part)[0].split(': ')[-1][:-1]
            val_m = re.search(r'\d+[.]*\d*', val_str)
            dtype = self.known_types[self.datatypes[attr]]
            attr_dict[attr] = dtype(val_m[0] if val_m else val_str)
        return attr_dict

    def load_data(self, path):
        """Retrieve the main data array from the log file.

        Parameters
        ----------
        path : str
            Path to the log file.

        Returns
        -------
        data : np.ndarray
            Data array retrieved from the log file.
        """
        return np.loadtxt(path, usecols=2, delimiter=';')
