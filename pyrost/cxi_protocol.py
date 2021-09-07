"""Examples
--------
Generate the default built-in CXI protocol as follows:

>>> import pyrost as rst
>>> rst.CXIProtocol()
{'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors':
'float', 'data': 'float', 'defocus': 'float', '...': '...'}, 'default_paths':
{'basis_vectors': '/entry_1/instrument_1/detector_1/basis_vectors',
'data': '/entry_1/data_1/data', 'defocus': '/speckle_tracking/defocus', '...': '...'}}

Or generate the default CXI loader:

>>> rst.CXILoader()
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
import h5py
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .data_processing import STData

CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class CXIProtocol(INIParser):
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
        Dictionary with attributes path in the CXI file.
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
    cxi_protocol : Full list of data attributes and configuration
        parameters.
    """
    known_types = {'int': np.int64, 'bool': np.bool, 'float': np.float64, 'str': str,
                   'uint': np.uint32}
    attr_dict = {'config': ('float_precision', ), 'datatypes': ('ALL', ),
                 'default_paths': ('ALL', )}
    fmt_dict = {'config': 'str','datatypes': 'str', 'default_paths': 'str'}

    def __init__(self, datatypes=None, default_paths=None, float_precision='float64'):
        if datatypes is None:
            datatypes = self._import_ini(CXI_PROTOCOL)['datatypes']
        if default_paths is None:
            default_paths = self._import_ini(CXI_PROTOCOL)['default_paths']
        datatypes = {attr: val for attr, val in datatypes.items() if attr in default_paths}
        default_paths = {attr: val for attr, val in default_paths.items() if attr in datatypes}
        super(CXIProtocol, self).__init__(config={'float_precision': float_precision},
                                          datatypes=datatypes, default_paths=default_paths)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        if not self.config['float_precision'] in ['float32', 'float64']:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    @classmethod
    def import_default(cls, datatypes=None, default_paths=None, float_precision=None):
        """Return the default :class:`CXIProtocol` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file.
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
        return cls.import_ini(CXI_PROTOCOL, datatypes, default_paths, float_precision)

    @classmethod
    def import_ini(cls, ini_file, datatypes=None, default_paths=None,
                   float_precision=None):
        """Initialize a :class:`CXIProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default CXI protocol if None.
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file. Initialized with
            `ini_file` if None.
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
        if float_precision is None:
            float_precision = kwargs['config']['float_precision']
        return cls(datatypes=kwargs['datatypes'], default_paths=kwargs['default_paths'],
                   float_precision=float_precision)

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

    def _read_from_dset(self, attr, dset, idxs=None, dtype=None):
        if idxs is None:
            data = dset[()]
        else:
            data = dset[idxs]
        if np.ndim(data) == 0 or np.size(data) == 1:
            data = self.get_dtype(attr, dtype)(data)
        else:
            data = data.astype(self.get_dtype(attr, dtype))
        return data

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

    def read_shape(self, attr, cxi_file, cxi_path=None):
        """Read `attr` data shapes from the CXI file `cxi_file` at
        the path defined by the protocol. If `cxi_path` or `dtype`
        argument are provided, it will override the protocol.

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
        list of tuples or None
            The attribute's data shapes extracted from the CXI file.
        """
        if cxi_path is None:
            cxi_path = self.get_default_path(attr, cxi_path)
        if cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                return cxi_obj.shape
            elif isinstance(cxi_obj, h5py.Group):
                return [dset.shape for dset in cxi_obj.values()]
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")
        else:
            return None

    def read_cxi(self, attr, cxi_file, idxs=None, cxi_path=None, dtype=None):
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
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                return self._read_from_dset(attr, cxi_obj, idxs, dtype)
            elif isinstance(cxi_obj, h5py.Group):
                if isinstance(idxs, list):
                    return np.stack([self._read_from_dset(attr, dset, didxs, dtype)
                                     for dset, didxs in zip(cxi_obj.values(), idxs)])
                else:
                    return np.stack([self._read_from_dset(attr, dset, idxs, dtype)
                                     for dset in cxi_obj.values()])
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")
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
            if not attr in self:
                raise ValueError(f"protocol doesn't contain the given attribute '{attr:s}'")
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

class CXILoader(CXIProtocol):
    """CXI file loader class. Loads data from a
    CXI file and returns a :class:`STData` container or a
    :class:`dict` with the data. Search data in the paths
    provided by `protocol` and `load_paths`.

    Parameters
    ----------
    protocol : CXIProtocol, optional
        Protocol object. The default protocol is used if None.
        Default protocol is used if not provided.
    load_paths : dict, optional
        Extra paths to the data attributes in a CXI file,
        which override `protocol`. Accepts only the attributes
        enlisted in `protocol`. Default paths are used if
        not provided.
    policy : dict, optional
        A dictionary with loading policy. Contains all the
        attributes that are available in `protocol` with their
        corresponding flags. If a flag is True, the attribute
        will be loaded from a file. Default policy is used if
        not provided.

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
    cxi_protocol : Full list of data attributes and configuration
        parameters.
    STData : Data container with all the data  necessary for
        Speckle Tracking.
    """
    attr_dict = {'config': ('float_precision', ), 'datatypes': ('ALL', ),
                 'default_paths': ('ALL', ), 'load_paths': ('ALL',),
                 'policy': ('ALL', )}
    fmt_dict = {'config': 'str','datatypes': 'str', 'default_paths': 'str',
                'load_paths': 'str', 'policy': 'str'}

    def __init__(self, protocol=None, load_paths=None, policy=None):
        if protocol is None:
            protocol = CXIProtocol()
        if load_paths is None:
            load_paths = self._import_ini(CXI_PROTOCOL)['load_paths']
        if policy is None:
            policy = self._import_ini(CXI_PROTOCOL)['policy']
        load_paths = {attr: paths for attr, paths in load_paths.items()
                      if attr in protocol}
        policy = {attr: flag for attr, flag in policy.items() if attr in protocol}
        super(CXIProtocol, self).__init__(config=protocol.config, datatypes=protocol.datatypes,
                                          default_paths=protocol.default_paths,
                                          load_paths=load_paths, policy=policy)

        if self.config['float_precision'] == 'float32':
            self.known_types['float'] = np.float32
        elif self.config['float_precision'] == 'float64':
            self.known_types['float'] = np.float64
        else:
            raise ValueError('Invalid float precision: {:s}'.format(self.config['float_precision']))

    @classmethod
    def import_default(cls, protocol=None, load_paths=None, policy=None):
        """Return the default :class:`CXILoader` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        protocol : CXIProtocol, optional
            Protocol object.
        load_paths : dict, optional
            Extra paths to the data attributes in a CXI file,
            which override `protocol`. Accepts only the attributes
            enlisted in `protocol`.
        policy : dict, optional
            A dictionary with loading policy. Contains all the
            attributes that are available in `protocol` and the
            corresponding flags. If a flag is True, the attribute
            will be loaded from a file.

        Returns
        -------
        CXILoader
            A :class:`CXILoader` object with the default parameters.

        See Also
        --------
        cxi_protocol : more details about the default CXI loader.
        """
        return cls.import_ini(CXI_PROTOCOL, protocol, load_paths, policy)

    @classmethod
    def import_ini(cls, ini_file, protocol=None, load_paths=None, policy=None):
        """Initialize a :class:`CXILoader` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Loads the default CXI loader if None.
        protocol : CXIProtocol, optional
            Protocol object. Initialized with `ini_file` if None.
        load_paths : dict, optional
            Extra paths to the data attributes in a CXI file,
            which override `protocol`. Accepts only the attributes
            enlisted in `protocol`. Initialized with `ini_file`
            if None.
        policy : dict, optional
            A dictionary with loading policy. Contains all the
            attributes that are available in `protocol` and the
            corresponding flags. If a flag is True, the attribute
            will be loaded from a file. Initialized with `ini_file`
            if None.

        Returns
        -------
        CXILoader
            A :class:`CXILoader` object with all the attributes imported
            from the ini file.

        See Also
        --------
        cxi_protocol : more details about the default CXI loader.
        """
        if protocol is None:
            protocol = CXIProtocol.import_ini(ini_file)
        kwargs = cls._import_ini(ini_file)
        if not load_paths is None:
            kwargs['load_paths'].update(**load_paths)
        if not policy is None:
            kwargs['policy'].update(**policy)
        return cls(protocol=protocol, load_paths=kwargs['load_paths'],
                   policy=kwargs['policy'])

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
        paths = [super(CXILoader, self).get_default_path(attr, value)]
        if attr in self.load_paths:
            paths.extend(self.load_paths[attr])
        return paths

    def get_policy(self, attr, value=None):
        """Return the atrribute's loding policy.

        Parameters
        ----------
        attr : str
            The attribute to look for.
        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        bool
            Attributes' loding policy.
        """
        policy = self.policy.get(attr, value)
        if isinstance(policy, str):
            return policy in ['True', 'true', '1', 'y', 'yes']
        else:
            return bool(policy)

    def get_protocol(self):
        """Return a CXI protocol from the loader.

        Returns
        -------
        CXIProtocol
            CXI protocol.
        """
        return CXIProtocol(datatypes=self.datatypes,
                           default_paths=self.default_paths,
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
            if path in cxi_file:
                return path
        return None

    def load_attributes(self, path, **attributes):
        """Return attributes' values from a CXI file at
        the given `path`.

        Parameters
        ----------
        path : str
            Path to the CXI file.

        Returns
        -------
        attr_dict : dict
            Dictionary with the attributes retrieved from
            the CXI file.
        """
        if not isinstance(path, str):
            raise ValueError('path must be a string')
        attrs = list(self)
        attrs.remove('data')
        attr_dict = {}
        with h5py.File(path, 'r') as cxi_file:
            for attr in attrs:
                if attr in attributes and not attributes[attr] is None:
                    attr_dict[attr] = np.asarray(attributes[attr], dtype=self.get_dtype(attr))
                elif self.get_policy(attr, False):
                    cxi_path = self.find_path(attr, cxi_file)
                    attr_dict[attr] = self.read_cxi(attr, cxi_file, cxi_path=cxi_path)
                else:
                    attr_dict[attr] = None
        if attr_dict.get('defocus'):
            attr_dict['defocus_ss'] = attr_dict['defocus']
            attr_dict['defocus_fs'] = attr_dict['defocus']
        return attr_dict

    def load_data_shape(self, data_files):
        """Retrieve the shape of the main data array from the CXI files.

        Parameters
        ----------
        data_files : str or list of str
            Paths to the data CXI files.

        Returns
        -------
        shapes : dict
            Dictionary of the data shapes for every file in `data_files`.
        """
        if isinstance(data_files, (str, list)):
            if isinstance(data_files, str):
                data_files = [data_files,]
        else:
            raise ValueError('data_files must be a string or a list of strings')
        shapes = {}
        for path in data_files:
            with h5py.File(path, 'r') as cxi_file:
                cxi_path = self.find_path('data', cxi_file)
                shapes[path] = self.read_shape('data', cxi_file, cxi_path)
        return shapes

    def load_data(self, data_files, idxs=None):
        """Retrieve the main data array from the CXI files.

        Parameters
        ----------
        data_files : str or list of str
            Paths to the data CXI files.
        idxs : dict, optional
            Indices of the data to retrieve for the each file in
            `data_files`.

        Returns
        -------
        data : dict
            Data arrays retrieved from the CXI files.
        """
        if isinstance(data_files, (str, list)):
            if isinstance(data_files, str):
                data_files = [data_files,]
        else:
            raise ValueError('data_files must be a string or a list of strings')
        if idxs is None:
            idxs = {path: None for path in data_files}
        data = {}
        for path in data_files:
            with h5py.File(path, 'r') as cxi_file:
                cxi_path = self.find_path('data', cxi_file)
                data[path] = self.read_cxi('data', cxi_file, idxs=idxs[path],
                                           cxi_path=cxi_path)
        return data

    def load_to_dict(self, paths, **attributes):
        """Load CXI files and return a :class:`dict` with
        all the data fetched from the files.

        Parameters
        ----------
        paths : str or list of str
            Paths to CXI files.
        **attributes : dict, optional
            Dictionary of attribute values, that override the loaded
            values.

        Returns
        -------
        dict
            Dictionary with all the data fetched from the CXI files.
        """
        data_dict = {}
        if isinstance(paths, str):
            data_dict.update(self.load_attributes(paths, **attributes))
        elif isinstance(paths, list):
            data_dict.update(self.load_attributes(paths[0], **attributes))
        else:
            raise ValueError('paths must be a string or a list of strings')
        data_dict['data'] = np.concatenate(list(self.load_data(paths).values()), axis=-3)
        return data_dict

    def load(self, path, **attributes):
        """Load a CXI file and return an :class:`STData` class object.

        Parameters
        ----------
        path : str
            Path to the cxi file.
        **attributes : dict
            Dictionary of attribute values,
            which will be parsed to the `STData` object instead.

        Returns
        -------
        STData
            Data container object with all the necessary data
            for the Speckle Tracking algorithm.
        """
        return STData(self.get_protocol(), **self.load_to_dict(path, **attributes))
