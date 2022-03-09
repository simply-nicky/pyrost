"""CXI protocol (:class:`pyrost.CXIProtocol`) is a helper class for a :class:`pyrost.STData`
data container, which tells it where to look for the necessary data fields in a CXI
file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file.

Examples:
    Generate the default built-in CXI protocol as follows:

    >>> import pyrost as rst
    >>> rst.CXIProtocol.import_default()
    {'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors': 'float',
    'data': 'uint', 'defocus_x': 'float', '...': '...'}, 'default_paths': {'basis_vectors':
    '/speckle_tracking/basis_vectors', 'data': '/entry/data/data', 'defocus_x':
    '/speckle_tracking/defocus_x', '...': '...'}, 'is_data': {'basis_vectors': 'False',
    'data': 'True', 'defocus_x': 'False', '...': '...'}}
"""
from __future__ import annotations
from multiprocessing import Pool
import os
from configparser import ConfigParser
from types import TracebackType
from typing import (Dict, ItemsView, Iterable, Iterator, KeysView,
                    List, Optional, Tuple, Union, ValuesView)
import h5py
import numpy as np
from tqdm.auto import tqdm
from .ini_parser import ROOT_PATH, INIParser

CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class CXIProtocol(INIParser):
    """CXI protocol class. Contains a CXI file tree path with
    the paths written to all the data attributes necessary for
    the Speckle Tracking algorithm, corresponding attributes'
    data types, and floating point precision.

    Attributes:
        config : Protocol configuration.
        datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
            or 'bool' are allowed.
        is_data : Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`.
        load_paths : Dictionary with attributes' CXI default file paths.
    """
    known_types = {'int': np.integer, 'bool': bool, 'float': np.floating, 'str': str,
                   'uint': np.unsignedinteger}
    default_types = {'int': np.int64, 'bool': bool, 'float': np.float64, 'str': str,
                     'uint': np.uint64}
    known_ndims = {'stack': (3,), 'frame': (2, 3), 'sequence': (1, 2, 3), 'scalar': (0, 1, 2)}
    attr_dict = {'datatypes': ('ALL', ), 'load_paths': ('ALL', ), 'kinds': ('ALL', )}
    fmt_dict = {'datatypes': 'str', 'load_paths': 'str', 'kinds': 'str'}

    datatypes   : Dict[str, str]
    load_paths  : Dict[str, List[str]]
    kinds       : Dict[str, str]

    def __init__(self, datatypes: Dict[str, str], load_paths: Dict[str, Union[str, List[str]]],
                 kinds: Dict[str, str]) -> None:
        """
        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed.
            load_paths : Dictionary with attributes path in the CXI file.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`.
        """
        load_paths = {attr: self.str_to_list(val)
                      for attr, val in load_paths.items() if attr in datatypes}
        kinds = {attr: val for attr, val in kinds.items() if attr in datatypes}
        super(CXIProtocol, self).__init__(datatypes=datatypes, load_paths=load_paths,
                                          kinds=kinds)

    @classmethod
    def import_default(cls, datatypes: Optional[Dict[str, str]]=None,
                       load_paths: Optional[Dict[str, Union[str, List[str]]]]=None,
                       kinds: Optional[Dict[str, str]]=None) -> CXIProtocol:
        """Return the default :class:`CXIProtocol` object. Extra arguments
        override the default values if provided.

        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed.
            load_paths : Dictionary with attributes path in the CXI file.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
        return cls.import_ini(CXI_PROTOCOL, datatypes, load_paths, kinds)

    @classmethod
    def import_ini(cls, ini_file: str, datatypes: Optional[Dict[str, str]]=None,
                   load_paths: Optional[Dict[str, Union[str, List[str]]]]=None,
                   kinds: Optional[Dict[str, str]]=None) -> CXIProtocol:
        """Initialize a :class:`CXIProtocol` object class with an
        ini file.

        Args:
            ini_file : Path to the ini file. Load the default CXI protocol if None.
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed. Initialized with `ini_file` if None.
            load_paths : Dictionary with attributes path in the CXI file. Initialized
                with `ini_file` if None.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`. Initialized with `ini_file` if None.

        Returns:
            A :class:`CXIProtocol` object with all the attributes imported
            from the ini file.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not load_paths is None:
            kwargs['load_paths'].update(**load_paths)
        if not kinds is None:
            kwargs['kinds'].update(**kinds)
        return cls(datatypes=kwargs['datatypes'], load_paths=kwargs['load_paths'],
                   kinds=kwargs['kinds'])

    def find_path(self, attr: str, cxi_file: h5py.File) -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Atrribute's path in the CXI file, returns an empty
            string if the attribute is not found.
        """
        paths = self.get_load_paths(attr, list())
        for path in paths:
            if path in cxi_file:
                return path
        return str()

    def parser_from_template(self, path: str) -> ConfigParser:
        """Return a :class:`configparser.ConfigParser` object using
        an ini file template.

        Args:
            path : Path to the ini file template.

        Returns:
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
        return self.datatypes.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.datatypes

    def get_load_paths(self, attr: str, value: Optional[List[str]]=None) -> List[str]:
        """Return the atrribute's default path in the CXI file.
        Return `value` if `attr` is not found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.load_paths.get(attr, value)

    def get_dtype(self, attr: str, dtype: Optional[np.dtype]=None) -> type:
        """Return the attribute's data type. Return `dtype` if the attribute's
        data type is not found.

        Args:
            attr : The data attribute.
            dtype : Data type which is returned if the attribute's data type
                is not found.

        Returns:
            Attribute's data type.
        """
        if attr not in self:
            raise ValueError(f'Invalid attribute: {attr:s}')

        attr_type = self.known_types.get(self.datatypes.get(attr))
        if dtype is not None and np.issubdtype(dtype, attr_type):
            return dtype
        return self.default_types.get(self.datatypes.get(attr))

    def get_kind(self, attr: str, value: str='scalar') -> str:
        """Return if the attribute is of data type. Data type is 2- or
        3-dimensional and has the same data shape as `data`.

        Args:
            attr : The data attribute.
            value : value which is returned if the `attr` is not found.

        Returns:
            True if `attr` is of data type.
        """
        return self.kinds.get(attr, value)

    def get_ndim(self, attr: str, value: int=0) -> Tuple[int, ...]:
        """Return the acceptable number of dimenstions that the attribute's data
        may have.

        Args:
            attr : The data attribute.
            value : value which is returned if the `attr` is not found.

        Returns:
            Number of dimensions acceptable for the attribute.
        """
        return self.known_ndims.get((self.get_kind(attr)), value)

    def cast(self, attr: str, array: np.ndarray) -> np.ndarray:
        """Cast the attribute's data to the right data type.

        Args:
            attr : The data attribute.
            array : The attribute's data.

        Returns:
            Data array casted to the right data type.
        """
        return np.asarray(array, dtype=self.get_dtype(attr, array.dtype))

    @staticmethod
    def read_dataset_shapes(cxi_path: str, cxi_file: h5py.File) -> Dict[str, Tuple[int, ...]]:
        """Visit recursively all the underlying datasets and return
        their names and shapes.

        Args:
            cxi_path : Path of the HDF5 group.
            cxi_obj: Group object.

        Returns:
            List of all the datasets and their shapes inside `cxi_obj`.
        """
        shapes = {}

        def caller(sub_path, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[os.path.join(cxi_path, sub_path)] = obj.shape

        if cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                shapes[cxi_path] = cxi_obj.shape
            elif isinstance(cxi_obj, h5py.Group):
                cxi_obj.visititems(caller)
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")

        return shapes

    def read_attribute_shapes(self, attr: str, cxi_file: h5py.File) -> Dict[str, Tuple[int, ...]]:
        """Return a shape of the dataset containing the attribute's data inside
        a file.

        Args:
            attr : Attribute's name.
            cxi_file : HDF5 file object.

        Returns:
            List of all the datasets and their shapes inside `cxi_file`.
        """
        cxi_path = self.find_path(attr, cxi_file)
        return self.read_dataset_shapes(cxi_path, cxi_file)

    def read_attribute_indices(self, attr: str, cxi_files: List[h5py.File]) -> np.ndarray:
        """Return a set of indices of the dataset containing the attribute's data
        inside a set of files.

        Args:
            attr : Attribute's name.
            cxi_files : A list of HDF5 file objects.

        Returns:
            Dataset indices of the data pertined to the attribute `attr`.
        """
        files, cxi_paths, fidxs = [], [], []
        kind = self.get_kind(attr)

        for cxi_file in cxi_files:
            shapes = self.read_attribute_shapes(attr, cxi_file)
            for cxi_path, shape in shapes.items():
                if len(shape) not in self.get_ndim(attr):
                    err_txt = f'Dataset at {cxi_file.filename}:'\
                              f' {cxi_path} has invalid shape: {str(shape)}'
                    raise ValueError(err_txt)

                if kind in ['stack', 'sequence']:
                    files.extend(np.repeat(cxi_file.filename, shape[0]).tolist())
                    cxi_paths.extend(np.repeat(cxi_path, shape[0]).tolist())
                    fidxs.extend(np.arange(shape[0]).tolist())
                if kind in ['frame', 'scalar']:
                    files.append(cxi_file.filename)
                    cxi_paths.append(cxi_path)
                    fidxs.append(tuple())

        return np.array([files, cxi_paths, fidxs], dtype=object).T

class CXIStore():
    """File handler class for HDF5 and CXI files. Provides an interface to
    save and load data attributes to a file. Support multiple input files.
    Input and output file can be the same file.

    Attributes:
        inp_dict : Dictionary of paths to the input files and their file
            objects.
        out_fname : Path to the output file.
        out_file : Output file object.
    """

    out_file : h5py.File
    inp_dict : Dict[str, h5py.File]

    def __init__(self, input_files: Union[str, List[str]], output_file: str,
                 protocol: CXIProtocol=CXIProtocol.import_default()) -> None:
        """
        Args:
            input_files : Paths to the input files.
            output_file : Path to the output file.
            protocol : CXI protocol. Uses the default protocol if not provided.
        """
        input_files = protocol.str_to_list(input_files)

        self.out_fname, self.out_file = output_file, None

        self.inp_dict = {data_file: None for data_file in input_files}
        self.protocol = protocol

        with self:
            self.update_indices()

    def __bool__(self) -> bool:
        isopen = True
        for cxi_file in self.input_files():
            isopen &= bool(cxi_file)
        return isopen & bool(self.out_file)

    def __contains__(self, attr: str) -> bool:
        return attr in self._indices

    def __iter__(self) -> Iterable:
        return self._indices.__iter__()

    def __repr__(self) -> str:
        return self._indices.__repr__()

    def __str__(self) -> str:
        return self._indices.__str__()

    def __enter__(self) -> CXIStore:
        self.open()
        return self

    def __exit__(self, exc_type: Optional[BaseException], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.close()

    def update_indices(self) -> None:
        """Read input files for the data attributes contained in the protocol."""
        indices = {}
        if self:
            for attr in self.protocol:
                kind = self.protocol.get_kind(attr)
                try:
                    if kind in ['stack', 'sequence', 'scalar']:
                        idxs = self.protocol.read_attribute_indices(attr, self.input_files())
                    if kind == 'frame':
                        idxs = self.protocol.read_attribute_indices(attr, self.input_files())
                except ValueError as err:
                    print(f'{attr:s} is not loaded: {err}')
                else:
                    if idxs.size:
                        indices[attr] = idxs

        self._indices = indices

    def open(self) -> None:
        """Open input and output files to read and write."""
        if not self:
            self.out_file = h5py.File(self.out_fname, mode='a')

            for data_file in self.inp_dict:
                if data_file == self.out_fname:
                    self.inp_dict[data_file] = self.out_file
                else:
                    self.inp_dict[data_file] = h5py.File(data_file, mode='r')


    def close(self) -> None:
        """Close input and output files."""
        for cxi_file in self.input_files():
            cxi_file.close()
        self.out_file.close()

    def input_filenames(self) -> List[str]:
        """Return a list of paths to the input files.

        Returns:
            List of paths to the input files.
        """
        return list(self.inp_dict.keys())

    def input_files(self) -> List[h5py.File]:
        """Return a list of file objects of the input files.

        Returns:
            List of file objects of the input files.
        """
        return list(self.inp_dict.values())

    def indices(self) -> np.ndarray:
        """Return a list of frame indices of the data contained in the input
        files.

        Returns:
            A list of frame indices of the data in the input files.
        """
        return np.arange(self._indices.get('data', np.array([])).shape[0])

    def keys(self) -> KeysView:
        """Return a set of the attribute's names contained in the input files.

        Returns:
            A set of the attribute's names in the input files.
        """
        return self._indices.keys()

    def values(self) -> ValuesView:
        """Return a set of the attribute's indices in the input files.

        Returns:
            A set of the attribute's indices in the input files.
        """
        return self._indices.values()

    def items(self) -> ItemsView:
        """Return a set of `(name, index)` pairs. `name` is the attribute's name
        contained in the input files, and `index` it's corresponding set of indices.

        Returns:
            A set of `(name, index)` pairs of the data attributes in the input files.
        """
        return self._indices.items()

    def _read_chunk(self, index: np.ndarray) -> np.ndarray:
        return self.inp_dict[index[0]][index[1]][index[2]]

    @staticmethod
    def _read_worker(index: np.ndarray) -> np.ndarray:
        with h5py.File(index[0], 'r') as cxi_file:
            return cxi_file[index[1]][index[2]]

    def _load_stack(self, attr: str, indices: Optional[Iterable[int]]=None, processes: int=1,
                    verbose: bool=True) -> np.ndarray:
        stack = []
        if indices is None:
            indices = self.indices()

        with Pool(processes=processes) as pool:
            for frame in tqdm(pool.imap(type(self)._read_worker, self._indices[attr][indices]),
                              total=self.indices()[indices].size, disable=not verbose,
                              desc=f'Loading {attr:s}'):
                stack.append(frame)

        return self.protocol.cast(attr, np.stack(stack, axis=0))

    def _load_frame(self, attr: str) -> np.ndarray:
        return self.protocol.cast(attr, self._read_chunk(self._indices[attr][0]))

    def _load_sequence(self, attr: str, indices: Optional[Iterable[int]]=None) -> np.ndarray:
        sequence = []
        if indices is None:
            indices = self.indices()

        for index in self._indices[attr][indices]:
            sequence.append(self._read_chunk(index))

        return self.protocol.cast(attr, np.array(sequence))

    def load_attribute(self, attr: str, indices: Optional[Iterable[int]]=None, processes: int=1,
                       verbose: bool=True) -> np.ndarray:
        """Load a data attribute from the input files.

        Args:
            attr : Attribute's name to load.
            indices : A list of frames' indices to load.
            processes : Number of parallel workers used during the loading.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If the attribute's kind is invalid.

        Returns:
            Attribute's data array.
        """
        kind = self.protocol.get_kind(attr)

        if self:
            if kind == 'stack':
                return self._load_stack(attr=attr, indices=indices, processes=processes,
                                        verbose=verbose)
            elif kind in ['frame', 'scalar']:
                return self._load_frame(attr=attr)
            elif kind == 'sequence':
                return self._load_sequence(attr=attr, indices=indices)
            else:
                raise ValueError(f'Invalid kind: {kind:s}')

        return np.array([], dtype=self.protocol.get_dtype(attr))

    def find_dataset(self, attr: str) -> str:
        """Return the path to the attribute from the output file. Return the default
        path if the attribute is not found inside the output file.

        Args:
            attr : Attribute's name.

        Returns:
            Path to the attribute inside the output file.
        """
        cxi_path = self.protocol.find_path(attr, self.out_file)

        if cxi_path:
            return cxi_path
        return self.protocol.get_load_paths(attr)[0]

    def _save_stack(self, attr: str, data: np.ndarray, mode: str='overwrite',
                    idxs: Optional[Iterable[int]]=None) -> None:
        cxi_path = self.find_dataset(attr)

        if cxi_path in self.out_file and self.out_file[cxi_path].shape[1:] == data.shape[1:]:
            if mode == 'append':
                self.out_file[cxi_path].resize(self.out_file[cxi_path].shape[0] + data.shape[0],
                                               axis=0)
                self.out_file[cxi_path][-data.shape[0]:] = data
            elif mode == 'overwrite':
                self.out_file[cxi_path].resize(data.shape[0], axis=0)
                self.out_file[cxi_path][...] = data
            elif mode == 'insert':
                if idxs is None or len(idxs) != data.shape[0]:
                    raise ValueError('Incompatible indices')
                self.out_file[cxi_path].resize(max(self.out_file[cxi_path].shape[0], max(idxs)),
                                               axis=0)
                self.out_file[cxi_path][idxs] = data

        else:
            if cxi_path in self.out_file:
                del self.out_file[cxi_path]
            self.out_file.create_dataset(cxi_path, data=data, shape=data.shape,
                                         chunks=(1,) + data.shape[1:],
                                         maxshape=(None,) + data.shape[1:],
                                         dtype=self.protocol.get_dtype(attr, data.dtype))

    def _save_data(self, attr: str, data: np.ndarray) -> None:
        cxi_path = self.find_dataset(attr)

        if cxi_path in self.out_file and self.out_file[cxi_path].shape == data.shape:
            self.out_file[cxi_path][...] = data

        else:
            if cxi_path in self.out_file:
                del self.out_file[cxi_path]
            self.out_file.create_dataset(cxi_path, data=data, shape=data.shape,
                                         dtype=self.protocol.get_dtype(attr, data.dtype))

    def save_attribute(self, attr: str, data: np.ndarray, mode: str='overwrite',
                       idxs: Optional[Iterable[int]]=None) -> None:
        """Save a data array pertained to the data attribute into the output file.

        Args:
            attr : Attribute's name.
            data : Data array.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices `idxs`.
                * `overwrite` : Overwrite the existing dataset.

            idxs : Indices where the data is saved. Used only if `mode` is set to
                'insert'.

        Raises:
            ValueError : If the attribute's kind is invalid.
        """
        kind = self.protocol.get_kind(attr)

        if self:
            if kind in ['stack', 'sequence']:
                return self._save_stack(attr=attr, data=data, mode=mode, idxs=idxs)
            elif kind in ['frame', 'scalar']:
                return self._save_data(attr=attr, data=data)
            else:
                raise ValueError(f'Invalid kind: {kind:s}')

        raise ValueError('Invalid file objects: the output file is closed')
