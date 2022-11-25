"""CXI protocol (:class:`pyrost.CXIProtocol`) is a helper class for a :class:`pyrost.STData`
data container, which tells it where to look for the necessary data fields in a CXI file. The
class is fully customizable so you can tailor it to your particular data structure of CXI file.

Examples:
    Generate the default built-in CXI protocol as follows:

    >>> import pyrost as rst
    >>> rst.CXIProtocol.import_default()
    CXIProtocol(datatypes={...}, load_paths={...}, kinds={...})
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
import os
from types import TracebackType
from typing import (Any, Callable, ClassVar, Dict, ItemsView, Iterator, KeysView,
                    List, Optional, Tuple, Union, ValuesView)
import h5py
import numpy as np
from tqdm.auto import tqdm
from .data_container import INIContainer

ROOT_PATH = os.path.dirname(__file__)
CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')
Indices = Union[int, slice, np.ndarray, List[int], Tuple[int]]

@dataclass
class CXIProtocol(INIContainer):
    """CXI protocol class. Contains a CXI file tree path with the paths written to all the data
    attributes necessary for the :class:`pyrost.STData` detector data container, their
    corresponding attributes' data types, and data structure.

    Args:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint', or 'bool' are
            allowed.
        load_paths : Dictionary with attributes' CXI default file paths.
        kinds : The attribute's kind, that specifies data dimensionality. The following keywords
            are allowed:

            * `scalar` : Data is either 0D, 1D, or 2D. The data is saved and loaded plainly
              without any transforms or indexing.
            * `sequence` : A time sequence array. Data is either 1D, 2D, or 3D. The data is
              indexed, so the first dimension of the data array must be a time dimension. The
              data points for the given index are not transformed.
            * `frame` : Frame array. Data must be 2D, it may be transformed with any of
              :class:`pyrost.Transform` objects. The data shape is identical to the detector
              pixel grid.
            * `stack` : A time sequnce of frame arrays. The data must be 3D. It's indexed in the
              same way as `sequence` attributes. Each frame array may be transformed with any of
              :class:`pyrost.Transform` objects.
    """
    __ini_fields__ = {'datatypes': 'datatypes', 'load_paths': 'load_paths', 'kinds': 'kinds'}

    datatypes   : Dict[str, str]
    load_paths  : Dict[str, List[str]]
    kinds       : Dict[str, str]

    known_types: ClassVar[Dict[str, Any]] = {'int': np.integer, 'bool': bool, 'float': np.floating,
                                             'str': str, 'uint': np.unsignedinteger}
    default_types: ClassVar[Dict[str, Any]] = {'int': np.int64, 'bool': bool, 'float': np.float64,
                                               'str': str, 'uint': np.uint64}
    known_ndims: ClassVar[Dict[str, Tuple[int, ...]]] = {'stack': (3,), 'frame': (2, 3),
                                                         'sequence': (1, 2, 3), 'scalar': (0, 1, 2)}

    def __post_init__(self):
        self.load_paths = {attr: self.str_to_list(val) for attr, val in self.load_paths.items()
                           if attr in self.datatypes}
        self.kinds = {attr: val for attr, val in self.kinds.items() if attr in self.datatypes}

    @classmethod
    def import_default(cls) -> CXIProtocol:
        """Return the default :class:`CXIProtocol` object.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
        return cls.import_ini(CXI_PROTOCOL)

    def find_path(self, attr: str, cxi_file: h5py.File) -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Attribute's path in the CXI file, returns an empty string if the attribute is not
            found.
        """
        paths = self.get_load_paths(attr, list())
        for path in paths:
            if path in cxi_file:
                return path
        return str()

    def __iter__(self) -> Iterator:
        return self.datatypes.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.datatypes

    def get_load_paths(self, attr: str, value: Optional[List[str]]=None) -> List[str]:
        """Return the attribute's default path in the CXI file. Return ``value`` if ``attr`` is not
        found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.load_paths.get(attr, value)

    def get_dtype(self, attr: str, dtype: Optional[np.dtype]=None) -> type:
        """Return the attribute's data type. Return ``dtype`` if the attribute's data type is not
        found.

        Args:
            attr : The data attribute.
            dtype : Data type which is returned if the attribute's data type is not found.

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
        """Return the attribute's kind, that specifies data dimensionality. Return ``value`` if the
        attribute is not found.

        Args:
            attr : The data attribute.
            value : value which is returned if the ``attr`` is not found.

        Returns:
            The attribute's kind, that specifies data dimensionality. The following keywords are
            allowed:

            * `scalar` : Data is either 0D, 1D, or 2D. The data is saved and loaded plainly without
              any transforms or indexing.
            * `sequence` : A time sequence array. Data is either 1D, 2D, or 3D. The data is indexed,
              so the first dimension of the data array must be a time dimension. The data points
              for the given index are not transformed.
            * `frame` : Frame array. Data must be 2D, it may be transformed with any of
              :class:`pyrost.Transform` objects. The data shape is identical to the detector pixel
              grid.
            * `stack` : A time sequnce of frame arrays. The data must be 3D. It's indexed in the
              same way as `sequence` attributes. Each frame array may be transformed with any of
              :class:`pyrost.Transform` objects.
        """
        return self.kinds.get(attr, value)

    def get_ndim(self, attr: str, value: int=0) -> Tuple[int, ...]:
        """Return the acceptable number of dimensions that the attribute's data may have.

        Args:
            attr : The data attribute.
            value : value which is returned if the ``attr`` is not found.

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
        """Visit recursively all the underlying datasets and return their names and shapes.

        Args:
            cxi_path : Path of the dataset inside the ``cxi_file``.
            cxi_file : HDF5 file handler.

        Returns:
            List of all the datasets and their shapes under the ``cxi_path`` of the ``cxi_file``
            HDF5 file.
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
        """Return a shape of the dataset containing the attribute's data inside a file.

        Args:
            attr : Attribute's name.
            cxi_file : HDF5 file object.

        Returns:
            List of all the datasets and their shapes inside ``cxi_file``.
        """
        cxi_path = self.find_path(attr, cxi_file)
        return self.read_dataset_shapes(cxi_path, cxi_file)

    def read_attribute_indices(self, attr: str, cxi_files: List[h5py.File]) -> np.ndarray:
        """Return a set of indices of the dataset containing the attribute's data inside a set
        of files.

        Args:
            attr : Attribute's name.
            cxi_files : A list of HDF5 file objects.

        Returns:
            Dataset indices of the data pertined to the attribute ``attr``.
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

def initializer(read_worker: Callable[[np.ndarray, Indices, Indices], np.ndarray],
                ss_idxs: np.ndarray, fs_idxs: np.ndarray):
    global worker
    worker = partial(read_worker, ss_idxs=ss_idxs, fs_idxs=fs_idxs)

def read_frame(index: np.ndarray) -> np.ndarray:
    return worker(index)

class CXIStore():
    """File handler class for HDF5 and CXI files. Provides an interface to save and load data
    attributes to a file. Support multiple files. The handler saves data to the first file.

    Args:
        names : Paths to the files.
        mode : Mode in which to open file; one of ('w', 'r', 'r+', 'a', 'w-').
        protocol : CXI protocol. Uses the default protocol if not provided.

    Attributes:
        file_dict : Dictionary of paths to the files and their file
            objects.
        file : :class:`h5py.File` object of the first file.
        protocol : :class:`pyrost.CXIProtocol` protocol object.
        mode : File mode. Valid modes are:

            * 'r' : Readonly, file must exist (default).
            * 'r+' : Read/write, file must exist.
            * 'w' : Create file, truncate if exists.
            * 'w-' or 'x' : Create file, fail if exists.
            * 'a' : Read/write if exists, create otherwise.
    """

    def __init__(self, names: Union[str, List[str]], mode: str='r',
                 protocol: CXIProtocol=CXIProtocol.import_default()) -> None:
        if mode not in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError(f'Wrong file mode: {mode}')
        names = protocol.str_to_list(names)

        self.file_dict: Dict[str, Optional[h5py.File]] = {data_file: None for data_file in names}
        self.protocol = protocol
        self.mode = mode

        with self:
            self.update_indices()

    def __bool__(self) -> bool:
        isopen = True
        for cxi_file in self.files():
            isopen &= bool(cxi_file)
        return isopen

    def __contains__(self, attr: str) -> bool:
        return attr in self._indices

    def __iter__(self) -> Iterator:
        return self._indices.__iter__()

    def __len__(self) -> int:
        return len(self._indices)

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
        """Read the files for the data attributes contained in the protocol."""
        indices = {}
        if self:
            for attr in self.protocol:
                kind = self.protocol.get_kind(attr)
                try:
                    if kind in ['stack', 'sequence', 'scalar']:
                        idxs = self.protocol.read_attribute_indices(attr, self.files())
                    if kind == 'frame':
                        idxs = self.protocol.read_attribute_indices(attr, self.files())
                except ValueError as err:
                    print(f'{attr:s} is not loaded: {err}')
                else:
                    if idxs.size:
                        indices[attr] = idxs

        self._indices = indices

    @property
    def file(self) -> h5py.File:
        return self.files()[0]

    def open(self) -> None:
        """Open the files."""
        if not self:
            for data_file in self.file_dict:
                self.file_dict[data_file] = h5py.File(data_file, mode=self.mode)

    def close(self) -> None:
        """Close the files."""
        for cxi_file in self.files():
            cxi_file.close()

    def filenames(self) -> List[str]:
        """Return a list of paths to the files.

        Returns:
            List of paths to the files.
        """
        return list(self.file_dict.keys())

    def files(self) -> List[h5py.File]:
        """Return a list of file objects of the files.

        Returns:
            List of file objects of the files.
        """
        return list(self.file_dict.values())

    def indices(self) -> np.ndarray:
        """Return a list of frame indices of the data contained in the files.

        Returns:
            A list of frame indices of the data in the files.
        """
        return np.arange(self._indices.get('data', np.array([])).shape[0])

    def keys(self) -> KeysView:
        """Return a set of the attribute's names contained in the files.

        Returns:
            A set of the attribute's names in the files.
        """
        return self._indices.keys()

    def values(self) -> ValuesView:
        """Return a set of the attribute's indices in the files.

        Returns:
            A set of the attribute's indices in the files.
        """
        return self._indices.values()

    def items(self) -> ItemsView:
        """Return a set of ``(name, index)`` pairs. ``name`` is the attribute's name contained in
        the files, and ``index`` it's corresponding set of indices.

        Returns:
            A set of ``(name, index)`` pairs of the data attributes in the files.
        """
        return self._indices.items()

    def read_shape(self) -> Tuple[int, int]:
        """Read the input files and return a shape of the `frame` type data attribute.

        Raises:
            RuntimeError : If the files are not opened.

        Returns:
            The shape of the 2D `frame`-like data attribute.
        """
        if self:
            for attr, indices in self._indices.items():
                kind = self.protocol.get_kind(attr)
                if kind in ['stack', 'frame']:
                    return self.file_dict[indices[0, 0]][indices[0, 1]].shape[-2:]

            return (0, 0)

        raise RuntimeError('Invalid file objects: the file is closed')

    @staticmethod
    def _read_worker_sequence(index: np.ndarray) -> np.ndarray:
        return h5py.File(index[0])[index[1]][index[2]]

    @staticmethod
    def _read_worker_frame(index: np.ndarray, ss_idxs: Indices, fs_idxs: Indices) -> np.ndarray:
        return h5py.File(index[0])[index[1]][index[2]][..., ss_idxs, fs_idxs]

    def _load_stack(self, attr: str, idxs: Optional[Indices], ss_idxs: Indices,
                    fs_idxs: Indices, processes: int, verbose: bool) -> np.ndarray:
        stack = []
        if idxs is None:
            idxs = self.indices()
        idxs = np.atleast_1d(idxs)

        with Pool(processes=processes, initializer=initializer,
                  initargs=(type(self)._read_worker_frame, ss_idxs, fs_idxs)) as pool:
            for frame in tqdm(pool.imap(read_frame, self._indices[attr][idxs]),
                              total=self.indices()[idxs].size, disable=not verbose,
                              desc=f'Loading {attr:s}'):
                stack.append(frame)

        return self.protocol.cast(attr, np.stack(stack, axis=0))

    def _load_frame(self, attr: str, ss_idxs: Indices, fs_idxs: Indices) -> np.ndarray:
        dset = self._read_worker_frame(self._indices[attr][0], ss_idxs, fs_idxs)
        return self.protocol.cast(attr, dset)

    def _load_sequence(self, attr: str, idxs: Optional[Indices]) -> np.ndarray:
        sequence = []
        if idxs is None:
            idxs = self.indices()
        idxs = np.atleast_1d(idxs)

        for index in self._indices[attr][idxs]:
            sequence.append(self._read_worker_sequence(index))

        return self.protocol.cast(attr, np.array(sequence))

    def load_attribute(self, attr: str, idxs: Optional[Indices]=None, processes: int=1,
                       ss_idxs: Indices=slice(None), fs_idxs: Indices=slice(None),
                       verbose: bool=True) -> np.ndarray:
        """Load a data attribute from the files.

        Args:
            attr : Attribute's name to load.
            idxs : A list of frames' indices to load.
            processes : Number of parallel workers used during the loading.
            ss_idxs : Slow (vertical) axis indices used to load the data attributes of
                `frame` and `stack` types.
            fs_idxs : Fast (horizontal) axis indices used to lead the data attributes
                of `frame` and `stack` types.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If the attribute's kind is invalid.
            RuntimeError : If the files are not opened.

        Returns:
            Attribute's data array.
        """
        kind = self.protocol.get_kind(attr)

        if kind not in ('stack', 'frame', 'scalar', 'sequence'):
            raise ValueError(f'Invalid kind: {kind:s}')

        if self:
            if kind == 'stack':
                return self._load_stack(attr=attr, idxs=idxs, processes=processes,
                                        ss_idxs=ss_idxs, fs_idxs=fs_idxs, verbose=verbose)
            if kind == 'frame':
                return self._load_frame(attr=attr, ss_idxs=ss_idxs, fs_idxs=fs_idxs)
            if kind == 'scalar':
                return self._load_sequence(attr, idxs=0)
            if kind == 'sequence':
                return self._load_sequence(attr=attr, idxs=idxs)

        raise RuntimeError('Invalid file objects: the file is closed')

    def find_dataset(self, attr: str) -> str:
        """Return the path to the attribute from the first file. Return the default path if the
        attribute is not found inside the first file.

        Args:
            attr : Attribute's name.

        Returns:
            Path to the attribute inside the first file.
        """
        cxi_path = self.protocol.find_path(attr, self.file)

        if cxi_path:
            return cxi_path
        return self.protocol.get_load_paths(attr)[0]

    def _save_stack(self, attr: str, data: np.ndarray, mode: str='overwrite',
                    idxs: Optional[Indices]=None) -> None:
        cxi_path = self.find_dataset(attr)

        if cxi_path in self.file and self.file[cxi_path].shape[1:] == data.shape[1:]:
            if mode == 'append':
                self.file[cxi_path].resize(self.file[cxi_path].shape[0] + data.shape[0],
                                               axis=0)
                self.file[cxi_path][-data.shape[0]:] = data
            elif mode == 'overwrite':
                self.file[cxi_path].resize(data.shape[0], axis=0)
                self.file[cxi_path][...] = data
            elif mode == 'insert':
                if idxs is None or len(idxs) != data.shape[0]:
                    raise ValueError('Incompatible indices')
                self.file[cxi_path].resize(max(self.file[cxi_path].shape[0], max(idxs) + 1),
                                           axis=0)
                self.file[cxi_path][idxs] = data

        else:
            if cxi_path in self.file:
                del self.file[cxi_path]
            self.file.create_dataset(cxi_path, data=data, shape=data.shape,
                                     chunks=(1,) + data.shape[1:],
                                     maxshape=(None,) + data.shape[1:],
                                     dtype=self.protocol.get_dtype(attr, data.dtype))

    def _save_data(self, attr: str, data: np.ndarray) -> None:
        cxi_path = self.find_dataset(attr)

        if cxi_path in self.file and self.file[cxi_path].shape == data.shape:
            self.file[cxi_path][...] = data

        else:
            if cxi_path in self.file:
                del self.file[cxi_path]
            self.file.create_dataset(cxi_path, data=data, shape=data.shape,
                                     dtype=self.protocol.get_dtype(attr, data.dtype))

    def save_attribute(self, attr: str, data: np.ndarray, mode: str='overwrite',
                       idxs: Optional[Indices]=None) -> None:
        """Save a data array pertained to the data attribute into the first file.

        Args:
            attr : Attribute's name.
            data : Data array.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices ``idxs``.
                * `overwrite` : Overwrite the existing dataset.

            idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

        Raises:
            ValueError : If the attribute's kind is invalid.
            ValueError : If the file is opened in read-only mode.
            RuntimeError : If the file is not opened.
        """
        if self.mode == 'r':
            raise ValueError('File is open in read-only mode')
        kind = self.protocol.get_kind(attr)

        if self:
            if kind in ['stack', 'sequence']:
                return self._save_stack(attr=attr, data=data, mode=mode, idxs=idxs)

            if kind in ['frame', 'scalar']:
                return self._save_data(attr=attr, data=data)

            raise ValueError(f'Invalid kind: {kind:s}')

        raise RuntimeError('Invalid file objects: the file is closed')
