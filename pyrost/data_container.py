"""Transforms are common image transformations. They can be chained together using
:class:`pyrost.ComposeTransforms`. You pass a :class:`pyrost.Transform` instance to a data
container :class:`pyrost.CrystData`. All transform classes are inherited from the abstract
:class:`pyrost.Transform` class.
"""
from __future__ import annotations
from configparser import ConfigParser
from dataclasses import dataclass
import os
import re
from typing import (Any, Callable, Dict, Generic, ItemsView, Iterator, List, Optional, Tuple, Type, Union,
                    ValuesView, TypeVar)
import numpy as np

T = TypeVar('T')
Self = TypeVar('Self')

class ReferenceType(Generic[T]):
    __callback__: Callable[[ReferenceType[T]], Any]
    def __new__(cls: type[Self], o: T,
                callback: Optional[Callable[[ReferenceType[T]], Any]]=...) -> Self:
        ...
    def __call__(self) -> Optional[T]:
        ...

D = TypeVar('D', bound='DataContainer')

class DataContainer():
    """Abstract data container class based on :class:`dataclass`. Has :class:`dict` intefrace,
    and :func:`DataContainer.replace` to create a new obj with a set of data attributes replaced.
    """
    def __getitem__(self, attr: str) -> Any:
        return self.__getattribute__(attr)

    def contents(self) -> List[str]:
        """Return a list of the attributes stored in the container that are initialised.

        Returns:
            List of the attributes stored in the container.
        """
        return [attr for attr in self.keys() if self.get(attr) is not None]

    def get(self, attr: str, value: Any=None) -> Any:
        """Retrieve a dataset, return ``value`` if the attribute is not found.

        Args:
            attr : Data attribute.
            value : Data which is returned if the attribute is not found.

        Returns:
            Attribute's data stored in the container, ``value`` if ``attr`` is not found.
        """
        if attr in self.keys():
            return self[attr]
        return value

    def keys(self) -> List[str]:
        """Return a list of the attributes available in the container.

        Returns:
            List of the attributes available in the container.
        """
        return [attr for attr, field in self.__dataclass_fields__.items()
                if str(field._field_type) == '_FIELD']

    def values(self) -> ValuesView:
        """Return the attributes' data stored in the container.

        Returns:
            List of data stored in the container.
        """
        return dict(self).values()

    def items(self) -> ItemsView:
        """Return (key, value) pairs of the datasets stored in the container.

        Returns:
            (key, value) pairs of the datasets stored in the container.
        """
        return dict(self).items()

    def replace(self: D, **kwargs: Any) -> D:
        """Return a new container object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new container object with updated attributes.
        """
        return type(self)(**dict(self, **kwargs))

I = TypeVar('I', bound='INIContainer')

class INIContainer(DataContainer):
    """Abstract data container class based on :class:`dataclass` with an interface to read from
    and write to INI files.
    """
    __ini_fields__ : Dict[str, Union[str, Tuple[str]]]

    @classmethod
    def _format_list(cls, string: str, f: Callable=str) -> List:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return [f(p.strip('\'\"')) for p in re.split(r'\s*,\s*', is_list.group(1)) if p]
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_tuple(cls, string: str, f: Callable=str) -> Tuple:
        is_tuple = re.search(r'^\(([\s\S]*)\)$', string)
        if is_tuple:
            return tuple(f(p.strip('\'\"')) for p in re.split(r'\s*,\s*', is_tuple.group(1)) if p)
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_array(cls, string: str) -> List:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return np.fromstring(is_list.group(1), sep=',')
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_bool(cls, string: str) -> bool:
        return string in ('yes', 'True', 'true', 'T')

    @classmethod
    def get_formatter(cls, t: str) -> Callable:
        _f1 = {'list': cls._format_list, 'List': cls._format_list,
               'tuple': cls._format_tuple, 'Tuple': cls._format_tuple}
        _f2 = {'ndarray': cls._format_array, 'float': float, 'int': int,
               'bool': cls._format_bool, 'complex': complex}
        for k1, f1 in _f1.items():
            if k1 in t:
                idx = t.index(k1) + len(k1)
                for k2, f2 in _f2.items():
                    if k2 in t[idx:]:
                        return lambda string: f1(string, f2)
                return f1
        for k2, f2 in _f2.items():
            if k2 in t:
                return f2
        return str

    @classmethod
    def _format_dict(cls, ini_dict: Dict[str, Any]) -> Dict[str, Any]:
        for attr, val in ini_dict.items():
            formatter = cls.get_formatter(str(cls.__dataclass_fields__[attr].type))
            if isinstance(val, dict):
                ini_dict[attr] = {k: formatter(v) for k, v in val.items()}
            if isinstance(val, str):
                ini_dict[attr] = formatter(val)
        return ini_dict

    @classmethod
    def import_ini(cls: Type[I], ini_file: str) -> I:
        """Initialize the container object with an INI file ``ini_file``.

        Args:
            ini_file : Path to the ini file.

        Returns:
            A new container with all the attributes imported from the ini file.
        """
        if not os.path.isfile(ini_file):
            raise ValueError(f"File {ini_file} doesn't exist")
        ini_parser = ConfigParser()
        ini_parser.read(ini_file)

        ini_dict: Dict[str, Any] = {}
        for section, attrs in cls.__ini_fields__.items():
            if isinstance(attrs, str):
                ini_dict[attrs] = dict(ini_parser[section])
            elif isinstance(attrs, tuple):
                for attr in attrs:
                    ini_dict[attr] = ini_parser[section][attr]
            else:
                raise TypeError(f"Invalid '__ini_fields__' values: {attrs}")

        return cls(**cls._format_dict(ini_dict))

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

    def _get_string(self, attr: Any) -> Union[str, Dict[str, str]]:
        val = self.get(attr)
        if isinstance(val, np.ndarray):
            return np.array2string(val, separator=',')
        if isinstance(val, dict):
            return {k: str(v) for k, v in val.items()}
        return str(val)

    def ini_dict(self) -> Dict[str, Any]:
        ini_dict: Dict[str, Any] = {}
        for section, attrs in self.__ini_fields__.items():
            if isinstance(attrs, str):
                ini_dict[section] = self._get_string(attrs)
            if isinstance(attrs, tuple):
                ini_dict[section] = {attr: self._get_string(attr) for attr in attrs}
        return ini_dict

    def to_ini(self, ini_file: str):
        """Save all the attributes stored in the container to an INI file ``ini_file``.

        Args:
            ini_file : Path to the ini file.
        """
        ini_parser = ConfigParser()
        for section, val in self.ini_dict().items():
            ini_parser[section] = val

        with open(ini_file, 'w') as out_file:
            ini_parser.write(out_file)

class Transform(DataContainer):
    """Abstract transform class."""

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Return a transformed image.

        Args:
            inp : Input image.

        Returns:
            Transformed image.
        """
        ss_idxs, fs_idxs = np.indices(inp.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        return inp[..., ss_idxs, fs_idxs]

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def backward(self, inp: np.ndarray, out: np.ndarray) -> np.ndarray:
        ss_idxs, fs_idxs = np.indices(out.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        out[..., ss_idxs, fs_idxs] = inp
        return out

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

@dataclass
class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Args:
        roi : Region of interest. Comprised of four elements ``[y_min, y_max, x_min, x_max]``.
    """
    roi : Union[List[int], Tuple[int, int, int, int], np.ndarray]

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] == obj.roi[0] and self.roi[1] == obj.roi[1] and \
                   self.roi[2] == obj.roi[2] and self.roi[3] == obj.roi[3]
        return NotImplemented

    def __ne__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] != obj.roi[0] or self.roi[1] != obj.roi[1] or \
                   self.roi[2] != obj.roi[2] or self.roi[3] != obj.roi[3]
        return NotImplemented

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the cropping
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        if ss_idxs.shape[0] == 1:
            return (ss_idxs[:, self.roi[2]:self.roi[3]],
                    fs_idxs[:, self.roi[2]:self.roi[3]])

        if ss_idxs.shape[1] == 1:
            return (ss_idxs[self.roi[0]:self.roi[1], :],
                    fs_idxs[self.roi[0]:self.roi[1], :])

        return (ss_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]],
                fs_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        return x - self.roi[2], y - self.roi[0]

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return x + self.roi[2], y + self.roi[0]

@dataclass
class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Args:
        scale : Downscaling integer ratio.
    """
    scale : int

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the downscaling
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        return (ss_idxs[::self.scale, ::self.scale], fs_idxs[::self.scale, ::self.scale])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        return x / self.scale, y / self.scale

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return x * self.scale, y * self.scale

@dataclass
class Mirror(Transform):
    """Mirror the data around an axis.

    Args:
        axis : Axis of reflection.
        shape : Shape of the input array.
    """
    axis: int
    shape: Tuple[int, int]

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the mirroring
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        if self.axis == 0:
            return (ss_idxs[::-1], fs_idxs[::-1])
        if self.axis == 1:
            return (ss_idxs[:, ::-1], fs_idxs[:, ::-1])
        raise ValueError('Axis must equal to 0 or 1')

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        if self.axis:
            return x, self.shape[0] - y
        return self.shape[1] - x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return self.forward_points(x, y)

@dataclass
class ComposeTransforms(Transform):
    """Composes several transforms together.

    Args:
        transforms: List of transforms.
    """
    transforms : List[Transform]

    def __post_init__(self):
        if len(self.transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = [transform.replace() for transform in self.transforms]

    def __iter__(self) -> Iterator[Transform]:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Union[int, slice]) -> Union[Transform, List[Transform]]:
        return self.transforms[idx]

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the composed transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        for transform in self:
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
        return ss_idxs, fs_idxs

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        for transform in self:
            x, y = transform.forward_points(x, y)
        return x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        for transform in list(self)[::-1]:
            x, y = transform.backward_points(x, y)
        return x, y
