"""Transforms are common image transformations. They can be chained together
using :class:`pyrost.ComposeTransforms`. You pass a :class:`pyrost.Transform`
instance to a data container :class:`pyrost.STData`. All transform classes
are inherited from the abstract :class:`pyrost.Transform` class.

:class:`pyrost.STData` contains all the necessary data for the Speckle
Tracking algorithm, and provides a suite of data processing tools to work
with the data.

Examples:
    Load all the necessary data using a :func:`pyrost.STData.load` function.

    >>> import pyrost as rst
    >>> files = rst.CXIStore(input_files='data.cxi', output_file='data.cxi')
    >>> data = rst.STData(files)
    >>> data = data.load()
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from weakref import ref
from multiprocessing import cpu_count
from tqdm.auto import tqdm
import numpy as np
from .aberrations_fit import AberrationsFit
from .data_container import DataContainer, dict_to_object
from .cxi_protocol import CXIStore
from .rst_update import SpeckleTracking
from .bin import median, median_filter, fft_convolve, ct_integrate

Indices = Union[int, slice]

class Transform():
    """Abstract transform class.

    Attributes:
        shape : Data frame shape.

    Raises:
        AttributeError : If shape isn't initialized.
    """
    def __init__(self, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            shape : Data frame shape.
        """
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, int]):
        if self._shape is None:
            self._shape = value
        else:
            raise ValueError("Shape is already defined.")

    def check_shape(self, shape: Tuple[int, int]) -> bool:
        """Check if shape is equal to the saved shape.

        Args:
            shape : shape to check.

        Returns:
            True if the shapes are equal.
        """
        if self.shape is None:
            self.shape = shape
            return True
        return self.shape == shape

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        raise NotImplementedError

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def integrate(self, axis: int) -> Transform:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return type(self)(**pdict)

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[y_min, y_max,
            x_min, x_max]`.
        shape : Data frame shape.
    """
    def __init__(self, roi: Iterable[int], shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            roi : Region of interest. Comprised of four elements `[y_min, y_max,
                x_min, x_max]`.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        self.roi = roi

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return inp[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return pts - self.roi[::2]

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.zeros(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = inp
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return pts + self.roi[::2]

    def integrate(self, axis: int) -> Crop:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        if axis == 0:
            pdict['roi'] = (0, 1, pdict['roi'][2], pdict['roi'][3])
        elif axis == 1:
            pdict['roi'] = (pdict['roi'][0], pdict['roi'][1], 0, 1)
        else:
            raise ValueError('Axis must be equal to 0 or 1')

        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return Crop(**pdict)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'roi': self.roi[:], 'shape': self.shape}

class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Attributes:
        scale : Downscaling integer ratio.
        shape : Data frame shape.
    """
    def __init__(self, scale: int, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            scale : Downscaling integer ratio.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        self.scale = scale

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return inp[..., ::self.scale, ::self.scale]

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return pts / self.scale

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.empty(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[...] = np.repeat(np.repeat(inp, self.scale, axis=-2),
                                 self.scale, axis=-1)[..., :self.shape[0], :self.shape[1]]
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return pts * self.scale

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'scale': self.scale, 'shape': self.shape}

class Mirror(Transform):
    """Mirror the data around an axis.

    Attributes:
        axis : Axis of reflection.
        shape : Data frame shape.
    """
    def __init__(self, axis: int, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            axis : Axis of reflection.
            shape : Data frame shape.
        """
        if axis not in [0, 1]:
            raise ValueError('Axis must equal to 0 or 1')

        super().__init__(shape=shape)
        self.axis = axis

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return np.flip(inp, axis=self.axis - 2)

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        pts[:, self.axis] = self.shape[self.axis] - pts[:, self.axis]
        return pts

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.empty(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[...] = self.forward(inp)
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return self.forward_points(pts)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'axis': self.axis, 'shape': self.shape}

class ComposeTransforms(Transform):
    """Composes several transforms together.

    Attributes:
        transforms: List of transforms.
        shape : Data frame shape.
    """
    def __init__(self, transforms: List[Transform], shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            transforms: List of transforms.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        if len(transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        pdict = transforms[0].state_dict()
        pdict['shape'] = self.shape
        self.transforms = [type(transforms[0])(**pdict),]

        for transform in transforms[1:]:
            pdict = transform.state_dict()
            pdict['shape'] = None
            self.transforms.append(type(transform)(**pdict))

    def __iter__(self) -> Iterable:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Indices) -> Transform:
        return self.transforms[idx]

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        for transform in self:
            inp = transform.forward(inp)
        return inp

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        for transform in self:
            pts = transform.forward_points(pts)
        return pts

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        for transform in self[1::-1]:
            inp = transform.backward(inp)
        return self[0].backward(inp, out)

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        for transform in self[::-1]:
            pts = transform.backward_points(pts)
        return pts

    def integrate(self, axis: int) -> ComposeTransforms:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        pdict['transforms'] = [transform.integrate(axis) for transform in pdict['transforms']]

        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return ComposeTransforms(**pdict)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'transforms': self.transforms, 'shape': self.shape}

class STData(DataContainer):
    """Speckle tracking data container class. Needs a :class:`pyrost.CXIStore` file
    handler. Provides an interface to work with the data and contains a suite of
    tools for the R-PXST data processing pipeline. Also provides an interface to load
    from a file and save to a file any of the data attributes. The data frames can
    be tranformed using any of the :class:`pyrost.Transform` classes.

    Attributes:
        files : HDF5 or CXI file handler.
        transform : Frames transform object.

    Notes:
        **Necessary attributes**:

        * basis_vectors : Detector basis vectors
        * data : Measured intensity frames.
        * distance : Sample-to-detector distance [m].
        * frames : List of frame indices.
        * translations : Sample's translations [m].
        * wavelength : Incoming beam's wavelength [m].
        * x_pixel_size : Pixel's size along the horizontal detector axis [m].
        * y_pixel_size : Pixel's size along the vertical detector axis [m].

        **Optional attributes**:

        * defocus_x : Defocus distance for the horizontal detector axis [m].
        * defocus_y : Defocus distance for the vertical detector axis [m].
        * good_frames : An array of good frames' indices.
        * mask : Bad pixels mask.
        * num_threads : Number of threads used in computations.
        * phase : Phase profile of lens' aberrations.
        * pixel_aberrations : Lens' aberrations along the horizontal and
          vertical axes in pixels.
        * pixel_translations : Sample's translations in the detector's
          plane in pixels.
        * reference_image : The unabberated reference image of the sample.
        * scale_map : Huber scale map.
        * whitefield : Measured frames' white-field.
        * whitefields : Set of dynamic white-fields for each of the measured
          images.
    """
    attr_set = {'files'}
    init_set = {'basis_vectors', 'data', 'distance', 'frames', 'translations', 'wavelength',
                'x_pixel_size', 'y_pixel_size', 'defocus_x', 'defocus_y', 'good_frames',
                'mask', 'num_threads', 'phase', 'pixel_aberrations',  'pixel_translations',
                'reference_image', 'scale_map', 'transform', 'whitefield', 'whitefields'}
    is_points = {}

    # Necessary attributes
    files               : CXIStore
    transform           : Transform

    # Optional attributes
    basis_vectors       : Optional[np.ndarray]
    data                : Optional[np.ndarray]
    defocus_x           : Optional[float]
    defocus_y           : Optional[float]
    distance            : Optional[np.ndarray]
    frames              : Optional[np.ndarray]
    phase               : Optional[np.ndarray]
    pixel_aberrations   : Optional[np.ndarray]
    reference_image     : Optional[np.ndarray]
    scale_map           : Optional[np.ndarray]
    translations        : Optional[np.ndarray]
    wavelength          : Union[float, np.ndarray, None]
    whitefields         : Optional[np.ndarray]
    x_pixel_size        : Union[float, np.ndarray, None]
    y_pixel_size        : Union[float, np.ndarray, None]

    # Automatially generated attributes
    good_frames         : Optional[np.ndarray]
    mask                : Optional[np.ndarray]
    num_threads         : Optional[int]
    pixel_translations  : Optional[np.ndarray]
    whitefield          : Optional[np.ndarray]

    def __init__(self, files: CXIStore, transform: Optional[Transform]=None,
                 **kwargs: Union[int, float, np.ndarray]) -> None:
        """
        Args:
            files : HDF5 or CXI file handler.
            transform : Frames transform object.
            kwargs : Dictionary of the necessary and optional data attributes specified
                in :class:`pyrost.STData` notes. All the necessary attributes must be
                provided

        Raises:
            ValueError : If any of the necessary attributes specified in :class:`pyrost.STData`
                notes have not been provided.
        """
        super(STData, self).__init__(files=files, transform=transform, **kwargs)

        self._init_functions(num_threads=lambda: np.clip(1, 64, cpu_count()))
        if self._isdata:
            self._init_functions(good_frames=lambda: np.arange(self.shape[0]),
                                 mask=lambda: np.ones(self.shape, dtype=bool),
                                 whitefield=self._whitefield)
        if self._isdefocus:
            self._init_functions(defocus_y=lambda: self.get('defocus_x', None),
                                 pixel_translations=self._pixel_translations)

        self._init_attributes()

    @property
    def _isdata(self) -> bool:
        return self.data is not None

    @property
    def _isdefocus(self) -> bool:
        return self.defocus_x is not None

    @property
    def _isphase(self) -> bool:
        return not self.pixel_aberrations is None and not self.phase is None

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if attr in self.files.protocol and data is not None:
                kind = self.files.protocol.get_kind(attr)
                if kind == 'stack':
                    shape[:] = data.shape
                if kind == 'frame':
                    shape[1:] = data.shape
        return tuple(shape)

    def _basis_vectors(self) -> np.ndarray:
        def get_axis(transform: Transform):
            return transform.state_dict().get('axis', -1)

        axes = []
        if isinstance(self.transform, Transform):
            if isinstance(self.transform, ComposeTransforms):
                for t in self.transform:
                    axes.append(get_axis(t))
            else:
                axes.append(get_axis(self.transform))

        basis_vectors = np.copy(self.basis_vectors)
        for axis in axes:
            if axis > 0:
                basis_vectors[:, axis] *= -1
        return basis_vectors

    def _pixel_translations(self) -> np.ndarray:
        basis_vectors = self._basis_vectors()
        pixel_translations = (self.translations[:, None] * basis_vectors).sum(axis=-1)
        mag = np.abs(self.distance / np.array([self.defocus_y, self.defocus_x]))
        pixel_translations *= mag / (basis_vectors**2).sum(axis=-1)
        pixel_translations -= pixel_translations[0]
        pixel_translations -= pixel_translations.mean(axis=0)
        return pixel_translations

    def _whitefield(self) -> np.ndarray:
        return median(data=self.data[self.good_frames], axis=0,
                      mask=self.mask[self.good_frames],
                      num_threads=self.num_threads)

    def _transform_attribute(self, attr: str, data: np.ndarray, transform: Transform,
                             mode: str='forward') -> np.ndarray:
        kind = self.files.protocol.get_kind(attr)
        if kind in ['stack', 'frame']:
            if mode == 'forward':
                data = transform.forward(data)
            elif mode == 'backward':
                data = transform.backward(data)
            else:
                raise ValueError(f'Invalid mode keyword: {mode}')
        if attr in self.is_points:
            if data.shape[-1] != 2:
                raise ValueError(f"'{attr}' has invalid shape: {str(data.shape)}")

            data = self.transform.forward_points(data)

        return data

    def pixel_map(self, dtype: np.dtype=np.float64) -> np.ndarray:
        """Return a preliminary pixel mapping.

        Args:
            dtype : The data type of the output pixel mapping.

        Returns:
            Pixel mapping array.
        """
        if sum(self.shape[1:]):
            pixel_map = np.indices(self.shape[1:], dtype=dtype)

            if self._isdefocus:
                if self.defocus_y < 0.0 and pixel_map[0, 0, 0] < pixel_map[0, -1, 0]:
                    pixel_map = np.flip(pixel_map, axis=1)
                if self.defocus_x < 0.0 and pixel_map[1, 0, 0] < pixel_map[1, 0, -1]:
                    pixel_map = np.flip(pixel_map, axis=2)
            return np.asarray(pixel_map, order='C')

        raise AttributeError('Data has not been loaded')

    @dict_to_object
    def load(self, attributes: Union[str, List[str], None]=None, indices: Iterable[int]=None,
             processes: int=1, verbose: bool=True) -> STData:
        """Load data attributes from the input files in `files` file handler object.

        Args:
            attributes : List of attributes to load. Loads all the data attributes
                contained in the file(s) by default.
            indices : List of frame indices to load.
            processes : Number of parallel workers used during the loading.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If attribute is not existing in the input file(s).
            ValueError : If attribute is invalid.

        Returns:
            New :class:`STData` object with the attributes loaded.
        """
        with self.files:
            self.files.update_indices()

            if attributes is None:
                attributes = [attr for attr in self.files.keys()
                              if attr in self.init_set]
            else:
                attributes = self.files.protocol.str_to_list(attributes)

            if indices is None:
                indices = self.files.indices()
            data_dict = {'frames': indices}

            for attr in attributes:
                if attr not in self.files.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.init_set:
                    raise ValueError(f"Invalid attribute: '{attr}'")

                data = self.files.load_attribute(attr, indices, processes, verbose)

                if self.transform and data is not None:
                    data = self._transform_attribute(attr, data, self.transform)

                data_dict[attr] = data

        return data_dict

    def save(self, attributes: Union[str, List[str], None]=None, apply_transform=False,
             mode: str='append', idxs: Optional[Iterable[int]]=None) -> None:
        """Save data arrays of the data attributes contained in the container to
        an output file.

        Args:
            attributes : List of attributes to save. Saves all the data attributes
                contained in the container by default.
            apply_transform : Apply `transform` to the data arrays if True. The
                saved data will be expanded to comply with the original shape of
                detector grid.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices `idxs`.
                * `overwrite` : Overwrite the existing dataset.

            idxs : A set of frame indices where the data is saved if `mode` is
                `insert`.

            verbose : Set the verbosity of the loading process.
        """
        if attributes is None:
            attributes = list(self.contents())
        with self.files:
            for attr in self.files.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.files.protocol and data is not None:
                    kind = self.files.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        data = self._transform_attribute(attr, data, self.transform,
                                                         mode='backward')

                    self.files.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

    @dict_to_object
    def clear(self, attributes: Union[str, List[str], None]=None) -> STData:
        """Clear the container.

        Args:
            attributes : List of attributes to clear in the container.

        Returns:
            New :class:`STData` object with the attributes cleared.
        """
        if attributes is None:
            attributes = self.keys()
        data_dict = {}
        for attr in attributes:
            data = self.get(attr)
            if attr in self.files.protocol and data is not None:
                data_dict[attr] = None
        return data_dict

    @dict_to_object
    def integrate_data(self, axis: int=0) -> STData:
        """Return a new :class:`STData` object with the `data` summed
        over the `axis`.

        Args:
            axis : Axis along which a sum is performed.

        Returns:
            New :class:`STData` object with the stack of measured
            frames integrated along the given axis.
        """
        if self._isdata:
            data_dict = {}

            if self.transform:
                data_dict['transform'] = self.transform.integrate(axis)

            for attr, data in self.items():
                if attr in self.files.protocol and data is not None:
                    kind = self.files.protocol.get_kind(attr)
                    if kind in ['stack', 'frame']:
                        data_dict[attr] = None

            data = np.zeros(self.shape, self.data.dtype)
            data[self.good_frames] = (self.data * self.mask)[self.good_frames]
            data_dict['data'] = data.sum(axis=axis - 2, keepdims=True)

            return data_dict

        raise AttributeError('data has not been loaded')

    @dict_to_object
    def mask_frames(self, good_frames: Optional[Iterable[int]]=None) -> STData:
        """Return a new :class:`STData` object with the updated
        good frames mask. Mask empty frames by default.

        Args:
            good_frames : List of good frames' indices. Masks empty
                frames if not provided.

        Returns:
            New :class:`STData` object with the updated `good_frames`
            and `whitefield`.
        """
        if good_frames is None:
            good_frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return {'good_frames': np.asarray(good_frames), 'whitefield': None}

    @dict_to_object
    def update_mask(self, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> STData:
        """Return a new :class:`STData` object with the updated bad pixels
        mask.

        Args:
            method : Bad pixels masking methods:

                * `no-bad` (default) : No bad pixels.
                * `range-bad` : Mask the pixels which values lie outside
                  of (`vmin`, `vmax`) range.
                * `perc-bad` : Mask the pixels which values lie outside
                  of the (`pmin`, `pmax`) percentiles.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            pmin : Lower percentage bound of 'perc-bad' masking method.
            pmax : Upper percentage bound of 'perc-bad' masking method.
            update : Multiply the new mask and the old one if `multiply`,
                use the new one if `reset`.

        Returns:
            New :class:`STData` object with the updated `mask`.
        """
        if update == 'reset':
            data = self.data
        elif update == 'multiply':
            data = self.data * self.mask
        else:
            raise ValueError(f'Invalid update keyword: {update}')

        if method == 'no-bad':
            mask = np.ones(data.shape, dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            offsets = (data - np.median(data))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError(f'Invalid method argument: {method}')

        if update == 'reset':
            return {'mask': mask, 'whitefield': None}
        if update == 'multiply':
            return {'mask': mask * self.mask, 'whitefield': None}
        raise ValueError(f'Invalid update keyword: {update}')

    @dict_to_object
    def update_transform(self, transform: Transform) -> STData:
        """Return a new :class:`STData` object with the updated transform object.

        Args:
            transform : New :class:`Transform` object.

        Returns:
            New :class:`STData` object with the updated transform object.
        """
        data_dict = {'transform': transform}

        if self.transform is None:
            for attr, data in self.items():
                if attr in self.files.protocol and data is not None:
                    data_dict[attr] = self._transform_attribute(attr, data, transform)
            return data_dict

        for attr, data in self.items():
            if attr in self.files.protocol and data is not None:
                kind = self.files.protocol.get_kind(attr)
                if kind in ['stack', 'frame'] or attr in self.is_points:
                    data_dict[attr] = None
        return data_dict

    @dict_to_object
    def update_whitefield(self) -> STData:
        """Return a new :class:`STData` object with the updated `whitefield`.

        Returns:
            New :class:`STData` object with the updated `whitefield`.
        """
        return {'whitefield': None}

    @dict_to_object
    def update_defocus(self, defocus_x: float, defocus_y: Optional[float]=None) -> STData:
        """Return a new :class:`STData` object with the updated defocus
        distances `defocus_x` and `defocus_y` for the horizontal and
        vertical detector axes accordingly. Update `pixel_translations`
        based on the new defocus distances.

        Args:
            defocus_x : Defocus distance for the horizontal detector axis [m].
            defocus_y : Defocus distance for the vertical detector axis [m].
                Equals to `defocus_x` if it's not provided.

        Returns:
            New :class:`STData` object with the updated `defocus_y`,
            `defocus_x`, and `pixel_translations`.
        """
        if defocus_y is None:
            defocus_y = defocus_x
        return {'defocus_y': defocus_y, 'defocus_x': defocus_x,
                'pixel_translations': None}

    def import_st(self, st_obj: SpeckleTracking) -> None:
        """Update `pixel_aberrations`, `phase`, `reference_image`, and `scale_map`
        based on the data from `st_obj` object. `st_obj` must be derived from this
        data container, an error is raised otherwise.

        Args:
            st_obj : :class:`SpeckleTracking` object derived from this
                data container.

        Raises:
            ValueError : If `st_obj` wasn't derived from this data container.
        """
        if st_obj.parent() is not self:
            raise ValueError("'st_obj' wasn't derived from this data container")
        # Update phase, pixel_aberrations, and reference_image
        dpm_y, dpm_x = (st_obj.pixel_map - self.pixel_map())
        dpm_y -= dpm_y.mean()
        dpm_x -= dpm_x.mean()
        self.pixel_aberrations = np.stack((dpm_y, dpm_x))

        # Calculate magnification for horizontal and vertical axes
        mag_y = np.abs((self.distance + self.defocus_y) / self.defocus_y)
        mag_x = np.abs((self.distance + self.defocus_x) / self.defocus_x)

        # Calculate the distance between the reference and the detector plane
        dist_y = self.distance * (1 - mag_y**-1)
        dist_x = self.distance * (1 - mag_x**-1)

        # dTheta = delta_pix / distance / magnification * du
        # Phase = 2 * pi / wavelength * Integrate[dTheta, delta_pix]
        phase = ct_integrate(sy_arr=self.y_pixel_size**2 / dist_y / mag_y * dpm_y,
                             sx_arr=self.x_pixel_size**2 / dist_x / mag_x * dpm_x)
        self.phase = 2.0 * np.pi / self.wavelength * phase
        self.reference_image = st_obj.reference_image
        self.scale_map = st_obj.scale_map

    def fit_phase(self, center: int=0, axis: int=1, max_order: int=2, xtol: float=1e-14,
                  ftol: float=1e-14, loss: str='cauchy') -> Dict[str, Union[float, np.ndarray]]:
        """Fit `pixel_aberrations` with the polynomial function using nonlinear
        least-squares algorithm. The function uses least-squares algorithm from
        :func:`scipy.optimize.least_squares`.

        Args:
            center : Index of the zerro scattering angle or direct beam pixel.
            axis : Axis along which `pixel_aberrations` is fitted.
            max_order : Maximum order of the polynomial model function.
            xtol : Tolerance for termination by the change of the independent
                variables.
            ftol : Tolerance for termination by the change of the cost function.
            loss : Determines the loss function. The following keyword values are
                allowed:

                * `linear` : ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * `soft_l1` : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * `huber` : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * `cauchy` (default) : ``rho(z) = ln(1 + z)``. Severely weakens
                  outliers influence, but may cause difficulties in optimization
                  process.
                * `arctan` : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.

        Returns:
            A dictionary with the model fit information. The following fields
            are contained:

            * `c_3` : Third order aberrations coefficient [rad / mrad^3].
            * `c_4` : Fourth order aberrations coefficient [rad / mrad^4].
            * `fit` : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * `ph_fit` : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * `rel_err` : Vector of relative errors of the fit coefficients.
            * `r_sq` : ``R**2`` goodness of fit.

        See Also:
            :func:`pyrost.AberrationsFit.fit` : Full details of the aberrations
            fitting algorithm.
        """
        if not self._isphase:
            raise ValueError("'phase' is not defined inside the container.")
        return self.get_fit(center=center, axis=axis).fit(max_order=max_order,
                                                          xtol=xtol, ftol=ftol,
                                                          loss=loss)

    def defocus_sweep(self, defoci_x: np.ndarray, defoci_y: Optional[np.ndarray]=None, size: int=51,
                      hval: Optional[float]=None, extra_args: Dict[str, Union[float, bool, str]]={},
                      return_extra: bool=False, verbose: bool=True) -> Tuple[List[float], Dict[str, np.ndarray]]:
        r"""Calculate a set of reference images for each defocus in `defoci` and
        return an average R-characteristic of an image (the higher the value the
        sharper reference image is). The kernel bandwidth `hval` is automatically
        estimated by default. Return the intermediate results if `return_extra`
        is True.

        Args:
            defoci_x : Array of defocus distances along the horizontal detector axis [m].
            defoci_y : Array of defocus distances along the vertical detector axis [m].
            hval : Kernel bandwidth in pixels for the reference image update. Estimated
                with :func:`pyrost.SpeckleTracking.find_hopt` for an average defocus value
                if None.
            size : Local variance filter size in pixels.
            extra_args : Extra arguments parser to the :func:`STData.get_st` and
                :func:`SpeckleTracking.update_reference` methods. The following
                keyword values are allowed:

                * `ds_y` : Reference image sampling interval in pixels along the
                  horizontal axis. The default value is 1.0.
                * `ds_x` : Reference image sampling interval in pixels along the
                  vertical axis. The default value is 1.0.
                * `aberrations` : Add `pixel_aberrations` to `pixel_map` of
                  :class:`SpeckleTracking` object if it's True. The default value
                  is False.
                * `ff_correction` : Apply dynamic flatfield correction if it's True.
                  The default value is False.
                * `ref_method` : Choose the reference image update algorithm. The
                  following keyword values are allowed:

                  * `KerReg` : Kernel regression algorithm.
                  * `LOWESS` : Local weighted linear regression.

                  The default value is 'KerReg'.

            return_extra : Return a dictionary with the intermediate results if True.
            verbose : Set the verbosity of the process.

        Returns:
            A tuple of two items ('r_vals', 'extra'). The elements are as
            follows:

            * `r_vals` : Array of the average values of `reference_image` gradients
              squared.
            * `extra` : Dictionary with the intermediate results. Only if `return_extra`
              is True. Contains the following data:

              * reference_image : The generated set of reference profiles.
              * r_images : The set of local variance images of reference profiles.

        Notes:
            R-characteristic is called a local variance and is given by:

            .. math::
                R[i, j] = \frac{\sum_{i^{\prime} = -N / 2}^{N / 2}
                \sum_{j^{\prime} = -N / 2}^{N / 2} (I[i - i^{\prime}, j - j^{\prime}]
                - \bar{I}[i, j])^2}{\bar{I}^2[i, j]},

            where :math:`\bar{I}[i, j]` is a local mean and defined as follows:

            .. math::
                \bar{I}[i, j] = \frac{1}{N^2} \sum_{i^{\prime} = -N / 2}^{N / 2}
                \sum_{j^{\prime} = -N / 2}^{N / 2} I[i - i^{\prime}, j - j^{\prime}]

        See Also:
            :func:`pyrost.SpeckleTracking.update_reference` : reference image update
            algorithm.
        """
        if defoci_y is None:
            defoci_y = defoci_x.copy()

        ds_y = extra_args.get('ds_y', 1.0)
        ds_x = extra_args.get('ds_x', 1.0)
        aberrations = extra_args.get('aberrations', False)
        ff_correction = extra_args.get('ff_correction', False)
        ref_method = extra_args.get('ref_method', 'KerReg')

        r_vals = []
        extra = {'reference_image': [], 'r_image': []}
        kernel = np.ones(int(size)) / size
        df0_x, df0_y = defoci_x.mean(), defoci_y.mean()
        st_obj = self.update_defocus(df0_x, df0_y).get_st(ds_y=ds_y, ds_x=ds_x,
                                                          aberrations=aberrations,
                                                          ff_correction=ff_correction)
        if hval is None:
            hval = st_obj.find_hopt(method=ref_method)

        for df1_x, df1_y in tqdm(zip(defoci_x.ravel(), defoci_y.ravel()),
                               total=len(defoci_x), disable=not verbose,
                               desc='Generating defocus sweep'):
            st_obj.di_pix *= np.abs(df0_y / df1_y)
            st_obj.dj_pix *= np.abs(df0_x / df1_x)
            df0_x, df0_y = df1_x, df1_y
            st_obj.update_reference.inplace_update(hval=hval, method=ref_method)
            extra['reference_image'].append(st_obj.reference_image)
            mean = st_obj.reference_image.copy()
            mean_sq = st_obj.reference_image**2
            if st_obj.reference_image.shape[0] > size:
                mean = fft_convolve(mean, kernel, mode='reflect', axis=0,
                                    num_threads=self.num_threads)[size // 2:-size // 2]
                mean_sq = fft_convolve(mean_sq, kernel, mode='reflect', axis=0,
                                       num_threads=self.num_threads)[size // 2:-size // 2]
            if st_obj.reference_image.shape[1] > size:
                mean = fft_convolve(mean, kernel, mode='reflect', axis=1,
                                    num_threads=self.num_threads)[:, size // 2:-size // 2]
                mean_sq = fft_convolve(mean_sq, kernel, mode='reflect', axis=1,
                                       num_threads=self.num_threads)[:, size // 2:-size // 2]
            r_image = (mean_sq - mean**2) / mean**2
            extra['r_image'].append(r_image)
            r_vals.append(np.mean(r_image))

        if return_extra:
            return r_vals, extra
        return r_vals

    def get_st(self, ds_y: float=1.0, ds_x: float=1.0, aberrations: bool=False,
               ff_correction: bool=False) -> SpeckleTracking:
        """Return :class:`SpeckleTracking` object derived from the container.
        Return None if `defocus_x` or `defocus_y` doesn't exist in the container.

        Args:
            ds_y : Reference image sampling interval in pixels along the vertical
                axis.
            ds_x : Reference image sampling interval in pixels along the
                horizontal axis.
            aberrations : Add `pixel_aberrations` to `pixel_map` of
                :class:`SpeckleTracking` object if it's True.
            ff_correction : Apply dynamic flatfield correction if it's True.

        Returns:
            An instance of :class:`SpeckleTracking` derived from the container.
            None if `defocus_x` or `defocus_y` are not defined.
        """
        if not self._isdefocus:
            raise ValueError("'defocus_x' is not defined inside the container.")

        if np.issubdtype(self.data.dtype, np.uint32):
            dtypes = SpeckleTracking.dtypes_32
        else:
            dtypes = SpeckleTracking.dtypes_64

        data = np.asarray((self.mask * self.data)[self.good_frames],
                          order='C', dtype=dtypes['data'])
        whitefield = np.asarray(self.whitefield, order='C', dtype=dtypes['whitefield'])
        dij_pix = np.asarray(np.swapaxes(self.pixel_translations[self.good_frames], 0, 1),
                             order='C', dtype=dtypes['dij_pix'])

        if ff_correction and self.whitefields is not None:
            np.rint(data * np.where(self.whitefields > 0, whitefield / self.whitefields, 1.),
                    out=data, casting='unsafe')

        pixel_map = self.pixel_map(dtype=dtypes['pixel_map'])

        if aberrations:
            pixel_map += self.pixel_aberrations
            if self.scale_map is None:
                scale_map = None
            else:
                scale_map = np.asarray(self.scale_map, order='C', dtype=dtypes['scale_map'])
            return SpeckleTracking(parent=ref(self), data=data, dj_pix=dij_pix[1],
                                   di_pix=dij_pix[0], num_threads=self.num_threads,
                                   pixel_map=pixel_map, scale_map=scale_map, ds_y=ds_y,
                                   ds_x=ds_x, whitefield=whitefield)

        return SpeckleTracking(parent=ref(self), data=data, dj_pix=dij_pix[1], di_pix=dij_pix[0],
                               num_threads=self.num_threads, pixel_map=pixel_map, ds_y=ds_y,
                               ds_x=ds_x, whitefield=whitefield)

    def get_fit(self, center: int=0, axis: int=1) -> AberrationsFit:
        """Return an :class:`AberrationsFit` object for parametric regression
        of the lens' aberrations profile. Raises an error if 'defocus_x' or
        'defocus_y' is not defined.

        Args:
            center : Index of the zerro scattering angle or direct beam pixel.
            axis : Detector axis along which the fitting is performed.

        Raises:
            ValueError : If 'defocus_x' or 'defocus_y' is not defined in the
                container.

        Returns:
            An instance of :class:`AberrationsFit` class.
        """
        if not self._isphase:
            raise ValueError("'phase' or 'pixel_aberrations' are not defined inside the container.")

        data_dict = {attr: self.get(attr) for attr in AberrationsFit.attr_set if attr in self}
        if axis == 0:
            data_dict.update({attr: self.get(data_attr)
                              for attr, data_attr in AberrationsFit.y_lookup.items()})
        elif axis == 1:
            data_dict.update({attr: self.get(data_attr)
                              for attr, data_attr in AberrationsFit.x_lookup.items()})
        else:
            raise ValueError(f'invalid axis value: {axis:d}')

        data_dict['defocus'] = np.abs(data_dict['defocus'])
        if center <= self.shape[axis - 2]:
            data_dict['pixels'] = np.arange(self.shape[axis - 2]) - center
            data_dict['pixel_aberrations'] = data_dict['pixel_aberrations'][axis].mean(axis=1 - axis)
        elif center >= self.shape[axis - 2] - 1:
            data_dict['pixels'] = center - np.arange(self.shape[axis - 2])
            idxs = np.argsort(data_dict['pixels'])
            data_dict['pixel_aberrations'] = -data_dict['pixel_aberrations'][axis].mean(axis=1 - axis)[idxs]
            data_dict['pixels'] = data_dict['pixels'][idxs]
        else:
            raise ValueError('Origin must be outside of the region of interest')

        return AberrationsFit(parent=ref(self), **data_dict)

    def get_pca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the Principal Component Analysis [PCA]_ of the measured data and
        return a set of eigen flatfields (EFF).

        Returns:
            A tuple of ('cor_data', 'effs', 'eig_vals'). The elements are
            as follows:

            * `cor_data` : Background corrected stack of measured frames.
            * `effs` : Set of eigen flat-fields.
            * `eig_vals` : Corresponding eigen values for each of the eigen
              flat-fields.

        References:
            .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                    Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                    normalization using eigen flat fields in X-ray imaging," Opt.
                    Express 23, 27975-27989 (2015).
        """
        if self._isdata:

            dtype = np.promote_types(self.whitefield.dtype, int)
            cor_data = np.zeros(self.shape, dtype=dtype)
            np.subtract(self.data, self.whitefield, dtype=dtype,
                        where=self.mask, out=cor_data)
            mat_svd = np.tensordot(cor_data, cor_data, axes=((1, 2), (1, 2)))
            eig_vals, eig_vecs = np.linalg.eig(mat_svd)
            effs = np.tensordot(eig_vecs, cor_data, axes=((0,), (0,)))
            return cor_data, effs, eig_vals / eig_vals.sum()

        raise AttributeError('Data has not been loaded')

    @dict_to_object
    def update_whitefields(self, method: str='median', size: int=11,
                           cor_data: Optional[np.ndarray]=None,
                           effs: Optional[np.ndarray]=None) -> STData:
        """Return a new :class:`STData` object with a new set of dynamic whitefields.
        The flatfields are generated by the dint of median filtering or Principal
        Component Analysis [PCA]_.

        Args:
            method : Method to generate a set of dynamic white-fields. The following
                keyword values are allowed:

                * `median` : Median `data` along the first axis.
                * `pca` : Generate a set of dynamic white-fields based on eigen flatfields
                  `effs` from the PCA. `effs` can be obtained with :func:`STData.get_pca`
                  method.

            size : Size of the filter window in pixels used for the 'median' generation
                method.
            cor_data : Background corrected stack of measured frames.
            effs : Set of Eigen flatfields used for the 'pca' generation method.

        Raises:
            ValueError : If the `method` keyword is invalid.
            AttributeError : If the `whitefield` is absent in the :class:`STData`
                container when using the 'pca' generation method.
            ValueError : If `effs` were not provided when using the 'pca' generation
                method.

        Returns:
            New :class:`STData` object with the updated `whitefields`.

        See Also:
            :func:`pyrost.STData.get_pca` : Method to generate eigen flatfields.
        """
        if self._isdata:

            if method == 'median':
                outliers = np.abs(self.data - self.whitefield) < 3 * np.sqrt(self.whitefield)
                whitefields = median_filter(self.data, size=(size, 1, 1), mask=outliers,
                                            num_threads=self.num_threads)
            elif method == 'pca':
                if cor_data is None:
                    dtype = np.promote_types(self.whitefield.dtype, int)
                    cor_data = np.zeros(self.shape, dtype=dtype)
                    np.subtract(self.data, self.whitefield, dtype=dtype,
                                where=self.mask, out=cor_data)
                if effs is None:
                    raise ValueError('No eigen flat fields were provided')

                weights = np.tensordot(cor_data, effs, axes=((1, 2), (1, 2))) / \
                        np.sum(effs * effs, axis=(1, 2))
                whitefields = np.tensordot(weights, effs, axes=((1,), (0,))) + self.whitefield
            else:
                raise ValueError('Invalid method argument')

            return {'whitefields': whitefields}

        raise ValueError('Data has not been loaded')
