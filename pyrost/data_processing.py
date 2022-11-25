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
    >>> inp_file = rst.CXIStore('data.cxi')
    >>> data = rst.STData(input_file=inp_file)
    >>> data = data.load()
"""
from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from weakref import ref
from multiprocessing import cpu_count
from tqdm.auto import tqdm
import numpy as np
from .aberrations_fit import AberrationsFit
from .data_container import DataContainer, Transform
from .cxi_protocol import CXIStore, Indices
from .rst_update import SpeckleTracking
from .bin import median, median_filter, fft_convolve, ct_integrate

S = TypeVar('S', bound='STData')

@dataclass
class STData(DataContainer):
    """Speckle tracking data container class. Needs a :class:`pyrost.CXIStore` file
    handler. Provides an interface to work with the data and contains a suite of
    tools for the R-PXST data processing pipeline. Also provides an interface to load
    from a file and save to a file any of the data attributes. The data frames can
    be tranformed using any of the :class:`pyrost.Transform` classes.

    Args:
        input_file : HDF5 or CXI file handler of input files.
        output_file : Output file handler.
        transform : Frames transform object.
        kwargs : Dictionary of the necessary and optional data attributes specified
            in :class:`pyrost.STData` notes. All the necessary attributes must be
            provided

    Raises:
        ValueError : If any of the necessary attributes specified in :class:`pyrost.STData`
            notes have not been provided.

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
        * output_file : Output file handler.
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
    # Necessary attributes
    input_file          : CXIStore
    transform           : Optional[Transform] = None

    # Optional attributes
    basis_vectors       : Optional[np.ndarray] = None
    data                : Optional[np.ndarray] = None
    defocus_x           : Optional[float] = None
    defocus_y           : Optional[float] = None
    distance            : Optional[np.ndarray] = None
    frames              : Optional[np.ndarray] = None
    output_file         : Optional[CXIStore] = None
    phase               : Optional[np.ndarray] = None
    pixel_aberrations   : Optional[np.ndarray] = None
    reference_image     : Optional[np.ndarray] = None
    scale_map           : Optional[np.ndarray] = None
    translations        : Optional[np.ndarray] = None
    wavelength          : Optional[float] = None
    whitefields         : Optional[np.ndarray] = None
    x_pixel_size        : Optional[float] = None
    y_pixel_size        : Optional[float] = None

    # Automatially generated attributes
    good_frames         : Optional[np.ndarray] = None
    mask                : Optional[np.ndarray] = None
    num_threads         : int = field(default=np.clip(1, 64, cpu_count()))
    pixel_translations  : Optional[np.ndarray] = None
    whitefield          : Optional[np.ndarray] = None

    _no_data_exc        : ClassVar[ValueError] = ValueError('No data in the container')
    _no_defocus_exc     : ClassVar[ValueError] = ValueError('No defocus in the container')

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind == 'sequence':
                    shape[0] = data.shape[0]
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind == 'frame':
                    shape[1:] = data.shape
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind == 'stack':
                    shape[:] = data.shape
        return tuple(shape)

    def replace(self: S, **kwargs: Any) -> S:
        """Return a new :class:`pyrost.STData` container with replaced data.

        Args:
            kwargs : Replaced attributes.

        Returns:
            A new :class:`pyrost.STData` container.
        """
        dct = dict(self, **kwargs)
        if dct['data'] is not None:
            if dct['defocus_x'] is not None:
                return STDataFull(**dct)
            return STDataPart(**dct)
        return STData(**dct)

    def pixel_map(self, dtype: np.dtype=np.float64) -> np.ndarray:
        """Return a preliminary pixel mapping.

        Args:
            dtype : The data type of the output pixel mapping.

        Returns:
            Pixel mapping array.
        """
        raise self._no_defocus_exc

    def load(self: S, attributes: Union[str, List[str], None]=None, idxs: Optional[Indices]=None,
             processes: int=1, verbose: bool=True) -> S:
        """Load data attributes from the input files in `files` file handler object.

        Args:
            attributes : List of attributes to load. Loads all the data attributes contained in
                the file(s) by default.
            idxs : List of frame indices to load.
            processes : Number of parallel workers used during the loading.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If attribute is not existing in the input file(s).
            ValueError : If attribute is invalid.

        Returns:
            New :class:`STData` object with the attributes loaded.
        """
        with self.input_file:
            self.input_file.update_indices()
            shape = self.input_file.read_shape()

            if attributes is None:
                attributes = [attr for attr in self.input_file.keys()
                              if attr in self.keys()]
            else:
                attributes = self.input_file.protocol.str_to_list(attributes)

            if idxs is None:
                idxs = self.input_file.indices()
            data_dict = {'frames': np.asarray(idxs)}

            for attr in attributes:
                if attr not in self.input_file.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.keys():
                    raise ValueError(f"Invalid attribute: '{attr}'")

                if self.transform and shape[0] * shape[1]:
                    ss_idxs, fs_idxs = np.indices(shape)
                    ss_idxs, fs_idxs = self.transform.index_array(ss_idxs, fs_idxs)
                    data = self.input_file.load_attribute(attr, idxs=idxs, ss_idxs=ss_idxs,
                                                          fs_idxs=fs_idxs, processes=processes,
                                                          verbose=verbose)
                else:
                    data = self.input_file.load_attribute(attr, idxs=idxs, processes=processes,
                                                          verbose=verbose)

                data_dict[attr] = data

        return self.replace(**data_dict)


    def save(self, attributes: Union[str, List[str], None]=None, apply_transform: bool=False,
             mode: str='append', idxs: Optional[Indices]=None):
        """Save data arrays of the data attributes contained in the container to an output file.

        Args:
            attributes : List of attributes to save. Saves all the data attributes contained in
                the container by default.
            apply_transform : Apply `transform` to the data arrays if True.
            mode : Writing modes. The following keyword values are allowed:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices `idxs`.
                * `overwrite` : Overwrite the existing dataset.

            idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

        Raises:
            ValueError : If the ``output_file`` is not defined inside the container.
        """
        if self.output_file is None:
            raise ValueError("'output_file' is not defined inside the container")

        if attributes is None:
            attributes = list(self.contents())

        with self.input_file:
            shape = self.input_file.read_shape()

        with self.output_file:
            for attr in self.output_file.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.output_file.protocol and data is not None:
                    kind = self.output_file.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        if kind in ['stack', 'frame']:
                            out = np.zeros(shape, dtype=data.dtype)
                            data = self.transform.backward(data, out)

                    self.output_file.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

    def clear(self: S, attributes: Union[str, List[str], None]=None) -> S:
        """Clear the data inside the container.

        Args:
            attributes : List of attributes to clear in the container.

        Returns:
            New :class:`STData` object with the attributes cleared.
        """
        if attributes is None:
            attributes = self.contents()

        data_dict = dict(self)
        for attr in self.input_file.protocol.str_to_list(attributes):
            if attr not in self.keys():
                raise ValueError(f"Invalid attribute: '{attr}'")

            if isinstance(self[attr], np.ndarray):
                data_dict[attr] = None

        return self.replace(**data_dict)

    def update_output_file(self: S, output_file: CXIStore) -> S:
        """Return a new :class:`STData` object with the new output file handler.

        Args:
            output_file : A new output file handler.

        Returns:
            New :class:`STData` object with the new output file handler.
        """
        return self.replace(output_file=output_file)

    def update_transform(self: S, transform: Transform) -> S:
        """Return a new :class:`STData` object with the updated transform object.

        Args:
            transform : New :class:`Transform` object.

        Returns:
            New :class:`STData` object with the updated transform object.
        """
        data_dict = {'transform': transform}

        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind in ['stack', 'frame']:
                    if self.transform is None:
                        data_dict[attr] = transform.forward(data)
                    else:
                        data_dict[attr] = None

        return self.replace(**data_dict)

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
        raise self._no_data_exc

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
        raise self._no_data_exc

    def integrate_data(self: S, axis: int=0) -> S:
        """Return a new :class:`STData` object with the `data` summed
        over the `axis`. Clear all the 2D and 3D data attributes inside the
        container.

        Args:
            axis : Axis along which a sum is performed.

        Returns:
            New :class:`STData` object with the stack of measured
            frames integrated along the given axis.
        """
        raise self._no_data_exc

    def mask_frames(self: S, frames: Optional[Indices]=None) -> S:
        """Return a new :class:`STData` object with the updated good frames mask.
        Mask empty frames by default.

        Args:
            frames : List of good frames' indices. Masks empty frames if not provided.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`STData` object with the updated ``frames`` and ``whitefield``.
        """
        raise self._no_data_exc

    def update_defocus(self: S, defocus_x: float, defocus_y: Optional[float]=None) -> S:
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
        raise self._no_data_exc

    def update_mask(self: S, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> S:
        """Return a new :class:`STData` object with the updated bad pixels mask.

        Args:
            method : Bad pixels masking methods. The following keyword values are
                allowed:

                * 'no-bad' (default) : No bad pixels.
                * 'range-bad' : Mask the pixels which values lie outside of (`vmin`,
                  `vmax`) range.
                * 'perc-bad' : Mask the pixels which values lie outside of the (`pmin`,
                  `pmax`) percentiles.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            pmin : Lower percentage bound of 'perc-bad' masking method.
            pmax : Upper percentage bound of 'perc-bad' masking method.
            update : Multiply the new mask and the old one if 'multiply', use the new
                one if 'reset'.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If ``method`` keyword is invalid.
            ValueError : If ``update`` keyword is invalid.
            ValueError : If ``vmin`` is larger than ``vmax``.
            ValueError : If ``pmin`` is larger than ``pmax``.

        Returns:
            New :class:`STData` object with the updated ``mask``.
        """
        raise self._no_data_exc

    def update_whitefield(self) -> STData:
        """Return a new :class:`STData` object with the updated `whitefield`.

        Returns:
            New :class:`STData` object with the updated `whitefield`.
        """
        raise self._no_data_exc

    def update_whitefields(self: S, method: str='median', size: int=11,
                           cor_data: Optional[np.ndarray]=None,
                           effs: Optional[np.ndarray]=None) -> S:
        """Return a new :class:`STData` object with a new set of dynamic whitefields.
        A set of whitefields are generated by the dint of median filtering or Principal
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
                offsets = np.abs(self.data - self.whitefield)
                outliers = offsets < 3 * np.sqrt(self.whitefield)
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

                weights = np.tensordot(cor_data, effs, axes=((1, 2), (1, 2)))
                weights /= np.sum(effs * effs, axis=(1, 2))
                whitefields = np.tensordot(weights, effs, axes=((1,), (0,)))
                whitefields += self.whitefield
            else:
                raise ValueError('Invalid method argument')

            return {'whitefields': whitefields}

        raise ValueError('Data has not been loaded')

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
        raise self._no_defocus_exc

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
        raise self._no_defocus_exc

    def get_st(self, ds_y: float=1.0, ds_x: float=1.0, test_ratio: float=0.1,
               aberrations: bool=False, ff_correction: bool=False) -> SpeckleTracking:
        """Return :class:`SpeckleTracking` object derived from the container.
        Return None if `defocus_x` or `defocus_y` doesn't exist in the container.

        Args:
            ds_y : Reference image sampling interval in pixels along the vertical
                axis.
            ds_x : Reference image sampling interval in pixels along the
                horizontal axis.
            test_ratio : Ratio between the size of the test subset and the whole dataset.
            aberrations : Add `pixel_aberrations` to `pixel_map` of
                :class:`SpeckleTracking` object if it's True.
            ff_correction : Apply dynamic flatfield correction if it's True.

        Returns:
            An instance of :class:`SpeckleTracking` derived from the container.
            None if `defocus_x` or `defocus_y` are not defined.
        """
        raise self._no_defocus_exc

    def import_st(self, st_obj: SpeckleTracking):
        """Update `pixel_aberrations`, `phase`, `reference_image`, and `scale_map`
        based on the data from `st_obj` object. `st_obj` must be derived from this
        data container, an error is raised otherwise.

        Args:
            st_obj : :class:`SpeckleTracking` object derived from this
                data container.

        Raises:
            ValueError : If `st_obj` wasn't derived from this data container.
        """
        raise self._no_defocus_exc

@dataclass
class STDataPart(STData):
    # Necessary attributes
    input_file          : CXIStore
    transform           : Optional[Transform] = None

    # Optional attributes
    basis_vectors       : Optional[np.ndarray] = None
    data                : Optional[np.ndarray] = None
    defocus_x           : Optional[float] = None
    defocus_y           : Optional[float] = None
    distance            : Optional[np.ndarray] = None
    frames              : Optional[np.ndarray] = None
    output_file         : Optional[CXIStore] = None
    phase               : Optional[np.ndarray] = None
    pixel_aberrations   : Optional[np.ndarray] = None
    reference_image     : Optional[np.ndarray] = None
    scale_map           : Optional[np.ndarray] = None
    translations        : Optional[np.ndarray] = None
    wavelength          : Optional[float] = None
    whitefields         : Optional[np.ndarray] = None
    x_pixel_size        : Optional[float] = None
    y_pixel_size        : Optional[float] = None

    # Automatially generated attributes
    good_frames         : Optional[np.ndarray] = None
    mask                : Optional[np.ndarray] = None
    num_threads         : int = field(default=np.clip(1, 64, cpu_count()))
    pixel_translations  : Optional[np.ndarray] = None
    whitefield          : Optional[np.ndarray] = None

    def __post_init__(self):
        if self.good_frames is None:
            self.good_frames = np.arange(self.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        if self.whitefield is None:
            self.whitefield = median(inp=self.data[self.good_frames], axis=0,
                                     mask=self.mask[self.good_frames],
                                     num_threads=self.num_threads)

    def defocus_sweep(self, defoci_x: np.ndarray, defoci_y: Optional[np.ndarray]=None, size: int=51,
                      hval: Optional[float]=None, extra_args: Dict[str, Union[float, bool, str]]={},
                      return_extra: bool=False, verbose: bool=True) -> Tuple[List[float], Dict[str, np.ndarray]]:
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
                               total=defoci_x.size, disable=not verbose,
                               desc='Generating defocus sweep'):
            st_obj.di_pix *= np.abs(df0_y / df1_y)
            st_obj.dj_pix *= np.abs(df0_x / df1_x)
            df0_x, df0_y = df1_x, df1_y
            st_obj = st_obj.update_reference(hval=hval, method=ref_method)
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

    def get_pca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dtype = np.promote_types(self.whitefield.dtype, int)
        cor_data = np.zeros(self.shape, dtype=dtype)[self.good_frames]
        np.subtract(self.data[self.good_frames], self.whitefield, dtype=dtype,
                    where=self.mask[self.good_frames], out=cor_data)
        mat_svd = np.tensordot(cor_data, cor_data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, cor_data, axes=((0,), (0,)))
        return cor_data, effs, eig_vals / eig_vals.sum()

    def integrate_data(self: S, axis: int=0) -> S:
        data_dict = {}
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind in ['stack', 'frame']:
                    data_dict[attr] = None

        data_dict['data'] = (self.data * self.mask).sum(axis=axis - 2, keepdims=True)

        return self.replace(**data_dict)

    def mask_frames(self: S, frames: Optional[Indices]=None) -> S:
        if frames is None:
            frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return self.replace(good_frames=np.asarray(frames), whitefield=None)


    def update_defocus(self, defocus_x: float, defocus_y: Optional[float]=None) -> STDataFull:
        if defocus_y is None:
            defocus_y = defocus_x
        return self.replace(defocus_y=defocus_y, defocus_x=defocus_x, pixel_translations=None)

    def update_mask(self: S, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> S:
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if pmin >= pmax:
            raise ValueError('pmin must be less than pmax')

        if update == 'reset':
            data = self.data
        elif update == 'multiply':
            data = self.data * self.mask
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if method == 'no-bad':
            mask = np.ones(self.shape, dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            average = median_filter(data, (1, 3, 3), num_threads=self.num_threads)
            offsets = (data.astype(np.int32) - average.astype(np.int32))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError('invalid method argument')

        if update == 'reset':
            return self.replace(mask=mask, whitefield=None)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask, whitefield=None)
        raise ValueError(f'Invalid update keyword: {update:s}')

    def update_whitefield(self: S) -> S:
        return self.replace(whitefield=None)

    def update_whitefields(self: S, method: str='median', size: int=11,
                           cor_data: Optional[np.ndarray]=None,
                           effs: Optional[np.ndarray]=None) -> S:
        if method == 'median':
            offsets = np.abs(self.data - self.whitefield)
            outliers = offsets < 3 * np.sqrt(self.whitefield)
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

            weights = np.tensordot(cor_data, effs, axes=((1, 2), (1, 2)))
            weights /= np.sum(effs * effs, axis=(1, 2))
            whitefields = np.tensordot(weights, effs, axes=((1,), (0,)))
            whitefields += self.whitefield
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefields=whitefields)

@dataclass
class STDataFull(STDataPart):
    # Necessary attributes
    input_file          : CXIStore
    transform           : Optional[Transform] = None

    # Optional attributes
    basis_vectors       : Optional[np.ndarray] = None
    data                : Optional[np.ndarray] = None
    defocus_x           : Optional[float] = None
    defocus_y           : Optional[float] = None
    distance            : Optional[np.ndarray] = None
    frames              : Optional[np.ndarray] = None
    output_file         : Optional[CXIStore] = None
    phase               : Optional[np.ndarray] = None
    pixel_aberrations   : Optional[np.ndarray] = None
    reference_image     : Optional[np.ndarray] = None
    scale_map           : Optional[np.ndarray] = None
    translations        : Optional[np.ndarray] = None
    wavelength          : Optional[float] = None
    whitefields         : Optional[np.ndarray] = None
    x_pixel_size        : Optional[float] = None
    y_pixel_size        : Optional[float] = None

    # Automatially generated attributes
    good_frames         : Optional[np.ndarray] = None
    mask                : Optional[np.ndarray] = None
    num_threads         : int = field(default=np.clip(1, 64, cpu_count()))
    pixel_translations  : Optional[np.ndarray] = None
    whitefield          : Optional[np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        if self.defocus_y is None:
            self.defocus_y = self.defocus_x
        if self.pixel_translations is None:
            self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
            mag = np.abs(self.distance / np.array([self.defocus_y, self.defocus_x]))
            self.pixel_translations *= mag / (self.basis_vectors**2).sum(axis=-1)
            self.pixel_translations -= self.pixel_translations[0]
            self.pixel_translations -= self.pixel_translations.mean(axis=0)

    def fit_phase(self, center: int=0, axis: int=1, max_order: int=2, xtol: float=1e-14,
                  ftol: float=1e-14, loss: str='cauchy') -> Dict[str, Union[float, np.ndarray]]:
        return self.get_fit(center=center, axis=axis).fit(max_order=max_order,
                                                          xtol=xtol, ftol=ftol,
                                                          loss=loss)
    def get_fit(self, center: int=0, axis: int=1) -> AberrationsFit:
        if axis == 0:
            defocus, pixel_size = np.abs(self.defocus_y), self.y_pixel_size
        elif axis == 1:
            defocus, pixel_size = np.abs(self.defocus_x), self.x_pixel_size
        else:
            raise ValueError(f'invalid axis value: {axis:d}')

        pixel_aberrations = np.copy(self.pixel_aberrations)
        if center <= self.shape[axis - 2]:
            pixels = np.arange(self.shape[axis - 2]) - center
            pixel_aberrations = pixel_aberrations[axis].mean(axis=1 - axis)
        elif center >= self.shape[axis - 2] - 1:
            pixels = center - np.arange(self.shape[axis - 2])
            idxs = np.argsort(pixels)
            pixel_aberrations = -pixel_aberrations[axis].mean(axis=1 - axis)[idxs]
            pixels = pixels[idxs]
        else:
            raise ValueError('Origin must be outside of the region of interest')

        return AberrationsFit(parent=ref(self), defocus=defocus, distance=self.distance,
                              pixels=pixels, pixel_aberrations=pixel_aberrations,
                              pixel_size=pixel_size, wavelength=self.wavelength)

    def get_st(self, ds_y: float=1.0, ds_x: float=1.0, test_ratio: float=0.1,
               aberrations: bool=False, ff_correction: bool=False) -> SpeckleTracking:
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
            np.rint(data * np.where(self.whitefields > 0, whitefield / self.whitefields, 1.0),
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
                                   ds_x=ds_x, whitefield=whitefield, test_ratio=test_ratio)

        return SpeckleTracking(parent=ref(self), data=data, dj_pix=dij_pix[1], di_pix=dij_pix[0],
                               num_threads=self.num_threads, pixel_map=pixel_map, ds_y=ds_y,
                               ds_x=ds_x, whitefield=whitefield, test_ratio=test_ratio)

    def import_st(self, st_obj: SpeckleTracking):
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
        dist_y = self.distance * (mag_y - 1.0) / mag_y
        dist_x = self.distance * (mag_x - 1.0) / mag_x

        # dTheta = delta_pix / distance / magnification * du
        # Phase = 2 * pi / wavelength * Integrate[dTheta, delta_pix]
        phase = ct_integrate(sy_arr=self.y_pixel_size**2 / dist_y / mag_y * dpm_y,
                             sx_arr=self.x_pixel_size**2 / dist_x / mag_x * dpm_x)
        self.phase = 2.0 * np.pi / self.wavelength * phase
        self.reference_image = st_obj.reference_image
        self.scale_map = st_obj.scale_map

    def pixel_map(self, dtype: np.dtype=np.float64) -> np.ndarray:
        with self.input_file:
            self.input_file.update_indices()
            shape = self.input_file.read_shape()

        # Check if STData is integrated
        if self.shape[1] == 1:
            shape = (1, shape[1])
        if self.shape[2] == 1:
            shape = (shape[0], 1)

        ss_idxs, fs_idxs = np.indices(shape, dtype=dtype)
        if self.transform:
            ss_idxs, fs_idxs = self.transform.index_array(ss_idxs, fs_idxs)
        pixel_map = np.stack((ss_idxs, fs_idxs))

        if self.defocus_y < 0.0:
            pixel_map = np.flip(pixel_map, axis=1)
        if self.defocus_x < 0.0:
            pixel_map = np.flip(pixel_map, axis=2)
        return np.asarray(pixel_map, order='C')
