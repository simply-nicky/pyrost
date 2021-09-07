"""Robust Speckle Tracking data processing algorithm.
:class:`pyrost.STData` contains all the necessary data for the Speckle
Tracking algorithm, and handy data processing tools to work
with the data. :class:`pyrost.SpeckleTracking` performs the main
Robust Speckle Tracking algorithm and yields reference image
and pixel mapping. :class:`pyrost.AberrationsFit` fit the lens'
aberrations profile with the polynomial function using
nonlinear least-squares algorithm.

Examples
--------
Extract all the necessary data using a :func:`pyrost.cxi_loader` function.

>>> import pyrost as rst
>>> loader = rst.cxi_loader()
>>> rst_data = loader.load('results/test/data.cxi')

Perform the Robust Speckle Tracking using a :class:`pyrost.SpeckleTracking` object.

>>> rst_obj = rst_data.get_st()
>>> rst_res, errors = rst_obj.iter_update(sw_fs=150, ls_pm=3., ls_ri=10.,
...                                       verbose=True, n_iter=10)
Iteration No. 0: Total MSE = 0.150
Iteration No. 1: Total MSE = 0.077
Iteration No. 2: Total MSE = 0.052
Iteration No. 3: Total MSE = 0.050

Extract lens' aberrations wavefront and fit it with a polynomial.

>>> rst_data.update_phase(rst_res)
>>> fit_res = rst_data.fit_phase()
>>> fit_res['ph_fit']
array([-5.19025587e+07, -8.63773622e+05,  3.42849675e+03,  2.98523995e+01,
        1.19773905e-02])
"""
from multiprocessing import cpu_count
from tqdm.auto import tqdm
import numpy as np
from .aberrations_fit import AberrationsFit
from .data_container import DataContainer, dict_to_object
from .bin import median, make_reference, update_pixel_map_gs, update_pixel_map_nm
from .bin import update_translations_gs, mse_frame, mse_total, ct_integrate
from .bin import gaussian_filter, fft_convolve

class STData(DataContainer):
    """Speckle Tracking algorithm data container class.
    Contains all the necessary data for the Robust Speckle
    Tracking algorithm (specified in `attr_set` and `init_set`),
    the list of all the :class:`SpeckleTracking` objects derived
    from it, and two :class:`AberrationsFit` objects to fit the phase
    along the fast and slow detector axes.

    Parameters
    ----------
    protocol : CXIProtocol
        CXI :class:`CXIProtocol` object.
    num_threads : int, optional
        Specify number of threads that are used in all the calculations.

    **kwargs : dict
        Dictionary of the attributes' data specified in `attr_set`
        and `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.
    init_set : set
        Set of optional data attributes.

    Raises
    ------
    ValueError
        If an attribute specified in `attr_set` has not been provided.

    Notes
    -----
    Necessary attributes:

    * basis_vectors : Detector basis vectors
    * data : Measured intensity frames.
    * distance : Sample-to-detector distance [m].
    * translations : Sample's translations [m].
    * wavelength : Incoming beam's wavelength [m].
    * x_pixel_size : Pixel's size along the fast detector axis [m].
    * y_pixel_size : Pixel's size along the slow detector axis [m].

    Optional attributes:

    * defocus_ss : Defocus distance for the slow detector axis [m].
    * defocus_fs : Defocus distance for the fast detector axis [m].
    * error_frame : MSE (mean-squared-error) of the reference image
      and pixel mapping fit per pixel.
    * good_frames : An array of good frames' indices.
    * mask : Bad pixels mask.
    * phase : Phase profile of lens' aberrations.
    * pixel_map : The pixel mapping between the data at the detector's
      plane and the reference image at the reference plane.
    * pixel_aberrations : Lens' aberrations along the fast and
      slow axes in pixels.
    * pixel_translations : Sample's translations in the detector's
      plane in pixels.
    * reference_image : The unabberated reference image of the sample.
    * roi : Region of interest in the detector plane.
    * whitefield : Measured frames' whitefield.
    """
    attr_set = {'basis_vectors', 'data', 'distance', 'translations', 'wavelength',
                'x_pixel_size', 'y_pixel_size'}
    init_set = {'defocus_fs', 'defocus_ss', 'error_frame', 'flatfields', 'good_frames',
                'mask', 'num_threads', 'phase', 'pixel_aberrations', 'pixel_map',
                'pixel_translations', 'reference_image', 'roi', 'whitefield'}

    def __init__(self, protocol, **kwargs):
        # Initialize protocol for the proper data type conversion in __setattr__
        self.__dict__['protocol'] = protocol

        # Initialize attr_dict
        super(STData, self).__init__(**kwargs)

        # Add protocol to the attr_dict in order to get the dict_update decorator working
        self.__dict__['attr_dict']['protocol'] = protocol

        # Initialize init_set attributes
        self._init_dict()

    def _init_dict(self):
        # Set number of threads, num_threads is not a part of the protocol
        if self.num_threads is None:
            self.num_threads = np.clip(1, 64, cpu_count())
        # Set ROI, good frames array, mask, and whitefield
        if self.roi is None:
            self.roi = np.array([0, self.data.shape[1], 0, self.data.shape[2]])
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.data.shape)
        if self.mask.shape == self.data.shape[1:]:
            self.mask = np.tile(self.mask[None, :], (self.data.shape[0], 1, 1))
        if self.whitefield is None:
            self.whitefield = median(data=self.data[self.good_frames],
                                     mask=self.mask[self.good_frames], axis=0,
                                     num_threads=self.num_threads)

        # Set a pixel map, deviation angles, and phase
        if not self.pixel_map is None:
            self.pixel_aberrations = self.pixel_map
        self.pixel_map = np.indices(self.whitefield.shape)

        if self._isdefocus:
            # Set a pixel translations
            if self.pixel_translations is None:
                self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
                mag = np.abs(self.distance / np.array([self.defocus_ss, self.defocus_fs]))
                self.pixel_translations *= mag / (self.basis_vectors**2).sum(axis=-1)
                # Remove first element to avoid losing precision of np.mean
                self.pixel_translations -= self.pixel_translations[0]
                self.pixel_translations -= self.pixel_translations.mean(axis=0)

            # Flip pixel mapping if defocus is negative
            if self.defocus_ss < 0:
                self.pixel_map = np.flip(self.pixel_map, axis=1)
            if self.defocus_fs < 0:
                self.pixel_map = np.flip(self.pixel_map, axis=2)

        # Initialize a list of SpeckleTracking objects
        self._st_objects = []

        # Initialize a list of AberrationsFit objects
        self._ab_fits = []
        if self._isphase:
            self._ab_fits.extend([AberrationsFit.import_data(self, axis=0),
                                  AberrationsFit.import_data(self, axis=1)])

    @property
    def _isdefocus(self):
        return not self.defocus_ss is None and not self.defocus_fs is None

    @property
    def _isphase(self):
        return not self.pixel_aberrations is None and not self.phase is None

    def __setattr__(self, attr, value):
        if attr in self.attr_set | self.init_set:
            dtype = self.protocol.get_dtype(attr)
            if not dtype is None:
                if isinstance(value, np.ndarray):
                    value = np.array(value, dtype=dtype)
                elif not value is None:
                    value = dtype(value)
            super(STData, self).__setattr__(attr, value)
        else:
            super(STData, self).__setattr__(attr, value)

    @dict_to_object
    def bin_data(self, bin_ratio=2):
        """Return a new :class:`STData` object with the data binned by
        a factor `bin_ratio`.

        Parameters
        ----------
        bin_ratio : int, optional
            Binning ratio. The frame size will decrease by the factor of
            `bin_ratio`.

        Returns
        -------
        STData
            New :class:`STData` object with binned `data`.
        """
        data = self.data[:, ::bin_ratio, ::bin_ratio]
        whitefield = self.whitefield[::bin_ratio, ::bin_ratio]
        mask = self.mask[:, ::bin_ratio, ::bin_ratio]
        data_dict = {'basis_vectors': bin_ratio * self.basis_vectors, 'data': data,
                     'whitefield': whitefield, 'mask': mask, 'roi': self.roi // bin_ratio,
                     'x_pixel_size': bin_ratio * self.x_pixel_size,
                     'y_pixel_size': bin_ratio * self.y_pixel_size}
        if self._isdefocus:
            data_dict['pixel_translations'] = self.pixel_translations / bin_ratio,
        return data_dict

    @dict_to_object
    def crop_data(self, roi):
        """Return a new :class:`STData` object with the updated `roi`.

        Parameters
        ----------
        roi : iterable
            Region of interest in the detector plane.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `roi`.
        """
        return {'roi': np.asarray(roi, dtype=int)}

    @dict_to_object
    def integrate_data(self, axis=0):
        """Return a new :class:`STData` object with the `data` summed
        over the `axis`.

        Parameters
        ----------
        axis : int, optional
            Axis along which a sum is performed.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `data`,
            `whitefield`, `mask`, and `roi`.
        """
        roi = self.roi.copy()
        roi[2 * axis:2 * (axis + 1)] = np.arange(2)

        data = np.zeros(self.data.shape, self.data.dtype)
        data[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = self.get('data') * self.get('mask')
        return {'data': np.sum(data, axis=axis + 1, keepdims=True),
                'whitefield': None, 'mask': None, 'roi': roi}

    @dict_to_object
    def mask_frames(self, good_frames=None):
        """Return a new :class:`STData` object with the updated
        good frames mask. Mask empty frames by default.

        Parameters
        ----------
        good_frames : iterable, optional
            List of good frames' indices. Keeps non-empty frames
            if not provided.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `good_frames`
            and `whitefield`.
        """
        if good_frames is None:
            good_frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return {'good_frames': np.asarray(good_frames, dtype=np.int),
                'whitefield': None}

    @dict_to_object
    def mirror_data(self, axis=1):
        """Return a new :class:`STData` object with the data mirrored
        along the given axis.

        Parameters
        ----------
        axis : int, optional
            Choose between the slow axis (0) and
            the fast axis (1).

        Returns
        -------
        STData
            New :class:`STData` object with the updated `data` and
            `basis_vectors`.
        """
        if not axis in [0, 1]:
            raise ValueError('Axis must equal to 0 or 1')
        basis_vectors = np.copy(self.basis_vectors)
        basis_vectors[:, axis] *= -1.
        data = np.flip(self.data, axis=axis + 1)
        whitefield = np.flip(self.whitefield, axis=axis)
        mask = np.flip(self.mask, axis=axis + 1)
        roi = np.copy(self.roi)
        roi[2 * axis] = self.whitefield.shape[axis] - self.roi[2 * axis + 1]
        roi[2 * axis + 1] = self.whitefield.shape[axis] - self.roi[2 * axis]
        return {'basis_vectors': basis_vectors, 'data': data, 'roi': roi,
                'mask': mask, 'whitefield': whitefield,  'pixel_translations': None}

    @dict_to_object
    def update_mask(self, method='perc-bad', pmin=0., pmax=99.99, vmin=0, vmax=65535,
                    update='reset'):
        """Return a new :class:`STData` object with the updated
        bad pixels mask.

        Parameters
        ----------
        method : {'no-bad', 'range-bad', 'perc-bad'}, optional
            Bad pixels masking methods:

            * 'no-bad' (default) : No bad pixels.
            * 'range-bad' : Mask the pixels which values lie outside
              of (`vmin`, `vmax`) range.
            * 'perc-bad' : Mask the pixels which values lie outside
              of the (`pmin`, `pmax`) percentiles.
        vmin, vmax : float, optional
            Lower and upper intensity values of 'range-bad' masking
            method.
        pmin, pmax : float, optional
            Lower and upper percentage values of 'perc-bad' masking
            method.
        update : {'reset', 'multiply'}, optional
            Multiply the new mask and the old one if 'multiply',
            use the new one if 'reset'.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `mask`.
        """
        data = self.get('data')
        if method == 'no-bad':
            mask = np.ones((self.good_frames.size, self.roi[1] - self.roi[0],
                            self.roi[3] - self.roi[2]), dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            offsets = (data - np.median(data))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError('invalid method argument')
        mask_full = self.mask.copy()
        if update == 'reset':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = mask
            return {'mask': mask_full, 'whitefield': None}
        if update == 'multiply':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] *= mask
            return {'mask': mask_full, 'whitefield': None}

    @dict_to_object
    def make_whitefield(self):
        """Return a new :class:`STData` object with the updated `whitefield`.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `whitefield`.
        """
        return {'whitefield': None}

    @dict_to_object
    def make_pixel_map(self):
        """Return a new :class:`STData` object with the default `pixel_map`.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `pixel_map`.
        """
        return {'pixel_map': None}

    @dict_to_object
    def update_defocus(self, defocus_fs, defocus_ss=None):
        """Return a new :class:`STData` object with the updated
        defocus distances `defocus_fs` and `defocus_ss` for
        the fast and slow detector axes accordingly. Update
        `pixel_translations` based on the new defocus distances.

        Parameters
        ----------
        defocus_fs : float
            Defocus distance for the fast detector axis [m].
        defocus_ss : float, optional
            Defocus distance for the slow detector axis [m].
            Equals to `defocus_fs` if it's not provided.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `defocus_ss`,
            `defocus_fs`, and `pixel_translations`.
        """
        if defocus_ss is None:
            defocus_ss = defocus_fs
        return {'defocus_ss': defocus_ss, 'defocus_fs': defocus_fs,
                'pixel_translations': None}

    def update_phase(self, st_obj):
        """Update `pixel_aberrations`, `phase`, and `reference_image`
        based on :class:`SpeckleTracking` object `st_obj` data. `st_obj`
        must be derived from the :class:`STData` object, an error is
        raised otherwise.

        Parameters
        ----------
        st_obj : SpeckleTracking
            :class:`SpeckleTracking` object derived from the :class:`STData` object.

        Returns
        -------
        STData
            :class:`STData` object with the updated `pixel_aberrations`,
            `phase`, and `reference_image`.

        Raises
        ------
        ValueError
            If `st_obj` doesn't belong to the :class:`STData` object.
        """
        if not st_obj in self._st_objects:
            raise ValueError("the SpeckleTracking object doesn't belong to the data container")
        # Update phase, pixel_aberrations, and reference_image
        dev_ss, dev_fs = (st_obj.pixel_map - self.get('pixel_map'))
        dev_ss -= dev_ss.mean()
        dev_fs -= dev_fs.mean()
        self.pixel_aberrations = np.zeros(self.pixel_map.shape)
        self.pixel_aberrations[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = np.stack((dev_ss, dev_fs))

        # Calculate magnification for fast and slow axes
        mag_ss = np.abs((self.distance + self.defocus_ss) / self.defocus_ss)
        mag_fs = np.abs((self.distance + self.defocus_fs) / self.defocus_fs)

        # Calculate the distance between the reference and the detector plane
        dist_ss = self.distance * (1 - mag_ss**-1)
        dist_fs = self.distance * (1 - mag_fs**-1)

        # dTheta = delta_pix / distance / magnification * du
        # Phase = 2 * pi / wavelength * Integrate[dTheta, delta_pix]
        phase = ct_integrate(self.y_pixel_size**2 / dist_ss / mag_ss * dev_ss,
                             self.x_pixel_size**2 / dist_fs / mag_fs * dev_fs)
        phase *= 2 * np.pi / self.wavelength
        self.phase = np.zeros(self.whitefield.shape)
        self.phase[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = phase
        self.error_frame = np.zeros(self.whitefield.shape)
        self.error_frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = st_obj.error_frame
        self.reference_image = st_obj.reference_image

        # Initialize AberrationsFit objects
        self._ab_fits.clear()
        self._ab_fits.extend([AberrationsFit.import_data(self, axis=0),
                                AberrationsFit.import_data(self, axis=1)])
        return self

    def fit_phase(self, center=0, axis=1, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """Fit `pixel_aberrations` with the polynomial function
        using nonlinear least-squares algorithm. The function uses
        least-squares algorithm from :func:`scipy.optimize.least_squares`.

        Parameters
        ----------
        center : int, optional
            Index of the zerro scattering angle or direct
            beam pixel.
        axis : int, optional
            Axis along which `pixel_aberrations` is fitted.
        max_order : int, optional
            Maximum order of the polynomial model function.
        xtol : float, optional
            Tolerance for termination by the change of the independent variables.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
            Determines the loss function. The following keyword values are
            allowed:

            * 'linear' : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy' (default) : ``rho(z) = ln(1 + z)``. Severely weakens
              outliers influence, but may cause difficulties in optimization
              process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * c_3 : Third order aberrations coefficient [rad / mrad^3].
            * c_4 : Fourth order aberrations coefficient [rad / mrad^4].
            * fit : Array of the polynomial function coefficients of the
              pixel aberrations fit.
            * ph_fit : Array of the polynomial function coefficients of
              the phase aberrations fit.
            * rel_err : Vector of relative errors of the fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
        AberrationsFit.fit - Full details of the aberrations fitting algorithm.
        """
        if self._isphase:
            ab_fit = self.get_fit(center, axis)
            return ab_fit.fit(max_order=max_order, xtol=xtol, ftol=ftol, loss=loss)
        else:
            return None

    def defocus_sweep(self, defoci_fs, defoci_ss=None, size=51, ls_ri=None, return_extra=False):
        r"""Calculate a set of reference images for each defocus in `defoci` and
        return an average R-characteristic of an image (the higher the value the sharper
        reference image is). Return the intermediate results if `return_extra` is True.

        Parameters
        ----------
        defoci_fs : numpy.ndarray
            Array of defocus distances along the fast detector axis [m].
        defoci_ss : numpy.ndarray, optional
            Array of defocus distances along the slow detector axis [m].
        size : int, optional
            Local variance filter size in pixels.
        ls_ri : float, optional
            Reference image kernel bandwidth in pixels.
        return_extra : bool, optional
            Return a dictionary with the intermediate results if True.

        Returns
        -------
        r_vals : numpy.ndarray
            Array of the average values of `reference_image` gradients squared.
        extra : dict
            Dictionary with the intermediate results. Only if `return_extra` is True.
            Contains the following data:

            * reference_image : The generated set of reference profiles.
            * r_images : The set of local variance images of reference profiles.

        Notes
        -----
        R-characteristic is called a local variance and is given by:

        .. math::
            R[i, j] = \frac{\sum_{i^{\prime} = -N / 2}^{N / 2}
            \sum_{j^{\prime} = -N / 2}^{N / 2} (I[i - i^{\prime}, j - j^{\prime}]
            - \bar{I}[i, j])^2}{\bar{I}^2[i, j]}

        where :math:`\bar{I}[i, j]` is a local mean and defined as follows:

        .. math::
            \bar{I}[i, j] = \frac{1}{N^2} \sum_{i^{\prime} = -N / 2}^{N / 2}
            \sum_{j^{\prime} = -N / 2}^{N / 2} I[i - i^{\prime}, j - j^{\prime}]

        See Also
        --------
        SpeckleTracking.update_reference : `reference_image` update algorithm.
        """
        if ls_ri is None:
            ls_ri = size / 2
        if defoci_ss is None:
            defoci_ss = defoci_fs.copy()
        r_vals = []
        extra = {'reference_image': [], 'r_image': []}
        kernel = np.ones(int(size)) / size
        for defocus_fs, defocus_ss in tqdm(zip(defoci_fs.ravel(), defoci_ss.ravel()),
                                           total=len(defoci_fs),
                                           desc='Generating defocus sweep'):
            st_data = self.update_defocus(defocus_fs, defocus_ss)
            st_obj = st_data.get_st().update_reference(ls_ri=ls_ri)
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

    def get(self, attr, value=None):
        """Return a dataset with `mask` and `roi` applied.
        Return `value` if the attribute is not found.

        Parameters
        ----------
        attr : str
            Attribute to fetch.
        value : object, optional
            Return if `attr` is not found.

        Returns
        -------
        numpy.ndarray or object
            `attr` dataset with `mask` and `roi` applied.
            `value` if `attr` is not found.
        """
        if attr in self:
            val = super(STData, self).get(attr)
            if not val is None:
                if attr in ['data', 'error_frame', 'mask', 'phase',
                            'pixel_aberrations', 'pixel_map', 'whitefield']:
                    val = np.ascontiguousarray(val[..., self.roi[0]:self.roi[1],
                                                        self.roi[2]:self.roi[3]])
                if attr in ['basis_vectors', 'data', 'mask', 'pixel_translations',
                            'translations']:
                    val = np.ascontiguousarray(val[self.good_frames])
            return val
        return value

    def get_st(self, aberrations=False, ff_correction=False):
        """Return :class:`SpeckleTracking` object derived
        from the container. Return None if `defocus_fs`
        or `defocus_ss` doesn't exist in the container.

        Parameters
        ----------
        aberrations : bool, optional
            Add `pixel_aberrations` to `pixel_map` of
            :class:`SpeckleTracking` object if it's True.
        ff_correction : bool, optional
            Apple dynamic flatfield correction if it's True.

        Returns
        -------
        SpeckleTracking or None
            An instance of :class:`SpeckleTracking` derived
            from the container. None if `defocus_fs` or
            `defocus_ss` is None.
        """
        if not self._isdefocus:
            return None
        return SpeckleTracking.import_data(self, aberrations, ff_correction)

    def get_st_list(self):
        """Return a list of all the :class:`SpeckleTracking`
        objects bound to the container.

        Returns
        -------
        list
            List of all the :class:`SpeckleTracking` objects
            bound to the container.
        """
        return self._st_objects.copy()

    def get_fit(self, center=0, axis=1):
        """Return an :class:`AberrationsFit` object for
        parametric regression of the lens' aberrations
        profile. Return None if `defocus_fs` or
        `defocus_ss` doesn't exist in the container.

        Parameters
        ----------
        center : int, optional
            Index of the zerro scattering angle or direct
            beam pixel.
        axis : int, optional
            Detector axis along which the fitting is performed.

        Returns
        -------
        AberrationsFit or None
            An instance of :class:`AberrationsFit` class.
            None if `defocus_fs` or `defocus_ss` is None.
        """
        if not self._isphase:
            return None
        return AberrationsFit.import_data(self, center=center, axis=axis)

    def get_pca(self):
        """Perform the Principal Component Analysis [PCA]_ of the measured data and
        return a set of eigen flat fields (EFF).

        Returns
        -------
        effs_var : numpy.ndarray
            Variance ratio for each EFF, that it describes.
        effs : numpy.ndarray
            Set of eigen flat fields.

        References
        ----------
        .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                 Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                 normalization using eigen flat fields in X-ray imaging," Opt. Express
                 23, 27975-27989 (2015).
        """
        data = self.get('data') * self.get('mask') - self.get('whitefield')
        mat_svd = np.tensordot(data, data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, data, axes=((0,), (0,)))
        return eig_vals / eig_vals.sum(), effs

    @dict_to_object
    def update_flatfields(self, effs):
        """Update flatfields based on a set of eigen flat fields `effs`.

        Parameters
        ----------
        effs : numpy.ndarray
            Set of the most important eigen flat fields.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `flatfields`.
        """
        data = self.get('data') * self.get('mask') - self.get('whitefield')
        weights = np.tensordot(data, effs, axes=((1, 2), (1, 2))) / np.sum(effs * effs, axis=(1, 2))
        flatfields = np.tensordot(weights, effs, axes=((1,), (0,))) + self.get('whitefield')
        return {'flatfields': flatfields}

    def write_cxi(self, cxi_file, overwrite=True):
        """Write all the `attr` to a CXI file `cxi_file`.

        Parameters
        ----------
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.
        overwrite : bool, optional
            Overwrite the content of `cxi_file` file if it's True.

        Raises
        ------
        ValueError
            If `overwrite` is False and the data is already present
            in `cxi_file`.
        """
        for attr, data in self.items():
            if attr in self.protocol:
                self.protocol.write_cxi(attr, data, cxi_file, overwrite=overwrite)

class SpeckleTracking(DataContainer):
    """Wrapper class for the  Robust Speckle Tracking algorithm.
    Performs `reference_image` and `pixel_map` updates.

    Parameters
    ----------
    st_data : STData
        The Speckle tracking data container, from which the
        object is derived.
    **kwargs : dict
        Dictionary of the attributes' data specified in `attr_set`
        and `init_set`.

    Attributes
    ----------
    attr_set : set
        Set of attributes in the container which are necessary
        to initialize in the constructor.
    init_set : set
        Set of optional data attributes.

    Raises
    ------
    ValueError
        If an attribute specified in `attr_set` has not been provided.

    Notes
    -----
    Necessary attributes:

    * data : Measured frames.
    * dss_pix : The sample's translations along the slow axis
      in pixels.
    * dfs_pix : The sample's translations along the fast axis in pixels.
    * whitefield : Measured frames' whitefield.
    * pixel_map : The pixel mapping between the data at the detector's
      plane and the reference image at the reference plane.
    * num_threads : Specify number of threads that are used in all the
      calculations.

    Optional attributes:

    * error_frame : MSE (mean-squared-error) of the reference image
      and pixel mapping fit per pixel.
    * ls_ri : Smoothing kernel bandwidth used in `reference_image`
      regression. The value is given in pixels.
    * reference_image : The unabberated reference image of the sample.
    * m0 : The lower bounds of the fast detector axis of
      the reference image at the reference frame in pixels.
    * n0 : The lower bounds of the slow detector axis of
      the reference image at the reference frame in pixels.

    See Also
    --------
    bin.make_reference : Full details of the `reference_image` update
        algorithm.
    bin.update_pixel_map_gs : Full details of the `pixel_map` update algorithm
        based on grid search.
    """
    attr_set = {'data', 'dfs_pix', 'dss_pix', 'num_threads', 'pixel_map', 'whitefield'}
    init_set = {'error_frame', 'ls_ri', 'n0', 'm0', 'reference_image'}

    def __init__(self, st_data, **kwargs):
        self.__dict__['_reference'] = st_data
        super(SpeckleTracking, self).__init__(**kwargs)
        self._reference._st_objects.append(self)

    def __repr__(self):
        with np.printoptions(threshold=6, edgeitems=2, suppress=True, precision=3):
            return {key: val.ravel() if isinstance(val, np.ndarray) else val
                    for key, val in self.attr_dict.items()}.__repr__()

    def __str__(self):
        with np.printoptions(threshold=6, edgeitems=2, suppress=True, precision=3):
            return {key: val.ravel() if isinstance(val, np.ndarray) else val
                    for key, val in self.attr_dict.items()}.__str__()

    @classmethod
    def import_data(cls, st_data, aberrations=False, ff_correction=False):
        """Return a new :class:`SpeckleTracking` object
        with all the necessary data attributes imported from
        the :class:`STData` container object `st_data`.

        Parameters
        ----------
        st_data : STData
            :class:`STData` container object.
        aberrations : bool, optional
            Add `pixel_aberrations` from `st_data` to
            `pixel_map` if it's True.
        ff_correction : bool, optional
            Apple dynamic flatfield correction if it's True.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object.
        """
        data = st_data.get('mask') * st_data.get('data')
        pixel_map = st_data.get('pixel_map')
        if aberrations:
            pixel_map += st_data.get('pixel_aberrations')
        whitefield = st_data.get('whitefield')
        if ff_correction:
            flatfields = st_data.get('flatfields')
            if not flatfields is None:
                data *= np.where(flatfields > 0, whitefield / flatfields, 1.)
        dij_pix = np.ascontiguousarray(np.swapaxes(st_data.get('pixel_translations'), 0, 1))
        return cls(data=data, st_data=st_data, dfs_pix=dij_pix[1], dss_pix=dij_pix[0],
                   num_threads=st_data.num_threads, pixel_map=pixel_map,
                   whitefield=whitefield)

    @dict_to_object
    def update_reference(self, ls_ri, sw_fs=0, sw_ss=0):
        """Return a new :class:`SpeckleTracking` object
        with the updated `reference_image`.

        Parameters
        ----------
        ls_ri : float
            Smoothing kernel bandwidth used in `reference_image`
            regression. The value is given in pixels.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        sw_fs : int, optional
            Search window size in pixels along the fast detector
            axis.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `reference_image`.

        See Also
        --------
        bin.make_reference : Full details of the `reference_image` update
            algorithm.
        """
        I0, n0, m0 = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                    di=self.dss_pix, dj=self.dfs_pix, sw_ss=sw_ss,
                                    sw_fs=sw_fs, ls=ls_ri, num_threads=self.num_threads)
        return {'st_data': self._reference, 'ls_ri': ls_ri, 'n0': n0, 'm0': m0,
                'reference_image': I0}

    @dict_to_object
    def update_pixel_map(self, ls_pm, sw_fs, sw_ss=0, method='search'):
        """Return a new :class:`SpeckleTracking` object with
        the updated `pixel_map`.

        Parameters
        ----------
        ls_pm : float
            Smoothing kernel bandwidth used in `pixel_map`
            regression. The value is given in pixels.
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        method : {'search', 'newton'}, optional
            `pixel_map` update algorithm. The following keyword
            values are allowed:

            * 'newton' : Iterative Newton's method based on
              finite difference gradient approximation.
            * 'search' : Grid search along the search window.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `pixel_map`.

        Raises
        ------
        AttributeError
            If `reference_image` was not generated beforehand.
        ValueError
            If `method` keyword value is not valid.

        See Also
        --------
        bin.update_pixel_map_gs, bin.update_pixel_map_nm : Full details
            of the `pixel_map` update algorithm.
        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        if method == 'search':
            pixel_map = update_pixel_map_gs(I_n=self.data, W=self.whitefield,
                                            I0=self.reference_image, u0=self.pixel_map,
                                            di=self.dss_pix - self.n0,
                                            dj=self.dfs_pix - self.m0,
                                            sw_ss=sw_ss, sw_fs=sw_fs, ls=ls_pm,
                                            num_threads=self.num_threads)
        elif method == 'newton':
            pixel_map = update_pixel_map_nm(I_n=self.data, W=self.whitefield,
                                            I0=self.reference_image, u0=self.pixel_map,
                                            di=self.dss_pix - self.n0,
                                            dj=self.dfs_pix - self.m0,
                                            sw_fs=sw_fs, ls=ls_pm,
                                            num_threads=self.num_threads)
        else:
            raise ValueError('Wrong method argument: {:s}'.format(method))
        return {'st_data': self._reference, 'pixel_map': pixel_map}

    @dict_to_object
    def update_errors(self):
        """Return a new :class:`SpeckleTracking` object with
        the updated `error_frame`.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `error_frame`.

        Raises
        ------
        AttributeError
            If `reference_image` was not generated beforehand.

        See Also
        --------
        bin.mse_frame : Full details of the error metric.
        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        error_frame = mse_frame(I_n=self.data, W=self.whitefield, ls=self.ls_ri,
                                I0=self.reference_image, u=self.pixel_map,
                                di=self.dss_pix - self.n0, dj=self.dfs_pix - self.m0,
                                num_threads=self.num_threads)
        return {'st_data': self._reference, 'error_frame': error_frame}

    @dict_to_object
    def update_translations(self, sw_fs, sw_ss=0):
        """Return a new :class:`SpeckleTracking` object with
        the updated sample pixel translations (`dss_pix`,
        `dfs_pix`).

        Parameters
        ----------
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        ls_ri: float, optional
            Smoothing kernel bandwidth used in `reference_image`
            regression. The value is given in pixels.
        method : {'search', 'newton'}, optional
            `pixel_map` update algorithm. The following keyword
            values are allowed:

            * 'newton' : Iterative Newton's method based on
              finite difference gradient approximation.
            * 'search' : Grid search along the search window.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `dss_pix`, `dfs_pix`.

        Raises
        ------
        AttributeError
            If `reference_image` was not generated beforehand.

        See Also
        --------
        bin.update_translations_gs : Full details of the sample
            translations update algorithm.
        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        dij = update_translations_gs(I_n=self.data, W=self.whitefield,
                                        I0=self.reference_image, u=self.pixel_map,
                                        di=self.dss_pix - self.n0,
                                        dj=self.dfs_pix - self.m0,
                                        sw_ss=sw_ss, sw_fs=sw_fs, ls=self.ls_ri,
                                        num_threads=self.num_threads)
        dss_pix = np.ascontiguousarray(dij[:, 0]) + self.n0
        dfs_pix = np.ascontiguousarray(dij[:, 1]) + self.m0
        return {'st_data': self._reference, 'dss_pix': dss_pix, 'dfs_pix': dfs_pix}

    def iter_update_gd(self, ls_ri, ls_pm, sw_fs, sw_ss=0, blur=None, n_iter=30, f_tol=0.,
                       momentum=0., learning_rate=1e1, gstep=.1, method='search',
                       update_translations=False, verbose=False, return_extra=False):
        """Perform iterative Robust Speckle Tracking update. `ls_ri` and
        `ls_pm` define high frequency cut-off to supress the noise. `ls_ri`
        is iteratively updated by dint of Gradient Descent with momentum
        update procedure. Iterative update terminates when the difference
        between total mean-squared-error (MSE) values of the two last iterations
        is less than `f_tol`.

        Parameters
        ----------
        ls_ri : float
            Smoothing kernel bandwidth used in `reference_image`
            regression. The value is given in pixels.
        ls_pm : float
            Smoothing kernel bandwidth used in `pixel_map`
            regression. The value is given in pixels.
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        blur : float, optional
            Smoothing kernel bandwidth used in `reference_image`
            post-update. The default value is equal to `ls_pm`. The value
            is given in pixels.
        n_iter : int, optional
            Maximum number of iterations.
        f_tol : float, optional
            Tolerance for termination by the change of the total MSE.
        momentum : float, optional
            hyperparameter in interval (0.0-1.0) that accelerates gradient
            descent of `ls_ri` in the relevant direction and dampens oscillations.
        learning_rate : float, optional
            Learning rate for `ls_ri` update procedure.
        gstep : float, optional
            `ls_ri` step used to numerically calculate the MSE gradient.
        method : {'search', 'newton'}, optional
            `pixel_map` update algorithm. The following keyword
            values are allowed:

            * 'newton' : Iterative Newton's method based on
              finite difference gradient approximation.
            * 'search' : Grid search along the search window.
        update_translations : bool, optional
            Update sample pixel translations if True.
        verbose : bool, optional
            Set verbosity of the computation process.
        return_extra : bool, optional
            Return errors and `ls_ri` array if True.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `pixel_map` and `reference_image`. `dss_pix` and `dfs_pix`
            are also updated if `update_translations` is True.
        list
            List of total MSE values for each iteration.  Only if
            `return_errors` is True.
        """
        if blur is None:
            blur = ls_pm
        velocity = 0.0
        obj = self.update_reference(ls_ri=ls_ri)
        obj.update_errors.inplace_update()
        extra = {'errors': [obj.error_frame.mean()],
                 'lss_ri': [ls_ri]}
        itor = range(1, n_iter + 1)
        if verbose:
            itor = tqdm(itor, bar_format='{desc} {percentage:3.0f}% {bar} '\
                      'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
            print(f"Initial MSE = {extra['errors'][-1]:.6f}, "\
                  f"Initial ls_ri = {extra['lss_ri'][-1]:.2f}")
        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(ls_pm=ls_pm, sw_fs=sw_fs, sw_ss=sw_ss, method=method)
            obj.pixel_map += gaussian_filter(new_obj.pixel_map - obj.pixel_map,
                                             (0, blur, blur), mode='nearest', num_threads=self.num_threads)

            # Update dss_pix, dfs_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_ss=sw_ss, sw_fs=sw_fs)
                obj.dss_pix, obj.dfs_pix = new_obj.dss_pix, new_obj.dfs_pix

            # Update ls_ri
            grad = (obj.mse_total(ls_ri + gstep) - extra['errors'][-1]) / gstep
            velocity = np.clip(momentum * velocity - learning_rate * grad,
                               -0.75 * ls_ri, 0.75 * ls_ri)
            ls_ri += velocity
            extra['lss_ri'].append(ls_ri)

            # Update reference_image
            obj.update_reference.inplace_update(ls_ri=ls_ri)
            obj.update_errors.inplace_update()
            extra['errors'].append(obj.error_frame.mean())
            if verbose:
                itor.set_description(f"Total MSE = {extra['errors'][-1]:.6f}, "\
                                     f"ls_ri = {extra['lss_ri'][-1]:.2f}")

            # Break if function tolerance is satisfied
            if (extra['errors'][-2] - extra['errors'][-1]) <= f_tol:
                break
        if return_extra:
            return obj, extra
        else:
            return obj

    def iter_update(self, ls_ri, ls_pm, sw_fs, sw_ss=0, blur=None, n_iter=5, f_tol=0.,
                    method='search', update_translations=False, verbose=False,
                    return_errors=False):
        """Perform iterative Robust Speckle Tracking update. `ls_ri` and
        `ls_pm` define high frequency cut-off to supress the noise.
        Iterative update terminates when the difference between total
        mean-squared-error (MSE) values of the two last iterations is
        less than `f_tol`.

        Parameters
        ----------
        ls_ri : float
            Smoothing kernel bandwidth used in `reference_image`
            regression. The value is given in pixels.
        ls_pm : float
            Smoothing kernel bandwidth used in `pixel_map`
            regression. The value is given in pixels.
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        blur : float, optional
            Smoothing kernel bandwidth used in `reference_image`
            post-update. The default value is equal to `ls_pm`. The value
            is given in pixels.
        n_iter : int, optional
            Maximum number of iterations.
        f_tol : float, optional
            Tolerance for termination by the change of the total MSE.
        method : {'search', 'newton'}, optional
            `pixel_map` update algorithm. The following keyword
            values are allowed:

            * 'newton' : Iterative Newton's method based on
              finite difference gradient approximation.
            * 'search' : Grid search along the search window.
        update_translations : bool, optional
            Update sample pixel translations if True.
        verbose : bool, optional
            Set verbosity of the computation process.
        return_errors : bool, optional
            Return errors array if True.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `pixel_map` and `reference_image`. `dss_pix` and `dfs_pix`
            are also updated if `update_translations` is True.
        list
            List of total MSE values for each iteration.  Only if
            `return_errors` is True.
        """
        if blur is None:
            blur = ls_pm
        obj = self.update_reference(ls_ri=ls_ri)
        obj.update_errors.inplace_update()
        errors = [obj.error_frame.mean()]
        itor = range(1, n_iter + 1)
        if verbose:
            itor = tqdm(itor, bar_format='{desc} {percentage:3.0f}% {bar} '\
                        'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
            print(f"Initial MSE = {errors[0]:.6f}")
        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(ls_pm=ls_pm, sw_fs=sw_fs, sw_ss=sw_ss, method=method)
            obj.pixel_map += gaussian_filter(new_obj.pixel_map - obj.pixel_map,
                                             (0, blur, blur), mode='nearest', num_threads=self.num_threads)

            # Update dss_pix, dfs_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_ss=sw_ss, sw_fs=sw_fs)
                obj.dss_pix, obj.dfs_pix = new_obj.dss_pix, new_obj.dfs_pix

            # Update reference_image
            obj.update_reference.inplace_update(ls_ri=ls_ri)
            obj.update_errors.inplace_update()

            # Calculate errors
            errors.append(obj.error_frame.mean())
            if verbose:
                itor.set_description(f"Total MSE = {errors[-1]:.6f}")

            # Break if function tolerance is satisfied
            if (errors[-2] - errors[-1]) <= f_tol * errors[0]:
                break
        if return_errors:
            return obj, errors
        else:
            return obj

    def mse_total(self, ls_ri):
        """Generate a reference image with the given `ls_ri` and return
        average total mean-squared-error (MSE).

        Parameters
        ----------
        ls_ri : float
            `reference_image` length scale in pixels.

        Returns
        -------
        float
            Average total mean-squared-error (MSE).


        See Also
        --------
        bin.mse_total : Full details of the error metric.
        """
        I0, n0, m0 = make_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                    di=self.dss_pix, dj=self.dfs_pix, sw_ss=0, sw_fs=0, ls=ls_ri,
                                    num_threads=self.num_threads)
        return mse_total(I_n=self.data, W=self.whitefield, I0=I0,
                         u=self.pixel_map, di=self.dss_pix - n0,
                         dj=self.dfs_pix - m0, ls=ls_ri,
                         num_threads=self.num_threads)

    def mse_curve(self, lss_ri):
        """Return a mean-squared-error (MSE) survace.

        Parameters
        ----------
        lss_ri : numpy.ndarray
            Set of `reference_image` length scales in pixels.

        Returns
        -------
        numpy.ndarray
            A mean-squared-error (MSE) surface.

        See Also
        --------
        bin.mse_total : Full details of the error metric.
        """
        mse_list = []
        for ls_ri in np.array(lss_ri, ndmin=1):
            mse_list.append(self.mse_total(ls_ri))
        return np.array(mse_list)
