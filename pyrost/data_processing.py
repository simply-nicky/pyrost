"""Robust Speckle Tracking data processing algorithm.
:class:`STData` contains all the necessary data for the Speckle
Tracking algorithm, and handy data processing tools to work
with the data. :class:`SpeckleTracking` performs the main
Robust Speckle Tracking algorithm and yields reference image
and pixel mapping. :class:`AbberationsFit` fit the lens'
abberations profile with the polynomial function using
nonlinear least-squares algorithm.

Examples
--------
Extract all the necessary data using a :func:`pyrost.protocol.loader` function.

>>> import pyrost as rst
>>> loader = rst.loader()
>>> rst_data = loader.load('results/test/data.cxi'

Perform the Robust Speckle Tracking using a :class:`SpeckleTracking` object.

>>> rst_obj = rst_data.get_st()
>>> rst_res, errors = rst_obj.iter_update([0, 150], ls_pm=3., ls_ri=10,
...                                       verbose=True, n_iter=10)
Iteration No. 0: Total MSE = 0.150
Iteration No. 1: Total MSE = 0.077
Iteration No. 2: Total MSE = 0.052
Iteration No. 3: Total MSE = 0.050

Extract lens' abberations wavefront and fit it with a polynomial.

>>> rst_data.update_phase(rst_res)
>>> fit_res = rst_data.fit_phase()
>>> fit_res['ph_fit']
array([-5.19025587e+07, -8.63773622e+05,  3.42849675e+03,  2.98523995e+01,
        1.19773905e-02])
"""
from __future__ import division

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from .abberations_fit import AbberationsFit
from .data_container import DataContainer, dict_to_object
from .bin import make_whitefield, make_reference, update_pixel_map_gs, update_pixel_map_nm, total_mse, ct_integrate

class STData(DataContainer):
    """Speckle Tracking algorithm data container class.
    Contains all the necessary data for the Robust Speckle
    Tracking algorithm (specified in `attr_set` and `init_set`),
    the list of all the :class:`SpeckleTracking` objects derived
    from it, and two :class:`AbberationsFit` objects to fit the phase
    along the fast and slow detector axes.

    Parameters
    ----------
    protocol : Protocol
        CXI :class:`Protocol` object.

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
    * good_frames : An array of good frames' indices.
    * mask : Bad pixels mask.
    * phase : Phase profile of lens' abberations.
    * pixel_map : The pixel mapping between the data at the detector's
      plane and the reference image at the reference plane.
    * pixel_abberations : Lens' abberations along the fast and
      slow axes in pixels.
    * pixel_translations : Sample's translations in the detector's
      plane in pixels.
    * reference_image : The unabberated reference image of the sample.
    * roi : Region of interest in the detector's plane.
    * whitefield : Measured frames' whitefield.
    """
    attr_set = {'basis_vectors', 'data', 'distance', 'translations', 'wavelength',
                'x_pixel_size', 'y_pixel_size'}
    init_set = {'defocus_fs', 'defocus_ss', 'good_frames', 'mask', 'phase',
                'pixel_abberations', 'pixel_map', 'pixel_translations',
                'reference_image', 'roi', 'whitefield'}

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
        # Set ROI, good frames array, mask, and whitefield
        if self.roi is None:
            self.roi = np.array([0, self.data.shape[1], 0, self.data.shape[2]])
        if self.good_frames is None:
            self.good_frames = np.arange(self.data.shape[0])
        if self.mask is None or self.mask.shape != self.data.shape[1:]:
            self.mask = np.ones(self.data.shape[1:])
        if self.whitefield is None:
            self.whitefield = make_whitefield(data=self.data[self.good_frames], mask=self.mask)

        # Initialize a list of SpeckleTracking objects
        self._st_objects = []

        # Initialize a list of AbberationsFit objects
        self._ab_fits = []

        # Set a pixel map, deviation angles, and phase
        if self.pixel_map is None:
            self._init_pixel_map()

        if self._isdefocus:
            # Set a pixel translations
            if self.pixel_translations is None:
                self.pixel_translations = (self.translations[:, None] * self.basis_vectors).sum(axis=-1)
                defocus = np.array([self.defocus_ss, self.defocus_fs])
                self.pixel_translations /= (self.basis_vectors**2).sum(axis=-1) * defocus / self.distance
                self.pixel_translations -= self.pixel_translations.mean(axis=0)

            # Initialize AbberationsFit objects
            self._ab_fits.extend([AbberationsFit.import_data(self, 0),
                                  AbberationsFit.import_data(self, 1)])

    @property
    def _isdefocus(self):
        return not self.defocus_ss is None and not self.defocus_fs is None

    def _init_pixel_map(self):
        self.pixel_map = np.indices(self.whitefield.shape)
        self.pixel_abberations = np.zeros(self.pixel_map.shape)
        self.phase = np.zeros(self.whitefield.shape)

    def __setattr__(self, attr, value):
        if attr in self.attr_set | self.init_set:
            value = np.array(value, dtype=self.protocol.get_dtype(attr))
            super(STData, self).__setattr__(attr, value)
        else:
            super(STData, self).__setattr__(attr, value)

    @dict_to_object
    def crop_data(self, roi):
        """Return a new :class:`STData` object with the updated `roi`.

        Parameters
        ----------
        roi : iterable
            Region of interest in the detector's plane.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `roi`.
        """
        return {'roi': np.asarray(roi, dtype=np.int)}

    @dict_to_object
    def mask_frames(self, good_frames):
        """Return a new :class:`STData` object with the updated
        good frames mask.

        Parameters
        ----------
        good_frames : iterable
            List of good frames' indices.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `good_frames`
            and `whitefield`.
        """
        return {'good_frames': np.asarray(good_frames, dtype=np.int),
                'whitefield': None}

    @dict_to_object
    def make_mask(self, method='no-bad', percentile=99.99):
        """Return a new :class:`STData` object with the updated
        bad pixels mask.

        Parameters
        ----------
        method : {'no-bad', 'eiger-bad', 'perc-bad'}, optional
            Bad pixels masking methods:

            * 'no-bad' (default) : No bad pixels.
            * 'eiger-bad' : Mask the pixels which value are higher
              than 65535 a.u. (for EIGER detector).
            * 'perc-bad' : mask the pixels which values lie outside
              of the q-th percentile. Provide percentile value with
              `percentile` argument.

        percentile : float, optional
            Percentile to compute. Defines the 'perc-bad' masking method.

        Returns
        -------
        STData
            New :class:`STData` object with the updated `mask`.
        """
        if method == 'no-bad':
            mask = np.ones(self.data.shape[1:])
        elif method == 'eiger-bad':
            mask = (self.data < 65535).all(axis=0)
        elif method == 'perc-bad':
            data_offset = (self.data - np.median(self.data))**2
            mask = (data_offset < np.percentile(data_offset, percentile)).all(axis=0)
        else:
            ValueError('invalid method argument')
        return {'mask': mask, 'whitefield': None}

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
        """Update `pixel_abberations`, `phase`, and `reference_image`
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
            :class:`STData` object with the updated `pixel_abberations`,
            `phase`, and `reference_image`.

        Raises
        ------
        ValueError
            If `st_obj` doesn't belong to the :class:`STData` object.
        """
        if st_obj in self._st_objects:
            self._init_pixel_map()
            dev_ss, dev_fs = (st_obj.pixel_map - self.get('pixel_map'))
            dev_ss -= dev_ss.mean()
            dev_fs -= dev_fs.mean()
            self.pixel_abberations[:, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = np.stack((dev_ss, dev_fs))
            phase = ct_integrate(self.y_pixel_size**2 * dev_ss * self.defocus_ss / self.wavelength,
                                 self.x_pixel_size**2 * dev_fs * self.defocus_fs / self.wavelength) \
                    * 2 * np.pi / self.distance**2
            self.phase[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = phase
            self.reference_image = st_obj.reference_image
            return self
        else:
            raise ValueError("the SpeckleTracking object doesn't belong to the data container")

    def fit_phase(self, axis=1, max_order=2, xtol=1e-14, ftol=1e-14, loss='cauchy'):
        """Fit `pixel_abberations` with the polynomial function
        using nonlinear least-squares algorithm. The function uses
        least-squares algorithm from :func:`scipy.optimize.least_squares`.

        Parameters
        ----------
        axis : int, optional
            Axis along which `pixel_abberations` is fitted.
        max_order : int, optional
            Maximum order of the polynomial model function.
        xtol : float, optional
            Tolerance for termination by the change of the independent variables.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}, optional
            Determines the loss function. The following keyword values are
            allowed:

            * 'linear : ``rho(z) = z``. Gives a standard
              least-squares problem.
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
              approximation of l1 (absolute value) loss. Usually a good
              choice for robust least squares.
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
              similarly to 'soft_l1'.
            * 'cauchy'' (default) : ``rho(z) = ln(1 + z)``. Severely weakens
              outliers influence, but may cause difficulties in optimization
              process.
            * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
              a single residual, has properties similar to 'cauchy'.

        Returns
        -------
        dict
            :class:`dict` with the following fields defined:

            * pix_fit : Array of the polynomial function coefficients of
              the `pixel_abberations` fit.
            * ang_fit : Array of the polynomial function coefficients of
              the deviation angles fit.
            * ph_fit : Array of the polynomial function coefficients of
              the `phase` fit.
            * pix_err : Vector of errors of the `pix_fit` fit coefficients.
            * r_sq : ``R**2`` goodness of fit.

        See Also
        --------
        :func:`scipy.optimize.least_squares` : Full nonlinear least-squares
            algorithm description.
        """
        fit_obj = self.get_fit(axis)
        if fit_obj is None:
            return None
        else:
            pixel_ab = self.get('pixel_abberations')[axis].mean(axis=1 - axis)
            return fit_obj.fit_pixel_abberations(pixel_ab, max_order=max_order,
                                                 xtol=xtol, ftol=ftol, loss=loss)

    def defocus_sweep(self, defoci_fs, defoci_ss=None, ls_ri=30):
        """Calculate an array of `reference_image` for `defoci`
        and return fugures of merit of `reference_image` sharpness
        (the higher the value the sharper `reference_image` is).
        `ls_ri` should be large enough in order to supress
        high frequency noise.

        Parameters
        ----------
        defoci_fs : numpy.ndarray
            Array of defocus distances along the fast detector axis [m].
        defoci_ss : numpy.ndarray, optional
            Array of defocus distances along the slow detector axis [m].
        ls_ri : float, optional
            `reference_image` length scale in pixels.

        Returns
        -------
        numpy.ndarray
            Array of the average values of `reference_image` gradients squared.

        See Also
        --------
        SpeckleTracking.update_reference : `reference_image` update algorithm.
        """
        if defoci_ss is None:
            defoci_ss = defoci_fs.copy()
        sweep_scan = []
        for defocus_fs, defocus_ss in zip(defoci_fs, defoci_ss):
            st_data = self.update_defocus(defocus_fs, defocus_ss)
            st_obj = st_data.get_st().update_reference(ls_ri=ls_ri)
            ri_gm = gaussian_gradient_magnitude(st_obj.reference_image, sigma=ls_ri)
            sweep_scan.append(np.mean(ri_gm**2))
        return np.array(sweep_scan)

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
                if attr in ['data', 'pixel_abberations', 'mask', 'phase', 'pixel_map', 'whitefield']:
                    val = val[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    val = np.ascontiguousarray(val)
                if attr in ['basis_vectors', 'data', 'pixel_translations', 'translations']:
                    val = val[self.good_frames]
                    val = np.ascontiguousarray(val)
            return val
        else:
            return value

    def get_st(self):
        """Return :class:`SpeckleTracking` object derived
        from the container. Return None if `defocus_fs`
        or `defocus_ss` doesn't exist in the container.

        Returns
        -------
        SpeckleTracking or None
            An instance of :class:`SpeckleTracking` derived
            from the container. None if `defocus_fs` or
            `defocus_ss` is None.
        """
        if self._isdefocus:
            if self._st_objects:
                return self._st_objects[0]
            else:
                return SpeckleTracking.import_data(self)
        else:
            return None

    def get_st_list(self):
        """Return a list of all the :class:`SpeckleTracking`
        objects bound to the container.

        Returns
        -------
        list
            List of all the :class:`SpeckleTracking` objects
            bound to the container.
        """
        return self._st_objects

    def get_fit(self, axis=1):
        """Return an :class:`AbberationsFit` object for
        parametric regression of the lens' abberations
        profile. Return None if `defocus_fs` or
        `defocus_ss` doesn't exist in the container.

        Parameters
        ----------
        axis : int
            Detector axis along which the fitting is performed.

        Returns
        -------
        AbberationsFit or None
            An instance of :class:`AbberationsFit` class.
            None if `defocus_fs` or `defocus_ss` is None.
        """
        if self._isdefocus:
            return self._ab_fits[axis]
        else:
            return None

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
            if isinstance(data, np.ndarray):
                self.protocol.write_cxi(attr, data, cxi_file, overwrite=overwrite)

class SpeckleTracking(DataContainer):
    """Wrapper class for the  Robust Speckle Tracking algorithm.
    Performs `reference_image` and `pixel_map` updates.

    Parameters
    ----------
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
    * dref : Reference to the :class:`STData` object, from which
      :class:`SpeckleTracking` object was derived.
    * dss_pix : The sample's translations along the slow axis
      in pixels.
    * dfs_pix : The sample's translations along the fast axis in pixels.
    * whitefield : Measured frames' whitefield.
    * pixel_map : The pixel mapping between the data at the detector's
      plane and the reference image at the reference plane.

    Optional attributes:

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
    attr_set = {'data', 'dref', 'dfs_pix', 'dss_pix', 'pixel_map', 'whitefield'}
    init_set = {'n0', 'm0', 'reference_image'}

    def __init__(self, **kwargs):
        super(SpeckleTracking, self).__init__(**kwargs)
        self.dref._st_objects.append(self)

    @classmethod
    def import_data(cls, st_data):
        """Return a new :class:`SpeckleTracking` object
        with all the necessary data attributes imported from
        the :class:`STData` container object `st_data`.

        Parameters
        ----------
        st_data : STData
            :class:`STData` container object.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object.
        """
        mask = st_data.get('mask')
        data = mask * st_data.get('data')
        whitefield = mask * st_data.get('whitefield')
        pixel_map = st_data.get('pixel_map')
        dij_pix = np.ascontiguousarray(np.swapaxes(st_data.get('pixel_translations'), 0, 1))
        return cls(data=data, dref=st_data, dfs_pix=dij_pix[1], dss_pix=dij_pix[0],
                   pixel_map=pixel_map, whitefield=whitefield)

    @dict_to_object
    def update_reference(self, ls_ri=10., sw_ss=0, sw_fs=0):
        """Return a new :class:`SpeckleTracking` object
        with the updated `reference_image`.

        Parameters
        ----------
        ls_ri : float, optional
            `reference_image` length scale in pixels.
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
                                    sw_fs=sw_fs, ls=ls_ri)
        return {'reference_image': I0, 'n0': n0, 'm0': m0}

    @dict_to_object
    def update_pixel_map(self, sw_fs, sw_ss=0, ls_pm=3., method='search'):
        """Return a new :class:`SpeckleTracking` object with
        the updated `pixel_map`.

        Parameters
        ----------
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        ls_pm : float, optional
            `pixel_map` length scale in pixels.
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
        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        else:
            if method == 'search':
                pixel_map = update_pixel_map_gs(I_n=self.data, W=self.whitefield,
                                                I0=self.reference_image, u0=self.pixel_map,
                                                di=self.dss_pix - self.n0,
                                                dj=self.dfs_pix - self.m0,
                                                sw_ss=sw_ss, sw_fs=sw_fs, ls=ls_pm)
            elif method == 'newton':
                pixel_map = update_pixel_map_nm(I_n=self.data, W=self.whitefield,
                                                I0=self.reference_image, u0=self.pixel_map,
                                                di=self.dss_pix - self.n0,
                                                dj=self.dfs_pix - self.m0,
                                                sw_fs=sw_fs, ls=ls_pm)
            else:
                raise ValueError('Wrong method argument: {:s}'.format(method))
            return {'pixel_map': pixel_map}

    def iter_update(self, sw_fs, sw_ss=0, ls_pm=3., ls_ri=10.,
                    n_iter=5, f_tol=3e-3, verbose=True, method='search',
                    return_errors=True):
        """Perform iterative Robust Speckle Tracking update. `ls_ri` and
        `ls_pm` define high frequency cut-off to supress the noise.
        Iterative update terminates when the difference between total
        mean-squared-error (MSE) values of the two last iterations is
        less than `f_tol`.

        Parameters
        ----------
        sw_fs : int
            Search window size in pixels along the fast detector
            axis.
        sw_ss : int, optional
            Search window size in pixels along the slow detector
            axis.
        ls_pm : float, optional
            `pixel_map` length scale in pixels.
        ls_ri : float, optional
            `reference_image` length scale in pixels.
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
        return_errors : bool, optional
            Return errors array if True.

        Returns
        -------
        SpeckleTracking
            A new :class:`SpeckleTracking` object with the updated
            `pixel_map` and `reference_image`.
        list
            List of total MSE values for each iteration.  Only if
            `return_errors` is True.
        """
        obj = self.update_reference(ls_ri=ls_ri, sw_ss=sw_ss, sw_fs=sw_fs)
        errors = [obj.total_mse(ls_ri=ls_ri)]
        if verbose:
            print('Iteration No. 0: Total MSE = {:.3f}'.format(errors[0]))
        for it in range(1, n_iter + 1):
            new_obj = obj.update_reference(ls_ri=ls_ri)
            new_obj.update_pixel_map.inplace_update(sw_ss=sw_ss, sw_fs=sw_fs,
                                                    ls_pm=ls_pm, method=method)

            obj.pixel_map += gaussian_filter(new_obj.pixel_map - obj.pixel_map,
                                             (0, ls_pm, ls_pm))
            ss_roi = np.clip([obj.n0 - new_obj.n0,
                              new_obj.reference_image.shape[0] + obj.n0 - new_obj.n0],
                             0, obj.reference_image.shape[0]) - obj.n0
            fs_roi = np.clip([obj.m0 - new_obj.m0,
                              new_obj.reference_image.shape[1] + obj.m0 - new_obj.m0],
                             0, obj.reference_image.shape[1]) - obj.m0
            ss_s0, fs_s0 = slice(*(ss_roi + obj.n0)), slice(*(fs_roi + obj.m0))
            ss_s1, fs_s1 = slice(*(ss_roi + new_obj.n0)), slice(*(fs_roi + new_obj.m0))
            d_ri = new_obj.reference_image[ss_s1, fs_s1] - obj.reference_image[ss_s0, fs_s0]
            obj.reference_image[ss_s0, fs_s0] += gaussian_filter(d_ri, (ls_ri, ls_ri))

            errors.append(obj.total_mse(ls_ri=ls_ri))
            if verbose:
                print('Iteration No. {:d}: Total MSE = {:.3f}'.format(it, errors[-1]))
            if errors[-2] - errors[-1] <= f_tol:
                break
        if return_errors:
            return obj, errors
        else:
            return obj

    def total_mse(self, ls_ri=3.):
        """Return average mean-squared-error (MSE) per pixel.

        Parameters
        ----------
        ls_ri : float
            `reference_image` length scale in pixels.

        Returns
        -------
        float
            Average mean-squared-error (MSE) per pixel.

        Raises
        ------
        AttributeError
            If `reference_image` was not generated beforehand.
        """
        if self.reference_image is None:
            raise AttributeError('The reference image hsa not been generated')
        else:
            return total_mse(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                             u=self.pixel_map, di=self.dss_pix - self.n0,
                             dj=self.dfs_pix - self.m0, ls=ls_ri)

    def var_pixel_map(self, ls_ri=20):
        """Return `pixel_map` variance.

        Parameters
        ----------
        ls_ri : float
            `reference_image` length scale in pixels.

        Returns
        -------
        float
            `pixel_map` variance.
        """
        var_psn = np.mean(self.data)
        ri = self.update_reference.dict_func(ls_ri=ls_ri) 
        dri_ss = np.mean(np.gradient(ri['reference_image'][0])**2)
        N, K = self.data.shape[0], self.data.shape[-1] / (self.dfs_pix[0] - self.dfs_pix[1])
        return var_psn * (1 / N + 1 / N / K) / dri_ss / np.mean(self.whitefield**2)
