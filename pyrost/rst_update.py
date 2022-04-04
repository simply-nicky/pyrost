""":class:`pyrost.SpeckleTracking` provides an interface to perform the reference
image and lens wavefront reconstruction and offers two methods
(:func:`pyrost.SpeckleTracking.train`, :func:`pyrost.SpeckleTracking.train_adapt`)
to perform the iterative R-PXST update until the error metric converges to a minimum.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from weakref import ReferenceType
from tqdm.auto import tqdm
import numpy as np
from .data_container import DataContainer, dict_to_object
from .bfgs import BFGS
from .bin import (gaussian_filter, KR_reference, LOWESS_reference, pm_gsearch,
                  pm_rsearch, pm_devolution, tr_gsearch, pm_errors, pm_total_error,
                  ref_errors, ct_integrate)

class SpeckleTracking(DataContainer):
    """Wrapper class for the  Robust Speckle Tracking algorithm.
    Provides an interface to perform the reference image and lens
    wavefront reconstruction.

    Attributes:
        attr_set : Set of attributes in the container which are necessary
            to initialize in the constructor.
        init_set : Set of optional data attributes.

    Notes:
        **Necessary attributes**:

        * data : Measured frames.
        * di_pix : The sample's translations along the vertical axis in
          pixels.
        * dj_pix : The sample's translations along the horizontal axis in
          pixels.
        * ds_y : `reference_image` sampling interval in pixels along
          the vertical axis.
        * ds_x : `reference_image` sampling interval in pixels along
          the horizontal axis.
        * num_threads : Specify number of threads that are used in all the
          calculations.
        * parent : The parent :class:`STData` container.
        * pixel_map : The pixel mapping between the data at the detector's
          plane and the reference image at the reference plane.
        * scale_map : Huber scaling map.
        * whitefield : Measured frames' white-field.

        **Optional attributes**:

        * error : Average error of the reference image and pixel mapping fit.
        * hval : Smoothing kernel bandwidth used in `reference_image`
          regression. The value is given in pixels.
        * m0 : The lower bounds of the horizontal detector axis of
          the reference image at the reference frame in pixels.
        * n0 : The lower bounds of the vertical detector axis of
          the reference image at the reference frame in pixels.
        * reference_image : The unabberated reference image of the sample.

    See Also:

        * :func:`pyrost.bin.KR_reference` : Full details of the `reference_image`
          update using kernel regression.
        * :func:`pyrost.bin.LOWESS_reference` : Full details of the `reference_image`
          update using local weighted linear regression.
        * :func:`pyrost.bin.pm_gsearch` : Full details of the `pixel_map` grid search
          update.

    """
    attr_set = {'data', 'dj_pix', 'di_pix', 'ds_y', 'ds_x', 'num_threads', 'pixel_map',
                'parent', 'whitefield'}
    init_set = {'error', 'hval', 'initial', 'ref_orig', 'reference_image',
                'scale_map', 'test_mask', 'train_mask'}

    dtypes_32 = {'data': np.uint32, 'dij_pix': np.float32, 'pixel_map': np.float32,
                 'scale_map': np.float32, 'whitefield': np.float32}
    dtypes_64 = {'data': np.uint64, 'dij_pix': np.float64, 'pixel_map': np.float64,
                 'scale_map': np.float64, 'whitefield': np.float64}

    # Necessary attributes
    data : np.ndarray
    dj_pix : np.ndarray
    di_pix : np.ndarray
    ds_x : float
    ds_y : float
    num_threads : int
    parent : ReferenceType
    pixel_map : np.ndarray
    whitefield : np.ndarray

    # Automatically generated attributes
    reference_image : np.ndarray
    ref_orig : np.ndarray
    scale_map : np.ndarray
    test_mask : np.ndarray
    train_mask : np.ndarray

    # Optional attributes
    hval : Optional[float]
    error : Optional[float]
    initial : Optional[SpeckleTracking]

    def __init__(self, parent: ReferenceType, **kwargs: Union[int, float, np.ndarray]) -> None:
        """
        Args:
            parent : The Speckle tracking data container, from which the
                object is derived.
            kwargs : Dictionary of the attributes' data specified in `attr_set`
                and `init_set`.

        Raises:
            ValueError : If an attribute specified in `attr_set` has not been
                provided.
        """
        super(SpeckleTracking, self).__init__(parent=parent, **kwargs)
        self._init_functions(test_mask=self._test_mask, train_mask=lambda: ~self.test_mask,
                             ref_orig=self._ref_orig, reference_image=self._reference_image,
                             scale_map=lambda: np.sqrt(self.whitefield))
        self._init_attributes()

    def _ref_orig(self) -> np.ndarray:
        y_orig = self.di_pix.max() - self.pixel_map[0].min()
        x_orig = self.dj_pix.max() - self.pixel_map[1].min()
        return np.array([y_orig, x_orig]).astype(int)

    def _reference_image(self) -> np.ndarray:
        shape = self.pixel_map.max(axis=(1, 2)) - np.array([self.di_pix.min(), self.dj_pix.min()])
        shape = ((shape + self.ref_orig) / np.array([self.ds_y, self.ds_x]))
        return np.ones(shape.astype(int) + 1, dtype=self.whitefield.dtype)

    def _test_mask(self, test_ratio: float=0.2) -> np.ndarray:
        idxs = np.random.choice(self.whitefield.size, size=int(self.whitefield.size * test_ratio),
                                replace=False)
        idxs = np.unravel_index(idxs, self.whitefield.shape)
        test_mask = np.zeros(self.whitefield.shape, dtype=bool)
        test_mask[idxs] = True
        return test_mask

    def __repr__(self) -> str:
        with np.printoptions(threshold=6, edgeitems=2, suppress=True, precision=3):
            return {key: val.ravel() if isinstance(val, np.ndarray) else val
                    for key, val in self.items()}.__repr__()

    def __str__(self) -> str:
        with np.printoptions(threshold=6, edgeitems=2, suppress=True, precision=3):
            return {key: val.ravel() if isinstance(val, np.ndarray) else val
                    for key, val in self.items()}.__str__()

    @dict_to_object
    def create_initial(self) -> SpeckleTracking:
        """Create a :class:`SpeckleTracking` object with preliminary approximation of
        the pixel mapping and reference profile. The object is saved into `initial`
        attribute. Necessary to calculate normalized error metrics.

        Returns:
            A new :class:`SpeckleTracking` object with the preliminary
            :class:`SpeckleTracking` object saved in the `initial` attribute.
        """
        initial = self.parent().get_st(ds_x=self.ds_x, ds_y=self.ds_y).update_errors()
        return {'initial': initial}

    @dict_to_object
    def test_train_split(self, test_ratio: float=0.1) -> SpeckleTracking:
        """Update test / train subsets split. Required to calculate the Cross-validation
        error metric.

        Args:
            test_ratio : Ratio between the size of the test subset and the whole dataset.

        Returns:
            A new :class:`SpeckleTracking` object with a new test / train subsets split.

        See Also:
            :func:`SpeckleTracking.CV` : Full details on the Cross-validation error
            metric.
        """
        test_mask = self._test_mask(test_ratio)
        return {'test_mask': test_mask, 'train_mask': ~test_mask}

    @dict_to_object
    def update_reference(self, hval: float, method: str='KerReg') -> SpeckleTracking:
        r"""Return a new :class:`SpeckleTracking` object with the updated
        `reference_image`. The reference profile is estimated either by
        Kernel regression ('KerReg') [KerReg]_ or Local weighted linear\
        egressin ('LOWESS') [LOWESS]_.

        Args:
            hval : Smoothing kernel bandwidth used in `reference_image`
                regression. The value is given in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

        Raises:
            ValueError : If `method` keyword value is not valid.

        Returns:
            A new :class:`SpeckleTracking` object with the updated
            `reference_image`.

        Notes:
            The reference profile estimator is given by:

            .. math::
                I_\text{ref}(f_x i, f_y j) = \frac{\sum_n \sum_{i^{\prime}}
                \sum_{j^{\prime}} K(f_x i - u^x_{i^{\prime}j^{\prime}} + \Delta i_n,
                f_y j - u^y_{i^{\prime}j^{\prime}} + \Delta j_n, h) \;
                W_{i^{\prime}j^{\prime}} I_{n i^{\prime} j^{\prime}}}
                {\sum_n \sum_{i^{\prime}} \sum_{j^{\prime}}
                K(f_x i - u^x_{i^{\prime}j^{\prime}} + \Delta i_n,
                f_y j - u^y_{i^{\prime}j^{\prime}} + \Delta j_n, h) \;
                W^2_{i^{\prime} j^{\prime}}},

            where :math:`K(i, j, h) = \exp(\frac{i\:\delta u + j\:\delta v}{h})
            / \sqrt{2 \pi}` is the Gaussian kernel, :math:`u^x_{ij}, u^y_{ij}`
            are the horizontal and vertical components of the pixel mapping,
            :math:`\Delta i_n, \Delta j_n` are the sample translation along the
            horizontal and vertical axes in pixels, :math:`I_{nij}` are the measured
            stack of frames, and `W_{ij}` is the white-field.

        See Also:
            :func:`pyrost.bin.make_reference` : Full details of the `reference_image`
                update algorithm.

        References:
            .. [KerReg] E. A. Nadaraya, “On estimating regression,” Theory Probab. & Its
                       Appl. 9, 141-142 (1964).

            .. [LOWESS] H.-G. Müller, “Weighted local regression and kernel methods for
                       nonparametric curve fitting,” J. Am. Stat. Assoc. 82, 231-238
                       (1987).
        """
        if method == 'KerReg':
            I0, n0, m0 = KR_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                      di=self.di_pix, dj=self.dj_pix, ds_y=self.ds_y,
                                      ds_x=self.ds_x, hval=hval, num_threads=self.num_threads)
        elif method == 'LOWESS':
            I0, n0, m0 = LOWESS_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                          di=self.di_pix, dj=self.dj_pix, ds_y=self.ds_y,
                                          ds_x=self.ds_x, hval=hval, num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')
        return {'hval': hval, 'ref_orig': np.array([n0, m0]), 'reference_image': I0}

    def ref_indices(self) -> np.ndarray:
        """Return an array of reference profile pixel indices.

        Returns:
            An array of reference profile pixel indices.
        """
        idxs = np.indices(self.reference_image.shape) 
        idxs[0] = idxs[0] * self.ds_y - self.ref_orig[0]
        idxs[1] = idxs[1] * self.ds_x - self.ref_orig[1]
        return idxs

    @dict_to_object
    def update_pixel_map(self, search_window: Tuple[float, float, float], blur: float=0.0,
                         integrate: bool=False, method: str='gsearch',
                         extra_args: Dict[str, Union[int, float]]={}) -> SpeckleTracking:
        r"""Return a new :class:`SpeckleTracking` object with the updated pixel mapping
        (`pixel_map`) and Huber scale mapping (`scale_map`). The update is performed
        with the adaptive Huber regression [HUBER]_. The Huber loss function is minimized
        by the non-gradient methods enlisted in the `method` argument.

        Args:
            search_window : A tuple of three elements ('sw_y', 'sw_x', 'sw_s'). The elements
                are the following:

                * `sw_y` : Search window size in pixels along the horizontal detector
                  axis.
                * `sw_x` : Search window size in pixels along the vertical detector
                  axis.
                * `sw_s` : Search window size of the Huber scaling map. Given as a ratio
                  (0.0 - 1.0) relative to the scaling map value before the update.

            blur : Smoothing kernel bandwidth used in `pixel_map` regularisation.
                The value is given in pixels.
            integrate : Ensure that the updated pixel map is irrotational by
                integrating and taking the derivative.
            method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * `gsearch` : Grid search algorithm.
                * `rsearch` : Random search algorithm.
                * `de`      : Differential evolution algorithm.

            extra_args : Extra arguments for pixel map update methods. Accepts the
                following keyword arguments:

                * `grid_size` : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * `n_trials` : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * `n_iter` : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * `pop_size` : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * `mutation` : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * `recombination` : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * `seed` : Specify seed for the random number generation.
                  Generated automatically if not provided.

        Raises:
            AttributeError : If `reference_image` was not generated beforehand.
            ValueError : If `method` keyword value is not valid.

        Returns:
            A new :class:`SpeckleTracking` object with the updated `pixel_map` and
            `scale_map`.

        Notes:
            The pixel mapping update is carried out separately at each pixel
            :math:`{i, j}` in the detector grid by minimizing the Huber error metric
            given by:

            .. math::

                \varepsilon_{ij}(\delta i, \delta j, s) = \frac{1}{N}
                \sum_{n = 1}^N \left[ s + \mathcal{H}_{1.35} \left( \frac{I_{nij} -
                W_{ij} I_\text{ref}(f_x i - u^x_{ij} - \delta i + \Delta i_n,
                f_y i - u^y_{ij} - \delta j + \Delta j_n)}{s} \right) s \right],

            where :math:`I_\text{ref}` is the reference profile, :math:`u^x_{ij},
            u^y_{ij}` are the horizontal and vertical components of the pixel mapping,
            :math:`\Delta i_n, \Delta j_n` are the sample translation along the
            horizontal and vertical axes in pixels, :math:`I_{nij}` are the measured
            stack of frames, and `W_{ij}` is the white-field.

        See Also:

            * :func:`pyrost.bin.pm_gsearch` : Full details of the grid search
              update method.
            * :func:`pyrost.bin.pm_rsearch` : Full details of the random search
              update method.
            * :func:`pyrost.bin.pm_devolution` : Full details of the differential
              evolution update method.

        References:
            .. [HUBER] A. B. Owen, “A robust hybrid of lasso and ridge regression,”
                      (2006).
        """
        n_iter = extra_args.get('n_iter', 5)
        mutation = extra_args.get('mutation', 0.75)
        recombination = extra_args.get('recombination', 0.7)
        seed = extra_args.get('seed', np.random.default_rng().integers(0, np.iinfo(np.int_).max,
                                                                       endpoint=False))
        grid_size = extra_args.get('grid_size', (int(0.5 * search_window[0]) + 1,
                                                 int(0.5 * search_window[1]) + 1,
                                                 int(50.0 * search_window[2]) + 1))
        n_trials = extra_args.get('n_trials', max(np.prod(grid_size), 2))
        pop_size = extra_args.get('pop_size', max(n_trials / n_iter, 4))

        if method == 'gsearch':
            pm, scale, derr = pm_gsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                         u0=self.pixel_map, di=self.di_pix - self.ref_orig[0],
                                         dj=self.dj_pix - self.ref_orig[1], search_window=search_window,
                                         grid_size=grid_size, ds_y=self.ds_y, ds_x=self.ds_x,
                                         sigma=self.scale_map, num_threads=self.num_threads)
        elif method == 'rsearch':
            pm, scale, derr = pm_rsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                         u0=self.pixel_map, di=self.di_pix - self.ref_orig[0],
                                         dj=self.dj_pix - self.ref_orig[1], search_window=search_window,
                                         n_trials=n_trials, seed=seed, ds_y=self.ds_y, ds_x=self.ds_x,
                                         sigma=self.scale_map, num_threads=self.num_threads)
        elif method == 'de':
            pm, scale, derr = pm_devolution(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                            u0=self.pixel_map, di=self.di_pix - self.ref_orig[0],
                                            dj=self.dj_pix - self.ref_orig[1], search_window=search_window,
                                            pop_size=pop_size, n_iter=n_iter, seed=seed,
                                            ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.scale_map,
                                            F=mutation, CR=recombination, num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')

        dpm = pm - self.pixel_map
        derr = (derr - derr.min() + 1.0) / derr.std()
        if blur > 0.0:
            norm = gaussian_filter(derr, (blur, blur))
            dpm = gaussian_filter(dpm * derr, (0, blur, blur),
                                  num_threads=self.num_threads) / norm

        uy_avg, ux_avg = dpm.mean(axis=(1, 2))
        pm[0] = (dpm[0] - uy_avg) + self.pixel_map[0]
        pm[1] = (dpm[1] - ux_avg) + self.pixel_map[1]

        if integrate:
            phi = ct_integrate(sy_arr=pm[0], sx_arr=pm[1], num_threads=self.num_threads)
            pm[0] = np.gradient(phi, axis=0)
            pm[1] = np.gradient(phi, axis=1)

        return {'pixel_map': pm, 'scale_map': scale}

    @dict_to_object
    def update_errors(self) -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object with
        the updated mean Huber error metric `error`.

        Returns:
            A new :class:`SpeckleTracking` object with the updated
            `error`.

        Raises:
            AttributeError : If `reference_image` was not generated
                beforehand.

        See Also:
            * :func:`pyrost.bin.ref_errors` : Full details of the reference
              update error metric.
            * :func:`pyrost.bin.pm_errors` : Full details of the pixel
              mapping update error metric.
        """
        error = pm_total_error(I_n=self.data, W=self.whitefield,
                               I0=self.reference_image, u=self.pixel_map,
                               di=self.di_pix - self.ref_orig[0], ds_y=self.ds_y,
                               dj=self.dj_pix - self.ref_orig[1], ds_x=self.ds_x,
                               sigma=self.scale_map, num_threads=self.num_threads)
        if self.initial is None:
            return {'error': error}
        return {'error': error / self.initial.error}

    @dict_to_object
    def update_translations(self, sw_x: float, sw_y: float=0.0, blur=0.0) -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object with the updated sample
        pixel translations (`di_pix`, `dj_pix`). The update is performed with the
        adaptive Huber regression [HUBER]_. The Huber loss function is minimized
        by the grid search algorithm.

        Args:
            sw_x : Search window size in pixels along the horizontal detector
                axis.
            sw_y : Search window size in pixels along the vertical detector
                axis.
            blur : Smoothing kernel bandwidth used in `dj_pix` and `di_pix`
                post-update. The value is given in pixels.

        Returns:
            A new :class:`SpeckleTracking` object with the updated
            `di_pix`, `dj_pix`.

        Raises:
            AttributeError : If `reference_image` was not generated beforehand.

        See Also:
            :func:`pyrost.bin.tr_gsearch` : Full details of the sample translations
                update algorithm.
        """
        dij = tr_gsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                         u=self.pixel_map, di=self.di_pix - self.ref_orig[0],
                         dj=self.dj_pix - self.ref_orig[1], sw_y=sw_y, sw_x=sw_x,
                         ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.scale_map,
                         num_threads=self.num_threads)
        dij += self.ref_orig

        di_pix = np.ascontiguousarray(dij[:, 0])
        dj_pix = np.ascontiguousarray(dij[:, 1])
        if blur > 0.0:
            di_pix = gaussian_filter(di_pix - self.di_pix, blur, mode='nearest',
                                      num_threads=self.num_threads) + self.di_pix
            dj_pix = gaussian_filter(dj_pix - self.dj_pix, blur, mode='nearest',
                                      num_threads=self.num_threads) + self.dj_pix

        return {'di_pix': di_pix, 'dj_pix': dj_pix}

    def error_profile(self, kind: str='pixel_map') -> np.ndarray:
        """Return a error metrics for the reference profile and pixel mapping
        updates. The error metrics may be normalized by the error of the initial
        estimations of the reference profile and pixel mapping function. The
        normalization is performed if :func:`SpeckleTracking.create_initial` was
        invoked before.

        Args:
            kind : Choose between generating the error metric of the pixel mapping
                ('pixel_map') or reference image ('reference')update.

        Raises:
            ValueError : If `kind` keyword value is not valid.

        Returns:
            Residual profile.

        See Also:
            :func:`SpeckleTracking.update_reference` : More details on the reference
            profile update procedure.
            :func:`SpeckleTracking.update_pixel_map` : More details on the pixel
            mapping update procedure.
        """

        if kind == 'pixel_map':
            pm_err = pm_errors(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                               I0=self.reference_image, sigma=self.scale_map,
                               di=self.di_pix - self.ref_orig[0], ds_y=self.ds_y,
                               dj=self.dj_pix - self.ref_orig[1], ds_x=self.ds_x,
                               num_threads=self.num_threads)
            if self.initial is None:
                return pm_err
            pm_tot = self.initial.error_profile(kind=kind)
            return np.where(pm_tot, pm_err / pm_tot, 0.0)

        if kind == 'reference':
            ref_err = ref_errors(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                 I0=self.reference_image, hval=self.hval,
                                 di=self.di_pix - self.ref_orig[0], ds_y=self.ds_y,
                                 dj=self.dj_pix - self.ref_orig[1], ds_x=self.ds_x,
                                 num_threads=self.num_threads)
            if self.initial is None:
                return ref_err
            self.initial.hval = self.hval
            ref_tot = self.initial.error_profile(kind=kind)

            idxs0 = np.moveaxis(self.initial.ref_indices(), 0, -1)
            idxs1 = np.moveaxis(self.ref_indices(), 0, -1)

            start = np.max((idxs0[0, 0], idxs1[0, 0]), axis=0)
            end = np.min((idxs0[-1, -1], idxs1[-1, -1]), axis=0)
            ps = 0.5 * np.array([self.ds_y, self.ds_x])

            roi0 = np.concatenate([np.argwhere(np.all(np.abs(idxs0 - start) < ps, axis=-1)),
                                   np.argwhere(np.all(np.abs(idxs0 - end) < ps, axis=-1)) + 1])
            roi1 = np.concatenate([np.argwhere(np.all(np.abs(idxs1 - start) < ps, axis=-1)),
                                   np.argwhere(np.all(np.abs(idxs1 - end) < ps, axis=-1)) + 1])

            ref_tot = ref_tot[roi0[0, 0]:roi0[1, 0], roi0[0, 1]:roi0[1, 1]]
            ref_err = ref_err[roi1[0, 0]:roi1[1, 0], roi1[0, 1]:roi1[1, 1]]
            return np.where(ref_tot, ref_err / ref_tot, 0.0)

        raise ValueError('kind keyword is invalid')

    def find_hopt(self, h0: float=1.0, method: str='KerReg',
                  epsilon: float=1e-3, maxiter: int=10, gtol: float=1e-5,
                  verbose: bool=False) -> float:
        """Find the optimal kernel bandwidth by finding the bandwidth, that minimized
        the Cross-validation error metric. The minimization process is performed with
        the quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno [BFGS]_.

        Args:
            h0 : Initial guess of kernel bandwidth in pixels.
            method : `reference_image` update algorithm. The following keyword values
                are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

            epsilon : The step size used in the estimation of the fitness gradient.
            maxiter : Maximum number of iterations in the minimization loop.
            gtol : Gradient norm must be less than `gtol` before successful
                termination.
            verbose : Print convergence message if True.

        Returns:
            Optimal kernel bandwidth in pixels.

        References:
            .. [BFGS] S. Wright, J. Nocedal et al., “Numerical optimization,” Springer
                     Sci. 35, 7 (1999).
        """
        optimizer = BFGS(lambda hval: self.CV(hval, method), np.atleast_1d(h0),
                         epsilon=epsilon)
        itor = tqdm(range(maxiter), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} Iteration {n_fmt}'\
                               ' / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        for _ in itor:
            optimizer.step()
            if verbose:
                itor.set_description(f"hopt = {optimizer.state_dict()['xk'].item():.3f}, " \
                                     f"gnorm = {optimizer.state_dict()['gnorm'].item():.3e}")
            if optimizer.state_dict()['gnorm'] < gtol:
                break
        return optimizer.state_dict()['xk'].item()

    def train_adapt(self, search_window: Tuple[float, float, float], h0: float,
                    blur: float=0.0, n_iter: int=30, f_tol: float=1e-8,
                    ref_method: str='KerReg', pm_method: str='rsearch',
                    pm_args: Dict[str, Union[bool, int, float, str]]={},
                    options: Dict[str, Union[bool, float, str]]={},
                    verbose: bool=True) -> Tuple[SpeckleTracking, Dict[str, List[float]]]:
        """Perform adaptive iterative Robust Speckle Tracking update. The reconstruction
        cycle consists of: (i) estimating an optimal kernel bandwidth for the reference
        image estimate (see :func:`SpeckleTracking.find_hopt`); (ii) generating the
        reference image (see :func:`SpeckleTracking.update_reference`); (iii) updating
        the pixel mapping between a stack of frames and the generated reference image
        (see :func:`SpeckleTracking.update_pixel_map`); (iv) updating the sample
        translation vectors (if needed, see :func:`SpeckleTracking.update_translations`);
        and (v) calculating the mean Huber error (see :func:`SpeckleTracking.update_error`).

        Args:
            search_window : A tuple of three elements ('sw_y', 'sw_x', 'sw_s'). The elements
                are the following:

                * `sw_y` : Search window size in pixels along the horizontal detector
                  axis.
                * `sw_x` : Search window size in pixels along the vertical detector
                  axis.
                * `sw_s` : Search window size of the Huber scaling map. Given as a ratio
                  (0.0 - 1.0) relative to the scaling map value before the update.

            h0 : Smoothing kernel bandwidth used in `reference_image` estimation.
                The value is given in pixels.
            blur : Smoothing kernel bandwidth used in `pixel_map` regularisation.
                The value is given in pixels.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the average error. The
                iteration stops when ``(error^k - error^{k+1})/max{|error^k|, |error^{k+1}|,
                1} <= f_tol``.
            ref_method : `reference_image` update algorithm. The following
                keyword values are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

            pm_method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * `gsearch` : Grid search algorithm.
                * `rsearch` : Random search algorithm.
                * `de`      : Differential evolution algorithm.

            pm_args : Pixel map update options. Accepts the following keyword
                arguments:

                * `integrate` : Ensure that the updated pixel map is irrotational
                  by integrating and taking the derivative. False by default.
                * `grid_size` : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * `n_trials` : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * `n_iter` : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * `pop_size` : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * `mutation` : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * `recombination` : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * `seed` : Specify seed for the random number generation.
                  Generated automatically if not provided.

            options : Extra options. Accepts the following keyword arguments:

                * `epsilon` : Increment to `h0` to use for determining the
                  function gradient for `h0` update algorithm. The default
                  value is 1.4901161193847656e-08.
                * `maxiter` : Maximum number of iterations in the line search at the
                  optimal kernel bandwidth update. The default value is 10.
                * `momentum` : Momentum used in the next error calculation. Helps to
                  smooth out the change of error. The default value is 0.0.
                * `update_translations` : Update sample pixel translations
                  if True. The default value is False.
                * `return_extra` : Return errors and `h0` array if True. The default
                  value is False.

            verbose : Set verbosity of the computation process.

        Returns:
            A tuple of two items ('st_obj', 'extra'). The elements of the tuple
            are as follows:

            * `st_obj` : A new :class:`SpeckleTracking` object with the
              updated `pixel_map` and `reference_image`. `di_pix` and `dj_pix`
              are also updated if `update_translations` is True.

            * `extra`: A dictionary with the given parameters:

              * `errors` : List of average error values for each iteration.
              * `hvals` : List of kernel bandwidths for each iteration.

              Only if `return_extra` is True.
        """
        integrate = pm_args.get('integrate', False)

        epsilon = options.get('epsilon', 1e-3)
        maxiter = options.get('maxiter', 10)
        momentum = options.get('momentum', 0.0)
        update_translations = options.get('update_translations', False)
        return_extra = options.get('return_extra', False)

        obj = self.update_reference(hval=h0, method=ref_method)
        obj.update_errors.inplace_update()

        optimizer = BFGS(lambda hval: obj.CV(hval, ref_method),
                         np.atleast_1d(h0), epsilon=epsilon)

        errors = [obj.error,]
        hvals = [h0,]

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} ' \
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        if verbose:
            print(f"Initial error = {errors[-1]:.6f}, Initial h0 = {hvals[-1]:.2f}")

        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(search_window=search_window, blur=blur,
                                           integrate=integrate, method=pm_method,
                                           extra_args=pm_args)

            # Update di_pix, dj_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_y=search_window[0],
                                                           sw_x=search_window[1],
                                                           blur=blur)

            # Update hval and step
            optimizer.update_loss(lambda hval: new_obj.CV(hval, ref_method))
            optimizer.step(maxiter=maxiter)
            h0 = (1.0 - momentum) * optimizer.state_dict()['xk'].item() + momentum * hvals[-1]
            hvals.append(h0)

            # Update reference_image
            new_obj.update_reference.inplace_update(hval=h0, method=ref_method)

            new_obj.update_errors.inplace_update()
            errors.append((1.0 - momentum) * new_obj.error + momentum * errors[-1])
            if verbose:
                itor.set_description(f"Error = {errors[-1]:.6f}, hval = {hvals[-1]:.2f}")

            # Break if function tolerance is satisfied
            if (errors[-2] - errors[-1]) / max(errors[-2], errors[-1]) > f_tol:
                obj = new_obj

            else:
                break

        if return_extra:
            return obj, {'errors': errors, 'hvals': hvals}
        return obj

    def train(self, search_window: Tuple[float, float, float], h0: float,
              blur: float=0.0, n_iter: int=30, f_tol: float=1e-8,
              ref_method: str='KerReg', pm_method: str='rsearch',
              pm_args: Dict[str, Union[bool, int, float, str]]={},
              options: Dict[str, Union[bool, float, str]]={},
              verbose: bool=True) -> Tuple[SpeckleTracking, List[float]]:
        """Perform iterative Robust Speckle Tracking update. The reconstruction cycle
        consists of: (i) generating the reference image (see
        :func:`SpeckleTracking.update_reference`); (ii) updating the pixel mapping between
        a stack of frames and the generated reference image (see
        :func:`SpeckleTracking.update_pixel_map`); (iii) updating the sample translation
        vectors (if needed, see :func:`SpeckleTracking.update_translations`); and (iv)
        calculating the mean Huber error (see :func:`SpeckleTracking.update_error`). The
        kernel bandwidth in the reference update is kept fixed during the iterative update
        procedure.

        Args:
            search_window : A tuple of three elements ('sw_y', 'sw_x', 'sw_s'). The elements
                are the following:

                * `sw_y` : Search window size in pixels along the horizontal detector
                  axis.
                * `sw_x` : Search window size in pixels along the vertical detector
                  axis.
                * `sw_s` : Search window size of the Huber scaling map. Given as a ratio
                  (0.0 - 1.0) relative to the scaling map value before the update.

            h0 : Smoothing kernel bandwidth used in `reference_image` regression.
                The value is given in pixels.
            blur : Smoothing kernel bandwidth used in `pixel_map` regularisation.
                The value is given in pixels.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the average error. The
                iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            ref_method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

            pm_method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * `gsearch` : Grid search algorithm.
                * `rsearch` : Random search algorithm.
                * `de`      : Differential evolution algorithm.

            pm_args : Pixel map update options. Accepts the following keyword
                arguments:

                * `integrate` : Ensure that the updated pixel map is irrotational
                  by integrating and taking the derivative. False by default.
                * `grid_size` : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * `n_trials` : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * `n_iter` : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * `pop_size` : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * `mutation` : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * `recombination` : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * `seed` : Specify seed for the random number generation.
                  Generated automatically if not provided.

            options : Extra options. Accepts the following keyword arguments:

                * `momentum` : Momentum used in the next error calculation. Helps to
                  smooth out the change of error.
                * `update_translations` : Update sample pixel translations
                  if True. The default value is False.
                * `return_extra` : Return errors at each iteration if True.
                  The default value is False.

            verbose : bool, optional
                Set verbosity of the computation process.

        Returns:
            A tuple of two items (`st_obj`, `errors`). The elements of the tuple
            are as follows:

            * `st_obj` : A new :class:`SpeckleTracking` object with the updated
              `pixel_map` and `reference_image`. `di_pix` and `dj_pix`
              are also updated if 'update_translations' in `options`
              is True.

            * `errors` : List of average error values for each iteration. Only if
              'return_extra' in `options` is True.
        """
        integrate = pm_args.get('integrate', False)

        momentum = options.get('momentum', 0.0)
        update_translations = options.get('update_translations', False)
        return_extra = options.get('return_extra', False)

        obj = self.update_reference(hval=h0, method=ref_method)
        obj.update_errors.inplace_update()
        errors = [obj.error]

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} '\
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        if verbose:
            print(f"Initial error = {errors[0]:.6f}, "\
                  f"Initial h0 = {h0:.2f}")

        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(search_window=search_window, blur=blur,
                                           integrate=integrate, method=pm_method,
                                           extra_args=pm_args)

            # Update di_pix, dj_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_y=search_window[0],
                                                           sw_x=search_window[1],
                                                           blur=blur)

            # Update reference_image
            new_obj.update_reference.inplace_update(hval=h0, method=ref_method)
            new_obj.update_errors.inplace_update()

            # Calculate errors
            errors.append((1.0 - momentum) * new_obj.error + momentum * errors[-1])
            if verbose:
                itor.set_description(f"Error = {errors[-1]:.6f}")

            # Break if function tolerance is satisfied
            if (errors[-2] - errors[-1]) / max(errors[-2], errors[-1]) > f_tol:
                obj = new_obj

            else:
                break

        if return_extra:
            return obj, errors
        return obj

    def CV(self, hval: float, method: str='KerReg') -> float:
        """Return cross-validation error for the given kernel bandwidth `hval`.
        The cross-validation error metric is calculated as follows: (i) generate
        a reference profile based on the training subset and (ii) find the
        mean-squared-error (MSE) for the test subset.

        Args:
            hval : `reference_image` kernel bandwidths in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

        Returns:
            Cross-validation error.

        Raises:
            ValueError : If `method` keyword value is not valid.

        See Also:
            :func:`pyrost.bin.pm_total_error` : Full details of the error metric.
        """
        if method == 'KerReg':
            I0, n0, m0 = KR_reference(I_n=self.data, W=self.whitefield * self.train_mask,
                                      u=self.pixel_map, di=self.di_pix, dj=self.dj_pix,
                                      ds_y=self.ds_y, ds_x=self.ds_x, hval=hval,
                                      num_threads=self.num_threads)
        elif method == 'LOWESS':
            I0, n0, m0 = LOWESS_reference(I_n=self.data, W=self.whitefield * self.train_mask,
                                          u=self.pixel_map, di=self.di_pix, dj=self.dj_pix,
                                          ds_y=self.ds_y, ds_x=self.ds_x, hval=hval,
                                          num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')
        error = pm_total_error(I_n=self.data, W=self.whitefield * self.test_mask, I0=I0,
                               u=self.pixel_map, di=self.di_pix - n0, dj=self.dj_pix - m0,
                               ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.scale_map,
                               num_threads=self.num_threads)
        return error

    def CV_curve(self, harr: np.ndarray, method: str='KerReg') -> np.ndarray:
        """Return a set of cross-validation errors for a set of kernel
        bandwidths.

        Args:
            harr : Set of `reference_image` kernel bandwidths in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * `KerReg` : Kernel regression algorithm.
                * `LOWESS` : Local weighted linear regression.

        Returns:
            An array of cross-validation errors.

        See Also:
            :func:`pyrost.bin.pm_total_error` : Full details of the error metric.
        """
        mse_list = []
        for hval in np.array(harr, ndmin=1):
            mse_list.append(self.CV(hval, method=method))
        return np.array(mse_list)
