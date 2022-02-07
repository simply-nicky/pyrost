""":class:`pyrost.SpeckleTracking` provides an interface to perform the reference
image and lens wavefront reconstruction and offers two methods
(:func:`pyrost.SpeckleTracking.iter_update`, :func:`pyrost.SpeckleTracking.iter_update_gd`)
to perform the iterative RST update until the error metric converges to a minimum.
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
        * sigma : Standard deviation of measured frames.
        * whitefield : Measured frames' whitefield.

        **Optional attributes**:

        * error : Average MSE (mean-squared-error) of the reference image
          and pixel mapping fit.
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
    attr_set = {'data', 'dj_pix', 'di_pix', 'ds_y', 'ds_x', 'num_threads',
                'parent', 'pixel_map', 'whitefield'}
    init_set = {'error', 'hval', 'n0', 'm0', 'reference_image', 'sigma', 'test_mask',
                'train_mask'}

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
        self._init_functions(sigma=lambda: np.sqrt(self.whitefield),
                             test_mask=self._test_mask,
                             train_mask=lambda: ~self.test_mask)
        self._init_attributes()

    def _test_mask(self, test_ratio=0.2):
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
    def test_train_split(self, test_ratio=0.1):
        test_mask = self._test_mask(test_ratio)
        return {'test_mask': test_mask, 'train_mask': ~test_mask}

    @dict_to_object
    def update_reference(self, hval: float, method: str='KerReg') -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object
        with the updated `reference_image`.

        Args:
            hval : Smoothing kernel bandwidth used in `reference_image`
                regression. The value is given in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

        Returns:
            A new :class:`SpeckleTracking` object with the updated
            `reference_image`.

        Raises:
            ValueError : If `method` keyword value is not valid.

        See Also:
            :func:`pyrost.bin.make_reference` : Full details of the `reference_image`
                update algorithm.
        """
        if method == 'KerReg':
            I0, n0, m0 = KR_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                      di=self.di_pix, dj=self.dj_pix, ds_y=self.ds_y,
                                      ds_x=self.ds_x, h=hval, num_threads=self.num_threads)
        elif method == 'LOWESS':
            I0, n0, m0 = LOWESS_reference(I_n=self.data, W=self.whitefield, u=self.pixel_map,
                                          di=self.di_pix, dj=self.dj_pix, ds_y=self.ds_y,
                                          ds_x=self.ds_x, h=hval, num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')
        return {'hval': hval, 'n0': n0, 'm0': m0, 'reference_image': I0}

    @dict_to_object
    def update_pixel_map(self, search_window: Tuple[float, float, float], blur: float=0.0,
                         integrate: bool=False, method: str='gsearch',
                         extra_args: Dict[str, Union[int, float]]={}) -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object with
        the updated `pixel_map`.

        Args:
            sw_x : Search window size in pixels along the horizontal detector
                axis.
            sw_y : Search window size in pixels along the vertical detector
                axis.
            blur : Smoothing kernel bandwidth used in `pixel_map`
                regularisation. The value is given in pixels.
            integrate : Ensure that the updated pixel map is irrotational by integrating
                and taking the derivative.
            method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * 'gsearch' : Grid search algorithm.
                * 'rsearch' : Random search algorithm.
                * 'de'      : Differential evolution algorithm.

            extra_args : Extra arguments for pixel map update methods. Accepts the
                following keyword arguments:

                * 'grid_size' : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * 'n_trials' : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * 'n_iter' : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * 'pop_size' : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * 'mutation' : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * 'recombination' : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * 'seed' : Specify seed for the random number generation.
                  Generated automatically if not provided.


        Returns:
            A new :class:`SpeckleTracking` object with the updated
            `pixel_map`.

        Raises:
            AttributeError : If `reference_image` was not generated beforehand.
            ValueError : If `method` keyword value is not valid.

        See Also:

            * :func:`pyrost.bin.pm_gsearch` : Full details of the grid search
              update method.
            * :func:`pyrost.bin.pm_rsearch` : Full details of the random search
              update method.
            * :func:`pyrost.bin.pm_devolution` : Full details of the differential
              evolution update method.

        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')

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
            pm, sigma, derr = pm_gsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                         u0=self.pixel_map, di=self.di_pix - self.n0,
                                         dj=self.dj_pix - self.m0, search_window=search_window,
                                         grid_size=grid_size, ds_y=self.ds_y, ds_x=self.ds_x,
                                         sigma=self.sigma, num_threads=self.num_threads)
        elif method == 'rsearch':
            pm, sigma, derr = pm_rsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                         u0=self.pixel_map, di=self.di_pix - self.n0,
                                         dj=self.dj_pix - self.m0, search_window=search_window,
                                         n_trials=n_trials, seed=seed, ds_y=self.ds_y, ds_x=self.ds_x,
                                         sigma=self.sigma, num_threads=self.num_threads)
        elif method == 'de':
            pm, sigma, derr = pm_devolution(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                                            u0=self.pixel_map, di=self.di_pix - self.n0,
                                            dj=self.dj_pix - self.m0, search_window=search_window,
                                            pop_size=pop_size, n_iter=n_iter, seed=seed,
                                            ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.sigma,
                                            F=mutation, CR=recombination, num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')

        dpm = pm - self.pixel_map
        if blur > 0.0:
            norm = gaussian_filter(derr, (blur, blur))
            dpm = gaussian_filter(dpm * derr, (0, blur, blur),
                                  num_threads=self.num_threads) / norm
            sigma = gaussian_filter(sigma * derr, (blur, blur),
                                    num_threads=self.num_threads) / norm

        uy_avg, ux_avg = dpm.mean(axis=(1, 2))
        pm[0] = (dpm[0] - uy_avg) + self.pixel_map[0]
        pm[1] = (dpm[1] - ux_avg) + self.pixel_map[1]

        if integrate:
            phi = ct_integrate(sy_arr=pm[0], sx_arr=pm[1], num_threads=self.num_threads)
            pm[0] = np.gradient(phi, axis=0)
            pm[1] = np.gradient(phi, axis=1)

        return {'pixel_map': pm, 'sigma': sigma}

    @dict_to_object
    def update_errors(self) -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object with
        the updated mean residual `error`.

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
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        error = pm_total_error(I_n=self.data, W=self.whitefield,
                               I0=self.reference_image, u=self.pixel_map,
                               di=self.di_pix - self.n0, dj=self.dj_pix - self.m0,
                               ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.sigma,
                               num_threads=self.num_threads)
        return {'error': error}

    @dict_to_object
    def update_translations(self, sw_x: float, sw_y: float=0.0, blur=0.0) -> SpeckleTracking:
        """Return a new :class:`SpeckleTracking` object with
        the updated sample pixel translations (`di_pix`, `dj_pix`).

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
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        dij = tr_gsearch(I_n=self.data, W=self.whitefield, I0=self.reference_image,
                         u=self.pixel_map, di=self.di_pix - self.n0, dj=self.dj_pix - self.m0,
                         sw_y=sw_y, sw_x=sw_x, ds_y=self.ds_y, ds_x=self.ds_x,
                         sigma=self.sigma, num_threads=self.num_threads)

        di_pix = np.ascontiguousarray(dij[:, 0]) + self.n0
        dj_pix = np.ascontiguousarray(dij[:, 1]) + self.m0
        if blur > 0.0:
            di_pix = gaussian_filter(di_pix - self.di_pix, blur, mode='nearest',
                                      num_threads=self.num_threads) + self.di_pix
            dj_pix = gaussian_filter(dj_pix - self.dj_pix, blur, mode='nearest',
                                      num_threads=self.num_threads) + self.dj_pix

        return {'di_pix': di_pix, 'dj_pix': dj_pix}

    def error_profile(self, kind: str='pixel_map') -> np.ndarray:
        """Return a residual profile.

        Args:
            kind : Choose between generating the error metric of the pixel mapping
                ('pixel_map') or reference image ('reference_image')update.

        Raises:
            AttributeError : If `reference_image` was not generated beforehand.
            ValueError : If `kind` keyword value is not valid.

        Returns:
            Residual profile.
        """
        if self.reference_image is None:
            raise AttributeError('The reference image has not been generated')
        if kind == 'pixel_map':
            return pm_errors(I_n=self.data, W=self.whitefield,
                             I0=self.reference_image, u=self.pixel_map,
                             di=self.di_pix - self.n0, dj=self.dj_pix - self.m0,
                             ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.sigma,
                             num_threads=self.num_threads)
        elif kind == 'reference':
            return ref_errors(I_n=self.data, W=self.whitefield, h=self.hval,
                              I0=self.reference_image, u=self.pixel_map,
                              di=self.di_pix - self.n0, dj=self.dj_pix - self.m0,
                              ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.sigma,
                              num_threads=self.num_threads)
        else:
            raise ValueError('kind keyword is invalid')

    def find_hopt(self, h0: float=1.0, method: str='KerReg',
                  epsilon: float=1e-4, maxiter: int=50, gtol: float=1e-5,
                  verbose: bool=False) -> float:
        """Find the optimal kernel bandwidth using the BFGS algorithm.

        Args:
            h0 : Initial guess of kernel bandwidth in pixels.
            alpha : Weight of the variance term in the error metric.
            method : `reference_image` update algorithm. The following
                keyword values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

            epsilon : The step size used in the estimation of the fitness
                gradient.
            verbose : Print convergence message if True.

        Returns:
            Optimal kernel bandwidth in pixels.
        """
        optimizer = BFGS(lambda hval: self.CV(hval, method), h0,
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

    def iter_update_gd(self, search_window: Tuple[float, float, float], blur: float=0.0,
                       h0: Optional[float]=None, n_iter: int=30, f_tol: float=1e-8,
                       ref_method: str='KerReg', pm_method: str='gsearch',
                       pm_args: Dict[str, Union[bool, int, float, str]]={},
                       options: Dict[str, Union[bool, float, str]]={},
                       verbose: bool=True) -> Tuple[SpeckleTracking, Dict[str, List[float]]]:
        """Perform iterative Robust Speckle Tracking update. `h0` and `blur` define
        high frequency cut-off to supress the noise. `h0` is iteratively updated by
        dint of Gradient Descent. Iterative update terminates when the change of the
        total mean-squared-error (MSE) becomes too small (see `f_tol` for more info).

        Args:
            sw_x : Search window size in pixels along the horizontal detector axis.
            sw_y : Search window size in pixels along the vertical detector axis.
            blur : Smoothing kernel bandwidth used in `pixel_map` regularisation.
                The value is given in pixels.
            h0 : Smoothing kernel bandwidth used in `reference_image` estimation.
                The value is given in pixels. The value is estimated with
                :func:`SpeckleTracking.find_hopt` by default.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the total MSE. The
                iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            ref_method : `reference_image` update algorithm. The following
                keyword values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

            pm_method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * 'gsearch' : Grid search algorithm.
                * 'rsearch' : Random search algorithm.
                * 'de'      : Differential evolution algorithm.

            pm_args : Pixel map update options. Accepts the following keyword
                arguments:

                * 'integrate' : Ensure that the updated pixel map is irrotational
                  by integrating and taking the derivative. False by default.
                * 'grid_size' : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * 'n_trials' : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * 'n_iter' : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * 'pop_size' : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * 'mutation' : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * 'recombination' : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * 'seed' : Specify seed for the random number generation.
                  Generated automatically if not provided.

            options : Extra options. Accepts the following keyword arguments:

                * 'h0' : Initial guess of the optimal bandwidth in
                  :func:`SpeckleTracking.find_hopt`. The value is used
                  if `h0` is None.
                * 'epsilon' : Increment to `h0` to use for determining the
                  function gradient for `h0` update algorithm. The default
                  value is 1.4901161193847656e-08.
                * 'update_translations' : Update sample pixel translations
                  if True. The default value is False.
                * 'return_extra' : Return errors and `h0` array if True.
                  The default value is False.

            verbose : Set verbosity of the computation process.

        Returns:
            A tuple of two items ('st_obj', 'extra'). The elements of the tuple
            are as follows:

            * 'st_obj' : A new :class:`SpeckleTracking` object with the
              updated `pixel_map` and `reference_image`. `di_pix` and `dj_pix`
              are also updated if `update_translations` is True.

            * 'extra': A dictionary with the given parameters:

              * 'errors' : List of total MSE values for each iteration.
              * 'hvals' : List of kernel bandwidths for each iteration.

              Only if `return_extra` is True.
        """
        integrate = pm_args.get('integrate', False)
        epsilon = options.get('epsilon', 1e-4)
        update_translations = options.get('update_translations', False)
        return_extra = options.get('return_extra', False)

        if h0 is None:
            if verbose:
                print("Finding the optimum kernel bandwidth...")
            h0 = self.find_hopt(method=ref_method, epsilon=epsilon,
                                verbose=verbose)
            if verbose:
                print(f"New hopt = {h0:.3f}")

        optimizer = BFGS(lambda hval: self.CV(hval, ref_method),
                         h0, epsilon=epsilon)

        obj = self.update_reference(hval=h0, method=ref_method)
        obj.update_errors.inplace_update()

        errors = [obj.error,]
        hvals = [h0,]

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} ' \
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        if verbose:
            print(f"Initial MSE = {errors[-1]:.6f}, Initial h0 = {hvals[-1]:.2f}")

        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(search_window=search_window, blur=blur, integrate=integrate,
                                           method=pm_method, extra_args=pm_args)

            # Update di_pix, dj_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_y=search_window[0], sw_x=search_window[1],
                                                           blur=blur)

            # Update hval and step
            optimizer.step()
            h0 = optimizer.state_dict()['xk'].item()
            hvals.append(h0)

            # Update reference_image
            new_obj.update_reference.inplace_update(hval=h0, method=ref_method)

            new_obj.update_errors.inplace_update()
            errors.append(new_obj.error)
            if verbose:
                itor.set_description(f"Total MSE = {errors[-1]:.6f}, hval = {hvals[-1]:.2f}")

            # Break if function tolerance is satisfied
            if (errors[-2] - errors[-1]) / max(errors[-2], errors[-1]) > f_tol:
                obj = new_obj

            else:
                break

        if return_extra:
            return obj, {'errors': errors, 'hvals': hvals}
        return obj

    def iter_update(self, search_window: Tuple[float, float, float], blur: float=0.0,
                    h0: Optional[float]=None, n_iter: int=30, f_tol: float=1e-8,
                    ref_method: str='KerReg', pm_method: str='gsearch',
                    pm_args: Dict[str, Union[bool, int, float, str]]={},
                    options: Dict[str, Union[bool, float, str]]={},
                    verbose: bool=True) -> Tuple[SpeckleTracking, List[float]]:
        """Perform iterative Robust Speckle Tracking update. `h0` and `blur` define
        high frequency cut-off to supress the noise and stay constant during the
        update. Iterative update terminates when the change of the total
        mean-squared-error (MSE) becomes too small (see `f_tol` for more info).

        Args:
            sw_x : Search window size in pixels along the horizontal detector axis.
            sw_y : Search window size in pixels along the vertical detector axis.
            blur : Smoothing kernel bandwidth used in `pixel_map` regularisation.
                The value is given in pixels.
            h0 : Smoothing kernel bandwidth used in `reference_image` regression.
                The value is given in pixels.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the total MSE. The
                iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            ref_method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

            pm_method : `pixel_map` update algorithm. The following keyword
                values are allowed:

                * 'gsearch' : Grid search algorithm.
                * 'rsearch' : Random search algorithm.
                * 'de'      : Differential evolution algorithm.

            pm_args : Pixel map update options. Accepts the following keyword
                arguments:

                * 'integrate' : Ensure that the updated pixel map is irrotational
                  by integrating and taking the derivative. False by default.
                * 'grid_size' : Grid size along one of the detector axes for
                  the 'gsearch' method. The grid shape is then (grid_size,
                  grid_size). The default value is :code:`int(0.5 * (sw_x + sw_y))`.
                * 'n_trials' : Number of points generated at each  pixel in
                  the detector grid for the 'rsearch' method. The default value
                  is :code:`int(0.25 * (sw_x + sw_y)**2)`.
                * 'n_iter' : The maximum number of generations over which the
                  entire population is evolved in the 'de' method. The default
                  value is 5.
                * 'pop_size' : The total population size in the 'de' method.
                  Must be greater or equal to 4. The default value is
                  :code:`max(int(0.25 * (sw_x + sw_y)**2) / n_iter, 4)`.
                * 'mutation' : The mutation constant in the 'de' method. It should
                  be in the range [0, 2]. The default value is 0.75.
                * 'recombination' : The recombination constant  in the 'de' method,
                  should be in the range [0, 1]. The default value is 0.7.
                * 'seed' : Specify seed for the random number generation.
                  Generated automatically if not provided.

            options : Extra options. Accepts the following keyword arguments:

                * 'epsilon' : Increment to `h0` to use for determining the
                  function gradient for `h0` update algorithm. The default
                  value is 1.4901161193847656e-08.
                * 'update_translations' : Update sample pixel translations
                  if True. The default value is False.
                * 'return_extra' : Return errors at each iteration if True.
                  The default value is False.

            verbose : bool, optional
                Set verbosity of the computation process.

        Returns:
            A tuple of two items ('st_obj', 'errors'). The elements of the tuple
            are as follows:

            * 'st_obj' : A new :class:`SpeckleTracking` object with the updated
              `pixel_map` and `reference_image`. `di_pix` and `dj_pix`
              are also updated if 'update_translations' in `options`
              is True.

            * 'errors' : List of total MSE values for each iteration. Only if
              'return_extra' in `options` is True.
        """
        integrate = pm_args.get('integrate', False)
        epsilon = options.get('epsilon', 1e-4)
        update_translations = options.get('update_translations', False)
        return_extra = options.get('return_extra', False)

        if h0 is None:
            if verbose:
                print("Finding the optimum kernel bandwidth...")
            h0 = self.find_hopt(method=ref_method, epsilon=epsilon, verbose=verbose)
            if verbose:
                print(f"New hopt = {h0:.3f}")

        obj = self.update_reference(hval=h0, method=ref_method)
        obj.update_errors.inplace_update()
        errors = [obj.error]

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} '\
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        if verbose:
            print(f"Initial MSE = {errors[0]:.6f}, "\
                  f"Initial h0 = {h0:.2f}")

        for _ in itor:
            # Update pixel_map
            new_obj = obj.update_pixel_map(search_window=search_window, blur=blur, integrate=integrate,
                                           method=pm_method, extra_args=pm_args)

            # Update di_pix, dj_pix
            if update_translations:
                new_obj.update_translations.inplace_update(sw_y=search_window[0], sw_x=search_window[1],
                                                           blur=blur)

            # Update reference_image
            new_obj.update_reference.inplace_update(hval=h0, method=ref_method)
            new_obj.update_errors.inplace_update()

            # Calculate errors
            errors.append(new_obj.error)
            if verbose:
                itor.set_description(f"Total MSE = {errors[-1]:.6f}")

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
        Generate a reference error based on the training subset and find the
        error for the test error.

        Args:
            hval : `reference_image` kernel bandwidths in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

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
                                      ds_y=self.ds_y, ds_x=self.ds_x, h=hval,
                                      num_threads=self.num_threads)
        elif method == 'LOWESS':
            I0, n0, m0 = LOWESS_reference(I_n=self.data, W=self.whitefield * self.train_mask,
                                          u=self.pixel_map, di=self.di_pix, dj=self.dj_pix,
                                          ds_y=self.ds_y, ds_x=self.ds_x, h=hval,
                                          num_threads=self.num_threads)
        else:
            raise ValueError('Method keyword is invalid')
        error = pm_total_error(I_n=self.data, W=self.whitefield * self.test_mask, I0=I0,
                               u=self.pixel_map, di=self.di_pix - n0, dj=self.dj_pix - m0,
                               ds_y=self.ds_y, ds_x=self.ds_x, sigma=self.sigma,
                               num_threads=self.num_threads)
        return error

    def CV_curve(self, harr: np.ndarray, method: str='KerReg') -> np.ndarray:
        """Return a set cross-validation errors for a set of kernel
        bandwidths.

        Args:
            harr : Set of `reference_image` kernel bandwidths in pixels.
            method : `reference_image` update algorithm. The following keyword
                values are allowed:

                * 'KerReg' : Kernel regression algorithm.
                * 'LOWESS' : Local weighted linear regression.

        Returns:
            An array of cross-validation errors.

        See Also:
            :func:`pyrost.bin.pm_total_error` : Full details of the error metric.
        """
        mse_list = []
        for hval in np.array(harr, ndmin=1):
            mse_list.append(self.CV(hval, method=method))
        return np.array(mse_list)
