
"""Object oriented implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
for a function of one parameter.
"""
from typing import Callable, Dict, Optional, Tuple, Union
from copy import deepcopy
from scipy.optimize import approx_fprime
from scipy.optimize.minpack2 import dcsrch
from numpy.linalg import norm
import numpy as np

class BFGS():
    """Minimize a function of one parameter using the the Broyden-Fletcher-
    Goldfarb-Shanno (BFGS) algorithm.
    """
    params: Dict[str, Union[int, float, np.ndarray]]

    def __init__(self, loss: Callable[[np.ndarray], float], x0: np.ndarray,
                 grad: Optional[Callable[[np.ndarray], float]]=None,
                 epsilon=1e-4, c1: float=1e-4, c2: float=0.9, xtol=1e-14,
                 line_search: str='minpack'):
        """
        Args:
            loss : Objective function to be minimized.
            x0 : Initial guess.
            grad : Gradient of the objective function.
            epsilon : If `grad` is approximated, use this value for the step
                size.
            c1 : Parameter for Armijo condition rule.
            c2 : Parameter for curvature condition rule.
            xtol : Relative tolerance for an acceptable step in the line search
                algorithm.
            line_search: Choose the implementation of the line search algorithm 
                to enforce strong Wolfe conditions. The following keyword values
                are allowed:

                * `minpack` : MINPACK line search algorithm.
                * `scipy` : SciPy line search algorithm.
        """
        if line_search == 'minpack':
            self.line_search = self._line_search_minpack
        elif line_search == 'scipy':
            self.line_search = self._line_search_scipy
        else:
            raise ValueError('Invalid line_search keyword argument')

        self._p = {'fcount': 0, 'gcount': 0, 'xk': x0,
                   'c1': c1, 'c2': c2, 'xtol': xtol, 'epsilon': epsilon}

        self.update_loss(loss, grad)

        self._p['fval'] = self.loss(self._p['xk'])
        self._p['I'] = np.eye(len(self._p['xk']), dtype=int)
        self._p['Hk'] = np.eye(len(self._p['xk']))
        self._update_gnorm()
        self._p['old_fval'] = self._p['fval'] + 0.5 * norm(self._p['gfk'])

    def update_loss(self, loss: Callable[[np.ndarray], float],
                    grad: Optional[Callable[[np.ndarray], float]]=None) -> None:
        """Update the objective function to minimize.

        Args:
            loss : Objective function to be minimized.
            grad : Gradient of the objective function.
        """
        if grad is None:
            grad = lambda x: approx_fprime(x, loss, self._p['epsilon'])
        self._loss = loss
        self._grad = grad
        self._p['gfk'] = self.grad(self._p['xk'])
        self._p['gfkp1'] = self._p['gfk']

    def loss(self, x: np.ndarray) -> float:
        """Return the objective value for a given argument.

        Args:
            x : Argument value.

        Returns:
            Objective value
        """
        self._p['fcount'] += 1
        return self._loss(x)

    def grad(self, x: np.ndarray) -> float:
        """Return the gradient value of the objective function for a given
        argument.

        Args:
            x : Argument value.

        Returns:
            A gradient value of the objective function.
        """
        self._p['gcount'] += 1
        return self._grad(x)

    def _update_pk(self):
        self._p['pk'] = -np.dot(self._p['Hk'], self._p['gfk'])

    def _update_gnorm(self):
        self._p['gnorm'] = np.amax(np.abs(self._p['gfk']))

    def _phi(self, s: np.ndarray) -> float:
        return self.loss(self._p['xk'] + s * self._p['pk'])

    def _derphi(self, s: np.ndarray) -> float:
        self._p['gfkp1'] = self.grad(self._p['xk'] + s * self._p['pk'])
        return np.dot(self._p['gfkp1'], self._p['pk'])

    @staticmethod
    def _cubicmin(a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float) -> Union[float, None]:
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                db = b - a
                dc = c - a
                denom = (db * dc) ** 2 * (db - dc)
                d1 = np.empty((2, 2))
                d1[0, 0] = dc ** 2
                d1[0, 1] = -db ** 2
                d1[1, 0] = -dc ** 3
                d1[1, 1] = db ** 3
                [A, B] = np.dot(d1, np.asarray([fb - fa - fpa * db,
                                                fc - fa - fpa * dc]).flatten())
                A /= denom
                B /= denom
                radical = B * B - 3.0 * A * fpa
                xmin = a + (-B + np.sqrt(radical)) / (3.0 * A)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    @staticmethod
    def _quadmin(a: float, fa: float, fpa: float, b: float, fb: float) -> Union[float, None]:
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                db = b - a
                xmin = a - fpa / (2.0 * (fb - fa - fpa * db) / (db * db))
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    def _zoom(self, a_lo: float, a_hi: float, phi_lo: float, phi_hi: float, derphi_lo: float,
              phi0: float, derphi0: float, maxiter: int=10) -> Tuple[float, float]:
        phi_rec = phi0
        a_rec = 0.0
        for i in range(maxiter):
            # interpolate to find a trial step length between a_lo and
            # a_hi Need to choose interpolation here. Use cubic
            # interpolation and then if the result is within delta *
            # dalpha or outside of the interval bounded by a_lo or a_hi
            # then use quadratic interpolation, if the result is still too
            # close, then use bisection

            dalpha = a_hi - a_lo
            if dalpha < 0.0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            if i > 0:
                cchk = 0.2 * dalpha # cubic interpolant check
                a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                                     a_rec, phi_rec)
            if i == 0 or a_j is None or a_j > b - cchk or a_j < a + cchk:
                qchk = 0.1 * dalpha # quadratic interpolant check
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if a_j is None or a_j > b - qchk or a_j < a + qchk:
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j

            phi_aj = self._phi(a_j)
            if (phi_aj > phi0 + self._p['c1'] * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = self._derphi(a_j)
                if abs(derphi_aj) <= -self._p['c2'] * derphi0:
                    break
                if derphi_aj * (a_hi - a_lo) >= 0.0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo
                else:
                    phi_rec = phi_lo
                    a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj

        a_star = a_j
        val_star = phi_aj

        return a_star, val_star

    def _line_search_scipy(self, maxiter: int, amin: float, amax: float):
        phi0 = self._p['fval']
        old_phi0 = self._p['old_fval']
        derphi0 = self._derphi(0.0)

        if derphi0 != 0.0:
            alpha1 = min(1.0, 2.02 * (phi0 - old_phi0) / derphi0)
        else:
            alpha1 = 1.0

        if alpha1 <= amin:
            alpha1 = 1.0

        alpha0 = 0.0
        alpha1 = min(alpha1, amax)
        phi_a1 = self._phi(alpha1)
        phi_a0 = phi0
        derphi_a0 = derphi0

        for i in range(maxiter):
            if alpha1 == amin or alpha0 == amax:
                break

            if phi_a1 > phi0 + self._p['c1'] * alpha1 * derphi0 or (phi_a1 >= phi_a0 and i > 0):
                alpha_star, phi_star = self._zoom(alpha0, alpha1, phi_a0,
                                                  phi_a1, derphi_a0, phi0, derphi0)
                break

            derphi_a1 = self._derphi(alpha1)
            if abs(derphi_a1) <= -self._p['c2'] * derphi0:
                alpha_star = alpha1
                phi_star = phi_a1
                break

            if derphi_a1 >= 0.0:
                alpha_star, phi_star = self._zoom(alpha1, alpha0, phi_a1,
                                                  phi_a0, derphi_a1, phi0, derphi0)
                break

            # increase by factor of two on each iteration
            alpha0 = max(alpha1, amin)
            alpha1 = min(2.0 * alpha1, amax)
            phi_a0 = phi_a1
            phi_a1 = self._phi(alpha1)
            derphi_a0 = derphi_a1
        else:
            alpha_star = alpha1
            phi_star = phi_a1

        # stopping test maxiter reached
        self._p['alpha_k'] = alpha_star
        self._p['fval'] = phi_star
        self._p['old_fval'] = phi0

    def _line_search_minpack(self, maxiter: int, amin: float, amax: float):
        phi0 = self._p['fval']
        old_phi0 = self._p['old_fval']
        derphi0 = self._derphi(0.0)

        if derphi0 != 0.0:
            alpha1 = min(1.0, 2.02 * (phi0 - old_phi0) / derphi0)
        else:
            alpha1 = 1.0

        if alpha1 <= amin:
            alpha1 = 1.0

        phi1 = phi0
        derphi1 = derphi0
        isave = np.zeros((2,), np.intc)
        dsave = np.zeros((13,), float)
        task = b'START'

        for _ in range(maxiter):
            stp, phi1, derphi1, task = dcsrch(alpha1, phi1, derphi1, self._p['c1'],
                                              self._p['c2'], self._p['xtol'],
                                              task, amin, amax, isave, dsave)

            if task[:2] == b'FG':
                alpha1 = stp
                phi1 = self._phi(stp)
                derphi1 = self._derphi(stp)
            else:
                break

        self._p['alpha_k'] = stp
        self._p['fval'] = phi1
        self._p['old_fval'] = phi0

    def step(self, maxiter: int=10, amin: float=1e-100, amax: float=1e100):
        """Performs a single optimization step.

        Args:
            maxiter : Maximum number of iteration of the line search algorithm
                to perform.
            amin : Minimum step size.
            amax : Maximum step size.
        """
        self._update_pk()
        self.line_search(maxiter=maxiter, amin=amin, amax=amax)

        xkp1 = self._p['xk'] + self._p['alpha_k'] * self._p['pk']
        sk = xkp1 - self._p['xk']
        self._p['xk'] = xkp1

        yk = self._p['gfkp1'] - self._p['gfk']
        self._p['gfk'] = self._p['gfkp1']
        self._update_gnorm()

        rhok_inv = np.dot(yk, sk)
        if rhok_inv == 0.0:
            rhok = 1000.0
        else:
            rhok = 1.0 / rhok_inv

        A1 = self._p['I'] - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = self._p['I'] - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        new_Hk = np.dot(A1, np.dot(self._p['Hk'], A2)) + \
                        (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])

        if np.trace(self._p['Hk']) > 0.0:
            self._p['Hk'] = new_Hk

    def state_dict(self):
        """Returns the state of the optimizer as a dict.

        Returns:
            A dictionary with all the parameters of the optimizer:

            * `fcount` : Number of functions evaluations made.
            * `gcount` : Number of gradient evaluations made.
            * `c1` : Parameter for Armijo condition rule.
            * `c2` : Parameter for curvature condition rule.
            * `xk` : The current point.
            * `fval` : Objective value of the current point.
            * `old_fval` : Objective value of the point prior to 'xk'.
            * `gfk` : Gradient value of the current point.
            * `gnorm` : Gradient norm value of the current point.
            * `Hk` : The current guess of the Hessian matrix.
            * `epsilon` : If `grad` is approximated, use this value for the
                step size.
            * `xtol` :  Relative tolerance for an acceptable step in the line
                search algorithm.
        """
        return deepcopy(self._p)
