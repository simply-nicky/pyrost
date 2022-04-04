
from typing import Callable, Dict, Optional, Union
from scipy.optimize import approx_fprime
from scipy.optimize.minpack2 import dcsrch
from numpy.linalg import norm
import numpy as np
from copy import deepcopy

class BFGS():
    """Minimize a function using the BFGS algorithm. Uses the line search
    algorithm to enforce strong Wolfe conditions. 
    """
    params: Dict[str, Union[int, float, np.ndarray]]

    def __init__(self, loss: Callable[[np.ndarray], float], x0: np.ndarray,
                 grad: Optional[Callable[[np.ndarray], float]]=None,
                 epsilon=1e-4, c1: float=1e-4, c2: float=0.9, xtol=1e-14):
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

        """
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

    def _line_search(self, maxiter: int, amin: float, amax: float):
        self._p['gfkp1'] = self._p['gfk']
        phi0 = self._p['fval']
        old_phi0 = self._p['old_fval']
        derphi0 = self._derphi(0.0)

        if derphi0 != 0.0:
            alpha1 = min(1.0, 2.02 * (phi0 - old_phi0) / derphi0)
        else:
            alpha1 = 1.0

        if alpha1 <= 0.0:
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
        self._line_search(maxiter=maxiter, amin=amin, amax=amax)

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
        self._p['Hk'] = np.dot(A1, np.dot(self._p['Hk'], A2)) + \
                        (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])

        if np.trace(self._p['Hk']) < 0.0:
            raise RuntimeError('Line search failed: inverse Hessian matrix is negative')

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
