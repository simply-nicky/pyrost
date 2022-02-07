
from typing import Callable, Dict, Optional, Union
from scipy.optimize import approx_fprime
from scipy.optimize.minpack2 import dcsrch
from numpy.linalg import norm
import numpy as np
from copy import deepcopy

class BFGS():
    params: Dict[str, Union[int, float, np.ndarray]]

    def __init__(self, loss: Callable[[np.ndarray], float],  x0: float,
                 grad: Optional[Callable[[np.ndarray], float]]=None,
                 epsilon=1e-4, c1: float=1e-4, c2: float=0.9, xtol=1e-14):
        if grad is None:
            grad = lambda x: approx_fprime(x, loss, epsilon)
        self._loss, self._grad = loss, grad

        self._p = {'fcount': 0, 'gcount': 0, 'xk': np.atleast_1d(x0),
                   'c1': c1, 'c2': c2, 'xtol': xtol}

        self._p['fval'] = self.loss(self._p['xk'])
        self._p['gfk'] = self.grad(self._p['xk'])
        self._p['I'] = np.eye(len(self._p['xk']), dtype=int)
        self._p['Hk'] = self._p['I']
        self._update_gnorm()
        self._p['old_fval'] = self._p['fval'] + 0.5 * norm(self._p['gfk'])

    def loss(self, x: float) -> float:
        self._p['fcount'] += 1
        return self._loss(x)

    def grad(self, x: float) -> float:
        self._p['gcount'] += 1
        return self._grad(x)

    def _update_pk(self):
        self._p['pk'] = -np.dot(self._p['Hk'], self._p['gfk'])

    def _update_gnorm(self):
        self._p['gnorm'] = np.amax(np.abs(self._p['gfk']))

    def _phi(self, s: float) -> float:
        return self.loss(self._p['xk'] + s * self._p['pk'])

    def _derphi(self, s: float) -> float:
        gval = self.grad(self._p['xk'] + s * self._p['pk'])
        return np.dot(gval, self._p['pk'])

    def _line_search(self, maxiter: int=5, amin: float=1e-8, amax: float=1e3):
        phi0 = self._p['fval']
        old_phi0 = self._p['old_fval']
        derphi0 = self._derphi(0.0)

        if derphi0 != 0.0:
            alpha1 = min(1.0, 1.01 * 2.0 * (phi0 - old_phi0) / derphi0)
            if alpha1 <= 0.0:
                alpha1 = 1.0
        else:
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

        if stp <= 0.0:
            stp = 1.0

        self._p['fval'] = phi1
        self._p['old_fval'] = phi0
        self._p['alpha_k'] = stp

    def step(self):
        self._update_pk()
        self._line_search()

        xkp1 = self._p['xk'] + self._p['alpha_k'] * self._p['pk']
        sk = xkp1 - self._p['xk']
        self._p['xk'] = xkp1

        gfkp1 = self.grad(self._p['xk'])
        yk = gfkp1 - self._p['gfk']
        self._p['gfk'] = gfkp1
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

    def state_dict(self):
        return deepcopy(self._p)
