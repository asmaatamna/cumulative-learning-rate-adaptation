#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/

# optimizers/clara_adam_numpy/clara_adam_numpy.py

import numpy as np
from typing import Any, Dict, Callable, Tuple


class NumpyClaraAdam:
    def __init__(self, hyperparameters: Dict[str, Any], param_shape: Tuple[int]):
        self.lr = hyperparameters.get('learning_rate', 0.001)
        self.b1 = hyperparameters.get('b1', 0.9)
        self.b2 = hyperparameters.get('b2', 0.999)
        self.eps = hyperparameters.get('eps', 1e-8)
        self.c = hyperparameters.get('clara_c', 0.2)
        self.d = hyperparameters.get('clara_d', 1.0)

        # Initialize states
        self.m = np.zeros(param_shape)
        self.v = np.zeros(param_shape)
        self.path = np.zeros(param_shape)
        self.t = 0  # time step

    def update(self, grad: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.t += 1

        # Update biased moments
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)

        step = m_hat / (np.sqrt(v_hat) + self.eps)

        # Normalize step for CLARA
        if np.linalg.norm(step) > 0:
            step /= np.linalg.norm(step)

        # Update path and learning rate
        self.path = (1 - self.c) * self.path + np.sqrt(self.c * (2 - self.c)) * step
        dim = len(step)
        lr_scaled = self.lr * np.exp(self.c / (2 * self.d) * (np.linalg.norm(self.path) ** 2 / dim - 1))

        # Parameter update
        update = -lr_scaled * step
        new_params = params + update

        return new_params, update


def build_optimizer(hyperparameters: Dict[str, Any]) -> Callable:
    """
    Returns a closure-based optimizer with the NumPy-based CLARA-Adam logic.
    This function must return a function that takes (grads, params) and returns (new_params, update).
    """

    def init_fn(param_shape: Tuple[int]):
        optimizer = NumpyClaraAdam(hyperparameters, param_shape)

        def optimizer_fn(grad: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return optimizer.update(grad, params)

        return optimizer_fn

    return init_fn


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\