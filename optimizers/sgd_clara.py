from torch.optim.optimizer import Optimizer, required
import copy
import math
import torch

class SGD_CLARA(Optimizer):
    r"""Implements vanilla Gradient Descent with Cumulative Learning Rate Adaptation (CLARA).
    Args: TODO

    Example:

    """

    def __init__(self, params, lr=1e-3, c=0.2, d=1.0, adapt_lr=True):
        c2 = math.sqrt(c * (2 - c))
        
        defaults = dict(lr=lr, c=c, c2=c2, d=d, adapt_lr=adapt_lr)
        
        super(SGD_CLARA, self).__init__(params, defaults)
        
        

    def __setstate__(self, state):
        super(SGD_CLARA, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            c = group['c']  # Smoothing factor for path
            c2 = group['c2']  # Precomputed value derived from c
            d = group['d']  # Damping factor used is CLARA
            adapt_lr = group['adapt_lr']  # Whether to update learning rate

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['path'] = torch.zeros_like(p.data)

                state['step'] += 1
                step = grad  # TODO: Check whether copying grad is necessary (Fabian: the logger callback sees the gradients before the optimizer step, so its fine)

                if adapt_lr:
                    step_norm = torch.linalg.norm(step)
                    if step_norm > 0:
                        step.div_(step_norm)

                # Calculate "gradient descent" update
                p.data.add_(step, alpha=-group['lr'])

                # Update path
                state['path'].mul_(1 - c).add_(step, alpha=c2)

                # Cumulative Learning Rate Adaptation (CLARA)
                if adapt_lr:
                    group['lr'] *= math.exp(c / (2 * d) * (torch.linalg.norm(state['path']) - 1))  # TODO: norm().item()?

        return loss