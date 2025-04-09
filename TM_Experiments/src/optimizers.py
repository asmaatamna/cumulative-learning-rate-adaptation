# ---------------------------------------------------------*\
# Title: Optimizer Loader
# Author: TM
# ---------------------------------------------------------*/

import torch
from torch.optim import Optimizer
import math

# -------------------- Custom Adam_Clara --------------------

class AdamClara(Optimizer):
    """Implements Adam optimizer with CLARA (Cumulative Learning Rate Adaptation)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), c=0.2, d=1.0, eps=1e-8, adapt_lr=True):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(base_lr=lr, betas=betas, c=c, d=d, eps=eps, adapt_lr=adapt_lr)
        super(AdamClara, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            base_lr = group['base_lr']
            beta1, beta2 = group['betas']
            c = group['c']
            d = group['d']
            eps = group['eps']
            adapt_lr = group['adapt_lr']

            path_norm_squared_sum = 0.0
            path_numel_sum = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['path'] = torch.zeros_like(p.data)

                m, v, path = state['m'], state['v'], state['path']
                state['step'] += 1

                # Adam update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                if adapt_lr:
                    step_norm = torch.norm(step)
                    if step_norm > 0:
                        step = step / step_norm

                # Update path
                sqrt_term = math.sqrt(c * (2 - c))
                path.mul_(1 - c).add_(step, alpha=sqrt_term)

                # Track path norm for lr scaling
                path_norm_squared_sum += path.norm() ** 2
                path_numel_sum += path.numel()

                # Save updated path
                state['path'] = path

            # Compute global lr scaling factor
            if adapt_lr and path_numel_sum > 0:
                path_norm_squared_avg = path_norm_squared_sum / path_numel_sum
                lr_factor = math.exp(c / (2 * d) * (path_norm_squared_avg.item() - 1))
            else:
                lr_factor = 1.0

            # Final parameter update (second loop)
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                m, v = state['m'], state['v']

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                if adapt_lr:
                    step_norm = torch.norm(step)
                    if step_norm > 0:
                        step = step / step_norm

                lr_scaled = base_lr * lr_factor
                p.data.add_(step, alpha=-lr_scaled)

        return loss


# -------------------- Optimizer Factory --------------------


def get_optimizer(optimizer_name, model_parameters, learning_rate=0.001):
    """Returns the requested optimizer initialized with model parameters."""

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name == "sgdmomentum":
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model_parameters, lr=learning_rate)
    elif optimizer_name == "adam_clara":
        optimizer = AdamClara(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")

    return optimizer


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
