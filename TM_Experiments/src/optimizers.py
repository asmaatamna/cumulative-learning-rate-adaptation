# ---------------------------------------------------------*\
# Title: Optimizer Loader
# Author: TM 
# ---------------------------------------------------------*/

import torch
from torch.optim import Optimizer
import math

# -------------------- Custom Adam_Clara Variants --------------------


class AdamClaraGlobal(Optimizer):
    """Improved Adam optimizer with CLARA (Global Averaging, Two Loops, Clipped Scaling)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), c=0.2, d=1.0, eps=1e-8, adapt_lr=True):
        defaults = dict(base_lr=lr, betas=betas, c=c, d=d, eps=eps, adapt_lr=adapt_lr, lr=lr)
        super(AdamClaraGlobal, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            base_lr = group['base_lr']
            beta1, beta2 = group['betas']
            c, d, eps, adapt_lr = group['c'], group['d'], group['eps'], group['adapt_lr']

            path_norm_squared_sum = 0.0
            path_numel_sum = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['path'] = torch.zeros_like(p.data)

                m, v, path = state['m'], state['v'], state['path']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                sqrt_term = math.sqrt(c * (2 - c))
                path.mul_(1 - c).add_(step, alpha=sqrt_term)

                path_norm_squared_sum += path.norm() ** 2
                path_numel_sum += path.numel()

                state['path'] = path

            # Calculate global scaling factor
            if adapt_lr and path_numel_sum > 0:
                path_norm_squared_avg = path_norm_squared_sum / path_numel_sum
                lr_factor = math.exp((c / (2 * d)) * (path_norm_squared_avg.item() - 1))
                lr_factor = min(max(lr_factor, 0.1), 10.0)  # ✅ Clipping neu eingebaut!
            else:
                lr_factor = 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                m, v = state['m'], state['v']
                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                p.data.add_(step, alpha=-base_lr * lr_factor)

            group['lr'] = base_lr * lr_factor

        return loss


# -------------------- AdamClaraLocal --------------------

class AdamClaraLocal(Optimizer):
    """Improved Adam optimizer with CLARA (Local Updates, Single Loop, Clipped Scaling)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), c=0.2, d=1.0, eps=1e-8, adapt_lr=True):
        defaults = dict(lr=lr, betas=betas, c=c, d=d, eps=eps, adapt_lr=adapt_lr)
        super(AdamClaraLocal, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            c, d, eps, adapt_lr = group['c'], group['d'], group['eps'], group['adapt_lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['path'] = torch.zeros_like(p.data)

                m, v, path = state['m'], state['v'], state['path']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                sqrt_term = math.sqrt(c * (2 - c))
                path.mul_(1 - c).add_(step, alpha=sqrt_term)

                norm_path_squared = path.norm() ** 2
                dim = path.numel()
                lr_factor = math.exp((c / (2 * d)) * (norm_path_squared.item() / dim - 1))
                lr_factor = min(max(lr_factor, 0.1), 10.0)  # ✅ Clipping neu eingebaut!

                if adapt_lr:
                    lr = lr * lr_factor

                p.data.add_(step, alpha=-lr)

            group['lr'] = lr

        return loss


# -------------------- AdamClaraSmoothed --------------------

class AdamClaraSmoothed(Optimizer):
    """Improved Adam optimizer with CLARA (Smoothed and Clipped LR, Boosted Exponent)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), c=0.2, d=1.0, eps=1e-8, adapt_lr=True):
        defaults = dict(base_lr=lr, betas=betas, c=c, d=d, eps=eps, adapt_lr=adapt_lr, lr_smooth=lr, lr=lr)
        super(AdamClaraSmoothed, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            base_lr = group['base_lr']
            lr_smooth = group['lr_smooth']
            beta1, beta2 = group['betas']
            c, d, eps, adapt_lr = group['c'], group['d'], group['eps'], group['adapt_lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['path'] = torch.zeros_like(p.data)

                m, v, path = state['m'], state['v'], state['path']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                step = m_hat / (v_hat.sqrt() + eps)

                sqrt_term = math.sqrt(c * (2 - c))
                path.mul_(1 - c).add_(step, alpha=2.0 * sqrt_term)

                norm_path_squared = path.norm() ** 2
                dim = path.numel()
                exponent = (c / (2 * d)) * (norm_path_squared.item() / dim - 1)
                scaling_factor = math.exp(5.0 * exponent)

                # Clip scaling factor (more aggressive)
                scaling_factor = min(max(scaling_factor, 0.01), 10.0)

                # Smooth learning rate
                lr_smooth = 0.9 * lr_smooth + 0.1 * (base_lr * scaling_factor)

                if adapt_lr:
                    p.data.add_(step, alpha=-lr_smooth)

                group['lr_smooth'] = lr_smooth

            group['lr'] = group['lr_smooth']

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
    elif optimizer_name == "adam_clara_global":
        return AdamClaraGlobal(model_parameters, lr=learning_rate)
    elif optimizer_name == "adam_clara_local":
        return AdamClaraLocal(model_parameters, lr=learning_rate)
    elif optimizer_name == "adam_clara_smoothed":
        return AdamClaraSmoothed(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")

    return optimizer


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
