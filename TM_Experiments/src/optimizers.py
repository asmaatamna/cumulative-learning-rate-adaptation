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


class SGD_CLARA(Optimizer):
    """Implements vanilla Gradient Descent with Cumulative Learning Rate Adaptation (CLARA)."""

    def __init__(self, params, lr=1e-3, c=0.2, d=1.0, adapt_lr=True):
        defaults = dict(lr=lr, c=c, d=d, adapt_lr=adapt_lr)
        super(SGD_CLARA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_CLARA, self).__setstate__(state)

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
                step = grad  # TODO: Check whether copying grad is necessary

                if adapt_lr:
                    step_norm = torch.linalg.norm(step)
                    if step_norm > 0:
                        step.div_(step_norm)

                # Calculate "gradient descent" update
                p.data.add_(step, alpha=-group['lr'])

                # Update path
                c2 = math.sqrt(c * (2 - c))  # TODO: Compute once only
                state['path'].mul_(1 - c).add_(step, alpha=c2)

                # Cumulative Learning Rate Adaptation (CLARA)
                if adapt_lr:
                    group['lr'] *= math.exp(c / (2 * d) * (torch.linalg.norm(state['path']) - 1))

        return loss

class Adam_CLARA(Optimizer):
    """Implements Adam with Cumulative Learning Rate Adaptation (CLARA)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 c=0.2, d=1, adapt_lr=True, clamp_lr=(1e-6, 1e-3), norm_update=False, log_interval=100):
        """
        Args:
            params (iterable): parameters to optimize
            lr (float): initial learning rate
            betas (Tuple[float, float]): coefficients used for computing running averages
            eps (float): term added to denominator to improve numerical stability
            c (float): CLARA smoothing factor
            d (float): CLARA damping factor
            adapt_lr (bool): whether to apply CLARA learning rate adaptation
            clamp_lr (tuple): min and max learning rate
            norm_update (bool): whether to normalize update vector before applying
            log_interval (int): how often to log path norm and lr (in steps)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, c=c, d=d,
                        adapt_lr=adapt_lr, clamp_lr=clamp_lr,
                        norm_update=norm_update, log_interval=log_interval)
        super(Adam_CLARA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_CLARA, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            c = group['c']
            d = group['d']
            adapt_lr = group['adapt_lr']
            clamp_min, clamp_max = group['clamp_lr']
            norm_update = group['norm_update']
            log_interval = group['log_interval']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['path'] = torch.zeros_like(p.data)
                    group['base_lr'] = group['lr']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                path = state['path']
                state['step'] += 1
                step_num = state['step']

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step_num
                bias_correction2 = 1 - beta2 ** step_num
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = group['lr'] / bias_correction1
                adam_step = -step_size * exp_avg / denom

                if norm_update:
                    norm = torch.linalg.norm(adam_step)
                    if norm > 0:
                        adam_step = adam_step / norm

                # Update parameter
                p.data.add_(adam_step)

                # CLARA path update
                c2 = math.sqrt(c * (2 - c))
                path.mul_(1 - c).add_(adam_step, alpha=c2)

                # CLARA learning rate adaptation
                if adapt_lr:
                    path_norm = torch.linalg.norm(path)
                    factor = math.exp(c / (2 * d) * (path_norm - 1))
                    new_lr = group['base_lr'] * factor
                    group['lr'] = max(clamp_min, min(new_lr, clamp_max))

                    # # # Optional logging
                    # if step_num % log_interval == 0:
                    #     print(f"[Step {step_num}] PathNorm={path_norm:.4f}, "
                    #         f"CLARA factor={factor:.6f}, LR={group['lr']:.6e}, BaseLR={group['base_lr']:.6e}")


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
    elif optimizer_name == "sgd_clara":
        return SGD_CLARA(model_parameters, lr=learning_rate)
    elif optimizer_name == "adam_clara":
        return Adam_CLARA(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")

    return optimizer


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
