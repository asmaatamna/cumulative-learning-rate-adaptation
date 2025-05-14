# ---------------------------------------------------------*\
# Title: Optimizer Loader
# Author: TM
# ---------------------------------------------------------*/

import torch
from torch.optim import Optimizer
import math


#---------------------------------------------------------*/
# SGD with CLARA
#---------------------------------------------------------*/
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


#---------------------------------------------------------*/
# Adam with CLARA
#---------------------------------------------------------*/
class Adam_CLARA(torch.optim.Optimizer):
    r"""
    Implements Adam with CLARA. 
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float): 
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        c (float): 
            CLARA smoothing factor (default: 0.2)
        d (float): 
            CLARA damping factor (default: 1.0)
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
    """
    def __init__(self, params, lr=1.0, 
                 betas=(0.9, 0.999), eps=1e-8, 
                 c=0.01, d=10,
                 weight_decay=0):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        c=c, d=d, weight_decay=weight_decay)
        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Collect all paths for global learning rate adjustment
        all_paths = []
        total_params = 0  # Total number of scalar parameters

        for group in self.param_groups:
            decay = group['weight_decay']
            c = group['c']
            d = group['d']
            eps = group['eps']
            lr = group['lr']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p)
                    state['path'] = torch.zeros_like(p)

                m, v, path = state['m'], state['v'], state['path']
                state['step'] += 1
                t = state['step']
                
                # Apply weight decay
                if decay != 0:
                    grad.add_(p, alpha=decay)

                # Adam EMA updates
                m.mul_(beta1).add_(grad, alpha=(1-beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                             
                # Compute Adam step
                m_hat = m / bias_correction1
                v_hat = (v / bias_correction2)
                adam_step = m_hat / (v_hat.sqrt() + eps)  

                
                # Normalize step direction
                step_norm = torch.norm(adam_step)
                if step_norm > 0:
                    adam_step.div_(step_norm)
                
                # Take step
                p.data.add_(adam_step, alpha=-lr)
                
                # Update CLARA path (exponential moving average of normalized steps)
                path.mul_(1 - c).add_(adam_step, alpha=math.sqrt(c * (2 - c)))
                
                # Collect path information
                all_paths.append(path.flatten())
                total_params += path.numel()

        # Global learning rate adaptation
        if total_params > 0:
            full_path = torch.cat(all_paths)
            
            # Calculate learning rate adjustment
            path_norm = torch.norm(full_path).item()
            lr_multiplier = math.exp(c / (2 * d) * (path_norm - 1))
            
            # Update learning rate for all groups
            for group in self.param_groups:
                group['lr'] *= lr_multiplier 
            

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
