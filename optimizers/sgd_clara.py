from torch.optim.optimizer import Optimizer, required
import copy
import math
import torch

class SGD_Clara(Optimizer):
    r"""Implements vanilla Gradient Descent with Cumulative Learning Rate Adaptation (CLARA).
    Args: TODO

    Example:

    """

    def __init__(self, params, lr=1e-3, c=0.2, d=None, adapt_lr=True, unit_step_direction=True):
        # Initialize optimizer first
        defaults = dict(lr=lr, c=c, d=d, adapt_lr=adapt_lr, unit_step_direction=unit_step_direction)  # partial init
        super(SGD_Clara, self).__init__(params, defaults)

        # Now self.param_groups is available
        self.total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.total_params += p.numel()

        # Compute dependent values
        mu = c / (2 - c)  # Mean of norm squared of uniformly sampled cumulated random steps. See Sebag et al. 2017
        sigma = math.sqrt(2) * mu * (1 - c) / math.sqrt(((1 - c) ** 2 + 1) * self.total_params)  # Standard deviation

        # Define damping if no default value passed
        if d is None:
            d = 2 * sigma

        # Save these to each param group
        for group in self.param_groups:
            group['mu'] = mu
            group['sigma'] = sigma
            group['d'] = d

    def __setstate__(self, state):
        super(SGD_Clara, self).__setstate__(state)

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

        # Collect all paths for global learning rate adjustment
        all_paths = []
        total_params = 0  # Total number of scalar parameters

        param_tensor_counter = 0
        for group in self.param_groups:
            c = group['c']  # Smoothing factor for path
            d = group['d']  # Damping factor used is CLARA
            adapt_lr = group['adapt_lr']  # Whether to update learning rate
            unit_step_direction = group['unit_step_direction']  # Whether to use normalized gradient to update parameters
            mu = group['mu']
            sigma = group['sigma']

            for p in group['params']:
                param_tensor_counter += 1
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['path'] = torch.zeros_like(p.data)
                    state['path2'] = torch.zeros_like(p.data)

                path = state['path']
                state['step'] += 1
                step = grad  # TODO: Check whether copying grad is necessary (Fabian: the logger callback sees the gradients before the optimizer step, so its fine)

                # Calculate step norm
                step_norm = torch.norm(step)

                if adapt_lr and unit_step_direction:
                    step.div_(torch.clamp(step_norm, min=1e-12))

                # Calculate "gradient descent" update
                p.data.add_(step, alpha=-group['lr'])

                # Update path
                if adapt_lr and unit_step_direction:
                    path.mul_(1 - c).add_(step, alpha=c)
                else:
                    path.mul_(1 - c).add_(step / torch.clamp(step_norm, min=1e-12), alpha=c)

                # Collect path information
                all_paths.append(path.flatten())
                total_params += path.numel()

        # Global learning rate adaptation
        if total_params > 0:
            full_path = torch.cat(all_paths)

            # Calculate learning rate adjustment
            path_norm = torch.linalg.norm(full_path).pow(2).item()

            # Cumulative Learning Rate Adaptation (CLARA)
            if adapt_lr:
                # Compute lr factor
                lr_multiplier = math.exp(d * (path_norm / (param_tensor_counter * mu) - 1))

                # Update learning rate for all groups
                for group in self.param_groups:
                    group['lr'] *= lr_multiplier

        return loss


class SGD_CLARA_global(Optimizer):
    r"""Implements vanilla Gradient Descent with Cumulative Learning Rate Adaptation (CLARA).
    Args: TODO

    Example:

    """

    def __init__(self, params, lr=1e-3, c=0.2, d=None, adapt_lr=True, unit_step_direction=True):
        # Initialize optimizer first
        defaults = dict(lr=lr, c=c, d=d, adapt_lr=adapt_lr, unit_step_direction=unit_step_direction)  # partial init
        super(SGD_CLARA_global, self).__init__(params, defaults)

        # Now self.param_groups is available
        self.total_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.total_params += p.numel()

        # Compute dependent values
        mu = c / (2 - c)  # Mean of norm squared of uniformly sampled cumulated random steps. See Sebag et al. 2017
        sigma = math.sqrt(2) * mu * (1 - c) / math.sqrt(((1 - c) ** 2 + 1) * self.total_params)  # Standard deviation

        # Define damping if no default value passed
        if d is None:
            d = 2 * sigma

        # Save these to each param group
        for group in self.param_groups:
            group['mu'] = mu
            group['sigma'] = sigma
            group['d'] = d

    def __setstate__(self, state):
        super(SGD_CLARA_global, self).__setstate__(state)

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

        grads = []  # To collect flattened gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()

                # Flatten and collect
                grads.append(grad.view(-1))

        if grads:
            all_grads = torch.cat(grads)
            # Now you can use `all_grads` as a single gradient vector

            # Example: compute its norm
            grad_norm = torch.linalg.norm(all_grads).item()
            # print("Gradient norm:", grad_norm)

        # Collect all paths for global learning rate adjustment
        all_paths = []
        total_params = 0  # Total number of scalar parameters

        param_tensor_counter = 0
        for group in self.param_groups:
            c = group['c']  # Smoothing factor for path
            d = group['d']  # Damping factor used is CLARA
            adapt_lr = group['adapt_lr']  # Whether to update learning rate
            unit_step_direction = group['unit_step_direction']  # Whether to use normalized gradient to update parameters
            mu = group['mu']
            sigma = group['sigma']

            for p in group['params']:
                param_tensor_counter += 1
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['path'] = torch.zeros_like(p.data)
                    state['path2'] = torch.zeros_like(p.data)

                path = state['path']
                state['step'] += 1
                step = grad  # TODO: Check whether copying grad is necessary (Fabian: the logger callback sees the gradients before the optimizer step, so its fine)

                if adapt_lr and unit_step_direction:
                    step.div_(torch.clamp(grad_norm, min=1e-12))

                # Calculate "gradient descent" update
                p.data.add_(step, alpha=-group['lr'])

                # Update path
                if adapt_lr and unit_step_direction:
                    path.mul_(1 - c).add_(step, alpha=c)
                else:
                    path.mul_(1 - c).add_(step / torch.clamp(grad_norm, min=1e-12), alpha=c)

                # Collect path information
                all_paths.append(path.flatten())
                total_params += path.numel()

        # Global learning rate adaptation
        if total_params > 0:
            full_path = torch.cat(all_paths)

            # Calculate learning rate adjustment
            path_norm = torch.linalg.norm(full_path).pow(2).item()

            # Cumulative Learning Rate Adaptation (CLARA)
            if adapt_lr:
                # Compute lr factor
                lr_multiplier = math.exp(d * (path_norm / mu - 1))

                # Update learning rate for all groups
                for group in self.param_groups:
                    group['lr'] *= lr_multiplier

        return loss
