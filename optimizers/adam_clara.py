import torch
import math


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
                 c=0.2, d=None, adapt_lr=True,
                 unit_step_direction=True,
                 weight_decay=0):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # Initialize optimizer first
        defaults = dict(lr=lr, adapt_lr=adapt_lr, unit_step_direction=unit_step_direction, betas=betas, eps=eps,
                        c=c, d=d, weight_decay=weight_decay)

        super().__init__(params, defaults)

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

        i = 0  # TODO: Rename
        for group in self.param_groups:
            decay = group['weight_decay']
            c = group['c']
            d = group['d']
            eps = group['eps']
            lr = group['lr']
            beta1, beta2 = group['betas']
            adapt_lr = group['adapt_lr']  # Whether to update learning rate
            unit_step_direction = group['unit_step_direction']  # Whether to use normalized gradient to update parameters
            mu = group['mu']
            sigma = group['sigma']

            for p in group['params']:
                i += 1
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
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Compute Adam step
                m_hat = m / bias_correction1
                v_hat = (v / bias_correction2)
                adam_step = m_hat / (v_hat.sqrt() + eps)

                # Calculate step norm
                step_norm = torch.norm(adam_step)

                if adapt_lr and unit_step_direction:
                    # Normalize step direction
                    if step_norm > 0:
                        adam_step.div_(step_norm)

                # Take step
                p.data.add_(adam_step, alpha=-lr)

                # Update CLARA path (exponential moving average of normalized steps)
                if adapt_lr and unit_step_direction:
                    path.mul_(1 - c).add_(adam_step, alpha=c)
                else:
                    path.mul_(1 - c).add_(adam_step / step_norm, alpha=c)  # TODO: Divide by norm only if not zero

                # Collect path information
                all_paths.append(path.flatten())
                total_params += path.numel()

        # Global learning rate adaptation
        if total_params > 0:
            full_path = torch.cat(all_paths)

            path_norm = torch.linalg.norm(full_path).pow(2).item()
            if adapt_lr:
                # Calculate learning rate adjustment
                lr_multiplier = math.exp(d * (path_norm / (self.total_params * mu) - 1))

                # Update learning rate for all groups
                for group in self.param_groups:
                    group['lr'] *= lr_multiplier

        return loss
