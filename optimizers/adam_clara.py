import torch
import math


class Adam_Clara(torch.optim.Optimizer):
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

        # Compute reference distribution parameters for cumulative learning rate adaptation
        # TODO: Handle mu and sigma values systematically
        if self.total_params == 2:
            mu = 0.5958
            sigma = 0.2553
        if self.total_params == 15:
            mu = 0.6729
            sigma = 0.0814
        if self.total_params == 42:
            mu = 0.6770
            sigma = 0.0489
        if self.total_params == 62:
            mu = 0.6853
            sigma = 0.0357
        if self.total_params == 650:
            mu = 0.6827
            sigma = 0.0119
        if self.total_params == 235146:
            mu = 0.6825
            sigma = 0.0006
        if self.total_params == 2193226:
            mu = 0.6825
            sigma = 0.0002
        if self.total_params == 2216356:
            mu = 0.6826
            sigma  = 0.0002

        # Define damping if no default value passed
        if d is None:
            d = 2 * sigma  # TODO: Rather arbitrary

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

        param_tensor_counter = 0
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
                param_tensor_counter += 1
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
                    adam_step.div_(torch.clamp(step_norm, min=1e-12))

                # Take step
                p.data.add_(adam_step, alpha=-lr)

                # Update CLARA path (exponential moving average of normalized steps)
                if adapt_lr and unit_step_direction:
                    path.mul_(1 - c).add_(adam_step, alpha=c)
                else:
                    path.mul_(1 - c).add_(adam_step / torch.clamp(step_norm, min=1e-12), alpha=c)

                # Collect path information
                all_paths.append(path.flatten())
                total_params += path.numel()

        # Global learning rate adaptation
        if total_params > 0:
            full_path = torch.cat(all_paths)

            path_norm = torch.linalg.norm(full_path).pow(2).item()
            if adapt_lr:
                # Calculate learning rate adjustment
                lr_multiplier = math.exp(d * (path_norm / (param_tensor_counter * mu) - 1))

                # Update learning rate for all groups
                for group in self.param_groups:
                    group['lr'] *= lr_multiplier

        return loss
