import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler


class CustomAdamLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr0: float, d: float = 0.1):
        super(CustomAdamLRScheduler, self).__init__(optimizer)
        self.lr = lr0
        self.path = 0  # Path, i.e. history of steps computed by Adam
        self.v = 0  # Adam steps' variance estimate
        self.c = 0.1  # Constant for cumul. path of Adam steps' variance
        self.s = 0
        self.d = d

    def get_lr(self):
        lrs = []
        updates = []

        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if 'exp_avg' in state and 'exp_avg_sq' in state:
                    mt = state['exp_avg']
                    vt = state['exp_avg_sq']
                    eps = group['eps']

                    # Compute Adam update step
                    update_step = -mt / (torch.sqrt(vt) + eps)
                    updates.append(update_step.view(-1))  # Flatten each update

        if updates:
            stacked_updates = torch.cat(updates).cpu().numpy()
            D = len(stacked_updates)

            # Update path variable
            self.path = (1 - self.c) * self.path + np.sqrt(self.c * (2 - self.c)) * stacked_updates
            # Update variance estimate
            self.s = self.s + np.linalg.norm(stacked_updates)**2
            self.v = self.s / (self.optimizer.state[self.optimizer.param_groups[0]['params'][-1]]['step'].item() * D)

            # Update learning rate starting from second iteration
            if self.optimizer.state[self.optimizer.param_groups[0]['params'][-1]]['step'].item() > 1:
                # print('Numerator: ', np.linalg.norm(self.path) ** 2)
                # print('Denominator: ', (D * self.v))
                self.lr = self.lr * np.exp(self.d * ((np.linalg.norm(self.path)**2 / (D * self.v)) - 1))  # (D * self.v)

            # Update learning rate for all optimizer parameter groups
            for _ in self.optimizer.param_groups:
                lrs.append(self.lr)

        # print('lrs: ', lrs)

        return lrs


class CustomSGDLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr0: float, d: float = 0.1, last_epoch: int = -1):
        self.lr = lr0
        self.path = 0  # Path, i.e. history of gradients
        self.c = 0.1   # Constant for cumul. path of gradients' variance
        self.d = d     # Damping factor
        self.s = 0
        super(CustomSGDLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        gradients = []
        iteration = self.last_epoch + 1

        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        if gradients:
            stacked_gradients = torch.cat(gradients).cpu().numpy()
            D = len(stacked_gradients)

            # Update path variable
            self.path = (1 - self.c) * self.path + np.sqrt(self.c * (2 - self.c)) * stacked_gradients
            # Update variance estimate
            self.s = self.s + np.linalg.norm(stacked_gradients)**2
            v = self.s / (iteration * D)

            # Update learning rate starting from second iteration
            if iteration > 1:
                self.lr = self.lr * np.exp(self.d * ((np.linalg.norm(self.path)**2 / (D * v)) - 1))

            # Update learning rate for all optimizer parameter groups
            for _ in self.optimizer.param_groups:
                lrs.append(self.lr)

        # print('lrs: ', lrs)

        return lrs


# Define the sphere function
def sphere(x):
    return torch.sum(x ** 2)


# Define the ellipsoid function
def ellipsoid(x):
    n = x.shape[0]
    c = torch.logspace(0, 6, steps=n, base=10, dtype=x.dtype, device=x.device)
    return torch.sum(c * x**2)


# Define the Sphere function
def sphere_numpy(x):
    return np.sum(x ** 2)


# Gradient of the Sphere function (known)
def sphere_gradient(x):
    return 2 * x


def optimize(f, x0, lr0, budget, algorithm=torch.optim.SGD, clara=False):
    x = torch.tensor(x0, requires_grad=True)
    optimizer = algorithm([x], lr=lr0)  # torch.optim.Adam([x], lr=lr0)

    if clara:
        # Initialize the scheduler
        if isinstance(optimizer, torch.optim.SGD):
            scheduler = CustomSGDLRScheduler(optimizer, lr0)
        if isinstance(optimizer, torch.optim.Adam):
            scheduler = CustomAdamLRScheduler(optimizer, lr0)

    # Keep track of learning rate, f-value and distance to optimum during optimization
    learning_rates = []
    f_values = []
    distance_to_opt = []
    grad_norm = []

    # Optimization loop
    for step in range(budget):
        optimizer.zero_grad()       # Clear previous gradients
        # print('x: ', x)
        loss = f(x)                 # Compute loss
        loss.backward()             # Compute gradients
        # Calculate the gradient norm
        grad_norm.append(torch.norm(x.grad).item())
        optimizer.step()            # Update parameters

        if clara:
            # Update the learning rate
            scheduler.step()

        for group in optimizer.param_groups:
            learning_rates.append(group['lr'].item() if isinstance(group['lr'], torch.Tensor) else group['lr'])

        f_values.append(loss.item())
        distance_to_opt.append(torch.norm(x).detach().numpy())

    return f_values, distance_to_opt, learning_rates, grad_norm


def vanilla_gd_clara_optimize(f, grad_f, x0, lr0, budget):
    x = x0
    lr = lr0
    dim = len(x0)

    path = 0
    s = 0
    c = 0.1
    d = 0.1

    learning_rates = []
    f_values = []
    distance_to_opt = []
    normalized_path = []

    for i in range(budget):
        # Update current solution using gradient information (sphere)
        grad = grad_f(x)
        x = x - lr * grad

        distance_to_opt.append(np.linalg.norm(x))
        f_values.append(f(x))

        # Update learning rate
        path = (1 - c) * path + np.sqrt(c * (2 - c)) * grad
        # Update variance estimate
        s = s + np.linalg.norm(grad)**2
        v = s / ((i + 1) * dim)

        # Update step-size starting from second iteration
        learning_rates.append(lr)
        # if i > 1:
        #     lr = lr * np.exp(d * ((np.linalg.norm(path)**2 / (dim * v)) - 1))

        normalized_path.append(np.linalg.norm(path)**2 / (dim * v))

    return f_values, distance_to_opt, learning_rates, normalized_path


x0 = np.ones(5)
lr0 = 1e-1
budget = 1000


f_values_clara, distance_to_opt_clara, learning_rates_clara, norm_path_clara = vanilla_gd_clara_optimize(sphere_numpy, sphere_gradient, x0, lr0=lr0, budget=budget)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))  # sharey='row'

fig.suptitle(f'Vanilla gradient descent with CLARA, sphere, D = {str(len(x0))}')

# Plot distance to optimum
axes[0].semilogy(distance_to_opt_clara, label='dist. to opt.', color='b')
axes[0].set_ylabel('Distance to opt.')
axes[0].legend()
axes[0].grid(True)
# axes[0].set_title('Vanilla Gradient Descent with CLARA')

# Plot learning rate
axes[1].semilogy(learning_rates_clara, label='learning rate', color='r')
axes[1].set_ylabel('Learning rate')
axes[1].legend()
axes[1].grid(True)
# axes[1].set_title('SGD')

# Plot normalized path
axes[2].semilogy(norm_path_clara, label='norm. path', color='g')
axes[2].set_xlabel('Iterations')
axes[2].set_ylabel('Normalized path')
axes[2].legend()
axes[2].grid(True)

# Adjust layout and show
plt.tight_layout()
plt.show()