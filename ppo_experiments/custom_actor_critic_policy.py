from typing import List
from stable_baselines3.common.policies import MultiInputActorCriticPolicy  # MultiInputActorCriticPolicy2
import torch
from torch import nn
import gymnasium.spaces as spaces
from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class CustomNetwork(nn.Module):
    def __init__(self, action_space, observation_space, net_arch=None):
        super(CustomNetwork, self).__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        # Get dimension of obs. space after flattening it
        self.n_obs = spaces.flatten_space(self.observation_space).shape[0]
        # Get dimension of action space
        self.n_action = self.action_space.n

        if net_arch is None:
            net_arch = [5]

        if isinstance(net_arch, dict):
            actor_network_layers = net_arch['actor']
            critic_network_layers = net_arch['critic']
        else:
            actor_network_layers = critic_network_layers = net_arch

        actor_net: List[nn.Module] = []
        critic_net: List[nn.Module] = []

        previous_actor_layer_dim = self.n_obs
        for current_actor_layer_dim in actor_network_layers:
            actor_net.append(nn.Linear(previous_actor_layer_dim, current_actor_layer_dim))
            actor_net.append(nn.ReLU()) #TODO: Don't add activation function to last layer
            previous_actor_layer_dim = current_actor_layer_dim

        previous_critic_layer_dim = self.n_obs
        for current_critic_layer_dim in critic_network_layers:
            critic_net.append(nn.Linear(previous_critic_layer_dim, current_critic_layer_dim))
            critic_net.append(nn.ReLU()) #TODO: Don't add activation function to last layer
            previous_critic_layer_dim = current_critic_layer_dim

        # Actor network
        self.actor_net = nn.Sequential(*actor_net)

        # Critic network
        self.critic_net = nn.Sequential(*critic_net)

        self.n_params = sum(p.numel() for p in self.actor_net.parameters()) + sum(p.numel() for p in self.critic_net.parameters())

        self.latent_dim_pi = actor_network_layers[-1]
        self.latent_dim_vf = critic_network_layers[-1]

    def forward(self, features):
        if isinstance(features, dict):
            features = torch.from_numpy(spaces.flatten(self.observation_space, features))
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.actor_net(features)

    def forward_critic(self, features):
        return self.critic_net(features)

    def set_network_params(self, theta):
        # TODO: Need to reimplement for custom network architectures
        raise NotImplementedError()
        assert theta.size == self.n_params, "Error: the dimension of the candidate solution must be " \
                                            "equal to the number of parameters of the policy network."

        theta = torch.from_numpy(theta)

        # Extract theta for each network
        theta_policy = theta[:self.actor_net.parameters()]
        theta_value = theta[self.actor_net.parameters():]

        # Extract weight and bias tensors from input candidate solution theta
        W_actor = torch.reshape(theta_policy[:self.n_hidden_units * self.n_obs], (self.n_hidden_units, -1))
        b_actor = theta_policy[self.n_hidden_units * self.n_obs:self.n_hidden_units * self.n_obs + self.n_hidden_units]
        W_critic = torch.reshape(theta_value[:self.n_hidden_units * self.n_obs], (self.n_hidden_units, -1))
        b_critic = theta_value[self.n_hidden_units * self.n_obs:self.n_hidden_units * self.n_obs + self.n_hidden_units]

        # Initialize policy network parameters with extracted values
        with torch.no_grad():
            self.state_dict()["policy_net.0.weight"].copy_(W_actor)
            self.state_dict()["policy_net.0.bias"].copy_(b_actor)
            self.state_dict()["value_net.0.weight"].copy_(W_critic)
            self.state_dict()["value_net.0.bias"].copy_(b_critic)


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space,
                                                      action_space,
                                                      lr_schedule,
                                                      net_arch,
                                                      activation_fn,
                                                      *args,
                                                      **kwargs)
        self.ortho_init = False

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.action_space, self.observation_space, self.net_arch)
