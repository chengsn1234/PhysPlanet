"""Small vector-observation networks for RoverLab skrl agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin
from skrl.models.torch.base import Model


def _activation(name: str) -> nn.Module:
    activations = {
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
    }
    return activations.get(name, nn.LeakyReLU(inplace=True))


def _mlp(input_dim: int, output_dim: int, layers, activation: str, output_activation: nn.Module | None = None):
    modules = []
    for width in layers:
        modules.extend((nn.Linear(input_dim, width), _activation(activation)))
        input_dim = width
    modules.append(nn.Linear(input_dim, output_dim))
    if output_activation is not None:
        modules.append(output_activation)
    return nn.Sequential(*modules)


class GaussianPolicy(GaussianMixin, Model):
    """Gaussian MLP policy for vector observations."""

    def __init__(
        self, observation_space, action_space, device, mlp_layers, mlp_activation,
        initial_log_std, min_log_std, max_log_std, **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True,
            min_log_std=min_log_std, max_log_std=max_log_std, reduction="sum",
        )
        output_dim = action_space.shape[0]
        self.mlp = _mlp(
            observation_space.shape[0], output_dim, mlp_layers, mlp_activation, nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.full((output_dim,), initial_log_std))

    def compute(self, inputs, role=""):
        return self.mlp(inputs["observations"]), {"log_std": self.log_std_parameter}


class DeterministicPolicy(DeterministicMixin, Model):
    """Deterministic MLP policy for off-policy agents."""

    def __init__(self, observation_space, action_space, device, mlp_layers, mlp_activation, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=True)
        self.mlp = _mlp(
            observation_space.shape[0], action_space.shape[0], mlp_layers, mlp_activation, nn.Tanh()
        )

    def compute(self, inputs, role=""):
        return self.mlp(inputs["observations"]), {}


class ValueNetwork(DeterministicMixin, Model):
    """Value MLP for vector observations."""

    def __init__(self, observation_space, action_space, device, mlp_layers, mlp_activation, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.mlp = _mlp(observation_space.shape[0], 1, mlp_layers, mlp_activation)

    def compute(self, inputs, role=""):
        return self.mlp(inputs["observations"]), {}


class QNetwork(DeterministicMixin, Model):
    """State-action value network shared by TD3 and SAC."""

    def __init__(self, observation_space, action_space, device, mlp_layers, mlp_activation, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.mlp = _mlp(
            observation_space.shape[0] + action_space.shape[0], 1, mlp_layers, mlp_activation
        )

    def compute(self, inputs, role=""):
        state_action = torch.cat((inputs["observations"], inputs["taken_actions"]), dim=-1)
        return self.mlp(state_action), {}
