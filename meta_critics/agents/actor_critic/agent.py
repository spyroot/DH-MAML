from collections import OrderedDict
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class Agent(nn.Module):
    def __init__(self, envs, out_features=64):

        super(Agent, self).__init__()

        self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), out_features)),
                nn.Tanh(),
                layer_init(nn.Linear(out_features, out_features)),
                nn.Tanh(),
                layer_init(nn.Linear(out_features, 1), std=1.0),
        )
        self.actor = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), out_features)),
                nn.Tanh(),
                layer_init(nn.Linear(out_features, out_features)),
                nn.Tanh(),
                layer_init(nn.Linear(out_features, envs.single_action_space.n), std=0.01),
        )

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critics_parameters(self):
        return self.critic.parameters()

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def update_critics_params(self, loss, step_size=0.5, first_order=False):
        """
        Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        grads = torch.autograd.grad(loss, list(self.critic.parameters()), create_graph=True)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params

    def get_action_and_value(self, x: torch.Tensor,
                             action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor,
                                                                             torch.Tensor]:
        """
        :param x:
        :param action:
        :return:
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
