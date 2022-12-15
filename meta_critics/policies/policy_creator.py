from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Optional, List

import gym
import torch

from meta_critics.base_trainer.torch_tools.tensor_tools import numpy_to_torch_dtype_dict, string_to_torch_remaping
from meta_critics.policies.categorical_mlp import CategoricalRLPPolicy
from meta_critics.policies.normal_mlp import NormalMLPPolicy
from meta_critics.running_spec import RunningSpec


class PolicyCreator:
    def __init__(self,
                 env: gym.Env,
                 spec: Optional[RunningSpec],
                 hidden_sizes: Optional[List] = None,
                 activation: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """

        :param env:
        :param spec:
        """
        self.env = env
        self.spec: RunningSpec

        if device is not None:
            self.device = device
        else:
            self.device = spec.get('device')

        if hidden_sizes is not None:
            self.hidden_sizes = hidden_sizes
        else:
            self.hidden_sizes = spec.get('hidden-sizes', 'policy_network')

        if activation is not None:
            self.activation = activation
        else:
            self.activation = spec.get('activation', 'policy_network')

        self.activation = getattr(torch, self.activation)
        self.is_continuous_actions = isinstance(env.action_space, gym.spaces.Box)
        self.input_size = self.get_input_size()

    def __call__(self):
        """ Intercept call to env and resets seed.
        :return:
        """
        return self.make_policy()

    def make_policy(self) -> tuple[NormalMLPPolicy | CategoricalRLPPolicy, bool]:
        """
        :return:
        """
        obs_dtype = string_to_torch_remaping[str(self.env.observation_space.dtype)]

        if self.is_continuous_actions:
            output_size = reduce(mul, self.env.action_space.shape, 1)
            policy = NormalMLPPolicy(self.input_size, output_size,
                                     hidden_sizes=tuple(self.hidden_sizes),
                                     activation=self.activation,
                                     device=self.device,
                                     observations_dtype=obs_dtype).to(self.device)
        else:
            output_size = self.env.action_space.n
            policy = CategoricalRLPPolicy(self.input_size, output_size,
                                          hidden_sizes=tuple(self.hidden_sizes),
                                          activation=self.activation,
                                          device=self.device, observations_dtype=obs_dtype).to(self.device)
        assert policy is not None
        return policy, self.is_continuous_actions

    def get_input_size(self):
        return reduce(mul, self.env.observation_space.shape, 1)
