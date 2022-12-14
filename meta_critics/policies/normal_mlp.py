"""Policy network based on a multi-layer perceptron (MLP), with a
`Normal` distribution output, with trainable standard deviation. This
policy network can be used on tasks with continuous action spaces (eg.
`HalfCheetahDir`).

The code is adapted from
https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
"""

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.nn import init
from meta_critics.policies.policy import Policy


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class NormalMLPPolicy(Policy, nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(),
                 activation=F.relu,
                 init_std: Optional[float] = 1.0,
                 min_std: Optional[float] = 1e-6,
                 device: torch.device = 'cpu',
                 observations_dtype: Optional[torch.dtype] = torch.float32):
        super(NormalMLPPolicy, self).__init__(input_size=input_size,
                                              output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.device = device
        self.observations_dtype = observations_dtype
        torch.set_default_dtype(self.observations_dtype)

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]).to(device))

        self.mu = nn.Linear(layer_sizes[-1], output_size).to(self.device)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.register_parameter("sigma", self.sigma)
        self.apply(weight_init)
        self.register_param_names()

    @staticmethod
    def weight_init_xavier(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    @staticmethod
    def weight_init_kaiming(module):
        """
        :return:
        """
        if hasattr(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            module.bias.data.zero_()

    def forward(self, x, W=None):
        """
        :param x:
        :param W:
        :return:
        """
        w_inited = True
        if W is None:
            W = OrderedDict(self.named_parameters())

        output = x.to(self.device)
        for i in range(1, self.num_layers):
            linear_out = F.linear(output,
                                  weight=W['layer{0}.weight'.format(i)],
                                  bias=W['layer{0}.bias'.format(i)])
            output = self.activation(linear_out)

        mu = F.linear(output, weight=W['mu.weight'], bias=W['mu.bias'])
        scale = torch.exp(torch.clamp(W['sigma'], min=self.min_log_std))

        try:
            Independent(Normal(loc=mu, scale=scale), 1)
        except Exception as er:
            print(f"Scale {scale}")
            print(f"mu {mu}")
            print(f"output {output}")
            print(f"x {x}")
            print(f"linear_out {linear_out}")
            print(f"W", w_inited)
            print(f"W", W)
            print(er)
            raise er

        return Independent(Normal(loc=mu, scale=scale), 1)
