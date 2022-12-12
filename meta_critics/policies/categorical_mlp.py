"""
Policy network based on a multi-layer perceptron (MLP), with a
`Categorical` distribution output. This policy network can be used on tasks
with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import OrderedDict
from meta_critics.policies.policy import Policy

from torch.nn.utils import weight_norm as wn
from torch.nn.utils import remove_weight_norm as wnr

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class CategoricalRLPPolicy(Policy, nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(),
                 activation=F.relu,
                 device: torch.device = 'cpu'):
        super(CategoricalRLPPolicy, self).__init__(input_size=input_size,
                                                   output_size=output_size)

        self.device = device
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.num_layers = len(hidden_sizes) + 1
        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]).to(device))

        self.apply(weight_init)
        self.register_param_names()

    def forward(self, x: torch.Tensor, W=None):
        """

        :param x:
        :param W:
        :return:
        """
        if W is None:
            W = OrderedDict(self.named_parameters())

        output = x.to(self.device)

        for i in range(1, self.num_layers):
            _W = W['layer{0}.weight'.format(i)]
            _B = W['layer{0}.bias'.format(i)]

            output = F.linear(output, weight=W['layer{0}.weight'.format(i)], bias=W['layer{0}.bias'.format(i)])
            output = self.activation(output).to(self.device)

        logits = F.linear(output,
                          weight=W['layer{0}.weight'.format(self.num_layers)],
                          bias=W['layer{0}.bias'.format(self.num_layers)])
        # wn(logits)
        return Categorical(logits=logits)
