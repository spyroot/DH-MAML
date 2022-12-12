"""
Policy network based on a multi-layer perceptron (MLP), with a
`Categorical` distribution output. This policy network can be used on tasks
with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import OrderedDict
from meta_critics.policies.policy import Policy
from torch.nn.utils import weight_norm as wn
from torch.nn.utils import remove_weight_norm as wnr


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        print("in_features out_feature", in_features, out_features)
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight', torch.FloatTensor(out_features, in_features))

        self.register_buffer(
                "weight_epsilon",
                torch.empty(out_features, in_features, device=device, dtype=dtype),
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias', torch.FloatTensor(out_features))

        self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
        )

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """
        :return:
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """

        :return:
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight.copy_(epsilon_out.ger(epsilon_in))
        self.bias.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    @property
    def weight(self) -> torch.Tensor:
        """
        :return:
        """
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """
        :return:
        """
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None


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
        self.num_layers = len(hidden_sizes) * 2 + 1
        layer_sizes = (input_size,) + hidden_sizes * 2 + (output_size,)
        before_last = self.num_layers
        for i in range(1, (self.num_layers + 1), 2):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]).to(device))
            if i != before_last:
                self.add_module('layer{0}'.format(i + 1), NoisyLinear(layer_sizes[i], layer_sizes[i]).to(device))

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

        #         ['layer1.weight', 'layer1.bias', 'layer2.weight_mu', 'layer2.weight_sigma', 'layer2.bias_mu',
        #          'layer2.bias_sigma', 'layer3.weight', 'layer3.bias', 'layer4.weight_mu', 'layer4.weight_sigma',
        #          'layer4.bias_mu', 'layer4.bias_sigma', 'layer5.weight', 'layer5.bias'])

        for i in range(1, self.num_layers):
            if i != 0 and i % 2 == 0:
                if self.training:
                    # mu and sigma
                    weight_mu = W['layer{0}.weight_mu'.format(i)]
                    weight_sigma = W['layer{0}.weight_sigma'.format(i)]
                    bias_mu = W['layer{0}.bias_mu'.format(i)]
                    bias_sigma = W['layer{0}.bias_sigma'.format(i)]

                    # weight = weight_mu + weight_sigma * weight_epsilon
                    # bias = bias_mu + bias_sigma * bias_epsilon

                    weight = weight_mu + weight_sigma
                    bias = bias_mu + bias_sigma

                    weight = weight_mu + weight_sigma.mul(Variable(weight))
                    bias = bias_mu + bias_sigma.mul(Variable(bias))
                else:
                    weight_mu = W['layer{0}.weight_mu'.format(i)]
                    bias_mu = W['layer{0}.bias_mu'.format(i)]
                    weight = weight_mu
                    bias = bias_mu

                output = F.linear(output, weight, bias)
                # output = self.activation(output).to(self.device)
            else:
                _W = W['layer{0}.weight'.format(i)]
                _B = W['layer{0}.bias'.format(i)]

                output = F.linear(output,
                                  weight=W['layer{0}.weight'.format(i)],
                                  bias=W['layer{0}.bias'.format(i)])
                output = self.activation(output).to(self.device)

        logits = F.linear(output,
                          weight=W['layer{0}.weight'.format(self.num_layers)],
                          bias=W['layer{0}.bias'.format(self.num_layers)])

        logits = torch.nn.functional.softmax(logits, dim=1)
        dist = Categorical(logits=logits)
        return dist
