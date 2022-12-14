# """
# Policy network based on a multi-layer perceptron (MLP), with a
# `Categorical` distribution output. This policy network can be used on tasks
# with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
# """
# import math
# from typing import Optional
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import OrderedDict
import torch.nn.functional as F

from torch.nn import init

from meta_critics.policies.noizy_categorical import weight_init
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
                 device: torch.device = 'cpu',
                 nm_size=5, nm_gate='hard',
                 observations_dtype: Optional[torch.dtype] = torch.float32):
        super(CategoricalRLPPolicy, self).__init__(input_size=input_size,
                                                   output_size=output_size)

        self.device = device
        self.hidden_sizes = hidden_sizes
        # self.activation = F.relu
        self.activation = torch.nn.LeakyReLU()
        torch.set_default_dtype(self.obs_dtype)

        self.activation.requires_grad = True
        self.nm_gate = nm_gate
        self.nm_size = nm_size

        self.num_layers = len(hidden_sizes) + 1
        layer_sizes = (input_size,) + hidden_sizes + (output_size,)

        for i in range(1, (self.num_layers + 1)):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]).to(device))

        self.apply(self.weight_init_xavier)
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
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            module.bias.data.zero_()

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
            output = F.linear(output,
                              weight=W['layer{0}.weight'.format(i)],
                              bias=W['layer{0}.bias'.format(i)])

            output = self.activation(output).to(self.device)

        logits = F.linear(output,
                          weight=W['layer{0}.weight'.format(self.num_layers)],
                          bias=W['layer{0}.bias'.format(self.num_layers)])

        # logits = F.softmax(logits, dim=0)

        dist = None
        try:
            dist = Categorical(logits=logits)
        except Exception as err:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    print(name, torch.isfinite(param.grad).all())
            raise err
        return dist
#
# class NMLinear(nn.Module):
#     def __init__(self, in_features, out_features, nm_features, bias=True, nm_gate='soft'):
#         super(NMLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.nm_features = nm_features
#         self.in_nm_act = F.relu
#         self.out_nm_act = torch.tanh
#         assert nm_gate in ['hard', 'soft'], '`gating` should be \'hard\' or \'soft\''
#         self.gating = nm_gate
#
#         self.std = nn.Linear(in_features, out_features, bias=bias)
#         self.in_nm = nn.Linear(in_features, nm_features, bias=bias)
#         self.out_nm = nn.Linear(nm_features, out_features, bias=bias)
#
#     def forward(self, data, params=None):
#         output = self.std(data)
#         mod_features = self.in_nm_act(self.in_nm(data))
#         sign_ = self.out_nm_act(self.out_nm(mod_features))
#         if self.gating == 'hard':
#             sign_ = torch.sign(sign_)
#             sign_[sign_ == 0.] = 1.  # a zero value should have sign of 1. and not 0.
#         output *= sign_
#         return output
#
#
# class CategoricalRLPPolicy(Policy, nn.Module):
#
#     def __init__(self, input_size, output_size,
#                  hidden_sizes=(), activation=F.relu, device: torch.device = 'cpu', nm_size=5, nm_gate='hard'):
#         """
#
#         :param input_size:
#         :param output_size:
#         :param hidden_sizes:
#         :param activation:
#         :param device:
#         :param nm_size:
#         :param nm_gate:
#         """
#         super(CategoricalRLPPolicy, self).__init__(
#                 input_size=input_size, output_size=output_size)
#         self.device = device
#         self.hidden_sizes = hidden_sizes
#         self.nonlinearity = torch.tanh
#         self.num_layers = len(hidden_sizes) + 1
#         self.nm_size = nm_size
#         self.nm_gate = nm_gate
#
#         layer_sizes = (input_size,) + hidden_sizes + (output_size,)
#         for i in range(1, self.num_layers):
#             self.add_module('layer{0}'.format(i),
#                             NMLinear(layer_sizes[i - 1], layer_sizes[i], nm_size, nm_gate))
#
#         self.add_module('layer{0}'.format(self.num_layers),
#                         nn.Linear(layer_sizes[self.num_layers - 1], layer_sizes[self.num_layers]))
#
#         self.apply(weight_init)
#         self.register_param_names()
#
#     def forward(self, x, W=None):
#         """
#         :param W:
#         :return:
#         """
#         if W is None:
#             W = OrderedDict(self.named_parameters())
#
#         output = x
#
#         for i in range(1, self.num_layers):
#             output_std = F.linear(output,
#                                   weight=W['layer{0}.std.weight'.format(i)],
#                                   bias=W['layer{0}.std.bias'.format(i)])
#             mod_features = self.nonlinearity(F.linear(output,
#                                                       weight=W['layer{0}.in_nm.weight'.format(i)],
#                                                       bias=W['layer{0}.in_nm.bias'.format(i)]))
#             sign_ = torch.tanh(F.linear(mod_features,
#                                         weight=W['layer{0}.out_nm.weight'.format(i)],
#                                         bias=W['layer{0}.out_nm.bias'.format(i)]))
#             if self.nm_gate == 'hard':
#                 sign_ = torch.sign(sign_)
#                 sign_[sign_ == 0.] = 1.  # a zero value should have sign of 1. and not 0.
#             output = self.nonlinearity(output_std * sign_)
#
#         logits = F.linear(output, weight=W['layer{0}.weight'.format(self.num_layers)],  bias=W['layer{0}.bias'.format(self.num_layers)])
#         return Categorical(logits=logits)
