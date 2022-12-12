"""Noisy Linear Layer.

Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
with gradient descent along with any other remaining network weights. Factorized Gaussian
noise is the type of noise usually employed.
"""

import math
from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter


class NoisyLinear(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 std_init: float = 0.1):
        """

        :param in_features:  input features dimension
        :param out_features: out features dimension
        :param bias:  if True, a bias term will be added to the matrix multiplication
        :param device: torch.device
        :param dtype:  dtype of the parameters.
        :param std_init: initial value of the Gaussian standard deviation before optimization
        """
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
                torch.empty(
                        out_features,
                        in_features,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                )
        )

        self.weight_sigma = nn.Parameter(
                torch.empty(
                        out_features,
                        in_features,
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                )
        )

        self.register_buffer(
                "weight_epsilon",
                torch.empty(out_features, in_features, device=device, dtype=dtype),
        )

        if bias:
            self.bias_mu = nn.Parameter(
                    torch.empty(
                            out_features,
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                    )
            )
            self.bias_sigma = nn.Parameter(
                    torch.empty(
                            out_features,
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                    )
            )
            self.register_buffer(
                    "bias_epsilon",
                    torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """

        :return:
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """

        :return:
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        """

        :param size:
        :return:
        """
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

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


class NoisyLazyLinear(LazyModuleMixin, NoisyLinear):
    def __init__(self, out_features: int, bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None, std_init: float = 0.1):
        super().__init__(0, 0, False, device=device)
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = UninitializedParameter(device=device, dtype=dtype)
        self.weight_sigma = UninitializedParameter(device=device, dtype=dtype)
        self.register_buffer("weight_epsilon", UninitializedBuffer(device=device, dtype=dtype))
        if bias:
            self.bias_mu = UninitializedParameter(device=device, dtype=dtype)
            self.bias_sigma = UninitializedParameter(device=device, dtype=dtype)
            self.register_buffer("bias_epsilon", UninitializedBuffer(device=device, dtype=dtype))
        else:
            self.bias_mu = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset if no parameters initialize
        :return:
        """
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def reset_noise(self) -> None:
        """

        :return:
        """
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_noise()

    def initialize_parameters(self, input: torch.Tensor) -> None:
        """
        :param input:
        :return:
        """
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight_mu.materialize((self.out_features, self.in_features))
                self.weight_sigma.materialize((self.out_features, self.in_features))
                self.weight_epsilon.materialize((self.out_features, self.in_features))
                if self.bias_mu is not None:
                    self.bias_mu.materialize((self.out_features,))
                    self.bias_sigma.materialize((self.out_features,))
                    self.bias_epsilon.materialize((self.out_features,))
                self.reset_parameters()
                self.reset_noise()

    @property
    def weight(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().weight

    @property
    def bias(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().bias


def reset_noise(layer: nn.Module) -> None:
    """Resets the noise of noisy layers."""
    if hasattr(layer, "reset_noise"):
        layer.reset_noise()
