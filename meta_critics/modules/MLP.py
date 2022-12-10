from numbers import Number
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn

from meta_critics.modules.n import NoisyLinear, NoisyLazyLinear
from meta_critics.policies.dist_utils import create_on_device
from torch_util.utils import prod

LazyMapping = {
    nn.Linear: nn.LazyLinear,
    NoisyLinear: NoisyLazyLinear,
}


class MLP(nn.Sequential):

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Union[int, Sequence[int]] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Type[nn.Module] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
        norm_class: Optional[Type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: Type[nn.Module] = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activate_last_layer: bool = False,
        device: Optional[torch.device] = None,
    ):
        if out_features is None:
            raise ValueError("out_features must be specified for MLP.")

        default_num_cells = 32
        if num_cells is None:
            if depth is None:
                num_cells = [default_num_cells] * 3
                depth = 3
            else:
                num_cells = [default_num_cells] * depth

        self.in_features = in_features

        _out_features_num = out_features
        if not isinstance(out_features, Number):
            _out_features_num = prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.activation_kwargs = (
activation_kwargs if activation_kwargs is not None else {})

        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}
        self.activate_last_layer = activate_last_layer
        if single_bias_last_layer:
            raise NotImplementedError

        if not (isinstance(num_cells, Sequence) or depth is not None):
            raise RuntimeError(" num_cells also require depth must be provided too.")

        self.num_cells = (list(num_cells) if isinstance(num_cells, Sequence) else [num_cells] * depth)

        self.depth = depth if depth is not None else len(self.num_cells)
        if not (len(self.num_cells) == depth or depth is None):
            raise RuntimeError("depth and num_cells length conflict, num_cell != despth")

        layers = self._make_net(device)
        super().__init__(*layers)

    def _make_net(self, device: Optional[torch.device]) -> List[nn.Module]:
        """
        :param device:
        :return:
        """
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]

        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            _bias = self.bias_last_layer if i == self.depth else True
            if _in is not None:
                layers.append(
                        create_on_device(
                                self.layer_class,
                                device,
                                _in,
                                _out,
                                bias=_bias,
                                **self.layer_kwargs,
                        )
                )
            else:
                try:
                    lazy_version = LazyMapping[self.layer_class]
                except KeyError:
                    raise KeyError(
                            f"The lazy version of {self.layer_class.__name__} is not implemented yet. "
                            "Consider providing the input feature dimensions explicitly when creating an MLP module"
                    )
                layers.append(
                        create_on_device(
                                lazy_version, device, _out, bias=_bias, **self.layer_kwargs
                        )
                )

            if i < self.depth or self.activate_last_layer:
                layers.append(create_on_device(self.activation_class, device, **self.activation_kwargs))
                if self.norm_class is not None:
                    layers.append(create_on_device(self.norm_class, device, **self.norm_kwargs))
        return layers

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        """

        :param inputs:
        :return:
        """
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)

        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)

        return
