from typing import Iterable
import torch
from torch.nn.utils.convert_parameters import _check_param_device


def vec2parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    """ vec2 param
    :param vec:
    :param parameters:
    :return:
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
    p_dev = None
    ptr = 0
    for p in parameters:
        p_dev = _check_param_device(p, p_dev)
        p_idx = p.numel()
        p.data.copy_(vec[ptr:ptr + p_idx].view_as(p).data)
        ptr += p_idx
