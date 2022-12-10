import torch
import numpy as np


def weighted_mean(tensor, lengths=None):
    """ This is same weight mean used in MAML
    :param tensor:
    :param lengths:
    :return:
    """
    if lengths is None:
        return torch.mean(tensor)

    if tensor.dim() < 2:
        raise ValueError('Expected tensor with at least 2 dimensions '
                         '(trajectory_length x batch_size), got {0}D '
                         'tensor.'.format(tensor.dim()))

    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.)

    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)
    out = torch.sum(tensor, dim=0)
    out.div_(lengths.view(-1, *extra_dims))
    return out


def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    """ This is same weight mean used in MAML.
    :param tensor:
    :param lengths:
    :param epsilon:
    :return:
    """
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()
    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.)

    std = torch.sqrt(weighted_mean(out ** 2, lengths=lengths).mean())
    out.div_(std + epsilon)
    return out


def to_numpy(tensor):
    """
    :param tensor:
    :return:
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0)
    else:
        raise NotImplementedError()
