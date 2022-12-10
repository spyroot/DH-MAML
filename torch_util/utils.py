import math

import scipy
import torch
import numpy as np


def prod(sequence):
    """General prod function, that generalised usage across math and np.
    Created for multiple python versions compatibility).
    """
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))


class DotDict(dict):
    """
    A helper dict that can be accessed using the property notation
    """
    __setattr__ = dict.__setitem__

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            return None


def flat(tensors):
    """Flatten give list of tensors in vector
    :param tensors: list of tensors
    :return: single tensor with single axis
    """
    return torch.concat([torch.reshape(v, [-1]) for v in tensors if v is not None], dim=0)


def reverse_flat(tensor, shapes):
    """ Extract back all tensors that from flat
    :param tensor: flattened tensor
    :param shapes: shapes of all tensors to be extracted
    :return: list of tensors with the given shapes
    """
    return [torch.reshape(t, shape) for t, shape
            in zip(torch.split(tensor, [shape.num_elements() for shape in shapes]), shapes)]


class ReplaceVariableManager:
    """This redirect get_variable calls to use the variables from a pre-specified `replace_dict`.
    This is useful to use tensors instead of variables, allowing for easy second order gradients.
    """

    def __init__(self):
        self.replace_dict = None

    def __call__(self, getter, name, *args, **kwargs):
        if self.replace_dict is not None:
            return self.replace_dict[name]
        return getter(name, *args, **kwargs)


def repeat(x, count):
    """repeat `x` `count` times along a newly inserted axis at the end
    :param x:
    :param count:
    :return:
    """
    tiled = torch.tile(x[..., torch.newaxis], [1] * x.shape.ndims + [count])
    return tiled


def correlation(x, y, sample_axis=0):
    x_mean = torch.reduce_mean(x, sample_axis)
    y_mean = torch.reduce_mean(y, sample_axis)
    return (torch.reduce_sum((x - x_mean) * (y - y_mean), sample_axis)
            / torch.sqrt(torch.reduce_sum(torch.square(x - x_mean), sample_axis)
                         * torch.reduce_sum(torch.square(y - y_mean), sample_axis)))


def discounted_cumsum(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[..., ::-1])[..., ::-1].astype(np.float32)


def tf_discounted_cumsum(t, gamma):
    return torch.py_func(lambda x: discounted_cumsum(x, gamma),
                         [t], torch.float32, stateful=False)


def calculate_gae(rewards, terminals, values, discount_factor, lambda_):
    terminals = torch.cast(terminals, torch.float32)
    not_terminals = 1.0 - terminals
    target = rewards + discount_factor * values[:, 1:] * not_terminals + terminals
    td_residual = target - values[:, :-1]
    advantage = tf_discounted_cumsum(td_residual, discount_factor * lambda_)
    advantage.set_shape(rewards.shape)
    return advantage


def discounted_cumsum_v2(t, gamma):
    """Calculates the cumsum in reverse along axis 1 with discount gamma
    :param t: tensor
    :param gamma: discount factor
    :return: tensor of size t with cumsum applied
    """
    t = torch.reverse(torch.transpose(t, perm=[1, 0, 2]), axis=[0])
    r = torch.scan(lambda acc, e: acc * gamma + e, t)
    return torch.transpose(torch.reverse(r, axis=[0]), perm=[1, 0, 2])


def merge_dicts(base: dict, update: dict):
    new = base.copy()
    new.update(update)
    return new


def get_vars(scope, trainable_only=True):
    if trainable_only:
        return [x for x in torch.trainable_variables() if scope in x.name]
    else:
        return [x for x in torch.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])
