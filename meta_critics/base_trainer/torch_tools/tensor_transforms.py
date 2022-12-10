import torch


def flat(tensors):
    """Flatten the given list of tensors into a single vector tensor
    :param tensors: list of tensors
    :return: single tensor with single axis
    """
    return torch.concat([torch.reshape(v, [-1]) for v in tensors if v is not None], dim=0)


def reverse_flat(tensor, shapes):
    """Extract back all tensors that were lost when applying
    :param tensor: flattened tensor
    :param shapes: shapes of all tensors to be extracted
    :return: list of tensors with the given shapes
    """
    return [torch.reshape(t, shape) for t, shape
            in zip(torch.split(tensor, [shape.num_elements() for shape in shapes]), shapes)]
