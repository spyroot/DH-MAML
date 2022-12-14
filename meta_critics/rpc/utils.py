import numpy as np
import torch

def format_num(n):
    """

    :param n:
    :return:
    """
    if isinstance(n, np.ndarray):
        return n
    elif isinstance(n, torch.Tensor):
        return n

    f = '{0:.3g}'.format(n)
    f = f.replace('+0', '+')
    f = f.replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n