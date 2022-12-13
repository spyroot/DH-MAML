# Trajectory sampler support dtype remapper.
# i.e  for example we can remap observation dtype GYM output to something else.
import numpy as np
import torch

# this remaps np.float16/32/64 to float 32
numpy_to_torch_remaping = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float32,
    np.float32: torch.float32,
    np.float64: torch.float32,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

string_to_torch_remaping = {
    "torch.bool": torch.bool,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float32": torch.float32,
    "torch.float64": torch.float32,
}


numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}

ALL_TENSORTYPES = [torch.float,
                   torch.double,
                   torch.half]


def to_numpy(tensor: torch.Tensor):
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
