#
# typing if output must convert to torch tensor for example.
#
# Mus
from enum import auto, Enum, unique


@unique
class EnvType(Enum):
    """
    Data type enum,  if we need use only torch or numpy to avoid
    changing data types.
    TODO evaluate option to compute everything on GPU.
    """
    Int = auto()
    Tensor = auto()
    NdArray = auto()
