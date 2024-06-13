import numpy as np
import torch

__all__ = {
}


def is_torch(obj):
    """
    The namespace here is not torch, hece renamce is_tensor to is_torch.

    >>> is_torch(np.zeros(3))
    False
    >>> is_torch(torch.zeros(3))
    True
    >>> is_torch(ComplexTensor(np.zeros(3)))
    True
    """
    if torch.is_tensor(obj):
        return True
    if type(obj).__name__ == 'ComplexTensor':
        from torch_complex import ComplexTensor
        if isinstance(obj, ComplexTensor):
            return True
    return False

