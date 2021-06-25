import numpy as np
import torch
import torch_complex
from torch_complex import ComplexTensor

__all__ = {
    'ComplexTensor',
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
    if torch.is_tensor(obj) or isinstance(obj, ComplexTensor):
        return True
    else:
        return False

