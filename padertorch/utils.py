import collections

import numpy as np
import torch


def normalize_axis(x, axis):
    """Here, `axis` is always understood to reference the unpacked axes.

    TODO: Do we need to assert, that the axes do not collide?

    >>> normalize_axis(torch.zeros(2, 3, 4), -1)
    (2,)

    >>> normalize_axis(torch.zeros(2, 3, 4), (-3, -2, -1))
    (0, 1, 2)
    """
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        ndim = len(x.data.size()) + 1
    else:
        ndim = len(x.size())
    return tuple(a % ndim for a in axis)


def to_list(x, length=None):
    """
    >>> to_list(1)
    [1]
    >>> to_list([1])
    [1]
    >>> to_list((i for i in range(3)))
    [0, 1, 2]
    >>> to_list(np.arange(3))
    [0, 1, 2]
    >>> to_list({'a': 1})
    [{'a': 1}]
    >>> to_list({'a': 1}.keys())
    ['a']
    """
    # Important cases (change type):
    #  - generator -> list
    #  - dict_keys -> list
    #  - dict_values -> list
    #  - np.array -> list (discussable)
    # Important cases (list of original object):
    #  - dict -> list of dict
    if not isinstance(x, collections.Iterable) \
            or isinstance(x, collections.Mapping):
        x = [x] * (1 if length is None else length)
    elif not isinstance(x, collections.Sequence):
        x = list(x)
    if length is not None:
        assert len(x) == length
    return x


def to_numpy(array):
    """
    >>> t = torch.zeros(2)
    >>> t
    tensor([0., 0.])
    >>> as_numpy(t), np.zeros(2, dtype=np.float32)
    (array([0., 0.], dtype=float32), array([0., 0.], dtype=float32))
    """
    if isinstance(array, torch.Tensor):
        array = array.cpu()

    # torch only supports np.asarray for cpu tensors
    return np.asarray(array)
