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
    Often a list is required, but for convenience it is desired to enter a
    single object, e.g. string.

    Complicated corner cases are e.g. `range()` and `dict.values()`, which are
    handled here.

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
    >>> to_list('ab')
    ['ab']
    >>> from pathlib import Path
    >>> to_list(Path('/foo/bar'))
    [PosixPath('/foo/bar')]
    """
    # Important cases (change type):
    #  - generator -> list
    #  - dict_keys -> list
    #  - dict_values -> list
    #  - np.array -> list (discussable)
    # Important cases (list of original object):
    #  - dict -> list of dict

    def to_list_helper(x_):
        return [x_] * (1 if length is None else length)

    if isinstance(x, collections.Mapping):
        x = to_list_helper(x)
    elif isinstance(x, str):
        x = to_list_helper(x)
    elif isinstance(x, collections.Sequence):
        pass
    elif isinstance(x, collections.Iterable):
        x = list(x)
    else:
        x = to_list_helper(x)

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
