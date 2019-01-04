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
    if not isinstance(x, list):
        x = [x] * (1 if length is None else length)
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
