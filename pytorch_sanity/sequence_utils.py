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
