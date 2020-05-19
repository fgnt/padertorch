import torch
import torch.nn.functional
from torch.nn.utils.rnn import PackedSequence
import padertorch as pt


__all__ = [
    'softmax_cross_entropy',
]


IGNORE_INDEX = -1


def softmax_cross_entropy(x, t):
    """Allow inputs to be of type `PackedSequence`.

    In my understanding, all dimensions but the last should be treated as
    independent dimensions. Therefore, I argue for x.size() == (..., K) where
    t.size() == (...). Similarly, for sequences x.size() == (T, B, ..., K) and
    t.size() == (T, B, ...).

    Check the test case for typical usage.

    Params:
        x: `Tensor` or `PackedSequence` holding a multidimensional array whose
            elements indicate unnormalized log probabilities (logits).
        t: Same object type as `x`. Holds integers of ground truth labels.

    Returns:

    >>> x = torch.randn(100, 3)
    >>> t = torch.randint(0, 3, size=(100,), dtype=torch.long)
    >>> softmax_cross_entropy(x, t).size()
    torch.Size([])
    """
    if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
        pass
    elif isinstance(x, PackedSequence) and isinstance(t, PackedSequence):
        # Data is already organized such that no padding is necessary.
        x, t = x.data, t.data
    else:
        raise ValueError(f'Incompatible types: {type(x)}, {type(t)}')

    assert x.size()[:-1] == t.size(), f'{x.size()}, {t.size()}'
    # remember torch.nn.CrossentropyLoss already includes softmax
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(pt.ops.move_axis(x, -1, 1), t)
