import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from pytorch_sanity.ops.tensor import move_axis


__all__ = [
    'softmax_cross_entropy',
    'deep_clustering_loss',
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
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(move_axis(x, -1, 1), t)


def deep_clustering_loss(x, t):
    """Allows `PackedSequence`.

    The trick to access x.data as in e.g. CE loss does not work, because this
    loss combines knowledge across all time frequency slots.

    Args:
        x: Shape (N, E), where it is assumed that each embedding vector
            is normalized to unit norm.
            Alternatively, packed sequence with data shape (sum_T, F, E).
        t: Target mask with shape (N, K).
            Alternatively, packed sequence with data shape (sum_T, F, K).

    Returns:

    """
    if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
        # This yields losses in the range 10^-2 to 10^0.
        N = x.size()[0]
        return (
            torch.sum(einsum('ne,nE->eE', x, x) ** 2)
            - 2 * torch.sum(einsum('ne,nK->eK', x, t) ** 2)
            + torch.sum(einsum('nk,nK->kK', t, t) ** 2)
        ) // N ** 2
    elif isinstance(x, PackedSequence) and isinstance(t, PackedSequence):
        x, _ = pad_packed_sequence(x)
        t, num_frames = pad_packed_sequence(t)
        return torch.mean(torch.stack([
            deep_clustering_loss(
                x[:num_frames_, b, :, :].view(-1, x.size()[-1]),
                t[:num_frames_, b, :, :].view(-1, t.size()[-1])
            )
            for b, num_frames_ in enumerate(num_frames)
        ]))
    else:
        raise ValueError(f'Incompatible types: {type(x)}, {type(t)}')
