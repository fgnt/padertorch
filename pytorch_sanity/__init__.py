import string
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


IGNORE_INDEX = -1


def move_axis(a: torch.Tensor, source: int, destination: int):
    """Move an axis from source location to destination location.

    API is a bit closer to Numpy but does not allow more than one source.

    Params:
        a: The Tensor whose axes should be reordered.
        source: Original positions of the axis to move.
        destination: Destination positions for each of the original axis.
    Returns: Tensor with moved axis.

    >>> x = zeros((3, 4, 5))
    >>> move_axis(x, 0, -1).size()
    torch.Size([4, 5, 3])

    >>> move_axis(x, -1, 0).size()
    torch.Size([5, 3, 4])
    """
    source = source % len(a.size())
    destination = destination % len(a.size())
    permutation = [d for d in range(len(a.size())) if not d == source]
    permutation.insert(destination, source)
    return a.permute(permutation)


def zeros(shape, dtype=None):
    return torch.zeros(*shape, dtype=dtype)


def einsum(operation: str, *operands):
    """Allows capital letters and collects operands as in `np.einsum`."""
    remaining_letters = set(string.ascii_lowercase)
    remaining_letters = remaining_letters - set(operation)
    for capital_letter, replacement in zip(set.intersection(
            set(string.ascii_uppercase),
            set(operation)
    ), remaining_letters):
        operation = operation.replace(capital_letter, replacement)
    return torch.einsum(operation, operands)


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
