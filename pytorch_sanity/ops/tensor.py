import torch

__all__ = [
    'move_axis'
]


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