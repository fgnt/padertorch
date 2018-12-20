import torch
from padertorch.utils import normalize_axis


def packed_batch_sizes_to_sequence_lengths(batch_sizes: list):
    """

    >>> packed_batch_sizes_to_sequence_lengths([2, 2, 1])
    [3, 2]

    >>> packed_batch_sizes_to_sequence_lengths([5, 3, 3, 2])
    [4, 4, 3, 1, 1]

    Args:
        batch_sizes:

    Returns:

    """
    # TODO: May need to respect batch_first argument.
    # TODO: This is a good candidate for Cython.
    # TODO: Neither we nor them support empty dimensions.
    lengths = []
    last_batch_size = 0
    for length, batch_size in reversed(list(enumerate(batch_sizes, 1))):
        if batch_size > last_batch_size:
            lengths.extend((int(batch_size) - last_batch_size) * [length])
            last_batch_size = int(batch_size)
        else:
            pass
    return lengths


def sequence_reduction(function, x, *args, axis=None, keepdims=False, **kwargs):
    """
    This also remaps some arguments. Enforces keyword arguments.

    Axis argument corresponds to the axis of the padded unpacked sequence.

    If desired, I can make the following examples a test case.

    T, B, F, K = 10, 2, 5, 3
    num_frames = [10, 9]
    packed_x = pack_padded_sequence(torch.randn(T, B, F, K), lengths=num_frames)
    x, _ = pad_packed_sequence(packed_x)

    torch.sum(x, dim=(2, 3), keepdim=False)
    # torch.sum(packed_x, dim=(2, 3), keepdim=False)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(2, 3), keepdims=False)
    sequence_reduction(torch.sum, packed_x, axis=(2, 3), keepdims=False)
    # If `keepdims` is True, type should match. If `keepdims` is False and we reduce along time axis, it should become a `Tensor`.

    torch.sum(x, dim=(2, 3), keepdim=True)
    # torch.sum(packed_x, dim=(2, 3), keepdim=True)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(2, 3), keepdims=True)
    sequence_reduction(torch.sum, packed_x, axis=(2, 3), keepdims=True)

    # Reduction over time
    torch.sum(x, dim=(0, 3), keepdim=False)
    # torch.sum(packed_x, dim=(0, 3), keepdim=False)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(0, 3), keepdims=False)
    sequence_reduction(torch.sum, packed_x, axis=(0, 3), keepdims=False)

    torch.sum(x, dim=(0, 3), keepdim=True)
    # torch.sum(packed_x, dim=(0, 3), keepdim=True)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(0, 3), keepdims=True)
    sequence_reduction(torch.sum, packed_x, axis=(0, 3), keepdims=True)

    # Reduction of time and batch
    torch.sum(x, dim=(0, 1, 3), keepdim=False)
    # torch.sum(packed_x, dim=(0, 1, 3), keepdim=False)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(0, 1, 3), keepdims=False)
    sequence_reduction(torch.sum, packed_x, axis=(0, 1, 3), keepdims=False)

    torch.sum(x, dim=(0, 1, 3), keepdim=True)
    # torch.sum(packed_x, dim=(0, 3), keepdim=True)  # Raises `TypeError` of course.
    sequence_reduction(torch.sum, x, axis=(0, 3), keepdims=True)
    sequence_reduction(torch.sum, packed_x, axis=(0, 1, 3), keepdims=True)


    TODO: May need to check for `batch_first` property of `PackedSequence`,
    TODO: but this is only known during creation time.
    """
    axis = normalize_axis(x, axis)
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        # May need to respect `batch_first` property?
        time_axis = 0  # Required when creating a `PackedSequence`.
        batch_axis = 1
        if time_axis in axis:
            if batch_axis in axis:
                # Adjust `axis` since time and batch axes are collapsed.
                axis = [a - 1 for a in axis if not a == 0]
                if keepdims:
                    return torch.nn.utils.rnn.PackedSequence(
                        function(x.data, *args, dim=axis, keepdim=keepdims),
                        [1]
                    )
                else:
                    function(x.data, *args, dim=axis, keepdim=keepdims)
            else:
                results = []
                lengths = packed_batch_sizes_to_sequence_lengths(x.batch_sizes)
                position = 0

                # Adjust `axis` since time and batch axes are collapsed.
                axis = [a - 1 if not a == 0 else 0 for a in axis]
                for length in lengths:
                    results.append(function(
                        x.data[position:position+length],
                        *args,
                        dim=axis,
                        keepdim=keepdims,
                        **kwargs
                    ))
                    position += length
                if keepdims:
                    return torch.nn.utils.rnn.PackedSequence(
                        torch.cat(results), [1 for b in x.batch_sizes]
                    )
                else:
                    return torch.stack(results)
        else:
            if batch_axis in axis:
                raise NotImplementedError(
                    'It is not well defined how to reduce along batch axis '
                    'when not reducing along time.'
                )
            else:
                # Adjust `axis` since time and batch axes are collapsed.
                axis = [a - 1 for a in axis]
                return torch.nn.utils.rnn.PackedSequence(
                    function(x.data, *args, dim=axis, keepdim=keepdims),
                    x.batch_sizes
                )
    else:
        return function(x, *args, dim=axis, keepdim=keepdims, **kwargs)
