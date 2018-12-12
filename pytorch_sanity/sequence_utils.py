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
