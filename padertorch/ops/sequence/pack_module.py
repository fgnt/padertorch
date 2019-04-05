"""
This module allows to switch between three types of tensors:
packed: PackedSequence
padded
list of tensor

# ToDo add contiguous to pack_padded_sequence if needed
# TODO: Improve speed of pack_sequence by avoiding padding inbetween
"""
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence

__all__ = [
    'pack_sequence',
    'unpack_sequence',
    'pad_sequence',
    'unpad_sequence',
    'pad_packed_sequence',
    'pack_padded_sequence',
    'pack_sequence_include_channel',
    'unpack_sequence_include_channel_like',
]


def unpack_sequence(packed_sequence: PackedSequence) -> list:
    return unpad_sequence(*pad_packed_sequence(packed_sequence))


def unpad_sequence(padded_sequence: torch.Tensor, lengths: list):
    return [padded_sequence[:l, b, ...] for b, l in enumerate(lengths)]


def pack_sequence_include_channel(list_of_tensors):
    """
    Similar to pack_sequence, but expect that the input has the following
    shape:
        batch channel batch_dependent_sequence_length ...
    while pack_sequence assumes:
        batch batch_dependent_sequence_length ...

    Example:
        2 channels
        3 to 4 frames
        5 features

    >>> list_of_tensors = [torch.zeros([2, 4, 5]), torch.ones([2, 3, 5 ])]
    >>> packed = pack_sequence_include_channel(list_of_tensors)
    >>> packed
    PackedSequence(data=tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]]), batch_sizes=tensor([4, 4, 4, 2]), sorted_indices=None, unsorted_indices=None)
    >>> packed.data.shape
    torch.Size([14, 5])
    >>> zero, one = unpack_sequence_include_channel_like(packed, like=list_of_tensors)
    >>> zero.shape, one.shape
    (torch.Size([2, 4, 5]), torch.Size([2, 3, 5]))
    >>> zero
    tensor([[[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]],
    <BLANKLINE>
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]])
    >>> one
    tensor([[[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]],
    <BLANKLINE>
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]]])

    >>> import numpy as np
    >>> import padertorch as pt
    >>> def test(list_of_shapes):
    ...     list_of_tensors = [torch.rand(shape) for shape in list_of_shapes]
    ...     packed = pack_sequence_include_channel(list_of_tensors)
    ...     list_of_tensors_hat = unpack_sequence_include_channel_like(packed, like=list_of_tensors)
    ...     assert len(list_of_tensors) == len(list_of_tensors_hat)
    ...     for t, t_hat in zip(list_of_tensors, list_of_tensors_hat):
    ...         np.testing.assert_equal(pt.utils.to_numpy(t), pt.utils.to_numpy(t_hat))
    >>> test([[2, 4, 5], [2, 3, 5]])
    >>> test([[3, 10, 5], [2, 3, 5]])
    >>> test([[3, 10, 5], [2, 3, 5], [2, 3, 5], [2, 2, 5]])

    """
    assert isinstance(list_of_tensors, (tuple, list))

    sequences = [
        sequence
        for entry in list_of_tensors
        # Assume splitting of the tensor entry is cheap enough and later
        # the python loop efficient enough with the memcopy.
        for sequence in entry
    ]
    return pack_sequence(sequences)


def unpack_sequence_include_channel_like(packed, like):
    """

    Unpacks a sequence that was packed with pack_channel_sequence.
    The channel size is obtained from the "like" tensor.

    """
    assert isinstance(like, (tuple, list))

    padded_sequence, lengths = pad_packed_sequence(packed)
    lengths = lengths.tolist()

    new = []
    index = 0

    for entry in like:
        channels = entry.shape[0]
        ls, lengths = lengths[:channels], lengths[channels:]

        assert len(set(ls)) == 1, ls
        length = ls[0]

        new.append(
            padded_sequence[:length, index:index+channels].transpose(0, 1)
        )
        index = index + channels

    return new
