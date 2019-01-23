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
    'pack_padded_sequence'
]


def unpack_sequence(packed_sequence: PackedSequence) -> list:
    return unpad_sequence(*pad_packed_sequence(packed_sequence))


def unpad_sequence(padded_sequence: torch.Tensor, lengths: list):
    return [padded_sequence[:l, b, ...] for b, l in enumerate(lengths)]
