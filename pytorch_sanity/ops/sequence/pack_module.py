import torch
from torch.nn.utils.rnn import pack_sequence as pack_sequence_
from torch.nn.utils.rnn import pad_packed_sequence as pad_packed_sequence_
from torch.nn.utils.rnn import PackedSequence


__all__ = [
    'pack_sequence',
    'unpack_sequence',
    'pad_sequence',
    'unpad_sequence',
    'pad_packed_sequence',
    'pack_padded_sequence'
]


def pack_sequence(sequences: list) -> PackedSequence:
    pass


def unpack_sequence(packed_sequence: PackedSequence) -> list:
    pass


def pad_sequence(sequences: list) -> (torch.Tensor, list):
    pass


def unpad_sequence(padded_sequence: torch.Tensor, lengths: list):
    pass


def pad_packed_sequence(packed_sequence: PackedSequence) -> (torch.Tensor, list):
    pass


def pack_padded_sequence(padded_sequence: torch.Tensor, lengths: list) -> PackedSequence:
    pass
