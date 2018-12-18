import torch
from torch.nn.utils.rnn import pack_sequence as pack_sequence_
from torch.nn.utils.rnn import pad_packed_sequence as pad_packed_sequence_
from torch.nn.utils.rnn import pack_padded_sequence as pack_padded_sequence_
from torch.nn.utils.rnn import pad_sequence as pad_sequence_
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
    return pack_padded_sequence_(pad_sequence(sequences),
                                 [v.size(0) for v in sequences])

def unpack_sequence(packed_sequence: PackedSequence) -> list:
    return unpad_sequence(*pad_packed_sequence(packed_sequence))


def pad_sequence(sequences: list) -> (torch.Tensor, torch.LongTensor):
    return pad_sequence_(sequences)


def unpad_sequence(padded_sequence: torch.Tensor, lengths: list):
    return [padded_sequence[:l, b, ...] for b, l in enumerate(lengths)]


def pad_packed_sequence(packed_sequence: PackedSequence) -> (torch.Tensor, torch.LongTensor):
    return pad_packed_sequence_(packed_sequence)


def pack_padded_sequence(padded_sequence: torch.Tensor, lengths: torch.LongTensor) -> PackedSequence:
    return pack_padded_sequence_(padded_sequence, lengths)
