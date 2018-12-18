import unittest
import pytorch_sanity as pts
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.sequence = [torch.zeros(5, 3), torch.ones(4, 3)]

        self.padded = torch.zeros(5, 2, 3)
        self.padded[:4, 1, :] = torch.ones(4, 3)

        self.lengths = torch.LongTensor([5, 4])

        self.packed = PackedSequence(
            torch.stack([self.sequence[0][0],
                         self.sequence[1][0],
                         self.sequence[0][1],
                         self.sequence[1][1],
                         self.sequence[0][2],
                         self.sequence[1][2],
                         self.sequence[0][3],
                         self.sequence[1][3],
                         self.sequence[0][4],
                         ], dim=0),
            torch.LongTensor(4 * [2] + 1 * [1])
        )

    def test_pack_sequence(self):
        actual = pts.ops.pack_sequence(self.sequence)
        assert isinstance(actual, type(self.packed))
        np.testing.assert_equal(
            actual[0].data.numpy(),
            self.packed.data.numpy(),
        )

    def test_unpack_sequence(self):
        actual = pts.ops.unpack_sequence(self.packed)
        assert isinstance(actual, type(self.sequence))
        for actual_, reference_ in zip(actual, self.sequence):
            np.testing.assert_equal(actual_.numpy(), reference_.numpy())

    def test_pad_sequence(self):
        actual = pts.ops.pad_sequence(self.sequence)
        assert isinstance(actual, type(self.padded))
        np.testing.assert_equal(actual.numpy(), self.padded.numpy())

    def test_unpad_sequence(self):
        actual = pts.ops.unpad_sequence(self.padded, self.lengths)
        assert isinstance(actual, type(self.sequence))
        for actual_, reference_ in zip(actual, self.sequence):
            np.testing.assert_equal(actual_.numpy(), reference_.numpy())

    def test_pad_packed_sequence(self):
        actual, lengths = pts.ops.pad_packed_sequence(self.packed)
        assert isinstance(actual, type(self.padded))
        np.testing.assert_equal(lengths.numpy(), self.lengths.numpy())
        np.testing.assert_equal(actual.numpy(), self.padded.numpy())

    def test_pack_padded_sequence(self):
        actual = pts.ops.pack_padded_sequence(self.padded, self.lengths)
        assert isinstance(actual, type(self.packed))
        np.testing.assert_equal(actual.data.numpy(), self.packed.data.numpy())
