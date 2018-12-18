import unittest
import pytorch_sanity as pts
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.sequence = [torch.zeros(100, 10), torch.ones(90, 10)]

        self.padded = torch.zeros(100, 2, 10)
        self.padded[:90, 1, :] = torch.ones(90, 10)

        self.lengths = torch.LongTensor([100, 90])

        self.packed = PackedSequence(
            torch.cat(self.sequence),
            90 * [2] + 10 * [1]
        )

    def test_pack_sequence(self):
        actual = pts.ops.pack_sequence(self.sequence)
        assert isinstance(actual, type(self.packed))
        np.testing.assert_equal(actual.data, self.packed.data)

    def test_unpack_sequence(self):
        actual = pts.ops.unpack_sequence(self.packed)
        assert isinstance(actual, type(self.sequence))
        for actual_, reference in zip(actual, self.sequence):
            np.testing.assert_equal(actual, reference)

    def test_pad_sequence(self):
        actual = pts.ops.pad_sequence(self.sequence)
        assert isinstance(actual, type(self.padded))
        np.testing.assert_equal(actual, self.packed)

    def test_unpad_sequence(self):
        actual = pts.ops.unpad_sequence(self.padded, self.lengths)
        assert isinstance(actual, type(self.padded))
        for actual_, reference in zip(actual, self.sequence):
            np.testing.assert_equal(actual, reference)

    def test_pad_packed_sequence(self):
        actual = pts.ops.pad_packed_sequence(self.packed)
        assert isinstance(actual, type(self.padded))
        np.testing.assert_equal(actual, self.padded)

    def test_pack_padded_sequence(self):
        actual = pts.ops.pack_padded_sequence(self.padded, self.lengths)
        assert isinstance(actual, type(self.packed))
        np.testing.assert_equal(actual.data, self.packed.data)
