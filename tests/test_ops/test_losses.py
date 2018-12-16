import unittest
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_sanity as pts
import numpy as np


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def setUp(self):
        T, B, F, K = 100, 2, 257, 3
        num_frames = [100, 98]

        self.packed_logits = pack_padded_sequence(
            torch.randn(T, B, F, K),
            lengths=num_frames
        )
        self.logits, _ = pad_packed_sequence(self.packed_logits)

        self.packet_targets = pack_padded_sequence(
            torch.randint(low=0, high=K, size=(T, B, F), dtype=torch.long),
            lengths=num_frames
        )
        self.targets, _ = pad_packed_sequence(
            self.packet_targets, padding_value=-1
        )

        loss = [
            num_frames_ * torch.nn.CrossEntropyLoss(ignore_index=-1)(
                self.logits[:num_frames_, b, :, :].permute(0, 2, 1),
                self.targets[:num_frames_, b, :]
            )
            for b, num_frames_ in enumerate(num_frames)
        ]
        self.reference_loss = (
            torch.sum(torch.stack(loss))
            / torch.sum(torch.Tensor(num_frames)).float()
        ).numpy()

    def test_vanilla_loss_when_padded_correctly(self):
        actual = torch.nn.CrossEntropyLoss(ignore_index=-1)(
            self.logits.permute(0, 3, 1, 2), self.targets
        ).numpy()
        np.testing.assert_allclose(actual, self.reference_loss, rtol=1e-4)

    def test_loss_when_padded_correctly(self):
        actual = pts.softmax_cross_entropy(self.logits, self.targets).numpy()
        np.testing.assert_allclose(actual, self.reference_loss, rtol=1e-4)

    def test_loss_with_packed_sequence(self):
        actual = pts.softmax_cross_entropy(
            self.packed_logits, self.packet_targets
        ).numpy()
        np.testing.assert_allclose(actual, self.reference_loss, rtol=1e-4)


class TestDeepClusteringLoss(unittest.TestCase):
    def setUp(self):
        T, B, F = 100, 2, 257
        K = 3
        E = 20
        embedding = torch.randn(T, B, F, E)
        target_mask = ...  # Needs one-hot mask


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def check_toy_example(self, estimate, target, reference_loss):
        estimate = torch.from_numpy(np.array(estimate, dtype=np.float32))
        target = torch.from_numpy(np.array(target, dtype=np.float32))
        num_frames = [1]
        actual_loss = pts.ops.loss.pit_mse_loss(estimate, target, num_frames)
        np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-4)

    def test_toy_example_1(self):
        self.check_toy_example([[[[0], [2]]]], [[[[0], [2]]]], 0)

    def test_toy_example_2(self):
        self.check_toy_example([[[[0], [2]]]], [[[[2], [0]]]], 0)

    def test_toy_example_3(self):
        self.check_toy_example([[[[0], [2]]]], [[[[-1], [0]]]], 2.5)

    def test_toy_example_4(self):
        self.check_toy_example([[[[0], [1]]]], [[[[0], [1]]]], 0)

    def test_toy_example_5(self):
        self.check_toy_example([[[[0], [1]]]], [[[[0], [1]]]], 0)

    def test_if_batch_result_equal_to_single_example(self):
        T, B, K, F = 100, 4, 2, 257
        estimate = torch.randn(T, B, K, F)
        target = torch.randn(T, B, K, F)
        num_frames = [100, 90, 80, 70]

        actual_loss = pts.ops.loss.pit_mse_loss(estimate, target, num_frames)

        reference_loss = list()
        for b, num_frames_ in enumerate(num_frames):
            reference_loss.append(pts.ops.loss.pit_mse_loss(
                estimate[:num_frames_, [b], :, :],
                target[:num_frames_, [b], :, :],
                num_frames=[num_frames_]
            ))
        reference_loss = torch.mean(torch.stack(reference_loss))

        np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-4)
