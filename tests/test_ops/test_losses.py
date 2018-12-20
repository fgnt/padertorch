import unittest

import numpy as np
import pytorch_sanity as pts
import torch
from torch.distributions import Normal, MultivariateNormal, kl_divergence
from torch.distributions.kl import _batch_diag
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


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
        actual_loss = pts.ops.losses.loss.pit_mse_loss(estimate, target)
        np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-4)

    def test_toy_example_1(self):
        self.check_toy_example([[[0], [2]]], [[[0], [2]]], 0)

    def test_toy_example_2(self):
        self.check_toy_example([[[0], [2]]], [[[2], [0]]], 0)

    def test_toy_example_3(self):
        self.check_toy_example([[[0], [2]]], [[[-1], [0]]], 2.5)

    def test_toy_example_4(self):
        self.check_toy_example([[[0], [1]]], [[[0], [1]]], 0)

    def test_toy_example_5(self):
        self.check_toy_example([[[0], [1]]], [[[0], [1]]], 0)


class TestKLLoss(unittest.TestCase):
    def test_against_multivariate_multivariate(self):
        B = 500
        K = 100
        D = 16

        scale = torch.randn((K, D, D))
        cov = scale @ scale.transpose(1, 2) + torch.diag(0.1*torch.ones(D))
        p = MultivariateNormal(torch.randn((K, D)), covariance_matrix=cov)

        q = MultivariateNormal(
            loc=torch.randn((B, 1, D)),
            scale_tril=torch.Tensor(
                np.broadcast_to(np.diag(np.random.rand(D)), (B, 1, D, D))
            )
        )
        q_ = Normal(loc=q.loc[:, 0], scale=_batch_diag(q.scale_tril[:, 0]))

        actual_loss = pts.ops.losses.loss.kl_normal_multivariatenormals(q_, p)
        reference_loss = kl_divergence(q, p)
        np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-4)

    def test_shapes(self):
        B1 = 100
        B2 = 50
        K1 = 20
        K2 = 4
        D = 16

        scale = torch.randn((K1, K2, D, D))
        cov = scale @ scale.transpose(-2, -1) + torch.diag(0.1*torch.ones(D))
        p = MultivariateNormal(torch.randn((K1, K2, D)), covariance_matrix=cov)
        p_ = MultivariateNormal(
            loc=p.loc.view(-1, D),
            scale_tril=p.scale_tril.view(-1, D, D)
        )

        q = Normal(
            loc=torch.randn((B1, B2, D)),
            scale=torch.rand((B1, B2, D))
        )
        q_ = Normal(
            loc=q.loc.view(-1, D),
            scale=q.scale.view(-1, D)
        )

        actual_loss = pts.ops.losses.loss.kl_normal_multivariatenormals(q, p)
        reference_loss = pts.ops.losses.loss.kl_normal_multivariatenormals(
            q_, p_).view(B1, B2, K1, K2)
        np.testing.assert_allclose(actual_loss, reference_loss, rtol=1e-4)
