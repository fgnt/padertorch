import unittest
import padertorch as pt
import numpy as np
import torch


class TestDeepClusteringLoss(unittest.TestCase):
    """First tests the reference implementation. Then tests against it."""
    embedding = np.asarray([
        [
            [[1, 0]],
            [[0, 1]],
            [[0, 1]]
        ],
        [
            [[1, 0]],
            [[0, 1]],
            [[0, 1]]
        ]
    ])
    target_mask = np.asarray([
        [
            [[1, 0]],
            [[0, 1]],
            [[0, 1]]
        ],
        [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]]
        ]
    ])

    @staticmethod
    def numpy_reference_loss(embedding, target_mask):
        """Numpy reference implementation

        :param embedding: Shape (N, E)
        :param target_mask: Shape (N, F)
        :return: Scalar
        """
        assert embedding.ndim == 2, embedding.shape
        assert target_mask.ndim == 2, target_mask.shape
        N = embedding.shape[0]
        embedding = embedding / np.sqrt(N)
        target_mask = target_mask / np.sqrt(N)
        return (
            np.sum(np.einsum('ne,nE->eE', embedding, embedding) ** 2)
            - 2 * np.sum(np.einsum('ne,nE->eE', embedding, target_mask) ** 2)
            + np.sum(np.einsum('ne,nE->eE', target_mask, target_mask) ** 2)
        )

    def test_equal(self):
        loss = self.numpy_reference_loss(
            self.embedding[0, :, 0, :],
            self.target_mask[0, :, 0, :]
        )
        np.testing.assert_allclose(loss, 0, atol=1e-6)

    def test_different(self):
        e = self.embedding[1, :, 0, :]
        t = self.target_mask[1, :, 0, :]
        loss = self.numpy_reference_loss(e, t)
        np.testing.assert_allclose(loss, 4 / e.shape[0] ** 2, atol=1e-6)

    def test_dc_loss_against_reference(self):
        embedding = np.random.normal(size=(100, 20))
        target_mask = np.random.choice([0, 1], size=(100, 3))
        loss_ref = self.numpy_reference_loss(embedding, target_mask)

        loss = pt.ops.loss.deep_clustering_loss(
            torch.Tensor(embedding.astype(np.float32)),
            torch.Tensor(target_mask.astype(np.float32)),
        )
        np.testing.assert_allclose(loss, loss_ref, atol=1e-6)


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.model = pt.models.bss.PermutationInvariantTrainingModel()

        self.T = 100
        self.B = 4
        self.K = 2
        self.F = 257
        self.num_frames = [100, 90, 80, 70]
        self.inputs = {
            'Y_abs': [
                np.abs(np.random.normal(
                    size=(num_frames_, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ],
            'X_abs': [
                np.abs(np.random.normal(
                    size=(num_frames_, self.K, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ]
        }

    def test_signature(self):
        assert callable(getattr(self.model, 'forward', None))
        assert callable(getattr(self.model, 'review', None))

    def test_forward(self):
        inputs = pt.data.batch_to_device(self.inputs)
        mask = self.model(inputs)

        for m, t in zip(mask, inputs['X_abs']):
            np.testing.assert_equal(m.size(), t.size())

    def test_review(self):
        inputs = pt.data.batch_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)

        assert 'losses' in review, review.keys()
        assert 'pit_mse_loss' in review['losses'], review['losses'].keys()

    def test_minibatch_equal_to_single_example(self):
        inputs = pt.data.batch_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)
        actual_loss = review['losses']['pit_mse_loss']

        reference_loss = list()
        for observation, target in zip(
            self.inputs['Y_abs'],
            self.inputs['X_abs'],
        ):
            inputs = {
                'Y_abs': [observation],
                'X_abs': [target],
            }
            inputs = pt.data.batch_to_device(inputs)
            mask = self.model(inputs)
            review = self.model.review(inputs, mask)
            reference_loss.append(review['losses']['pit_mse_loss'])

        reference_loss = torch.mean(torch.stack(reference_loss))

        np.testing.assert_allclose(
            actual_loss.detach().numpy(),
            reference_loss.detach().numpy(),
            atol=1e-6
        )
