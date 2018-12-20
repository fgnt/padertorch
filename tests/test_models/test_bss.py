import unittest
import padertorch as pts
import numpy as np
import torch


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.model = pts.models.bss.PermutationInvariantTrainingModel()

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
        inputs = pts.data.batch_to_device(self.inputs)
        mask = self.model(inputs)

        for m, t in zip(mask, inputs['X_abs']):
            np.testing.assert_equal(m.size(), t.size())

    def test_review(self):
        inputs = pts.data.batch_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)

        assert 'losses' in review, review.keys()
        assert 'pit_mse_loss' in review['losses'], review['losses'].keys()

    def test_minibatch_equal_to_single_example(self):
        inputs = pts.data.batch_to_device(self.inputs)
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
            inputs = pts.data.batch_to_device(inputs)
            mask = self.model(inputs)
            review = self.model.review(inputs, mask)
            reference_loss.append(review['losses']['pit_mse_loss'])

        reference_loss = torch.mean(torch.stack(reference_loss))

        np.testing.assert_allclose(
            actual_loss.detach().numpy(),
            reference_loss.detach().numpy(),
            atol=1e-6
        )
