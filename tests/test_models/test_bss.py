import unittest
import padertorch as pt
import numpy as np
import torch

from padertorch.contrib.examples.source_separation.pit.model import PermutationInvariantTrainingModel
from padertorch.contrib.tcl.dc import DeepClusteringModel

class TestDeepClusteringModel(unittest.TestCase):
    # TODO: Test forward deterministic if not train

    def setUp(self):
        self.model = DeepClusteringModel()

        self.T = 100
        self.B = 4
        self.E = 20
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
            'target_mask': [
                np.abs(np.random.choice(
                    [0, 1],
                    size=(num_frames_, self.K, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ]
        }

    def test_signature(self):
        assert callable(getattr(self.model, 'forward', None))
        assert callable(getattr(self.model, 'review', None))

    def test_forward(self):
        inputs = pt.data.example_to_device(self.inputs)
        model_out = self.model(inputs)

        for embedding, num_frames in zip(model_out, self.num_frames):
            expected_shape = (num_frames, self.E, self.F)
            assert embedding.shape == expected_shape, embedding.shape

    def test_review(self):
        inputs = pt.data.example_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)

        assert 'losses' in review, review.keys()
        assert 'dc_loss' in review['losses'], review['losses'].keys()

    def test_minibatch_equal_to_single_example(self):
        inputs = pt.data.example_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)
        actual_loss = review['losses']['dc_loss']

        reference_loss = list()
        for observation, target_mask in zip(
            self.inputs['Y_abs'],
            self.inputs['target_mask'],
        ):
            inputs = {
                'Y_abs': [observation],
                'target_mask': [target_mask],
            }
            inputs = pt.data.example_to_device(inputs)
            mask = self.model(inputs)
            review = self.model.review(inputs, mask)
            reference_loss.append(review['losses']['dc_loss'])

        reference_loss = torch.mean(torch.stack(reference_loss))

        np.testing.assert_allclose(
            actual_loss.detach().numpy(),
            reference_loss.detach().numpy(),
            atol=1e-6
        )


class TestPermutationInvariantTrainingModel(unittest.TestCase):
    # TODO: Test forward deterministic if not train

    def setUp(self):
        self.model = PermutationInvariantTrainingModel(
            dropout_input=0.5,
            dropout_hidden=0.5,
            dropout_linear=0.5
        )

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
            ],
            'Y_norm': [
                np.abs(np.random.normal(
                    size=(num_frames_, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ],
            'X_norm': [
                np.abs(np.random.normal(
                    size=(num_frames_, self.K, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ],
            'cos_phase_difference': [
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
        inputs = pt.data.example_to_device(self.inputs)
        mask = self.model(inputs)

        for m, t in zip(mask, inputs['X_abs']):
            np.testing.assert_equal(m.size(), t.size())

    def test_review(self):
        inputs = pt.data.example_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)

        assert 'losses' in review, review.keys()
        assert 'pit_mse_loss' in review['losses'], review['losses'].keys()

    def test_minibatch_equal_to_single_example(self):
        inputs = pt.data.example_to_device(self.inputs)

        self.model.eval()
        mask = self.model(inputs)

        review = self.model.review(inputs, mask)

        actual_loss = review['losses']['pit_mse_loss']

        reference_loss = list()
        for Y_abs, X_abs, Y_norm, X_norm, cos_phase_difference in zip(
            self.inputs['Y_abs'],
            self.inputs['X_abs'],
            self.inputs['Y_norm'],
            self.inputs['X_norm'],
            self.inputs['cos_phase_difference'],
        ):
            inputs = {
                'Y_abs': [Y_abs],
                'X_abs': [X_abs],
                'Y_norm': [Y_norm],
                'X_norm': [X_norm],
                'cos_phase_difference': [cos_phase_difference],
            }
            inputs = pt.data.example_to_device(inputs)

            self.model.eval()
            mask = self.model(inputs)

            review = self.model.review(inputs, mask)
            reference_loss.append(review['losses']['pit_mse_loss'])

        reference_loss = torch.mean(torch.stack(reference_loss))

        np.testing.assert_allclose(
            actual_loss.detach().numpy(),
            reference_loss.detach().numpy(),
            atol=1e-6
        )

    def test_evaluation_mode_deterministic(self):
        self.model.eval()

        inputs = pt.data.example_to_device(self.inputs)
        mask1 = self.model(inputs)[0]

        inputs = pt.data.example_to_device(self.inputs)
        mask2 = self.model(inputs)[0]

        np.testing.assert_allclose(
            mask1.detach().numpy(),
            mask2.detach().numpy(),
            atol=1e-6
        )
