import unittest
import pytorch_sanity as pts
from pytorch_sanity.models import bss
import numpy as np


class TestPermutationInvariantTrainingLoss(unittest.TestCase):
    def setUp(self):
        self.model = pts.models.bss.PermutationInvariantTrainingModel()

    def test_signature(self):
        assert callable(getattr(self.model, 'forward', None))
        assert callable(getattr(self.model, 'review', None))

    def test_smoke(self):
        T, B, K, F = 100, 4, 2, 257
        inputs = {
            'observation_amplitude_spectrum':
                np.abs(np.random.normal(size=(T, B, F))).astype(np.float32),
            'target_amplitude_spectrum':
                np.abs(np.random.normal(size=(T, B, K, F))).astype(np.float32),
            'num_frames': [100, 90, 80, 70]
        }
        mask = self.model.forward(inputs)
        np.testing.assert_allclose(mask.size(), (T, B, K, F))

        review = self.model.review(inputs)
        assert 'losses' in review, review.keys()
        assert 'pit_mse_loss' in review['losses'], review['losses'].keys()
