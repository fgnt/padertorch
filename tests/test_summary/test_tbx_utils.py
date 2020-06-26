import numpy as np
import torch

from padertorch.summary.tbx_utils import audio


def test_audio():
    # A CPU tensor and a numpy array share the data.
    # Verify that the input is not changed.
    
    # Test normalization
    tensor = torch.ones((16000,))
    array, _ = audio(tensor)
    np.testing.assert_allclose(tensor.numpy(), 1)
    np.testing.assert_allclose(array, 0.95)

    # Test zero signal
    tensor = torch.zeros((16000,))
    array, _ = audio(tensor)
    np.testing.assert_allclose(tensor.numpy(), 0)
    np.testing.assert_allclose(array, 0)
