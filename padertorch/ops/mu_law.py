import numpy as np
import torch

__all__ = [
    'mu_law_encode',
    'mu_law_decode'
]


def mu_law_decode(x, mu_quantization=256):
    assert(torch.max(x) <= mu_quantization)
    assert(torch.min(x) >= 0)
    x = x.float()
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**torch.abs(signal) - 1)
    return torch.sign(signal) * magnitude


def mu_law_encode(x, mu_quantization=256):
    assert(torch.max(x) <= 1.0)
    assert(torch.min(x) >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).long()
    return encoding
