import unittest
import torch
from padertorch.contrib.je.conv1d import TCN


class TestTCN(unittest.TestCase):
    def test_output_shapes(self):
        batch_size = 100
        n_frames = 128

        input_size = 40
        condition_size = 39
        latent_dim = 16

        x = torch.ones(batch_size, input_size, n_frames)
        h = torch.ones(batch_size, condition_size, n_frames)
        for n_scales in [None, 1, 2]:
            for pooling in ['max', 'avg']:
                for pool_size in [1, 2]:
                    for padding in ['both', None]:
                        enc = TCN.from_config(
                            TCN.get_config(
                                updates=dict(
                                    input_size=input_size, hidden_sizes=256,
                                    output_size=latent_dim,
                                    condition_size=condition_size,
                                    n_scales=n_scales, norm='batch',
                                    pooling=pooling, pool_sizes=pool_size,
                                    paddings=padding
                                )
                            )
                        )
                        z, pooling_data = enc(x, h)
                        dec = TCN.from_config(
                            TCN.get_config(
                                updates=dict(
                                    input_size=latent_dim, hidden_sizes=256,
                                    output_size=input_size,
                                    condition_size=condition_size,
                                    transpose=True,
                                    n_scales=n_scales, norm='batch',
                                    pooling=pooling, pool_sizes=pool_size,
                                    paddings=padding
                                )
                            )
                        )
                        x_hat = dec(z, h, pooling_data=pooling_data[::-1])
                        self.assertEqual(
                            x_hat.shape, (batch_size, input_size, n_frames))
