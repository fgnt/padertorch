import unittest
import torch
from padertorch.modules.conv1d import MSTCN


class TestMSTCN(unittest.TestCase):
    def test_output_shapes(self):
        batch_size = 100
        n_frames = 64

        input_size = 40
        condition_size = 39
        latent_dim = 16

        x = torch.ones(batch_size, input_size, n_frames)
        h = torch.ones(batch_size, condition_size, n_frames)

        enc = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=input_size, hidden_sizes=256,
                    output_size=latent_dim, condition_size=condition_size
                )
            )['kwargs']
        )
        z, pool_indices = enc(x, h)
        self.assertEquals(z.shape, (batch_size, latent_dim, n_frames))
        dec = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=latent_dim, hidden_sizes=256,
                    output_size=input_size, condition_size=condition_size,
                    transpose=True
                )
            )['kwargs']
        )
        x_hat = dec(z, h, pool_indices=pool_indices[::-1])
        self.assertEquals(x_hat.shape, (batch_size, input_size, n_frames))

        enc = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=input_size, hidden_sizes=256,
                    output_size=latent_dim, condition_size=condition_size,
                    n_scales=2, pool_sizes=2)
            )['kwargs']
        )
        y, pool_indices = enc(x, h)
        self.assertEquals(y.shape, (batch_size, latent_dim, n_frames//(2**5)))
        dec = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=latent_dim, hidden_sizes=256,
                    output_size=input_size, condition_size=condition_size,
                    transpose=True, n_scales=2, pool_sizes=2
                )
            )['kwargs']
        )
        x_hat = dec(z, h, pool_indices=pool_indices[::-1])
        self.assertEquals(x_hat.shape, (batch_size, input_size, n_frames))

        enc = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=input_size, hidden_sizes=[256, 128, 64, 32],
                    output_size=latent_dim, condition_size=condition_size,
                    n_scales=2, pool_sizes=2)
            )['kwargs']
        )
        y, pool_indices = enc(x, h)
        self.assertEquals(y.shape, (batch_size, latent_dim, n_frames//(2**5)))
        dec = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_size=latent_dim, hidden_sizes=[32, 64, 128, 256],
                    output_size=input_size, condition_size=condition_size,
                    transpose=True, n_scales=2, pool_sizes=2
                )
            )['kwargs']
        )
        x_hat = dec(z, h, pool_indices=pool_indices[::-1])
        self.assertEquals(x_hat.shape, (batch_size, input_size, n_frames))
