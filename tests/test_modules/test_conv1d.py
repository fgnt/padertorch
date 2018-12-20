import unittest
import torch
from padertorch.modules.conv1d import MSTCN


class TestMSTCN(unittest.TestCase):
    def test_output_shapes(self):
        batch_size = 100
        n_frames = 64

        input_dim = 40
        conditional_dim = 39
        latent_dim = 16

        x = torch.ones(batch_size, input_dim, n_frames)
        h = torch.ones(batch_size, conditional_dim, n_frames)

        enc = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_dim=input_dim, hidden_dim=256, output_dim=latent_dim,
                    condition_dim=conditional_dim
                )
            )['kwargs']
        )
        z, pool_indices = enc(x, h)
        self.assertEquals(z.shape, (batch_size, latent_dim, n_frames))
        dec = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_dim=latent_dim, hidden_dim=256, output_dim=input_dim,
                    condition_dim=conditional_dim, transpose=True
                )
            )['kwargs']
        )
        x_hat = dec(z, h, pool_indices=pool_indices[::-1])
        self.assertEquals(x_hat.shape, (batch_size, input_dim, n_frames))

        enc = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_dim=input_dim, hidden_dim=256, output_dim=latent_dim,
                    condition_dim=conditional_dim, n_scales=2, pool_sizes=2)
            )['kwargs']
        )
        y, pool_indices = enc(x, h)
        self.assertEquals(y.shape, (batch_size, latent_dim, n_frames//(2**5)))
        dec = MSTCN(
            **MSTCN.get_config(
                updates=dict(
                    input_dim=latent_dim, hidden_dim=256, output_dim=input_dim,
                    condition_dim=conditional_dim, transpose=True, n_scales=2,
                    pool_sizes=2
                )
            )['kwargs']
        )
        x_hat = dec(z, h, pool_indices=pool_indices[::-1])
        self.assertEquals(x_hat.shape, (batch_size, input_dim, n_frames))
