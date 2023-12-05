import numpy as np
import padertorch as pt
from einops import einops

import torch


from padertorch.contrib.je.modules.conv import CNN2d, Conv2d, Pool2d, Pool1d
from padertorch.contrib.je.modules.reduce import Mean

from tcrl.modules.loss import AngularPenaltySMLoss

class StudentdVectors(pt.Module):
    """
       The basic Speaker ID neural network based on ResNet34 without time average pooling.
       Extracts K speaker embeddings for a single input example.
       """

    def __init__(
            self,
            in_channels=1,
            channels=(64, 128, 256, 256),
            dvec_dim=256,
            num_spk=1,
            activation_fn='relu',
            norm='batch',
            pre_activation=True,
            encoder_context=3,
            pool_stride=1,
            pool_size=11
    ):
        super().__init__()
        # ResNet34
        out_channels = 3 * 2 * [channels[0]] + 4 * 2 * [channels[1]] + 6 * 2 * [channels[2]] + 3 * 2 * [channels[3]]
        assert len(out_channels) == 32, len(out_channels)
        kernel_size = 32 * [3]
        stride = 3 * 2 * [(1, 1)] + [(2, 2)] + (4 * 2 - 1) * [(1, 1)] + 6 * 2 * [(1, 1)] + [(2, 1)] + (
                3 * 2 - 1) * [(1, 1)]
        pool_size_resnet = 32 * [1]
        pool_stride_resnet = 32 * [1]
        pool_type = 32 * [None]
        residual_connections = 32 * [None]
        for i in range(0, 32, 2):
            residual_connections[i] = i + 2
        norm = norm
        self.embedding_dim = dvec_dim
        self.input_convolution = Conv2d(in_channels, channels[0], kernel_size=encoder_context, stride=2, bias=False,
                                        norm=norm)
        self.resnet = CNN2d(
            input_layer=False,
            output_layer=False,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_size=pool_size_resnet,
            pool_stride=pool_stride_resnet,
            pool_type=pool_type,
            residual_connections=residual_connections,
            activation_fn=activation_fn,
            pre_activation=pre_activation,
            norm=norm,
            normalize_skip_convs=True
        )
        self.output_convolution = Conv2d(channels[-1], dvec_dim * num_spk, kernel_size=3, stride=(2, 1), bias=False,
                                         activation_fn='relu', norm=norm, pre_activation=True)
        self.output_pooling = Pool1d(pool_type='avg', pool_size=pool_size, stride=pool_stride)
        self.num_spk = num_spk
        self.pool_size = pool_size
        self.aam = AngularPenaltySMLoss(in_features=channels[-1], out_features=5994)

    def forward(self, x, seq_len):
        """

        Args:
            x: Log fbank features, Shape (B F T)
            seq_len: Frame-lenghts of each example in the minibatch, Shape (B)
        Returns: Frame-level speaker embeddings for each output stream, shape (B K T)

        """
        # Add a singleton dimension for the convolutions
        # Shape (b t f) -> (b 1 t f)
        x = einops.rearrange(x, 'b f t -> b 1 f t')

        x, seq_len = self.input_convolution(x, seq_len)
        x, seq_len = self.resnet(x, seq_len)
        x, seq_len = self.output_convolution(x, seq_len)
        # Mean Pooling over reduced frequency dim (same len for each example)
        x = Mean(axis=-2)(x)
        x, seq_len, _ = self.output_pooling(x, seq_len)

        x = einops.rearrange(x, 'b (k e) t -> b k e t', k=self.num_spk)

        return x, seq_len

    def get_reduction(self, device=None):
        """
        Determine reduction factor across time dimension of the ResNet to match input and output time resolution.
        """
        x = torch.zeros((1, 1, 100, 100), device=device)
        seq_len = [100, ]
        seq_len_in = np.array(seq_len)
        x, seq_len = self.input_convolution(x, seq_len)
        x, seq_len = self.resnet(x, seq_len)
        x, out_seq_len = self.output_convolution(x, seq_len)
        return seq_len_in / out_seq_len
