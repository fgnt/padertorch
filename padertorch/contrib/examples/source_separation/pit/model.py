import einops
import torch
from torch.nn.utils.rnn import PackedSequence
import numpy as np

import padertorch as pt
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.summary import mask_to_image, stft_to_image
from paderbox.transform import istft

class PermutationInvariantTrainingModel(pt.Model):
    """
    Implements a variant of Permutation Invariant Training [1].

    An example notebook can be found here:
    /net/vol/ldrude/share/2018-12-14_pytorch.ipynb

    Check out this repository to see example code:
    git clone git@ntgit.upb.de:scratch/ldrude/pth_bss

    [1] Kolbaek 2017, https://arxiv.org/pdf/1703.06284.pdf

    TODO: Input normalization
    TODO: Batch normalization

    """
    def __init__(
            self,
            F=257,
            recurrent_layers=3,
            units=600,
            K=2,
            dropout_input=0.,
            dropout_hidden=0.,
            dropout_linear=0.,
            output_activation='relu'
    ):
        """

        Args:
            F: Number of frequency bins, fft_size / 2 + 1
            recurrent_layers:
            units: results in `units` forward and `units` backward units
            K: Number of output streams/ speakers
            dropout_input: Dropout forget ratio before first recurrent layer
            dropout_hidden: Vertical forget ratio dropout between each
                recurrent layer
            dropout_linear: Dropout forget ratio before first linear layer
            output_activation: Different activations. Default is 'ReLU'.
        """
        super().__init__()

        self.K = K
        self.F = F

        assert dropout_input <= 0.5, dropout_input
        self.dropout_input = torch.nn.Dropout(dropout_input)

        assert dropout_hidden <= 0.5, dropout_hidden
        self.blstm = torch.nn.LSTM(
            F,
            units,
            recurrent_layers,
            bidirectional=True,
            dropout=dropout_hidden,
        )

        assert dropout_linear <= 0.5, dropout_linear
        self.dropout_linear = torch.nn.Dropout(dropout_linear)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(2 * units, 2 * units)
        self.linear2 = torch.nn.Linear(2 * units, F * K)
        self.output_activation = ACTIVATION_FN_MAP[output_activation]()

    def forward(self, batch):
        """

        Args:
            batch: Dictionary with lists of tensors

        Returns: List of mask tensors
            Each list element has shape (T, K, F)

        """

        h = pt.ops.pack_sequence(batch['Y_abs'])

        _, F = h.data.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        h_data = self.dropout_input(h.data)

        h_data = pt.ops.sequence.log1p(h_data)
        h = PackedSequence(h_data, h.batch_sizes)

        # Returns tensor with shape (t, b, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h_data = self.dropout_linear(h.data)
        h_data = self.linear1(h_data)
        h_data = self.relu(h_data)
        h_data = self.linear2(h_data)
        h_data = self.output_activation(h_data)
        h = PackedSequence(h_data, h.batch_sizes)

        mask = PackedSequence(
            einops.rearrange(h.data, 'tb (k f) -> tb k f', k=self.K),
            h.batch_sizes,
        )
        return pt.ops.unpack_sequence(mask)

    def review(self, batch, model_out):

        pit_mse_loss = list()
        pit_ips_loss = list()

        for mask, observation, target, cos_phase_diff in zip(
                model_out,
                batch['Y_abs'],
                batch['X_abs'],
                batch['cos_phase_difference']
        ):
            # MSE loss
            pit_mse_loss.append(pt.ops.losses.pit_loss(
                mask * observation[:, None, :],
                target,
                axis=-2
            ))

            # Ideal Phase Sensitive loss
            pit_ips_loss.append(pt.ops.losses.pit_loss(
                mask * observation[:, None, :],
                target * cos_phase_diff,
                axis=-2
            ))

        losses = {
                'pit_mse_loss': torch.mean(torch.stack(pit_mse_loss)),
                'pit_ips_loss': torch.mean(torch.stack(pit_ips_loss)),
        }

        b = 0   # only print image of first example in a batch
        images = dict()
        images['observation'] = stft_to_image(batch['Y_abs'][b])
        for i in range(model_out[b].shape[1]):
            images[f'mask_{i}'] = mask_to_image(model_out[b][:, i, :])
            images[f'estimation_{i}'] = stft_to_image(batch['X_abs'][b][:, 0, :])

        return dict(losses=losses,
                    images=images
                    )
