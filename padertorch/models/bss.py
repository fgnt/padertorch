import einops
import padertorch as pt
import torch
from torch.nn.utils.rnn import PackedSequence


class PermutationInvariantTrainingModel(pt.base.Model):
    """
    Implements a variant of Permutation Invariant Training [1].

    An example notebook can be found here:
    /net/vol/ldrude/share/2018-12-14_pytorch.ipynb

    Check out this repository to see example code:
    git clone git@ntgit.upb.de:scratch/ldrude/pth_bss

    [1] Kolbaek 2017, https://arxiv.org/pdf/1703.06284.pdf

    TODO: Input normalization
    TODO: Mu-Law as input transform/ at least a logarithm
    TODO: Dropout
    TODO: Mask sensitive loss. See paderflow for more ideas
    TODO: Sigmoid output
    TODO: Batch normalization
    TODO: Phase discounted MSE loss

    """
    def __init__(self, F=257, recurrent_layers=2, units=600, K=2):
        """

        Args:
            F: Number of frequency bins, fft_size / 2 + 1
            recurrent_layers:
            units: results in `units` forward and `units` backward units
            K: Number of output streams/ speakers
        """
        super().__init__()
        self.K = K
        self.F = F
        self.blstm = torch.nn.LSTM(
            F, units, recurrent_layers, bidirectional=True
        )
        self.linear = torch.nn.Linear(2 * units, F * K)

    def forward(self, batch):
        """

        Args:
            batch: Dictionary with lists of tensors

        Returns: List of mask tensors

        """

        h = pt.ops.pack_sequence(batch['Y_abs'])

        _, F = h.data.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        # Why not mu-law?
        h = PackedSequence(h.data + 1e-10, h.batch_sizes)

        # Returns tensor with shape (t, b, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h = PackedSequence(self.linear(h.data), h.batch_sizes)

        mask = PackedSequence(
            einops.rearrange(h.data, 'tb (k f) -> tb k f', k=self.K),
            h.batch_sizes,
        )
        return pt.ops.unpack_sequence(mask)

    def review(self, batch, model_out):
        pit_mse_loss = list()
        for mask, observation, target in zip(
                model_out,
                batch['Y_abs'],
                batch['X_abs']
        ):
            pit_mse_loss.append(pt.ops.losses.loss.pit_mse_loss(
                mask * observation[:, None, :],
                target
            ))

        return {
            'losses': {'pit_mse_loss': torch.mean(torch.stack(pit_mse_loss))}
        }


class DeepClusteringModel(pt.base.Model):
    def __init__(self, F=257, recurrent_layers=2, units=600, E=20):
        """

        TODO: Dropout
        TODO: ...

        Args:
            F: Number of frequency bins, fft_size / 2 + 1
            recurrent_layers:
            units: results in `units` forward and `units` backward units
            E: Dimensionality of the embedding
        """
        super().__init__()
        self.E = E
        self.F = F
        self.blstm = torch.nn.LSTM(
            F, units, recurrent_layers, bidirectional=True
        )
        self.linear = torch.nn.Linear(2 * units, F * E)

    def forward(self, batch):
        """

        Args:
            batch: Dictionary with lists of tensors

        Returns: List of mask tensors

        """

        h = pt.ops.pack_sequence(batch['Y_abs'])

        _, F = h.data.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        # Why not mu-law?
        h = PackedSequence(h.data + 1e-10, h.batch_sizes)

        # Returns tensor with shape (t, b, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h = PackedSequence(self.linear(h.data), h.batch_sizes)

        # TODO: Normalize embedding vectors to unit norm

        mask = PackedSequence(
            einops.rearrange(h.data, 'tb (e f) -> tb e f', e=self.E),
            h.batch_sizes,
        )
        return pt.ops.unpack_sequence(mask)

    def review(self, batch, model_out):
        dc_loss = list()
        for embedding, target_mask in zip(model_out, batch['target_mask']):
            dc_loss.append(pt.ops.losses.deep_clustering_loss(
                einops.rearrange(embedding, 't e f -> (t f) e'),
                einops.rearrange(target_mask, 't k f -> (t f) k')
            ))

        return {'losses': {'dc_loss': torch.mean(torch.stack(dc_loss))}}
