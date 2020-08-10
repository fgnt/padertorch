import einops
import torch
from torch.nn.utils.rnn import PackedSequence

import padertorch as pt


class DeepClusteringModel(pt.Model):
    def __init__(
            self,
            F=257,
            recurrent_layers=2,
            units=600,
            E=20,
            input_feature_transform='identity'
    ):
        """

        TODO: Dropout
        TODO: Loss mask to avoid to assign embeddings to silent regions

        Args:
            F: Number of frequency bins, fft_size / 2 + 1
            recurrent_layers:
            units: results in `units` forward and `units` backward units
            E: Dimensionality of the embedding
        """
        super().__init__()
        self.E = E
        self.F = F
        self.input_feature_transform = input_feature_transform
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

        if self.input_feature_transform == 'identity':
            pass
        elif self.input_feature_transform == 'log1p':
            # This is equal to the mu-law for mu=1.
            h = pt.ops.sequence.log1p(h)
        elif self.input_feature_transform == 'log':
            h = PackedSequence(h.data + 1e-10, h.batch_sizes)
            h = pt.ops.sequence.log(h)
        else:
            raise NotImplementedError(self.input_feature_transform)

        _, F = h.data.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        # Returns tensor with shape (t, b, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h = PackedSequence(self.linear(h.data), h.batch_sizes)
        h_data = einops.rearrange(h.data, 'tb (e f) -> tb e f', e=self.E)

        # Hershey 2016 page 2 top right paragraph: Unit norm
        h_data = torch.nn.functional.normalize(h_data, dim=-2)

        embedding = PackedSequence(h_data, h.batch_sizes,)
        embedding = pt.ops.unpack_sequence(embedding)
        return embedding

    def review(self, batch, model_out):
        dc_loss = list()
        for embedding, target_mask in zip(model_out, batch['target_mask']):
            dc_loss.append(pt.ops.losses.deep_clustering_loss(
                einops.rearrange(embedding, 't e f -> (t f) e'),
                einops.rearrange(target_mask, 't k f -> (t f) k')
            ))

        return {'losses': {'dc_loss': torch.mean(torch.stack(dc_loss))}}
