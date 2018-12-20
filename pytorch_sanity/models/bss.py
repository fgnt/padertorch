import torch
import pytorch_sanity as pts
import einops
from paderbox.transform import stft
import numpy as np
from paderbox.database.keys import *
from paderbox.database.iterator import AudioReader
from torch.nn.utils.rnn import PackedSequence
from functools import partial


class PermutationInvariantTrainingModel(pts.base.Model):
    """

    TODO: Dropout
    TODO: Mask sensitive loss. See paderflow for more ideas

    """
    def __init__(self, F=257, recurrent_layers=2, units=600, K=2):
        super().__init__()
        self.K = K
        self.F = F
        self.blstm = torch.nn.LSTM(
            F, units, recurrent_layers, bidirectional=True
        )
        self.linear = torch.nn.Linear(2 * units, F * K)

    def prepare_iterable(self, db, dataset: str, batch_size, return_keys=None):
        audio_keys = [OBSERVATION, SPEECH_SOURCE]
        audio_reader = AudioReader(audio_keys=audio_keys, read_fn=db.read_fn)
        return (
            db.get_iterator_by_names(dataset)
            .map(audio_reader)
            .map(partial(self.pre_batch_transform, return_keys=return_keys))
            .batch(batch_size)
            .map(lambda batch: sorted(
                batch,
                key=lambda example: example["num_frames"],
                reverse=True,
            ))
            .map(pts.utils.collate_fn)
            .map(self.post_batch_transform)
            .prefetch(4, 8)
        )

    def pre_batch_transform(self, inputs, return_keys=None):
        s = inputs['audio_data']['speech_source']
        y = inputs['audio_data']['observation']
        S = stft(s, 512, 128)
        Y = stft(y, 512, 128)
        Y = einops.rearrange(Y, 't f -> t f')
        S = einops.rearrange(S, 'k t f -> t k f')
        num_frames = Y.shape[0]

        return_dict = dict()

        def maybe_add(key, value):
            if return_keys is None or key in return_keys:
                return_dict[key] = value

        maybe_add('example_id', inputs['example_id'])
        maybe_add('s', np.ascontiguousarray(s, np.float32))
        maybe_add('y', np.ascontiguousarray(y, np.float32))
        maybe_add('Y_stft', np.ascontiguousarray(Y, np.complex64))
        maybe_add('S_abs', np.ascontiguousarray(np.abs(S), np.float32))
        maybe_add('Y_abs', np.ascontiguousarray(np.abs(Y), np.float32))
        maybe_add('num_frames', num_frames)

        return return_dict

    def post_batch_transform(self, batch):
        return batch

    def forward(self, batch):
        """
        Parameters:
            batch: Dictionary with lists of tensors
        """

        h = pts.ops.pack_sequence(batch['Y_abs'])

        _, F = h.data.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        # Why not mu-law?
        h = pts.ops.pointwise.abs(h)
        h = PackedSequence(h.data + 1e-10, h.batch_sizes)

        # Returns tensor with shape (t, b, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h = PackedSequence(self.linear(h.data), h.batch_sizes)

        mask = PackedSequence(
            einops.rearrange(h.data, 'tb (k f) -> tb k f', k=self.K),
            h.batch_sizes,
        )
        return pts.ops.unpack_sequence(mask)

    def review(self, batch, model_out):
        pit_mse_loss = list()
        for mask, observation, target in zip(
                model_out,
                batch['Y_abs'],
                batch['S_abs']
        ):
            pit_mse_loss.append(pts.ops.loss.pit_mse_loss(
                mask * observation[:, None, :],
                target
            ))

        return {
            'losses': {
                'pit_mse_loss': torch.mean(torch.stack(pit_mse_loss))
            }
        }
