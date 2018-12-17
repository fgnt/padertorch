import torch
import pytorch_sanity as pts
import einops


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

    def forward(self, batch):
        """
        Parameters:
            batch: Dictionary
        """
        h = batch['observation_amplitude_spectrum']
        num_frames = batch['num_frames']

        if not isinstance(h, torch.Tensor):
            h = torch.from_numpy(h)

        T, B, F = h.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        h = torch.abs(h)
        h = torch.log(h + 1e-10)  # Why not mu-law?

        h = torch.nn.utils.rnn.pack_padded_sequence(h, lengths=num_frames)

        # Returns (seq_len, batch, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h, num_frames = torch.nn.utils.rnn.pad_packed_sequence(h)

        h = self.linear(h)  # Independent dimension?
        mask = einops.rearrange(h, 't b (k f) -> t b k f', k=self.K)
        return mask

    def review(self, batch, model_out):
        observation_amplitude_spectrum \
            = batch['observation_amplitude_spectrum']
        target_amplitude_spectrum = batch['target_amplitude_spectrum']
        num_frames = batch['num_frames']
        mask = model_out

        if not isinstance(observation_amplitude_spectrum, torch.Tensor):
            observation_amplitude_spectrum \
                = torch.from_numpy(observation_amplitude_spectrum)
        if not isinstance(target_amplitude_spectrum, torch.Tensor):
            target_amplitude_spectrum \
                = torch.from_numpy(target_amplitude_spectrum)

        return {
            'losses': {
                'pit_mse_loss': pts.ops.loss.pit_mse_loss(
                    mask * observation_amplitude_spectrum[:, :, None, :],
                    target_amplitude_spectrum,
                    num_frames
                )
            }
        }
