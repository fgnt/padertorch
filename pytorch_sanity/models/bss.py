import torch
import pytorch_sanity as pts


class PermutationInvariantTrainingModel(pts.base.Model):
    """

    TODO: Dropout
    TODO: Mask sensitive loss. See paderflow for more ideas

    """
    def __init__(self):
        super().__init__()
        self.F = 257
        self.recurrent_layers = 2
        self.units = 600
        self.K = 2
        self.blstm = torch.nn.LSTM(
            self.F, self.units, self.recurrent_layers, bidirectional=True
        )
        self.linear = torch.nn.Linear(2 * self.units, self.F * self.K)

    def forward(self, inputs):
        """
        Parameters:
            inputs:
        """
        h = inputs['observation_amplitude_spectrum']
        num_frames = inputs['num_frames']

        if not isinstance(h, torch.Tensor):
            h = torch.from_numpy(h)

        T, B, F = h.size()
        assert F == self.F, f'self.F = {self.F} != F = {F}'

        h = torch.abs(h)
        h = h + 1e-10
        h = torch.log(h + 1e-10)  # Why not mu-law?

        h = torch.nn.utils.rnn.pack_padded_sequence(h, lengths=num_frames)

        # Returns (seq_len, batch, num_directions * hidden_size)
        h, _ = self.blstm(h)

        h, num_frames = torch.nn.utils.rnn.pad_packed_sequence(h)

        h = self.linear(h.view(-1, 2 * self.units))  # Independent dimension?
        mask = torch.reshape(h, (T, B, self.K, self.F))
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
