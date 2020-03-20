import torch
from einops import rearrange
from padertorch import modules
from padertorch.contrib.je.modules.features import MelTransform
from padertorch.contrib.je.modules.norm import Norm
from padertorch.models.wavenet import WaveNet as BaseWaveNet


class WaveNet(BaseWaveNet):
    def __init__(
            self, wavenet, sample_rate, fft_length, n_mels, fmin=50, fmax=None
    ):
        super().__init__(
            wavenet=wavenet, sample_rate=sample_rate,
            feature_key="features", audio_key="audio_data"
        )
        self.mel_transform = MelTransform(
            n_mels=n_mels, sample_rate=sample_rate, fft_length=fft_length,
            fmin=fmin, fmax=fmax,
        )
        self.in_norm = Norm(
            data_format='bcft',
            shape=(None, 1, n_mels, None),
            statistics_axis='bt',
            scale=True,
            independent_axis=None,
            momentum=None,
            interpolation_factor=1.
        )

    def feature_extraction(self, x, seq_len=None):
        x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)
        x = self.in_norm(x, seq_len=seq_len)
        x = rearrange(x, 'b c f t -> b (c f) t')
        return x

    def forward(self, inputs):
        x_target = inputs['stft']
        seq_len = inputs['seq_len']
        x_target = self.feature_extraction(x_target, seq_len)
        return self.wavenet(x_target, inputs[self.audio_key])

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['wavenet']['factory'] = modules.WaveNet
        config['wavenet']['n_cond_channels'] = config['n_mels']
