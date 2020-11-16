import torch
from einops import rearrange
from padertorch import modules
from padertorch.base import Model
from padertorch.contrib.je.modules.features import MelTransform
from padertorch.modules.normalization import InputNormalization
from padertorch.ops import mu_law_decode


class WaveNet(Model):
    def __init__(
            self, wavenet, sample_rate, fft_length, n_mels, fmin=50, fmax=None
    ):
        super().__init__()
        self.wavenet = wavenet
        self.sample_rate = sample_rate
        self.mel_transform = MelTransform(
            n_mels=n_mels, sample_rate=sample_rate, fft_length=fft_length,
            fmin=fmin, fmax=fmax,
        )
        self.in_norm = InputNormalization(
            data_format='bcft',
            shape=(None, 1, n_mels, None),
            statistics_axis='bt',
            independent_axis=None,
        )

    def feature_extraction(self, x, seq_len=None):
        x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)
        x = self.in_norm(x, sequence_lengths=seq_len)
        x = rearrange(x, 'b c f t -> b (c f) t')
        return x

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = inputs['seq_len']
        x = self.feature_extraction(x, seq_len)
        return self.wavenet(x.squeeze(1), inputs['audio_data'].squeeze(1))

    def review(self, inputs, outputs):
        predictions, targets = outputs
        ce = torch.nn.CrossEntropyLoss(reduction='none')(predictions, targets)
        summary = dict(
            loss=ce.mean(),
            scalars=dict(),
            histograms=dict(reconstruction_ce=ce),
            audios=dict(
                target=(inputs['audio_data'][0], self.sample_rate),
                decode=(
                    mu_law_decode(
                        torch.argmax(outputs[0][0], dim=0),
                        mu_quantization=self.wavenet.n_out_channels),
                    self.sample_rate)
            ),
            images=dict()
        )
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['wavenet']['factory'] = modules.WaveNet
        config['wavenet']['n_cond_channels'] = config['n_mels']
