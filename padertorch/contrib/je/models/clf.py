import numpy as np
import torch
from einops import rearrange
from padertorch.base import Model
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.contrib.je.modules.features import MelTransform
from padertorch.contrib.je.modules.global_pooling import Mean
from padertorch.contrib.je.modules.norm import Norm
from torchvision.utils import make_grid


class Classifier(Model):
    def __init__(
            self, net: CNN1d, target_key, sample_rate, fft_length, n_mels, fmin=50, fmax=None,
    ):
        super().__init__()
        self.net = net
        self.target_key = target_key
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

    def inverse_feature_extraction(self, x):
        return self.mel_transform.inverse(
            (
                torch.sqrt(self.in_norm.running_var) * x
                + self.in_norm.running_mean
            ).transpose(-2, -1)
        )

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = inputs['seq_len']
        x = self.feature_extraction(x, seq_len)
        return x, self.net(x, seq_len)

    def review(self, inputs, outputs):
        targets = inputs[self.target_key].long()
        x, (logits, seq_len) = outputs
        if logits.dim() > 2 and targets.dim() == 1:
            assert logits.dim() == 3, logits.shape
            targets = targets.unsqueeze(-1)  # add time axis
            targets = targets.expand((targets.shape[0], logits.shape[-1]))
        predictions = torch.argmax(logits, dim=1)
        ce = torch.nn.CrossEntropyLoss(reduction='none')(logits, targets)
        ce = Mean(axis=-1)(ce, seq_len)
        return dict(
            loss=ce.mean(),
            scalars=dict(
                predictions=predictions,
                targets=targets,
            ),
            histograms=dict(
                ce_=ce.flatten(),
                logits_=logits.flatten(),
            ),
            images=dict(
                features=x[:3],
            )
        )

    def modify_summary(self, summary):
        if 'targets' in summary['scalars']:
            targets = summary['scalars'].pop('targets')
            predictions = summary['scalars'].pop('predictions')
            summary['scalars']['accuracy'] = (
                np.array(predictions) == np.array(targets)
            ).mean()
        for key, image in summary['images'].items():
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        summary = super().modify_summary(summary)
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['net']['factory'] = CNN1d
        if config['net']['factory'] == CNN1d:
            config['net']['in_channels'] = config['n_mels']
        else:
            raise ValueError(f'Factory {config["encoder"]["factory"]} not allowed.')
