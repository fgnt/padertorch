import numpy as np
import torch
from padertorch.base import Model
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import Mean
from torchvision.utils import make_grid
from einops import rearrange


class Classifier(Model):
    def __init__(
            self, net: CNN1d, feature_extractor=None, *,
            input_key='stft', input_seq_len_key='seq_len', target_key,
    ):
        super().__init__()
        self.net = net
        self.feature_extractor = feature_extractor
        self.input_key = input_key
        self.input_seq_len_key = input_seq_len_key
        self.target_key = target_key

    def forward(self, inputs):
        x = inputs[self.input_key]
        seq_len = inputs[self.input_seq_len_key]
        if self.feature_extractor is not None:
            x = self.feature_extractor(x, seq_len)
            if x.dim() == 4 and isinstance(self.net, CNN1d):
                x = rearrange(x, 'b c f t -> b (c f) t')
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
        config['feature_extractor'] = {
            'factory': NormalizedLogMelExtractor,
        }
        if config['net']['factory'] == CNN1d:
            if config['feature_extractor']['factory'] == NormalizedLogMelExtractor:
                config['net']['in_channels'] = config['feature_extractor']['n_mels']
        else:
            raise ValueError(f'Factory {config["encoder"]["factory"]} not allowed.')
