import numpy as np
import torch
from einops import rearrange
from padertorch.base import Model
from torchvision.utils import make_grid


class SpeakerClf(Model):
    def __init__(self, feature_extractor, cnn, enc, fcn):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.cnn = cnn
        self.enc = enc
        self.fcn = fcn

    def forward(self, inputs):
        x = inputs['features'][:, 0]
        seq_len = inputs['seq_len']

        x = self.feature_extractor(x, seq_len)

        # cnn
        x, seq_len = self.cnn(x, sequence_lengths=seq_len)

        # rnn
        if self.enc.batch_first:
            x = rearrange(x, 'b f t -> b t f')
        else:
            x = rearrange(x, 'b f t -> t b f')
        x, _ = self.enc(x)
        if not self.enc.batch_first:
            x = rearrange(x, 't b f -> b t f')
        x = x[torch.arange(len(seq_len)), seq_len - 1]

        x = self.fcn(x)
        return x

    def review(self, inputs, outputs):
        labels = inputs['speaker_id']
        ce = torch.nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        summary = dict(
            loss=ce.mean(),
            scalars=dict(
                labels=labels,
                predictions=torch.argmax(outputs, dim=-1)
            ),
            images=dict(
                features=inputs['features'][:3]
            )
        )
        return summary

    def modify_summary(self, summary):
        if 'labels' in summary['scalars']:
            labels = summary['scalars'].pop('labels')
            predictions = summary['scalars'].pop('predictions')
            summary['scalars']['accuracy'] = (
                    np.array(predictions) == np.array(labels)
            ).mean()
        summary = super().modify_summary(summary)
        for key, image in summary['images'].items():
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary
