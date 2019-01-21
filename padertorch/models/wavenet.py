import torch

from padertorch.base import Model
from padertorch import modules
from padertorch.ops import mu_law_decode


class WaveNet(Model):
    def __init__(self, wavenet, sample_rate=16000):
        super().__init__()
        self.wavenet = wavenet
        self.sample_rate = sample_rate

    @classmethod
    def get_signature(cls):
        signature = super().get_signature()
        signature['wavenet'] = {'cls': modules.WaveNet}
        return signature

    def forward(self, inputs):
        features, audio = inputs
        return self.wavenet(features, audio)

    def review(self, inputs, outputs):
        predictions, targets = outputs
        ce = torch.nn.CrossEntropyLoss(reduction='none')(predictions, targets)
        summary = dict(
            loss=ce.mean(),
            scalars=dict(),
            histograms=dict(reconstruction_ce=ce),
            audios=dict(
                target=(inputs[1][0], self.sample_rate),
                decode=(
                    mu_law_decode(
                        torch.argmax(outputs[0][0], dim=0),
                        mu_quantization=self.wavenet.n_out_channels),
                    self.sample_rate)
            ),
            images=dict()
        )
        return summary
