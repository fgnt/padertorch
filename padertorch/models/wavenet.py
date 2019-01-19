import torch

from padertorch.base import Model
from padertorch import modules
from padertorch.ops import mu_law_decode


class WaveNet(Model):
    def __init__(self, wavenet):
        super().__init__()
        self.wavenet = wavenet

    @classmethod
    def get_signature(cls):
        signature = super().get_signature()
        signature['wavenet'] = {'cls': modules.WaveNet}
        return signature

    def forward(self, inputs):
        features, forward_input = inputs
        return self.wavenet(features, forward_input)

    def review(self, inputs, outputs):
        targets = inputs[1].long()
        ce = torch.nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        summary = dict(
            loss=ce.mean(),
            scalars=dict(),
            histograms=dict(reconstruction_ce=ce),
            audios=dict(
                target=mu_law_decode(
                    inputs[1][0].long(),
                    mu_quantization=self.n_out_channels
                ),
                decode=mu_law_decode(
                    torch.argmax(outputs[0], dim=0),
                    mu_quantization=self.n_out_channels)
            ),
            images=dict()
        )
        return summary
