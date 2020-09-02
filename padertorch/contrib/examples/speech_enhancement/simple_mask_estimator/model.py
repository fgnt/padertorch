import padertorch as pt
import torch
from padertorch.summary import mask_to_image, stft_to_image


class SimpleMaskEstimator(pt.Model):
    def __init__(self, num_features, num_units=1024, dropout=0.5,
                 activation='elu'):
        """

        Args:
            num_features: number of input features
            num_units: number of units in linear layern
            dropout: dropout forget ratio
            activation: activation for the linear layer except the output layer

        >>> SimpleMaskEstimator(513)
        SmallExampleModel(
          (net): Sequential(
            (0): Dropout(p=0.5)
            (1): Linear(in_features=513, out_features=1024, bias=True)
            (2): ELU(alpha=1.0)
            (3): Dropout(p=0.5)
            (4): Linear(in_features=1024, out_features=1024, bias=True)
            (5): ELU(alpha=1.0)
            (6): Linear(in_features=1024, out_features=1026, bias=True)
            (7): Sigmoid()
          )
        )
        """
        super().__init__()
        self.num_features = num_features
        self.net = torch.nn.Sequential(
            pt.modules.Normalization(
                'btf', (1, 1, num_features), statistics_axis='t',
                independent_axis='f', batch_axis='b', sequence_axis='t'
            ),
            pt.modules.StatefulLSTM(
                num_features, num_units // 4,
                bidirectional=True, batch_first=True
            ),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((num_units // 4) * 2, num_units),
            pt.mappings.ACTIVATION_FN_MAP[activation](),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_units, num_units),
            pt.mappings.ACTIVATION_FN_MAP[activation](),
            # twice num_features for speech and noise_mask
            torch.nn.Linear(num_units, 2 * num_features),
            # Output activation to force outputs between 0 and 1
            torch.nn.Sigmoid()
        )

    def forward(self, batch):

        x = batch['observation_abs']
        out = self.net(x)
        return dict(
            speech_mask_prediction=out[..., :self.num_features],
            noise_mask_prediction=out[..., self.num_features:],
        )

    def review(self, batch, output):
        noise_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['noise_mask_prediction'], batch['noise_mask_target']
        )
        speech_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['speech_mask_prediction'], batch['speech_mask_target']
        )
        return dict(loss=noise_mask_loss + speech_mask_loss,
                    images=self.add_images(batch, output))

    @staticmethod
    def add_images(batch, output):
        speech_mask = output['speech_mask_prediction']
        observation = batch['observation_abs']
        images = dict()
        images['speech_mask'] = mask_to_image(speech_mask, True)
        images['observed_stft'] = stft_to_image(observation, True)

        if 'noise_mask_prediction' in output:
            noise_mask = output['noise_mask_prediction']
            images['noise_mask'] = mask_to_image(noise_mask, True)
        if batch is not None and 'speech_mask_prediction' in batch:
            images['speech_mask_target'] = mask_to_image(
                batch['speech_mask_target'], True)
            if 'speech_mask_target' in batch:
                images['noise_mask_target'] = mask_to_image(
                    batch['noise_mask_target'], True)
        return images
