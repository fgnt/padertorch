import numpy as np
import torch
from padercontrib.evaluation.event_detection import fscore
from padertorch import Model
from padertorch.contrib.je.modules.augment import MelWarping, \
    LogTruncNormalSampler, TruncExponentialSampler
from padertorch.contrib.je.modules.conv import CNN2d
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import Mean
from pb_sed.evaluation import instance_based
from sklearn import metrics
from torch import nn
from torchvision.utils import make_grid


class WALNet(Model):
    """
    >>> from paderbox.utils.nested import deflatten
    >>> tagger = WALNet(sample_rate=44100, fft_length=2048, output_size=10)
    >>> inputs = {'stft': torch.zeros(4,1,128,1025,2), 'events': torch.zeros((4, 10)), 'seq_len': 4*[128]}
    >>> outputs = tagger(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 10, 1])
    >>> review = tagger.review(inputs, outputs)
    """

    def __init__(self, sample_rate, fft_length, output_size):
        super().__init__()
        self.feature_extractor = NormalizedLogMelExtractor(
            n_mels=128, sample_rate=sample_rate, fft_length=fft_length,
            # augmentation
            frequency_warping_fn=MelWarping(
                alpha_sampling_fn=LogTruncNormalSampler(
                    scale=0.07, truncation=np.log(1.3)
                ),
                fhi_sampling_fn=TruncExponentialSampler(
                    scale=0.5, truncation=5.
                )
            ),
            n_time_masks=1,
            n_mel_masks=1,
        )
        self.cnn = CNN2d(
            in_channels=1,
            out_channels=[
                16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024, output_size
            ],
            kernel_size=11 * [3] + [2, 1],
            pad_type=11 * ['both'] + 2 * [None],
            pool_size=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1],
            norm='batch',
            activation_fn='relu',
        )

    def forward(self, inputs):
        x, seq_len = inputs['stft'], inputs['seq_len']
        x, *_ = self.feature_extractor(x, seq_len)
        y, seq_len = self.cnn(x, seq_len)
        return (nn.Sigmoid()(y.squeeze(2)), seq_len), x

    def review(self, inputs, outputs):
        # compute loss
        targets = inputs['events']
        (y, seq_len), x = outputs

        y = Mean(axis=-1)(y, seq_len)
        bce = nn.BCELoss(reduction='none')(y, targets).sum(-1).mean()

        # create review including metrics and visualizations
        review = dict(
            loss=bce,
            scalars=dict(),
            histograms=dict(),
            images=dict(
                features=x[:3],
            ),
            buffers=dict(predictions=y.data.cpu().numpy(), targets=targets.data.cpu().numpy()),
        )
        return review

    def modify_summary(self, summary):
        # compute precision, recall and fscore
        if f'predictions' in summary['buffers']:
            predictions = np.concatenate(summary['buffers'].pop('predictions'))
            targets = np.concatenate(summary['buffers'].pop('targets'))
            thresholds, f = instance_based.get_optimal_thresholds(
                targets, predictions, 'f1'
            )
            summary['scalars']['macro_fscore'] = f.mean()
            summary['scalars']['micro_fscore'] = fscore(
                targets, predictions > thresholds
            )[0]
            summary['scalars']['lwlrap'] = instance_based.lwlrap(
                targets, predictions
            )
            if (targets.sum(0) > 1).all():
                summary['scalars']['map'] = metrics.average_precision_score(targets, predictions)
                summary['scalars']['mauc'] = metrics.roc_auc_score(targets, predictions)

        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)

        for key, image in summary['images'].items():
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary
