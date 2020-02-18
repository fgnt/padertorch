from functools import partial

import numpy as np
import torch
from padertorch.contrib.je.modules.augment import (
    mel_warping, truncexponential_sampling_fn, log_truncnormal_sampling_fn,
    log_uniform_sampling_fn, Scale, Mixup, Mask,
)
from padertorch.contrib.je.modules.features import MelTransform
from padertorch.contrib.je.modules.hybrid import CRNN as BaseCRNN
from padertorch.contrib.je.modules.norm import Norm, MulticlassNorm
from torch import nn
from torchvision.utils import make_grid
from upb_audio_tagging_2019.lwlrap import lwlrap_from_precisions
from upb_audio_tagging_2019.lwlrap import positive_class_precisions


class CRNN(BaseCRNN):
    def __init__(
            self, cnn_2d, cnn_1d, rnn, fcn, *, sample_rate, fft_length, n_mels,
            post_rnn_pooling=None, decision_boundary=.5,
            max_scale=2.,
            fmin=50., fmax=None, frequency_warping=True,
            n_norm_classes=None,
            mixup_prob=0., shifted_mixup=False, max_seq_len=None,
            n_time_masks=1, max_masked_time_steps=70, max_masked_time_rate=.2,
            n_frequency_masks=1, max_masked_frequency_steps=70, max_masked_frequency_rate=.2,
    ):
        super().__init__(
            cnn_2d, cnn_1d, rnn, fcn, post_rnn_pooling=post_rnn_pooling
        )
        self.decision_boundary = decision_boundary
        self.n_norm_classes = n_norm_classes
        if max_scale is not None:
            self.scale = Scale(
                log_uniform_sampling_fn, scale=2*np.abs(np.log(max_scale))
            )
        else:
            self.scale = None
        self.mel_transform = MelTransform(
            n_mels=n_mels, sample_rate=sample_rate, fft_length=fft_length,
            fmin=fmin, fmax=fmax,
            warping_fn=mel_warping if frequency_warping else None,
            alpha_sampling_fn=partial(
                log_truncnormal_sampling_fn, scale=.1, truncation=np.log(1.5)
            ),
            fhi_sampling_fn=partial(
                truncexponential_sampling_fn, scale=.5, truncation=5.
            )
        )
        norm_kwargs = dict(
            data_format='bcft',
            shape=(None, 1, n_mels, None),
            statistics_axis='bt',
            scale=True,
            independent_axis=None,
            momentum=None,
            interpolation_factor=1.,
        )
        if n_norm_classes is None:
            self.in_norm = Norm(**norm_kwargs)
        else:
            self.in_norm = MulticlassNorm(
                n_classes=n_norm_classes, **norm_kwargs
            )

        if mixup_prob > 0.:
            self.mixup = Mixup(
                interpolate=False, p=mixup_prob,
                shift=shifted_mixup, max_seq_len=max_seq_len
            )
        else:
            self.mixup = None
        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1,  n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
            )
        else:
            self.time_masking = None
        if n_frequency_masks > 0:
            self.freq_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_steps,
                max_masked_rate=max_masked_frequency_rate,
            )
        else:
            self.freq_masking = None

    def forward(self, inputs):
        x = inputs['features']
        seq_len = inputs['seq_len']
        y = inputs['events']
        if self.scale is not None:
            x = self.scale(x)
        x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)

        if self.n_norm_classes is None:
            x = self.in_norm(x, seq_len=seq_len)
        else:
            x = self.in_norm(
                x, seq_len=seq_len, class_idx=inputs['norm_class_idx']
            )

        if self.mixup is not None:
            x, seq_len, mixup_params = self.mixup(
                x, seq_len=seq_len, sequence_axis=-1
            )
            y, *_ = self.mixup(y, mixup_params=mixup_params, cutoff_value=1)

        if self.time_masking is not None:
            x = self.time_masking(x, seq_len=seq_len)
        if self.freq_masking is not None:
            x = self.freq_masking(x)
        h, seq_len = self.cnn_2d(x, seq_len)
        h, seq_len = self.cnn_1d(h, seq_len)
        h = self.rnn(h)
        h, seq_len = self.post_rnn_pooling(h, seq_len)
        return nn.Sigmoid()(self.fcn(h)), y, x

    def review(self, inputs, outputs):
        # compute loss
        y_hat, y, x = outputs
        if y_hat.dim() == 3:  # (B, T, K)
            if y.dim() == 2:   # (B, K)
                y = y.unsqueeze(1).expand(y_hat.shape)
            y_hat = y_hat.contiguous().view((-1, y_hat.shape[-1]))
            y = y.contiguous().view((-1, y.shape[-1]))
        assert y_hat.dim() == y.dim() == 2
        bce = nn.BCELoss(reduction='none')(y_hat, y).sum(-1)

        # create review including metrics and visualizations
        labels, label_ranked_precisions = positive_class_precisions(
            y.cpu().data.numpy(),
            y_hat.cpu().data.numpy()
        )
        decision = (y_hat.detach() > self.decision_boundary).float()
        true_pos = (decision * y).sum()
        false_pos = (decision * (1.-y)).sum()
        false_neg = ((1.-decision) * y).sum()
        review = dict(
            loss=bce.mean(),
            scalars=dict(
                labels=labels,
                label_ranked_precisions=label_ranked_precisions,
                true_pos=true_pos.cpu().data.numpy(),
                false_pos=false_pos.cpu().data.numpy(),
                false_neg=false_neg.cpu().data.numpy()
            ),
            histograms=dict(),
            images=dict(
                features=x[:3],
            )
        )
        return review

    def modify_summary(self, summary):
        # compute lwlrap
        if 'labels' in summary['scalars']:
            labels = summary['scalars'].pop('labels')
            label_ranked_precisions = summary['scalars'].pop(
                'label_ranked_precisions'
            )
            summary['scalars']['lwlrap'] = lwlrap_from_precisions(
                label_ranked_precisions, labels
            )[0]

        # compute precision, recall and fscore for each decision boundary
        if 'true_pos' in summary['scalars']:
            tp = np.sum(summary['scalars'].pop('true_pos'))
            fp = np.sum(summary['scalars'].pop('false_pos'))
            fn = np.sum(summary['scalars'].pop('false_neg'))
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            summary['scalars'][f'precision'] = p
            summary['scalars'][f'recall'] = r
            summary['scalars'][f'fscore'] = 2*(p*r)/(p+r)

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

    input_size_key = 'n_mels'
