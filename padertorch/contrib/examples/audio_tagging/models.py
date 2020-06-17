import numpy as np
import torch
from einops import rearrange
from padertorch import Model
from padertorch.contrib.je.modules.augment import MelWarping, \
    LogTruncNormalSampler, TruncExponentialSampler
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.global_pooling import Mean
from padertorch.contrib.je.modules.rnn import GRU, reverse_sequence
from padertorch.modules.fully_connected import fully_connected_stack
from torch import nn
from torchvision.utils import make_grid


class WALNet(Model):
    """
    >>> from paderbox.utils.nested import deflatten
    >>> tagger = WALNet(sample_rate=44100, fft_length=2048, output_size=10)
    >>> inputs = {'stft': torch.zeros(4,1,128,1025,2), 'events': torch.zeros((4, 10)), 'seq_len': 4*[128]}
    >>> outputs = tagger(inputs)
    >>> outputs[1].shape
    torch.Size([4, 10, 1])
    >>> review = tagger.review(inputs, outputs)
    """

    def __init__(self, sample_rate, fft_length, output_size):
        super().__init__()
        self.feature_extractor = NormalizedLogMelExtractor(
            n_mels=128, sample_rate=sample_rate, fft_length=fft_length,
            # augmentation
            scale_sigma=.8,
            mixup_prob=.5,
            warping_fn=MelWarping(
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
            pad_side=11 * ['both'] + 2 * [None],
            pool_size=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1],
            norm='batch',
            activation_fn='relu',
        )

    def forward(self, inputs):
        x, seq_len = inputs['stft'], inputs['seq_len']
        x, *_ = self.feature_extractor(x, seq_len=seq_len)
        y, seq_len = self.cnn(x, seq_len=seq_len)
        return x, nn.Sigmoid()(y.squeeze(2)), seq_len

    @property
    def decision_thresholds(self):
        step = .1
        return np.arange(step, 1., step).tolist()

    def review(self, inputs, outputs):
        # compute loss
        y = inputs['events']
        x, frame_probs, seq_len = outputs

        seq_len = torch.Tensor(seq_len)[:, None, None].to(frame_probs.device)
        idx = torch.arange(frame_probs.shape[-1]).to(x.device)
        mask = (idx < seq_len.long()).float()
        y_hat = (frame_probs * mask).sum(dim=-1) / seq_len.squeeze(-1)
        bce = nn.BCELoss(reduction='none')(y_hat, y).sum(-1)

        # create review including metrics and visualizations
        review = dict(
            loss=bce.mean(),
            scalars=dict(),
            histograms=dict(
                sequence_probs=y_hat.flatten()
            ),
            images=dict(
                input=x[:3].flip(1).view((-1, x.shape[-1]))[None]
            )
        )

        with torch.no_grad():
            for threshold in self.decision_thresholds:
                decision = (y_hat.detach() > threshold).float()
                review['scalars'][f'tp_{threshold}'] = (decision * y).sum().cpu().data.numpy()
                review['scalars'][f'fp_{threshold}'] = (decision * (1.-y)).sum().cpu().data.numpy()
                review['scalars'][f'fn_{threshold}'] = ((1.-decision) * y).sum().cpu().data.numpy()
        return review

    def modify_summary(self, summary):
        # compute precision, recall and fscore
        if 'tp_0.5' in summary['scalars']:
            precisions = []
            recalls = []
            fscores = []
            for threshold in self.decision_thresholds:
                tp = np.sum(summary['scalars'].pop(f'tp_{threshold}'))
                fp = np.sum(summary['scalars'].pop(f'fp_{threshold}'))
                fn = np.sum(summary['scalars'].pop(f'fn_{threshold}'))
                p = tp/max(tp+fp, 1)
                r = tp/max(tp+fn, 1)
                precisions.append(p)
                recalls.append(r)
                fscores.append(2*(p*r)/max(p+r, 1e-6))
            best_idx = int(np.argmax(fscores))
            summary['scalars']['precision'] = precisions[best_idx]
            summary['scalars']['recall'] = recalls[best_idx]
            summary['scalars']['fscore'] = fscores[best_idx]
            summary['scalars']['threshold'] = self.decision_thresholds[best_idx]

        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)

        # normalize images
        for image in summary['images'].values():
            image -= image.min()
            image /= image.max()
        return summary


class CRNN(Model):
    """
    >>> config = CRNN.get_config({\
            'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
            'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
            'rnn_fwd': {'hidden_size': 64},\
            'fcn_fwd': {'hidden_size': 64, 'output_size': 10},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'fft_length': 512,\
                'n_mels': 80,\
            },\
        })
    >>> crnn = CRNN.from_config(config)
    >>> inputs = {'stft': torch.zeros((4, 1, 100, 257, 2)), 'seq_len': [70, 80, 90, 100], 'events': torch.zeros((4,10)), 'dataset':[0,1,1,0]}
    >>> outputs = crnn(inputs)
    >>> outputs[0].shape
    torch.Size([4, 100, 10])
    >>> review = crnn.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn_2d, cnn_1d,
            rnn_fwd, fcn_fwd, rnn_bwd, fcn_bwd,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._rnn_fwd = rnn_fwd
        self._fcn_fwd = fcn_fwd
        self._rnn_bwd = rnn_bwd
        self._fcn_bwd = fcn_bwd

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x, seq_len = self._cnn_2d(x, seq_len)
        if x.dim() != 3:
            assert x.dim() == 4
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x, seq_len = self._cnn_1d(x, seq_len)
        return x, seq_len

    def fwd_classification(self, x, seq_len=None):
        x = self._rnn_fwd(x, seq_len)
        y = self._fcn_fwd(x)
        return nn.Sigmoid()(y)

    def bwd_classification(self, x, seq_len=None):
        x = reverse_sequence(
            self._rnn_bwd(reverse_sequence(x, seq_len), seq_len),
            seq_len
        )
        y = self._fcn_bwd(x)
        return nn.Sigmoid()(y)

    def forward(self, inputs):
        x = inputs['stft']
        y = inputs['events']
        seq_len = inputs['seq_len']
        x, y, seq_len = self.feature_extractor(x, y, seq_len=seq_len)
        h, seq_len = self.cnn_2d(x, seq_len)
        h, seq_len = self.cnn_1d(h, seq_len)
        h = rearrange(h, 'b f t -> b t f')
        y_hat_fwd = self.fwd_classification(h, seq_len=seq_len)
        y_hat_bwd = self.bwd_classification(h, seq_len=seq_len)
        y_hat = torch.max(y_hat_fwd, y_hat_bwd)
        return y_hat, y, x

    @property
    def decision_thresholds(self):
        step = .1
        return np.arange(step, 1., step).tolist()

    def review(self, inputs, outputs):
        # compute loss
        y_hat, y, x = outputs
        if y.dim() == 2:   # (B, K)
            y = y.unsqueeze(1).expand(y_hat.shape)
        assert y_hat.dim() == y.dim() == 3, (y_hat.shape, y.shape)
        bce = nn.BCELoss(reduction='none')(y_hat, y).sum(-1)
        bce = Mean(axis=-1)(bce, inputs['seq_len'])
        review = dict(
            loss=bce.mean(),
            scalars=dict(
                mixup_prob=0. if self.feature_extractor.mixup is None
                else self.feature_extractor.mixup.p
            ),
            histograms=dict(),
            images=dict(
                features=x[:3],
            )
        )

        with torch.no_grad():
            for threshold in self.decision_thresholds:
                decision = (y_hat.detach() > threshold).float()
                review['scalars'][f'tp_{threshold}'] = (decision * y).sum().cpu().data.numpy()
                review['scalars'][f'fp_{threshold}'] = (decision * (1.-y)).sum().cpu().data.numpy()
                review['scalars'][f'fn_{threshold}'] = ((1.-decision) * y).sum().cpu().data.numpy()
        return review

    def modify_summary(self, summary):
        # compute precision, recall and fscore
        if 'tp_0.5' in summary['scalars']:
            precisions = []
            recalls = []
            fscores = []
            for threshold in self.decision_thresholds:
                tp = np.sum(summary['scalars'].pop(f'tp_{threshold}'))
                fp = np.sum(summary['scalars'].pop(f'fp_{threshold}'))
                fn = np.sum(summary['scalars'].pop(f'fn_{threshold}'))
                p = tp/max(tp+fp, 1)
                r = tp/max(tp+fn, 1)
                precisions.append(p)
                recalls.append(r)
                fscores.append(2*(p*r)/max(p+r, 1e-6))
            best_idx = int(np.argmax(fscores))
            summary['scalars']['precision'] = precisions[best_idx]
            summary['scalars']['recall'] = recalls[best_idx]
            summary['scalars']['fscore'] = fscores[best_idx]
            summary['scalars']['threshold'] = self.decision_thresholds[best_idx]

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

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['feature_extractor'] = {'factory': NormalizedLogMelExtractor}
        config['cnn_2d'] = {'factory': CNN2d}
        config['cnn_1d'] = {'factory': CNN1d}
        config['rnn_fwd'] = {'factory': GRU}
        config['fcn_fwd'] = {'factory': fully_connected_stack}
        input_size = config['feature_extractor']['n_mels']
        if config['cnn_2d'] is not None and input_size is not None:
            config['cnn_2d']['in_channels'] = 1
            in_channels = config['cnn_2d']['in_channels']
            cnn_2d = config['cnn_2d']['factory'].from_config(config['cnn_2d'])
            output_size = cnn_2d.get_shapes((1, in_channels, input_size, 1000))[-1][2]
            input_size = cnn_2d.out_channels[-1] * output_size

        if config['cnn_1d'] is not None:
            if input_size is not None:
                config['cnn_1d']['in_channels'] = input_size
            input_size = config['cnn_1d']['out_channels'][-1]

        if config['rnn_fwd'] is not None:
            if config['rnn_fwd']['factory'] == GRU:
                config['rnn_fwd'].update({
                    'num_layers': 1,
                    'bias': True,
                    'dropout': 0.,
                    'bidirectional': False
                })

            if input_size is not None:
                config['rnn_fwd']['input_size'] = input_size
            input_size = config['rnn_fwd']['hidden_size']

        if config['fcn_fwd'] is not None:
            config['fcn_fwd']['input_size'] = input_size

        config['rnn_bwd'] = config['rnn_fwd']
        config['fcn_bwd'] = config['fcn_fwd']
