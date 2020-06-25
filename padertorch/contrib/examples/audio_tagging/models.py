import numpy as np
import torch
from einops import rearrange
from padertorch import Model
from padertorch.contrib.je.modules.augment import MelWarping, \
    LogTruncNormalSampler, TruncExponentialSampler
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.global_pooling import Mean, TakeLast
from padertorch.contrib.je.modules.rnn import GRU, reverse_sequence
from torch import nn
from torchvision.utils import make_grid
from padertorch.contrib.je.data.transforms import Collate
from padercontrib.evaluation.event_detection import fscore, get_candidate_thresholds


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
            k = self.cnn.out_channels[-1]
            predictions = np.array(summary['buffers'].pop('predictions')).reshape((-1, k))
            targets = np.array(summary['buffers'].pop('targets')).reshape((-1, k))
            candidate_thresholds = Collate()([
                np.array(th)[np.linspace(0,len(th)-1,100).astype(np.int)]
                for th in get_candidate_thresholds(targets, predictions)
            ])
            decisions = predictions > candidate_thresholds.T[:, None]
            f, p, r = fscore(targets, decisions, event_wise=True)
            best_idx = np.argmax(f, axis=0)
            best_f = f[best_idx, np.arange(k)]
            best_thresholds = candidate_thresholds[np.arange(k), best_idx]
            summary['scalars'][f'macro_fscore'] = best_f.mean()
            summary['scalars'][f'micro_fscore'] = fscore(
                targets, predictions > best_thresholds
            )[0]

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


class CRNN(Model):
    """
    >>> config = CRNN.get_config({\
            'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
            'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
            'rnn_fwd': {'hidden_size': 64},\
            'clf_fwd': {'out_channels':[32,10], 'kernel_size': 1},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'fft_length': 512,\
                'n_mels': 80,\
            },\
        })
    >>> crnn = CRNN.from_config(config)
    >>> inputs = {'stft': torch.zeros((4, 1, 100, 257, 2)), 'seq_len': [100, 90, 80, 70], 'events': torch.zeros((4,10)), 'dataset':[0,1,1,0]}
    >>> outputs = crnn(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 10, 100])
    >>> review = crnn.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn_2d, cnn_1d,
            rnn_fwd, clf_fwd, rnn_bwd, clf_bwd,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._rnn_fwd = rnn_fwd
        self._clf_fwd = clf_fwd
        self._rnn_bwd = rnn_bwd
        self._clf_bwd = clf_bwd

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x, seq_len = self._cnn_2d(x, seq_len)
        if x.dim() != 3:
            assert x.dim() == 4, x.shape
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x, seq_len = self._cnn_1d(x, seq_len)
        return x, seq_len

    def fwd_classification(self, x, seq_len=None):
        x = rearrange(x, 'b f t -> b t f')
        x = self._rnn_fwd(x, seq_len)
        x = rearrange(x, 'b t f -> b f t')
        y, seq_len_y = self._clf_fwd(x, seq_len)
        return nn.Sigmoid()(y), seq_len_y

    def bwd_classification(self, x, seq_len=None):
        x = rearrange(x, 'b f t -> b t f')
        x = reverse_sequence(
            self._rnn_bwd(reverse_sequence(x, seq_len), seq_len), seq_len
        )
        x = rearrange(x, 'b t f -> b f t')
        y, seq_len_y = self._clf_bwd(x, seq_len)
        return nn.Sigmoid()(y), seq_len_y

    def predict(self, x, seq_len=None):
        x, seq_len = self.feature_extractor(x, seq_len=seq_len)
        h, seq_len = self.cnn_2d(x, seq_len)
        h, seq_len = self.cnn_1d(h, seq_len)
        y_fwd, seq_len_y = self.fwd_classification(h, seq_len=seq_len)
        if self._rnn_bwd is not None:
            y_bwd, _ = self.bwd_classification(h, seq_len=seq_len)
        else:
            y_bwd = None
        return (y_fwd, y_bwd, seq_len_y), x

    def prediction_pooling(self, y_fwd, y_bwd, seq_len):
        if y_bwd is None:
            y = TakeLast(axis=-1)(y_fwd, seq_len=seq_len)
            seq_len = None
        elif self.training:
            y = torch.max(y_fwd, y_bwd)
        else:
            y = (TakeLast(axis=-1)(y_fwd, seq_len=seq_len) + y_bwd[:, ..., 0]) / 2
            seq_len = None
        return y, seq_len

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])

        (y_fwd, y_bwd, seq_len_y), x = self.predict(x, seq_len)

        return (y_fwd, y_bwd, seq_len_y), x

    def review(self, inputs, outputs):
        # compute loss
        (y_fwd, y_bwd, seq_len_y), x = outputs

        y, seq_len_y = self.prediction_pooling(y_fwd, y_bwd, seq_len_y)
        targets = inputs['events']

        if y.dim() == 3 and targets.dim() == 2:   # (B, K)
            targets = targets.unsqueeze(-1).expand(y.shape)
        assert targets.dim() == y.dim(), (targets.shape, y.shape)
        bce = nn.BCELoss(reduction='none')(y, targets).sum(1)
        if bce.dim() > 1:
            assert bce.dim() == 2, bce.shape
            bce = Mean(axis=-1)(bce, seq_len_y)
        bce = bce.mean()

        if y.dim() == 3:
            y = (TakeLast(axis=-1)(y_fwd, seq_len=seq_len_y) + y_bwd[:, ..., 0]) / 2
            targets = targets.max(-1)[0]
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
            k = self._clf_fwd.out_channels[-1]
            predictions = np.array(summary['buffers'].pop('predictions')).reshape((-1, k))
            targets = np.array(summary['buffers'].pop('targets')).reshape((-1, k))
            candidate_thresholds = Collate()([
                np.array(th)[np.linspace(0,len(th)-1,100).astype(np.int)]
                for th in get_candidate_thresholds(targets, predictions)
            ])
            decisions = predictions > candidate_thresholds.T[:, None]
            f, p, r = fscore(targets, decisions, event_wise=True)
            best_idx = np.argmax(f, axis=0)
            best_f = f[best_idx, np.arange(k)]
            best_thresholds = candidate_thresholds[np.arange(k), best_idx]
            summary['scalars'][f'macro_fscore'] = best_f.mean()
            summary['scalars'][f'micro_fscore'] = fscore(
                targets, predictions > best_thresholds
            )[0]

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
        config['clf_fwd'] = {'factory': CNN1d}
        config['rnn_bwd'] = {'factory': GRU}
        config['clf_bwd'] = {'factory': CNN1d}
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

        if config['rnn_bwd'] is not None:
            if config['rnn_fwd'] is not None and config['rnn_bwd']['factory'] == config['rnn_fwd']['factory']:
                config['rnn_bwd'].update(config['rnn_fwd'].to_dict())
            elif config['rnn_bwd']['factory'] == GRU:
                config['rnn_bwd'].update({
                    'num_layers': 1,
                    'bias': True,
                    'dropout': 0.,
                    'bidirectional': False
                })

            if input_size is not None:
                config['rnn_bwd']['input_size'] = input_size

        if config['rnn_fwd'] is not None:
            input_size = config['rnn_fwd']['hidden_size']
        elif config['rnn_bwd'] is not None:
            input_size = config['rnn_bwd']['hidden_size']

        if config['clf_fwd'] is not None and config['clf_fwd']['factory'] == CNN1d:
            config['clf_fwd']['in_channels'] = input_size

        if config['clf_bwd'] is not None:
            if config['clf_fwd'] is not None and config['clf_bwd']['factory'] == config['clf_fwd']['factory']:
                config['clf_bwd'].update(config['clf_fwd'].to_dict())
            elif config['clf_bwd']['factory'] == CNN1d:
                config['clf_bwd']['in_channels'] = input_size
