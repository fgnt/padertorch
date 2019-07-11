import numpy as np
import torch
from paderbox.evaluation.event_detection import positive_class_precisions, \
    lwlrap_from_precisions
from padertorch.base import Module, Model
from padertorch.contrib.je.modules.attention import Transformer
from padertorch.contrib.je.modules.conv import CNN1d, CNN2d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.utils import to_list
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.utils import make_grid
from einops import rearrange


class HybridNet(Module):
    """
    >>> hybrid_net = HybridNet.from_config(HybridNet.get_config({\
        'input_size': 80,\
        'cnn_2d': {\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'cnn_1d': {\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'enc': {'hidden_size': 64},\
        'fcn': {'hidden_size': 32, 'output_size': 10}\
    }))
    >>> hybrid_net(torch.zeros(4, 1, 80, 100)).shape
    torch.Size([4, 100, 10])
    """
    def __init__(
            self, cnn_2d, cnn_1d, enc, fcn, pool, *, input_size=None
    ):
        super().__init__()
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self._enc = enc
        self._fcn = fcn
        self._pool = pool
        self.input_size = input_size

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x = self._cnn_2d(x)
            if seq_len is not None:
                in_shape = [
                    (1024 if self.input_size is None else self.input_size, n)
                    for n in seq_len
                ]
                out_shape = self._cnn_2d.get_out_shape(in_shape)
                seq_len = [s[-1] for s in out_shape]

        if x.dim() != 3:
            assert x.dim() == 4
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x = self._cnn_1d(x)
            if seq_len is not None:
                seq_len = self._cnn_1d.get_out_shape(seq_len)
        return x, seq_len

    def enc(self, x, seq_len=None):
        if self._enc is None:
            x = rearrange(x, 'b f t -> b t f')
        elif isinstance(self._enc, nn.RNNBase):
            if self._enc.batch_first:
                x = rearrange(x, 'b f t -> b t f')
            else:
                x = rearrange(x, 'b f t -> t b f')
            if seq_len is not None:
                x = pack_padded_sequence(
                    x, seq_len, batch_first=self._enc.batch_first
                )
            x, _ = self._enc(x)
            if seq_len is not None:
                x = pad_packed_sequence(x, batch_first=self._enc.batch_first)[0]
            if not self._enc.batch_first:
                x = rearrange(x, 't b f -> b t f')
        # elif isinstance(self.enc, Transformer):
        #     x = self.enc(x.transpose(1, 2))
        else:
            raise NotImplementedError
        return x

    def out(self, x, seq_len=None):
        if self._fcn is not None:
            x = self._fcn(x)
        if self._pool is not None:
            x = self._pool(x, seq_len)
        return x

    def forward(self, x, seq_len=None):
        x, seq_len = self.cnn_2d(x, seq_len)
        x, seq_len = self.cnn_1d(x, seq_len)
        x = self.enc(x, seq_len)
        x = self.out(x, seq_len)
        return x

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['cnn_2d'] = {'factory': CNN2d}
        config['cnn_1d'] = {'factory': CNN1d}
        config['enc'] = {'factory': nn.GRU}
        config['fcn'] = {'factory': fully_connected_stack}
        input_size = config['input_size']
        if config['cnn_2d'] is not None and input_size is not None:
            cnn_2d = config['cnn_2d']['factory'].from_config(config['cnn_2d'])
            output_size = cnn_2d.get_out_shape((input_size, 1000))[0]
            out_channels = cnn_2d.out_channels \
                if cnn_2d.out_channels is not None \
                else cnn_2d.hidden_channels[-1]
            input_size = out_channels * output_size

        if config['cnn_1d'] is not None and input_size is not None:
            config['cnn_1d']['in_channels'] = input_size
            input_size = config['cnn_1d']['out_channels'] \
                if config['cnn_1d']['out_channels'] is not None \
                else to_list(config['cnn_1d']['hidden_channels'])[-1]

        if config['enc'] is not None:
            if config['enc']['factory'] == nn.GRU:
                config['enc'].update({
                    'num_layers': 1,
                    'bias': True,
                    'batch_first': True,
                    'dropout': 0.,
                    'bidirectional': False
                })
            elif config['enc']['factory'] == Transformer:
                config['enc']['norm'] = config['cnn']['norm']

            if input_size is not None:
                config['enc']['input_size'] = input_size

        if config['fcn'] is not None:
            config['fcn']['input_size'] = config['enc']['hidden_size']
        config['pool'] = None


class MultinomialClassifier(HybridNet, Model):
    """
    >>> hybrid_cls = MultinomialClassifier.from_config(MultinomialClassifier.get_config({\
        'input_size': 80,\
        'feature_key': 'features',\
        'label_key': 'labels',\
        'cnn_2d': {\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'cnn_1d': {\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'enc': {'hidden_size': 64},\
        'fcn': {'hidden_size': 32, 'output_size': 10},\
        'pool': {'factory': TakeLast, 'n': 3}\
    }))
    >>> inputs = {'features': torch.zeros(4, 1, 80, 100), 'labels': torch.zeros(4).long(), 'seq_len': None}
    >>> logits = hybrid_cls(inputs)
    >>> logits.shape
    torch.Size([4, 3, 10])
    >>> review = hybrid_cls.review(inputs, logits)
    """
    def __init__(
            self, cnn_2d, cnn_1d, enc, fcn, pool, *,
            feature_key, label_key, input_size=None
    ):
        super().__init__(cnn_2d, cnn_1d, enc, fcn, pool, input_size=input_size)
        self.feature_key = feature_key
        self.label_key = label_key

    def forward(self, inputs):
        return super().forward(
            inputs[self.feature_key],
            inputs['seq_len']
        )

    def review(self, inputs, outputs):
        labels = inputs[self.label_key]
        logits = outputs
        if logits.dim() == 3:
            if labels.dim() == 1:
                labels = labels.unsqueeze(1).expand(logits.shape[:-1])
            logits = outputs.contiguous().view((-1, logits.shape[-1]))
            labels = labels.contiguous().flatten()
        ce = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
        summary = dict(
            loss=ce.mean(),
            scalars=dict(
                labels=labels,
                predictions=torch.argmax(logits, dim=-1)
            ),
            images=dict(
                features=inputs[self.feature_key][:3]
            ),
            histograms=dict(labels=labels)
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
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary


class BinaryClassifier(HybridNet, Model):
    """
    >>> hybrid_cls = BinaryClassifier.from_config(BinaryClassifier.get_config({\
        'input_size': 80,\
        'feature_key': 'features',\
        'label_key': 'labels',\
        'cnn_2d': {\
            'in_channels': 1,\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'cnn_1d': {\
            'hidden_channels': 32,\
            'num_layers': 3,\
            'out_channels': 16,\
            'kernel_size': 3\
        },\
        'enc': {'hidden_size': 64},\
        'fcn': {'hidden_size': 32, 'output_size': 10},\
        'pool': {'factory': TakeLast, 'n': 1}\
    }))
    >>> inputs = {'features': torch.zeros(4, 1, 80, 100), 'labels': torch.zeros(4, 10), 'seq_len': None}
    >>> logits = hybrid_cls(inputs)
    >>> logits.shape
    torch.Size([4, 1, 10])
    >>> review = hybrid_cls.review(inputs, logits)
    """
    def __init__(
            self, cnn_2d, cnn_1d, enc, fcn, pool, output_activation, *,
            feature_key, label_key, input_size=None, decision_boundary=.5
    ):
        super().__init__(cnn_2d, cnn_1d, enc, fcn, pool, input_size=input_size)
        self.output_activation = output_activation
        self.feature_key = feature_key
        self.label_key = label_key
        self.decision_boundary = decision_boundary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        super().finalize_dogmatic_config(config)
        config['output_activation'] = {'factory': nn.Sigmoid}  # {'factory': nn.Softmax, 'dim': 1}

    def forward(self, inputs):
        y = super().forward(
            inputs[self.feature_key],
            inputs['seq_len']
        )
        if self.output_activation is not None:
            y = self.output_activation(y)
        return y

    def review(self, inputs, outputs):
        # compute loss
        targets = inputs[self.label_key]
        if outputs.dim() == 3:  # (B, T, K)
            if targets.dim() == 2:   # (B, K)
                targets = targets.unsqueeze(1).expand(outputs.shape)
            outputs = outputs.contiguous().view((-1, outputs.shape[-1]))
            targets = targets.contiguous().view((-1, targets.shape[-1]))
        bce = nn.BCELoss(reduction='none')(outputs, targets).sum(-1)

        # create review including metrics and visualizations
        labels, label_ranked_precisions = positive_class_precisions(
            targets.cpu().data.numpy(),
            outputs.cpu().data.numpy()
        )
        review = dict(
            loss=bce.mean(),
            scalars=dict(
                labels=labels,
                label_ranked_precisions=label_ranked_precisions
            ),
            images=dict(
                features=inputs[self.feature_key][:3]
            )
        )
        for boundary in to_list(self.decision_boundary):
            decision = (outputs.detach() > boundary).float()
            true_pos = (decision * targets).sum()
            false_pos = (decision * (1.-targets)).sum()
            false_neg = ((1.-decision) * targets).sum()
            review['scalars'].update({
                f'true_pos_{boundary}': true_pos,
                f'false_pos_{boundary}': false_pos,
                f'false_neg_{boundary}': false_neg
            })
        return review

    def modify_summary(self, summary):
        # compute lwlrap
        if all([
            key in summary['scalars']
            for key in ['labels', 'label_ranked_precisions']
        ]):
            labels = summary['scalars'].pop('labels')
            label_ranked_precisions = summary['scalars'].pop(
                'label_ranked_precisions'
            )
            summary['scalars']['lwlrap'] = lwlrap_from_precisions(
                label_ranked_precisions, labels
            )[0]

        # compute precision, recall and fscore for each decision boundary
        for boundary in to_list(self.decision_boundary):
            true_pos_key = f'true_pos_{boundary}'
            false_pos_key = f'false_pos_{boundary}'
            false_neg_key = f'false_neg_{boundary}'
            if all([
                key in summary['scalars']
                for key in [true_pos_key, false_pos_key, false_neg_key]
            ]):
                tp = np.sum(summary['scalars'].pop(true_pos_key))
                fp = np.sum(summary['scalars'].pop(false_pos_key))
                fn = np.sum(summary['scalars'].pop(false_neg_key))
                p = tp/(tp+fp)
                r = tp/(tp+fn)
                summary['scalars'][f'precision_{boundary}'] = p
                summary['scalars'][f'recall_{boundary}'] = r
                summary['scalars'][f'f1_{boundary}'] = 2*(p*r)/(p+r)

        summary = super().modify_summary(summary)
        for key, image in summary['images'].items():
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary


class AvgPool(nn.Module):
    def __call__(self, x, seq_len=None):
        x = x.mean(1)
        return x


class MaxPool(nn.Module):
    def __call__(self, x, seq_len=None):
        x = x.max(1)
        return x


class TakeLast(nn.Module):
    def __init__(self, n, r=0.1):
        super(TakeLast, self).__init__()
        self.n = n
        self.r = r

    def __call__(self, x, seq_len=None):
        n = self.n if self.training else 1
        if seq_len is None:
            x = x[:, -n:]
        elif n == 1:
            x = x[torch.arange(x.shape[0]), seq_len - 1].unsqueeze(1)
        else:
            b, t, f = x.shape
            seq_len = np.array(seq_len)[..., None]
            n = max(int(min(self.n, min(self.r * seq_len))), 1)
            seq_len = torch.Tensor(seq_len)
            idx = torch.cumsum(torch.ones((b, t)), dim=1) - 1
            idx = (idx < seq_len) * (idx >= (seq_len - n))
            x = x.masked_select(idx.unsqueeze(-1).to(x.device)).view((b, n, f))
        return x
