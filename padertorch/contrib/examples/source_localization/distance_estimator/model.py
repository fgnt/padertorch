from einops import rearrange
import numpy as np
import torch
from torch import nn

from padertorch.base import Model
from padertorch.base import Module
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.ops.mappings import ACTIVATION_FN_MAP


class SamePadding(Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list)
        assert len(kernel_size) in [1, 2]

        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]
            pad_left, pad_right = self.split_padding(kernel_size)
            self.pad = nn.ConstantPad1d((pad_left, pad_right), 0)
        else:
            pad_top, pad_bottom = self.split_padding(kernel_size[0])
            pad_left, pad_right = self.split_padding(kernel_size[1])
            self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    @staticmethod
    def split_padding(kernel_size):
        if kernel_size % 2 == 0:
            pad_left = int(np.floor((kernel_size - 1) / 2))
            pad_right = int(np.ceil((kernel_size - 1) / 2))

        else:
            pad_left = int(kernel_size // 2)
            pad_right = pad_left
        return pad_left, pad_right

    def forward(self, x):
        return self.pad(x)


class Pool(Module):
    def __init__(self, pool_type, kernel_size):
        super().__init__()
        assert pool_type in ['max', 'avg']
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list)
        assert len(kernel_size) in [1, 2]
        if pool_type == 'max':
            if len(kernel_size) == 1:
                self.pool = nn.MaxPool1d(kernel_size)
            elif len(kernel_size) == 2:
                self.pool = nn.MaxPool2d(kernel_size)
        elif pool_type == 'avg':
            if len(kernel_size) == 1:
                self.pool = nn.AvgPool1d(kernel_size)
            elif len(kernel_size) == 2:
                self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        return self.pool(x)


class _Conv(Module):
    conv_cls = None

    def __init__(self, in_chs, out_chs, kernel_size, activation_fn='relu',
                 batch_norm=True, pre_activation=True, padding='same'):
        super().__init__()
        assert padding in ['same', False]
        assert isinstance(kernel_size, tuple) or isinstance(kernel_size, list)
        assert len(kernel_size) in [1, 2]
        self.conv = self.conv_cls(in_chs, out_chs, kernel_size)
        if padding:
            self.pad = SamePadding(kernel_size)
        else:
            self.pad = None
        if activation_fn == 'glu':
            self.activation_fn = 'glu'
            if batch_norm:
                if len(kernel_size) == 1:
                    self.bn_gate = nn.BatchNorm1d(out_chs)
                elif len(kernel_size) == 2:
                    self.bn_gate = nn.BatchNorm2d(out_chs)
            else:
                self.bn_gate = None
            self.conv_gate = self.conv_cls(in_chs, out_chs, kernel_size)
        else:
            self.activation_fn = ACTIVATION_FN_MAP[activation_fn]()
        self.pre_activation = pre_activation
        if batch_norm:
            if len(kernel_size) == 1:
                self.bn = nn.BatchNorm1d(out_chs)
            elif len(kernel_size) == 2:
                self.bn = nn.BatchNorm2d(out_chs)
        else:
            self.bn = None
        self.conv = self.conv_cls(in_chs, out_chs, kernel_size)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)

        y = self.conv(x)
        if self.activation_fn == 'glu':
            g = self.conv_gate(x)
            g = self.bn_gate(g)
            y = self.bn(y)
            y = y * torch.sigmoid(g)
        else:
            if self.pre_activation:
                if self.bn is not None:
                    y = self.bn(y)
            y = self.activation_fn(y)
            if not self.pre_activation:
                if self.bn is not None:
                    y = self.bn(y)
        return y


class Conv1D(_Conv):
    conv_cls = nn.Conv1d


class Conv2D(_Conv):
    conv_cls = nn.Conv2d


class CNN(Module):
    conv_cls = None

    def __init__(self, n_chs_input, n_chs, kernel_sizes, pool_layers,
                 activation_fn='relu', batch_norm=True, pre_activation=True,
                 padding='same', dropout_prob=0.):
        super().__init__()
        assert padding in ['same', False]
        assert len(n_chs) == len(kernel_sizes) == len(pool_layers)
        in_chs = \
            [n_chs[i-1] if i > 0 else n_chs_input for i in range(len(n_chs))]
        conv_layers = []
        for in_ch, out_ch, kernel_size in zip(in_chs, n_chs, kernel_sizes):
            new_layer = self.conv_cls(in_ch, out_ch, kernel_size,
                                      activation_fn, batch_norm,
                                      pre_activation, padding)
            conv_layers.append(new_layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        pool_layers = \
            [Pool(**pool_layer) if pool_layer is not None else None
             for pool_layer in pool_layers]
        self.pool_layers = nn.ModuleList(pool_layers)
        if dropout_prob > 0:
            dropout_layers = [nn.Dropout2d(dropout_prob)
                              for _ in range(len(conv_layers) - 1)]
            dropout_layers += [None]
            self.dropout_layers = nn.ModuleList(dropout_layers)
        else:
            self.dropout_layers = [None] * len(conv_layers)

    def forward(self, x):
        for conv_layer, pool_layer, dropout_layer in \
                zip(self.conv_layers, self.pool_layers, self.dropout_layers):
            x = conv_layer(x)
            if pool_layer is not None:
                x = pool_layer(x)
            if dropout_layer is not None:
                x = dropout_layer(x)
        return x


class CNN1D(CNN):
    conv_cls = Conv1D


class CNN2D(CNN):
    conv_cls = Conv2D


class HybridCNN(Module):
    def __init__(self,
                 cnn_2d: CNN2D,
                 cnn_1d: CNN1D,
                 **kwargs):
        super().__init__()
        self.cnn_2d = cnn_2d
        self.cnn_1d = cnn_1d

    def forward(self, x):
        x = self.cnn_2d(x)
        x = rearrange(x, 'b c f t -> b (c f) t')
        x = self.cnn_1d(x)
        return x

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['cnn_2d'] = {'factory': CNN2D}
        config['cnn_1d'] = {'factory': CNN1D}
        n_freq_bins_reduced = config['n_freq_bins']
        for pool_layer in config['cnn_2d']['pool_layers']:
            if pool_layer is not None:
                n_freq_bins_reduced = np.floor(
                    n_freq_bins_reduced / pool_layer['kernel_size'][0]
                )
        config['cnn_1d']['n_chs_input'] = \
            int(config['cnn_2d']['n_chs'][-1] * n_freq_bins_reduced)


class GRU(Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_prob):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True, dropout=dropout_prob)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x[:, -1, :]


class CRNN(Module):
    def __init__(self,
                 cnn: HybridCNN,
                 gru: GRU,
                 fcn: fully_connected_stack):
        super().__init__()
        self.cnn = cnn
        self.gru = gru
        self.fcn = fcn

    def forward(self, x):
        x = self.cnn(x)
        x = self.gru(x)
        x = self.fcn(x)
        return x

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['cnn'] = {'factory': HybridCNN}
        config['gru'] = {'factory': GRU}
        config['fcn'] = {'factory': fully_connected_stack}
        config['gru']['input_size'] = config['cnn']['cnn_1d']['n_chs'][-1]
        config['fcn']['input_size'] = config['gru']['hidden_size']


class DistanceEstimator(Model):
    def __init__(self, net, num_cls, quant_step=.1, d_min=0):
        """DNN-based distance estimator

        Args:
            net:
                Module implementing the neural network
            num_cls:
                Amount of distance classes
            quant_step:
                Step size of the quantization done to derive distance classes
                from the continuous distance values
            d_min:
                Lower limit of the considered distance range
        """
        super().__init__()
        self.net = net
        self.num_classes = num_cls
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.loss = nn.CrossEntropyLoss()
        self.quant_step = quant_step
        self.d_min = d_min

    def forward(self, inputs):
        # Rund the forward path of the network
        return self.net(inputs['features'])

    def review(self, inputs, outputs):
        # Calculate the loss and some further metrics
        target = inputs['label']
        loss = self.loss(outputs, target)
        cls_probs = nn.Softmax(dim=-1)(outputs)
        est_cls = cls_probs.argmax(dim=-1)
        est_dist = est_cls.float() * self.quant_step + self.d_min
        ae = self.l1_loss(est_dist, inputs['distance'])
        se = self.mse_loss(est_dist, inputs['distance'])
        review_dict = {
            'loss': loss,
            'scalars': {'mae': ae,
                        'rmse': se}
        }
        cls_dict = {'mae': ae,
                    'rmse': se,
                    'target': target,
                    'est_cls': est_cls}
        review_dict['scalars'].update(cls_dict)
        return review_dict

    def modify_summary(self, summary):
        if 'target' in summary['scalars'] and 'est_cls' in summary['scalars']:
            target = np.asarray(summary['scalars'].pop('target'))
            est_cls = np.asarray(summary['scalars'].pop('est_cls'))
            target_min_1 = target - 1
            target_pl_1 = target + 1
            acc_allow_neighbors = est_cls == target
            acc_allow_neighbors += est_cls == target_min_1
            acc_allow_neighbors += est_cls == target_pl_1
            summary['scalars']['acc_allow_neighbors'] = acc_allow_neighbors
            acc = est_cls == target
            summary['scalars']['acc'] = acc
        if 'rmse' in summary['scalars']:
            rmse = summary['scalars'].pop('rmse')
            summary['scalars']['rmse'] = np.sqrt(np.mean(rmse))
        super().modify_summary(summary)
        return summary
