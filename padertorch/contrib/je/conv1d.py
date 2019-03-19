import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn


class Pad(Module):
    def __init__(self, side='both', mode='constant'):
        super().__init__()
        self.side = side
        self.mode = mode

    def forward(self, x, size):
        """
        expects time axis to be on the last dim
        :param x:
        :return:
        """
        if self.side == 'front':
            pad = (size, 0)
        elif self.side == 'both':
            pad = (size // 2, math.ceil(size / 2))
        elif self.side == 'end':
            pad = (0, size)
        else:
            raise ValueError
        x = F.pad(x, pad, mode=self.mode)
        return x

    @staticmethod
    def get_size(nframes, kernel_size, stride):
        return kernel_size - 1 - ((nframes - 1) % stride)


class Cut(Module):
    def __init__(self, side='both'):
        super().__init__()
        self.side = side

    def forward(self, x, size):
        if size > 0:
            if self.side == 'front':
                x = x[..., size:]
            elif self.side == 'both':
                x = x[..., size//2: -math.ceil(size / 2)]
            elif self.side == 'end':
                x = x[..., :-size]
            else:
                raise ValueError
        return x


class Scale(Module):
    """
    >>> print(Scale()(torch.Tensor(np.arange(10)).view(1,1,10), 5))
    tensor([[[0.5000, 2.5000, 4.5000, 6.5000, 8.5000]]])
    >>> print(Scale(padding='front')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[0.2500, 3.5000, 7.5000]]])
    >>> print(Scale(padding='both')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[0.7500, 4.5000, 8.2500]]])
    >>> print(Scale(padding='end')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[1.5000, 5.5000, 8.7500]]])
    >>> print(Scale(padding=None)(torch.Tensor(np.arange(10)).view(1,1,10), 6))
    tensor([[[4., 5., 6., 7., 8., 9.]]])
    """
    def __init__(self, padding='both'):
        super().__init__()
        self.padding = padding

    def forward(self, x, size):
        if size == 1:
            return x.mean(dim=-1, keepdim=True)
        if self.padding is None:
            stride = x.shape[-1] // size
        else:
            stride = int(np.ceil((x.shape[-1] - 1) / (size - 1e-10)))
            assert stride <= (x.shape[-1] - 1) / (size - 1)
            x = Pad(side=self.padding, mode="replicate")(
                x, size=(stride - 1 - ((x.shape[-1] - 1) % stride)))
        if 1 < stride:
            x = F.avg_pool1d(x, stride)
        if x.shape[-1] > size:
            # Cut front because no padding was used
            # Is this the right behavior?
            x = Cut(side='front')(x, x.shape[-1] - size)
        if x.shape[-1] < size:
            x = F.interpolate(x, size, mode='linear')
        assert x.shape[-1] == size
        return x


class Pool1d(Module):
    def __init__(self, kernel_size, pooling='max', padding='both'):
        super().__init__()
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.padding = padding

    def forward(self, x):
        if self.kernel_size < 2:
            return x, (None, None)
        pad_size = self.kernel_size - 1 - ((x.shape[-1] - 1) % self.kernel_size)
        x = Pad(side=self.padding)(x, size=pad_size)
        if self.pooling == 'max':
            x, pool_indices = nn.MaxPool1d(
                kernel_size=self.kernel_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = nn.AvgPool1d(kernel_size=self.kernel_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, (pool_indices, pad_size)


class Unpool1d(Module):
    def __init__(self, kernel_size, padding='end'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, indices=None, pad_size=None):
        if self.kernel_size < 2:
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.kernel_size)
        else:
            x = nn.MaxUnpool1d(kernel_size=self.kernel_size)(
                x, indices=indices
            )
        if pad_size is not None:
            x = Cut(side=self.padding)(x, size=pad_size)
        return x


class Conv1d(Module):
    """
    Wrapper for torch.nn.Conv1d or torch.nn.ConvTransose1d, adds additonal
    options of applying an activation, using a gated convolution
    or normalizing the network output
    """
    def __init__(
            self, input_size, output_size, condition_size=0, kernel_size=5, # ToDo: why kernel_size=5 and not keep it without default?
            dilation=1, stride=1, transpose=False, padding='both', bias=True,
            groups=1, dropout=0., activation='leaky_relu', gated=False,
            norm=None
    ):
        """

        Args:
            input_size:
            output_size:
            condition_size:
            kernel_size:
            dilation:
            stride:
            transpose: if true uses ConvTransose1d (fractionally-strided convolution)
                    else: used Conv1d
            padding:
            bias:
            groups:
            dropout:
            activation:
            gated:
            norm: may be None or 'batch'
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.transpose = transpose
        self.padding = padding
        self.dropout = dropout
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.gated = gated

        conv_cls = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = conv_cls(
            input_size + condition_size, output_size,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias, groups=groups)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)

        # ToDo: do you need a batchnorm during the conv layer?
        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(output_size)
        else:
            raise ValueError(f'{norm} normalization  not known.')
        if self.gated:
            self.gate_conv = conv_cls(
                input_size + condition_size, output_size,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias, groups=groups)
            # ToDo: why this specific initialization?
            torch.nn.init.xavier_uniform_(self.gate_conv.weight)
            if bias:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x, h=None):
        # ToDo: what does h stand for?

        x_ = x
        if self.training and self.dropout > 0.:
            x_ = F.dropout(x_, self.dropout)

        if h is not None:
            x_ = torch.cat(
                (x_, Scale(padding=self.padding)(h, x_.shape[-1])), dim=1)

        if self.padding and not self.transpose:
            x_ = Pad(side=self.padding)(
                x_, size=((1 + self.dilation * (self.kernel_size - 1))
                          - 1 - ((x_.shape[-1] - 1) % self.stride)))

        y = self.conv(x_)
        # ToDo: why normalization after network and not before?
        if self.norm is not None:
            y = self.norm(y)
        y = self.activation(y)

        # ToDo: Does this still make sense if you are using transpose?
        if self.gated:
            g = self.gate_conv(x_)
            y = y * torch.sigmoid(g)

        if self.padding and self.transpose:
            k = 1 + self.dilation * (self.kernel_size - 1)
            y = Cut(side=self.padding)(y, size=k - self.stride)

        return y


class MultiScaleConv1d(Module):
    def __init__(
            self, input_size, hidden_size, output_size, condition_size=0,
            kernel_size=3, n_scales=1, dilation=1, stride=1, transpose=False,
            padding='both', dropout=0., activation='leaky_relu', gated=False,
            residual=False, norm=None
    ):
        assert hidden_size % n_scales == 0, (hidden_size, n_scales)
        super().__init__()

        # kernel_sizes = n_scales * [kernel_size]
        # dilations = [2 ** i for i in range(n_scales)]

        self.kernel_sizes = [
            1 + (kernel_size - 1) * 2**i for i in range(n_scales)
        ]
        dilations = n_scales * [dilation]

        self.convs = nn.ModuleList([
            Conv1d(
                input_size=input_size, output_size=hidden_size // n_scales,
                condition_size=condition_size, kernel_size=self.kernel_sizes[i],
                dilation=dilations[i], stride=stride, transpose=transpose,
                padding=padding, dropout=dropout, activation=activation,
                gated=gated, norm=None
            )
            for i in range(n_scales)
        ])
        self.out = Conv1d(
            input_size=hidden_size, output_size=output_size, kernel_size=1,
            activation='identity', norm=None
        )

        self.residual = residual
        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(output_size)
        else:
            raise ValueError(f'{norm} normalization not known.')

    def forward(self, x, h=None):
        y = [conv(x, h) for conv in self.convs]
        tails = [y_.shape[-1] - y[-1].shape[-1] for y_ in y]
        y = [
            Cut(side='both')(y_, size=tail) if tail >= 0
            else Pad(side='both')(y_, size=-tail)
            for y_, tail in zip(y, tails)
        ]
        y = self.out(torch.cat(y, dim=1))
        if self.residual and y.shape == x.shape:
            y = y + x
        if self.norm is not None:
            y = self.norm(y)
        return y


class TCN(Module):
    """
    Multi-Scale Temporal Convolutional Network
    """
    def __init__(
            self, input_size, output_size, hidden_sizes=256, condition_size=0,
            num_layers=5, kernel_sizes=3, n_scales=None, dilations=1,
            strides=1, transpose=False, pooling='max', pool_sizes=1,
            paddings='both', dropout=0., activation='leaky_relu', gated=False,
            residual=False, norm=None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = to_list(
            hidden_sizes, num_layers - int(n_scales is None)
        )
        self.output_size = output_size
        self.condition_size = condition_size
        self.num_layers = num_layers
        self.kernel_sizes = to_list(kernel_sizes, num_layers)
        self.n_scales = None if n_scales is None else to_list(
            n_scales, num_layers)
        self.dilations = to_list(dilations, num_layers)
        self.strides = to_list(strides, num_layers)
        self.pooling = pooling
        self.pool_sizes = to_list(pool_sizes, num_layers)
        self.transpose = transpose
        self.paddings = to_list(paddings, num_layers)
        self.residual = residual

        convs = list()
        for i in range(num_layers):
            if n_scales is None:
                assert residual is False
                if i == num_layers - 1:
                    output_size_ = output_size
                    norm = None
                    activation = 'identity'
                else:
                    output_size_ = self.hidden_sizes[i]
                convs.append(Conv1d(
                    input_size=input_size, output_size=output_size_,
                    condition_size=condition_size,
                    kernel_size=self.kernel_sizes[i],
                    dilation=self.dilations[i],
                    stride=self.strides[i], transpose=transpose,
                    padding=self.paddings[i], norm=norm, dropout=dropout,
                    activation=activation, gated=gated
                ))
            else:
                hidden_size = self.hidden_sizes[i]
                if i == num_layers - 1:
                    output_size_ = output_size
                    norm = None
                else:
                    output_size_ = hidden_size
                convs.append(MultiScaleConv1d(
                    input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size_, condition_size=condition_size,
                    kernel_size=self.kernel_sizes[i],
                    n_scales=self.n_scales[i], dilation=self.dilations[i],
                    stride=self.strides[i], transpose=transpose,
                    padding=self.paddings[i], dropout=dropout,
                    activation=activation, gated=gated, residual=residual,
                    norm=norm
                ))
            input_size = output_size_
        self.convs = nn.ModuleList(convs)

    def forward(self, x, h=None, pooling_data=None):
        pooling_data = to_list(pooling_data, self.num_layers)
        for i, conv in enumerate(self.convs):
            pool_size = self.pool_sizes[i]
            if self.transpose:
                unpool = Unpool1d(kernel_size=pool_size, padding='end')
                indices, pad_size = (None, None) if pooling_data[i] is None \
                    else pooling_data[i]
                x = unpool(
                    x, indices=indices, pad_size=pad_size
                )
            x = conv(x, h)
            if not self.transpose:
                pool = Pool1d(
                    kernel_size=pool_size, pooling=self.pooling, padding='end'
                )
                x, pooling_data[i] = pool(x)
        if self.transpose:
            return x
        return x, pooling_data
