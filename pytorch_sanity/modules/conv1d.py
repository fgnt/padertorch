import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

from pytorch_sanity.base import Module
from pytorch_sanity.utils import to_list
from pytorch_sanity.mapping import ACTIVATION_FN_MAP


class Pad(Module):
    def __init__(
            self, kernel_size=None, stride=1, side='both', mode='constant'
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.side = side
        self.mode = mode

    def forward(self, x):
        """
        expects time axis to be on the last dim
        :param x:
        :return:
        """
        k = self.kernel_size - 1
        if k > 0:
            tail = (x.shape[-1] - 1) % self.stride
            if self.side == 'front':
                pad = (k - tail, 0)
            elif self.side == 'both':
                pad = ((k - tail) // 2, math.ceil((k - tail) / 2))
            elif self.side == 'end':
                pad = (0, k - tail)
            else:
                raise ValueError
            x = F.pad(x, pad, mode=self.mode)
        return x


class Cut(Module):
    def __init__(self, side='both'):
        super().__init__()
        self.mode = side

    def forward(self, x, size):
        if size > 0:
            if self.mode == 'front':
                x = x[..., size:]
            elif self.mode == 'both':
                x = x[..., size//2: -math.ceil(size / 2)]
            elif self.mode == 'end':
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
        if self.padding:
            stride = int(np.ceil((x.shape[-1] - 1) / (size - 1e-10)))
            assert stride <= (x.shape[-1] - 1) / (size - 1)
            x = Pad(
                kernel_size=stride, stride=stride,
                side=self.padding, mode="replicate"
            )(x)
        else:
            stride = x.shape[-1] // size
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


class MaxPool1d(Module):
    def __init__(self, kernel_size, padding='both'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        if self.kernel_size < 2:
            return x, None
        x = Pad(
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            side=self.padding
        )(x)
        return nn.MaxPool1d(
            kernel_size=self.kernel_size, return_indices=True
        )(x)


class MaxUnpool1d(Module):
    def __init__(self, kernel_size=None, padding='both'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, indices):
        if self.kernel_size < 2:
            return x
        x = Cut(side=self.padding)(x, size=(x.shape[-1] - indices.shape[-1]))
        return nn.MaxUnpool1d(kernel_size=self.kernel_size)(x, indices=indices)


class Conv1d(Module):
    def __init__(
            self, in_channels=None, out_channels=None, condition_channels=0,
            kernel_size=5, dilation=1, stride=1, transpose=False,
            padding='both', bias=True, groups=1, batch_norm=False, dropout=0.,
            activation='leaky_relu', gated=False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.transpose = transpose
        self.padding = padding
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = ACTIVATION_FN_MAP[activation]
        self.gated = gated

        if batch_norm:
            self.bn = nn.BatchNorm1d(in_channels)
        conv_cls = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = conv_cls(
            in_channels + condition_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias, groups=groups)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)
        if self.gated:
            self.gate_conv = conv_cls(
                in_channels + condition_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias, groups=groups)
            torch.nn.init.xavier_uniform_(self.gate_conv.weight)
            if bias:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x, h=None):
        if self.batch_norm:
            x = self.bn(x)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)

        if h is not None:
            x = torch.cat(
                (x, Scale(padding=self.padding)(h, x.shape[-1])), dim=1)

        if not self.transpose:
            x = Pad(
                kernel_size=1 + self.dilation * (self.kernel_size - 1),
                stride=self.stride,
                side=self.padding
            )(x)

        y = self.conv(x)
        if self.activation is not None:
            y = self.activation(y)

        if self.gated:
            g = self.gate_conv(x)
            y = y * torch.sigmoid(g)

        if self.transpose:
            k = 1 + self.dilation * (self.kernel_size - 1)
            y = Cut(side=self.padding)(y, size=k - self.stride)
        return y


class TCN(Module):
    def __init__(
            self, input_dim=None, output_dim=None, hidden_dim=256,
            condition_dim=0, depth=5, kernel_sizes=3, dilations=1, strides=1,
            transpose=False, pool_sizes=1, padding='both', batch_norm=False,
            dropout=0., activation='leaky_relu', gated=False, groups=1
    ):
        super().__init__()

        self.in_channels = input_dim
        self.hidden_channels = hidden_dim
        self.out_channels = output_dim
        self.depth = depth
        self.kernel_sizes = to_list(kernel_sizes, depth)
        self.dilations = to_list(dilations, depth)
        self.strides = to_list(strides, depth)
        self.pool_sizes = to_list(pool_sizes, depth)
        self.transpose = transpose
        self.padding = padding

        batch_norm_ = False
        convs = list()
        for i in range(depth):
            if i == depth - 1:
                activation = 'linear'
                hidden_dim = output_dim
            convs.append(Conv1d(
                in_channels=input_dim, out_channels=hidden_dim,
                condition_channels=condition_dim,
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i],
                stride=self.strides[i], transpose=transpose, padding=padding,
                batch_norm=batch_norm_, dropout=dropout, activation=activation,
                gated=gated, groups=groups
            ))
            batch_norm_ = batch_norm
            input_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

    def forward(self, x, h=None, pool_indices=None):
        pool_indices = to_list(pool_indices, self.depth)
        for i, conv in enumerate(self.convs):
            pool_size = self.pool_sizes[i]
            if self.transpose:
                pool = MaxUnpool1d(kernel_size=pool_size, padding=self.padding)
                x = pool(x, indices=pool_indices[i])
            x = conv(x, h)
            if not self.transpose:
                pool = MaxPool1d(kernel_size=pool_size, padding=self.padding)
                x, pool_indices[i] = pool(x)
        if self.transpose:
            return x
        return x, pool_indices


class MultiScaleConv1d(Module):
    def __init__(
            self, in_channels=None, out_channels=None, condition_channels=0,
            kernel_size=3, n_scales=1, dilated=False, stride=1, transpose=False,
            padding='both', batch_norm=False, dropout=0.,
            activation='leaky_relu', gated=False
    ):
        assert out_channels % n_scales == 0
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.stride = stride
        self.transpose = transpose
        self.padding = padding
        self.condition_channels = condition_channels
        self.activation = activation
        self.gated = gated
        self.dropout = dropout

        if dilated:
            kernel_sizes = n_scales * [kernel_size]
            dilations = [2 ** i for i in range(n_scales)]
        else:
            kernel_sizes = [
                1 + (kernel_size - 1) * 2**i for i in range(n_scales)
            ]
            dilations = n_scales * [1]
        self.convs = nn.ModuleList([
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels // n_scales,
                condition_channels=condition_channels,
                kernel_size=kernel_sizes[i], dilation=dilations[i],
                stride=stride, transpose=transpose, padding=padding,
                batch_norm=batch_norm, dropout=dropout, activation=activation,
                gated=gated
            )
            for i in range(n_scales)
        ])

    def forward(self, x, h=None):
        return torch.cat([conv(x, h) for conv in self.convs], dim=1)


class MSTCN(Module):
    def __init__(
            self, input_dim=None, output_dim=None, hidden_dim=256,
            condition_dim=0, depth=5, kernel_sizes=3, n_scales=1, dilated=False,
            strides=1, transpose=False, pool_sizes=1, padding='both',
            batch_norm=False, dropout=0., activation='leaky_relu', gated=False
    ):
        super().__init__()

        self.in_channels = input_dim
        self.hidden_channels = hidden_dim
        self.out_channels = output_dim
        self.depth = depth
        self.kernel_sizes = to_list(kernel_sizes, depth)
        self.n_scales = to_list(n_scales, depth - 1)
        self.strides = to_list(strides, depth)
        self.pool_sizes = to_list(pool_sizes, depth)
        self.transpose = transpose
        self.padding = padding

        batch_norm_ = False
        convs = list()
        for i in range(depth):
            if i == depth - 1:
                activation = 'linear'
                hidden_dim = output_dim
                n_scales = 1
            else:
                n_scales = self.n_scales[i]
            convs.append(MultiScaleConv1d(
                in_channels=input_dim, out_channels=hidden_dim,
                condition_channels=condition_dim,
                kernel_size=self.kernel_sizes[i], n_scales=n_scales,
                dilated=dilated, stride=self.strides[i], transpose=transpose,
                padding=padding, batch_norm=batch_norm_, dropout=dropout,
                activation=activation, gated=gated
            ))
            batch_norm_ = batch_norm
            input_dim = hidden_dim
        self.convs = nn.ModuleList(convs)

    def forward(self, x, h=None, pool_indices=None):
        pool_indices = to_list(pool_indices, self.depth)
        for i, conv in enumerate(self.convs):
            pool_size = self.pool_sizes[i]
            if self.transpose:
                pool = MaxUnpool1d(kernel_size=pool_size, padding=self.padding)
                x = pool(x, indices=pool_indices[i])
            x = conv(x, h)
            if not self.transpose:
                pool = MaxPool1d(kernel_size=pool_size, padding=self.padding)
                x, pool_indices[i] = pool(x)
        if self.transpose:
            return x
        return x, pool_indices
