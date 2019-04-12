import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn


class CNN(Module):
    """
    (Multi-Scale) Convolutional Neural Network
    ToDo: allow hybrid CNN combining 2d and 1d Convs
    ToDo: allow 2d kernels, scales, dilations, strides, pooling, padding
    """
    def __init__(
            self, input_size, output_size, kernel_sizes, hidden_sizes,
            num_layers, ndim=1, n_scales=None, dilations=1, strides=1,
            transpose=False, pooling='max', pool_sizes=1, paddings='both',
            dropout=0., activation='leaky_relu', gated=False, residual=False,
            norm=None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = to_list(
            hidden_sizes, num_layers - int(n_scales is None)
        )
        self.output_size = output_size
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
                convs.append(Conv(
                    input_size=input_size, output_size=output_size_,
                    kernel_size=self.kernel_sizes[i], ndim=ndim,
                    dilation=self.dilations[i], stride=self.strides[i],
                    transpose=transpose, padding=self.paddings[i], norm=norm,
                    dropout=dropout, activation=activation, gated=gated
                ))
            else:
                hidden_size = self.hidden_sizes[i]
                if i == num_layers - 1:
                    output_size_ = output_size
                    norm = None
                else:
                    output_size_ = hidden_size
                convs.append(MultiScaleConv(
                    input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size_, kernel_size=self.kernel_sizes[i],
                    ndim=ndim, n_scales=self.n_scales[i],
                    dilation=self.dilations[i], stride=self.strides[i],
                    transpose=transpose, padding=self.paddings[i],
                    dropout=dropout, activation=activation, gated=gated,
                    residual=residual, norm=norm
                ))
            input_size = output_size_
        self.convs = nn.ModuleList(convs)

    def forward(self, x, pooling_data=None):
        pooling_data = to_list(pooling_data, self.num_layers)
        for i, conv in enumerate(self.convs):
            pool_size = self.pool_sizes[i]
            if self.transpose:
                unpool = Unpool(kernel_size=pool_size, padding='end')
                indices, pad_size = (None, None) if pooling_data[i] is None \
                    else pooling_data[i]
                x = unpool(
                    x, indices=indices, pad_size=pad_size
                )
            x = conv(x)
            if not self.transpose:
                pool = Pool(
                    kernel_size=pool_size, pooling=self.pooling, padding='end'
                )
                x, pooling_data[i] = pool(x)
        if self.transpose:
            return x
        return x, pooling_data


class Conv(Module):
    """
    Wrapper for torch.nn.ConvXd and torch.nn.ConvTransoseXd for X in {1,2}
    including additional options of applying an (gated) activation or
    normalizing the network output
    """
    def __init__(
            self, input_size, output_size, kernel_size, ndim=1,
            dilation=1, stride=1, transpose=False, padding='both', bias=True,
            groups=1, dropout=0., activation='leaky_relu', gated=False,
            norm=None
    ):
        """

        Args:
            input_size:
            output_size:
            kernel_size:
            ndim: if 1 using 1d Modules else 2d Modules
            dilation:
            stride:
            transpose: if true uses ConvTransoseXd else ConvXd
            padding:
            bias:
            groups:
            dropout:
            activation:
            gated:
            norm: may be None or 'batch'
        """
        super().__init__()
        self.kernel_size = to_list(kernel_size, ndim)
        self.dilation = to_list(dilation, ndim)
        self.stride = to_list(stride, ndim)
        self.padding = None if padding is None else to_list(padding, ndim)
        self.transpose = transpose
        self.dropout = dropout
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.gated = gated

        if ndim == 1:
            conv_cls = nn.ConvTranspose1d if transpose else nn.Conv1d
        elif ndim == 2:
            conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
        else:
            raise ValueError
        self.conv = conv_cls(
            input_size, output_size,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias, groups=groups
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)

        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(output_size) if ndim == 1 \
                else nn.BatchNorm2d(output_size)
        else:
            raise ValueError(f'{norm} normalization  not known.')
        if self.gated:
            self.gate_conv = conv_cls(
                input_size, output_size,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias, groups=groups)
            torch.nn.init.xavier_uniform_(self.gate_conv.weight)
            if bias:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x):
        x_ = x
        if self.training and self.dropout > 0.:
            x_ = F.dropout(x_, self.dropout)

        if self.padding is not None and not self.transpose:
            sizes = [
                (1 + self.dilation[i] * (self.kernel_size[i] - 1))
                - ((x_.shape[2 + i] - 1) % self.stride[i]) - 1
                for i in range(len(self.kernel_size))
            ]
            x_ = Pad(side=self.padding)(x_, size=sizes)

        y = self.conv(x_)
        if self.norm is not None:
            y = self.norm(y)
        y = self.activation(y)

        if self.gated:
            g = self.gate_conv(x_)
            y = y * torch.sigmoid(g)

        if self.padding is not None and self.transpose:
            sizes = [
                1 + self.dilation[i] * (self.kernel_size[i] - 1) - self.stride[i]
                for i in range(len(self.kernel_size))
            ]
            y = Cut(side=self.padding)(y, size=sizes)
        return y


class MultiScaleConv(Module):
    """
    Concatenates outputs of multiple convolutions with different kernel sizes,
    followed by an output layer
    """
    def __init__(
            self, input_size, hidden_size, output_size, ndim=1,
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
            Conv(
                input_size=input_size, output_size=hidden_size // n_scales,
                ndim=ndim, kernel_size=self.kernel_sizes[i],
                dilation=dilations[i], stride=stride, transpose=transpose,
                padding=padding, dropout=dropout, activation=activation,
                gated=gated, norm=None
            )
            for i in range(n_scales)
        ])
        self.out = Conv(
            input_size=hidden_size, output_size=output_size, ndim=ndim,
            kernel_size=1, activation='identity', norm=None
        )

        self.residual = residual
        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(output_size) if ndim == 1 \
                else nn.BatchNorm2d(output_size)
        else:
            raise ValueError(f'{norm} normalization not known.')

    def forward(self, x):
        y = [conv(x) for conv in self.convs]
        tails = [
            [y_.shape[2 + i] - y[-1].shape[2 + i] for i in range(x.dim() - 2)]
            for y_ in y
        ]
        y = [
            Cut(side='both')(y_, size=tail) if tail[0] >= 0
            else Pad(side='both')(y_, size=[-t for t in tail])
            for y_, tail in zip(y, tails)
        ]
        y = self.out(torch.cat(y, dim=1))
        if self.residual and y.shape == x.shape:
            y = y + x
        if self.norm is not None:
            y = self.norm(y)
        return y


class Pad(Module):
    def __init__(self, side='both', mode='constant'):
        super().__init__()
        self.side = side
        self.mode = mode

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        pad = []
        for side, size in list(zip(sides, sizes))[::-1]:
            if side == 'front':
                pad.extend([size, 0])
            elif side == 'both':
                pad.extend([size // 2, math.ceil(size / 2)])
            elif side == 'end':
                pad.extend([0, size])
            else:
                raise ValueError(f'pad side {side} unknown')

        x = F.pad(x, tuple(pad), mode=self.mode)
        return x

    @staticmethod
    def get_size(nframes, kernel_size, stride):
        return kernel_size - 1 - ((nframes - 1) % stride)


class Cut(Module):
    def __init__(self, side='both'):
        super().__init__()
        self.side = side

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        slc = [slice(None)] * x.dim()
        for i, (side, size) in enumerate(zip(sides, sizes)):
            if size > 0:
                idx = 2 + i
                if side == 'front':
                    slc[idx] = slice(size, x.shape[idx])
                elif side == 'both':
                    slc[idx] = slice(size//2, -math.ceil(size / 2))
                elif side == 'end':
                    slc[idx] = slice(0, -size)
                else:
                    raise ValueError
        x = x[tuple(slc)]
        return x


class Pool(Module):
    def __init__(self, kernel_size, pooling='max', padding='both'):
        super().__init__()
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.padding = padding

    def forward(self, x):
        if x.dim() == 3:
            return Pool1d(
                kernel_size=self.kernel_size,
                pooling=self.pooling,
                padding=self.padding
            )(x)
        elif x.dim() == 4:
            return Pool2d(
                kernel_size=self.kernel_size,
                pooling=self.pooling,
                padding=self.padding
            )(x)
        else:
            raise ValueError


class Unpool(Module):
    def __init__(self, kernel_size, padding='both'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, indices=None, pad_size=None):
        if x.dim() == 3:
            return Unpool1d(
                kernel_size=self.kernel_size,
                padding=self.padding
            )(x, indices=indices, pad_size=pad_size)
        elif x.dim() == 4:
            return Unpool2d(
                kernel_size=self.kernel_size,
                padding=self.padding
            )(x, indices=indices, pad_size=pad_size)
        else:
            raise ValueError


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
    def __init__(self, kernel_size, padding='both'):
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


class Pool2d(Module):
    def __init__(self, kernel_size, pooling='max', padding='both'):
        super().__init__()
        self.kernel_size = to_list(kernel_size, 2)
        self.pooling = pooling
        self.padding = padding

    def forward(self, x):
        if all(np.array(self.kernel_size) < 2):
            return x, (None, None)
        pad_size = (
                self.kernel_size[0] - 1 - ((x.shape[-2] - 1) % self.kernel_size[0]),
                self.kernel_size[1] - 1 - ((x.shape[-1] - 1) % self.kernel_size[1])
        )
        x = Pad(side=self.padding)(x, size=pad_size)
        if self.pooling == 'max':
            x, pool_indices = nn.MaxPool2d(
                kernel_size=self.kernel_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = nn.AvgPool2d(kernel_size=self.kernel_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, (pool_indices, pad_size)


class Unpool2d(Module):
    def __init__(self, kernel_size, padding='both'):
        super().__init__()
        self.kernel_size = to_list(kernel_size, 2)
        self.padding = to_list(padding, 2)

    def forward(self, x, indices=None, pad_size=None):
        if all(np.array(self.kernel_size) < 2):
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.kernel_size)
        else:
            x = nn.MaxUnpool2d(kernel_size=self.kernel_size)(
                x, indices=indices
            )
        if pad_size is not None:
            x = Cut(side=self.padding)(x, size=pad_size)
        return x


class Scale1d(Module):
    """
    >>> print(Scale1d()(torch.Tensor(np.arange(10)).view(1,1,10), 5))
    tensor([[[0.5000, 2.5000, 4.5000, 6.5000, 8.5000]]])
    >>> print(Scale1d(padding='front')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[0.2500, 3.5000, 7.5000]]])
    >>> print(Scale1d(padding='both')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[0.7500, 4.5000, 8.2500]]])
    >>> print(Scale1d(padding='end')(torch.Tensor(np.arange(10)).view(1,1,10), 3))
    tensor([[[1.5000, 5.5000, 8.7500]]])
    >>> print(Scale1d(padding=None)(torch.Tensor(np.arange(10)).view(1,1,10), 6))
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
