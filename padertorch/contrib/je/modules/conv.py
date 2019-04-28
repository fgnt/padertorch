import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn
from copy import copy


class Pad(Module):
    """
    Adds padding of a certain size either to front, end or both.
    ToDo: Exception if side is None but size != 0 (requires adjustments in _Conv)
    """
    def __init__(self, side='both', mode='constant'):
        super().__init__()
        self.side = side
        self.mode = mode

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        pad = []
        for side, size in list(zip(sides, sizes))[::-1]:
            if side is None or size < 1:
                pad.extend([0, 0])
            elif side == 'front':
                pad.extend([size, 0])
            elif side == 'both':
                pad.extend([size // 2, math.ceil(size / 2)])
            elif side == 'end':
                pad.extend([0, size])
            else:
                raise ValueError(f'pad side {side} unknown')

        x = F.pad(x, tuple(pad), mode=self.mode)
        return x


class Cut(Module):
    """
    Removes a certain number of values either from front, end or both.
    (Counter part to Pad)
    ToDo: Exception if side is None but size != 0 (requires adjustments in _Conv)
    """
    def __init__(self, side='both'):
        super().__init__()
        self.side = side

    def forward(self, x, size):
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        slc = [slice(None)] * x.dim()
        for i, (side, size) in enumerate(zip(sides, sizes)):
            idx = 2 + i
            if side is None or size < 1:
                continue
            elif side == 'front':
                slc[idx] = slice(size, x.shape[idx])
            elif side == 'both':
                slc[idx] = slice(size//2, -math.ceil(size / 2))
            elif side == 'end':
                slc[idx] = slice(0, -size)
            else:
                raise ValueError
        x = x[tuple(slc)]
        return x


class Pool1d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool1d including padding
    """
    def __init__(self, pooling, pool_size, padding='both'):
        super().__init__()
        self.pool_size = pool_size
        self.pooling = pooling
        self.padding = padding

    def forward(self, x):
        if self.pool_size < 2:
            return x, None
        if self.padding is not None:
            pad_size = self.pool_size - 1 - ((x.shape[-1] - 1) % self.pool_size)
            x = Pad(side=self.padding)(x, size=pad_size)
        x = Cut(side='both')(x, size=x.shape[2] % self.pool_size)
        if self.pooling == 'max':
            x, pool_indices = nn.MaxPool1d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = nn.AvgPool1d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, pool_indices


class Unpool1d(Module):
    """
    1d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x, indices=None):
        if self.pool_size < 2:
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool1d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        return x


class Pool2d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool2d including padding
    """
    def __init__(self, pooling, pool_size, padding='both'):
        super().__init__()
        self.pooling = pooling
        self.pool_size = to_pair(pool_size)
        self.padding = to_pair(padding)

    def forward(self, x):
        if all(np.array(self.pool_size) < 2):
            return x, None
        pad_size = (
            self.pool_size[0] - 1 - ((x.shape[-2] - 1) % self.pool_size[0]),
            self.pool_size[1] - 1 - ((x.shape[-1] - 1) % self.pool_size[1])
        )
        pad_size = np.where([pad is None for pad in self.padding], 0, pad_size)
        if any(pad_size > 0):
            x = Pad(side=self.padding)(x, size=pad_size)
        x = Cut(side='both')(x, size=np.array(x.shape[2:]) % self.pool_size)
        if self.pooling == 'max':
            x, pool_indices = nn.MaxPool2d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pooling == 'avg':
            x = nn.AvgPool2d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pooling} pooling unknown.')
        return x, pool_indices


class Unpool2d(Module):
    """
    2d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = to_pair(pool_size)

    def forward(self, x, indices=None, pad_size=None):
        if all(np.array(self.pool_size) < 2):
            return x
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool2d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        return x


class _Conv(Module):
    """
    Wrapper for torch.nn.ConvXd and torch.nn.ConvTransoseXd for X in {1,2}
    including additional options of applying an (gated) activation or
    normalizing the network output. Base Class for Conv(Transpose)Xd.
    """
    conv_cls = None

    @property
    def is_transpose(self):
        return self.conv_cls in [nn.ConvTranspose1d, nn.ConvTranspose2d]

    @property
    def is_2d(self):
        return self.conv_cls in [nn.Conv2d, nn.ConvTranspose2d]

    def __init__(
            self, in_channels, out_channels, kernel_size, dropout=0.,
            padding='both', dilation=1, stride=1, bias=True, norm=None,
            activation='relu', gated=False, pooling=None, pool_size=1,
            return_pool_data=False
    ):
        """

        Args:
            in_channels:
            out_channels:
            kernel_size:
            dilation:
            stride:
            bias:
            dropout:
            norm: may be None or 'batch'
            activation:
            gated:
            pooling:
            pool_size:
        """
        super().__init__()
        if self.is_2d:
            padding = to_pair(padding)
            kernel_size = to_pair(kernel_size)
            dilation = to_pair(dilation)
            stride = to_pair(stride)
        self.dropout = dropout
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.gated = gated
        self.pooling = pooling
        self.pool_size = pool_size
        self.return_pool_data = return_pool_data

        self.conv = self.conv_cls(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            torch.nn.init.zeros_(self.conv.bias)

        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels) if self.is_2d \
                else nn.BatchNorm1d(out_channels)
        else:
            raise ValueError(f'{norm} normalization not known.')

        if self.gated:
            self.gate_conv = self.conv_cls(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias)
            torch.nn.init.xavier_uniform_(self.gate_conv.weight)
            if bias:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x, pool_indices=None, out_shape=None):
        in_shape = x.shape[2:]
        x = self.unpool(x, pool_indices)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)

        x = self.pad(x)

        y = self.conv(x)

        if self.norm is not None:
            y = self.norm(y)

        y = self.activation(y)
        if self.gated:
            g = self.gate_conv(x)
            y = y * torch.sigmoid(g)

        y = self.unpad(y, out_shape)

        y, pool_indices = self.pool(y)

        if self.return_pool_data:
            return y, pool_indices, in_shape
        return y

    def pool(self, x):
        if self.is_transpose or self.pooling is None or self.pool_size == 1:
            return x, None

        if self.is_2d:
            pool = Pool2d(
                pooling=self.pooling,
                pool_size=self.pool_size,
                padding=self.padding
            )
        else:
            pool = Pool1d(
                pooling=self.pooling,
                pool_size=self.pool_size,
                padding=self.padding
            )
        x, pool_indices = pool(x)
        return x, pool_indices

    def unpool(self, x, pool_indices=None):
        if not self.is_transpose or not self.pooling or self.pool_size == 1:
            assert pool_indices is None, (self.is_transpose, self.pooling, self.pool_size, pool_indices is None)
            return x
        if self.is_2d:
            unpool = Unpool2d(pool_size=self.pool_size)
        else:
            unpool = Unpool1d(pool_size=self.pool_size)
        x = unpool(x, indices=pool_indices)
        return x

    def pad(self, x):
        if self.is_transpose:
            return x
        padding = [pad is not None for pad in to_list(self.padding)]
        if any(padding):
            size = (
                np.array(self.dilation) * (np.array(self.kernel_size) - 1)
                - ((np.array(x.shape[2:]) - 1) % np.array(self.stride))
            ).tolist()
            x = Pad(side=self.padding)(x, size=size)
        if not all(padding):
            size = (
                (np.array(x.shape[2:]) - np.array(self.kernel_size))
                % np.array(self.stride)
            ).tolist()
            x = Cut(side=('both' if not pad else None for pad in padding))(x, size)
        return x

    def unpad(self, y, out_shape=None):
        if out_shape is not None:
            assert self.is_transpose
            size = np.array(y.shape[2:]) - np.array(out_shape)
            padding = [
                'both' if side is None else side
                for side in to_list(self.padding)
            ]
            if any(size > 0):
                y = Cut(side=padding)(y, size=size)
            if any(size < 0):
                y = Pad(side=padding, mode='constant')(y, size=-size)
        elif self.is_transpose and self.padding is not None:
            size = (
                    np.array(self.dilation) * (np.array(self.kernel_size) - 1)
                    - np.array(self.stride) + 1
            ).tolist()
            y = Cut(side=self.padding)(y, size=size)
        return y

    def get_out_shape(self, in_shape):
        if self.is_2d:
            assert len(in_shape) == 2
        else:
            assert len(in_shape) == 1
        if self.is_transpose:
            raise NotImplementedError
        else:
            out_shape = np.array(in_shape)
            out_shape_ = out_shape - (
                np.array(self.dilation) * (np.array(self.kernel_size) - 1)
            )
            out_shape = np.where(
                [pad is None for pad in to_list(self.padding)],
                out_shape_, out_shape
            )
            out_shape = np.ceil(out_shape/np.array(self.stride))
            if self.pooling is not None:
                out_shape = out_shape / np.array(self.pool_size)
                out_shape_ = np.floor(out_shape)
                out_shape = np.where(
                    [pad is None for pad in to_list(self.padding)],
                    out_shape_, out_shape
                )
                out_shape = np.ceil(out_shape)
        return out_shape.astype(np.int64)


class Conv1d(_Conv):
    conv_cls = nn.Conv1d


class ConvTranspose1d(_Conv):
    conv_cls = nn.ConvTranspose1d


class Conv2d(_Conv):
    conv_cls = nn.Conv2d


class ConvTranspose2d(_Conv):
    conv_cls = nn.ConvTranspose2d


def to_pair(x):
    return tuple(to_list(x, 2))


class _CNN(Module):
    """
    Stack of Convolutional Layers. Base Class for CNN(Transpose)Xd.
    """
    conv_cls = None
    conv_transpose_cls = None

    def __init__(
            self, in_channels, hidden_channels, out_channels, kernel_size,
            num_layers, dropout=0., padding='both', dilation=1, stride=1,
            norm=None, activation='relu', gated=False,
            pooling='max', pool_size=1, return_pool_data=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        num_hidden_layers = num_layers - int(out_channels is not None)
        self.hidden_channels = to_list(
            hidden_channels, num_hidden_layers
        )
        self.kernel_sizes = to_list(kernel_size, num_layers)
        self.paddings = to_list(padding, num_layers)
        self.dilations = to_list(dilation, num_layers)
        self.strides = to_list(stride, num_layers)
        self.poolings = to_list(pooling, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.out_channels = out_channels
        self.return_pool_data = return_pool_data

        convs = list()
        for i in range(num_hidden_layers):
            hidden_channels = self.hidden_channels[i]
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=hidden_channels,
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i],
                stride=self.strides[i], padding=self.paddings[i], norm=norm,
                dropout=dropout, activation=activation, gated=gated,
                pooling=self.poolings[i], pool_size=self.pool_sizes[i],
                return_pool_data=return_pool_data
            ))
            in_channels = hidden_channels
        if self.out_channels is not None:
            convs.append(self.conv_cls(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=self.kernel_sizes[-1], dilation=self.dilations[-1],
                stride=self.strides[-1], padding=self.paddings[-1], norm=None,
                dropout=dropout, activation='identity', gated=False,
                pooling=self.poolings[-1], pool_size=self.pool_sizes[-1],
                return_pool_data=return_pool_data
            ))

        self.convs = nn.ModuleList(convs)

    def forward(self, x, pool_indices=None, out_shapes=None):
        pool_indices = to_list(copy(pool_indices), self.num_layers)[::-1]
        shapes = to_list(copy(out_shapes), self.num_layers)[::-1]
        for i, conv in enumerate(self.convs):
            x = conv(x, pool_indices[i], shapes[i])
            if isinstance(x, tuple):
                x, pool_indices[i], shapes[i] = x
        if self.return_pool_data:
            return x, pool_indices, shapes
        return x

    @classmethod
    def get_transpose_config(cls, config, transpose_config=None):
        assert config['factory'] == cls
        if transpose_config is None:
            transpose_config = dict()
        if config['factory'] == CNN1d:
            transpose_config['factory'] = CNNTranspose1d
        if config['factory'] == CNNTranspose1d:
            transpose_config['factory'] = CNN1d
        if config['factory'] == CNN2d:
            transpose_config['factory'] = CNNTranspose2d
        if config['factory'] == CNNTranspose2d:
            transpose_config['factory'] = CNN2d

        transpose_config['in_channels'] = config['out_channels']
        transpose_config['out_channels'] = config['in_channels']

        for kw in [
            'hidden_channels', 'kernel_size', 'num_layers', 'dropout',
            'padding', 'dilation', 'stride', 'norm', 'activation', 'gated',
            'pooling', 'pool_size'
        ]:
            if kw not in config.keys():
                continue
            if isinstance(config[kw], list):
                transpose_config[kw] = config[kw][::-1]
            else:
                transpose_config[kw] = config[kw]
        return transpose_config

    def get_out_shape(self, in_shape):
        out_shape = in_shape
        for conv in self.convs:
            out_shape = conv.get_out_shape(out_shape)
        return out_shape


class CNN1d(_CNN):
    conv_cls = Conv1d


class CNNTranspose1d(_CNN):
    conv_cls = ConvTranspose1d


class CNN2d(_CNN):
    conv_cls = Conv2d


class CNNTranspose2d(_CNN):
    conv_cls = ConvTranspose2d


class _MultiScaleConv(Module):
    """
    Concatenates outputs of multiple convolutions with different kernel sizes,
    followed by an output layer. Base Class for MultiScaleConv(Transpose)Xd.
    """
    conv_cls = None

    @property
    def is_transpose(self):
        return self.conv_cls in [ConvTranspose1d, ConvTranspose2d]

    @property
    def is_2d(self):
        return self.conv_cls in [Conv2d, ConvTranspose2d]

    def __init__(
            self, in_channels, hidden_channels, out_channels, kernel_size,
            dropout=0., padding='both', num_scales=1, dilated=False, stride=1,
            norm=None, activation='relu', gated=False, residual=False,
            pooling=None, pool_size=1, return_pool_data=False

    ):
        assert hidden_channels % num_scales == 0, (hidden_channels, num_scales)
        super().__init__()

        if dilated:
            self.kernel_sizes = num_scales * [kernel_size]
            self.dilations = [2 ** i for i in range(num_scales)]
        else:
            self.kernel_sizes = [
                1 + (kernel_size - 1) * 2**i for i in range(num_scales)
            ]
            self.dilations = num_scales * [1]

        self.padding = padding
        self.stride = stride

        self.convs = nn.ModuleList([
            self.conv_cls(
                in_channels=in_channels,
                out_channels=hidden_channels // num_scales,
                padding=padding, dropout=dropout,
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i],
                stride=stride,
                norm=None, activation=activation, gated=gated,
                pooling=None
            )
            for i in range(num_scales)
        ])
        self.out = self.conv_cls(
            in_channels=hidden_channels, out_channels=out_channels,
            kernel_size=1, activation='identity', norm=None, pooling=None
        )

        self.residual = residual

        if norm is None:
            self.norm = None
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels) if self.is_2d \
                else nn.BatchNorm1d(out_channels)
        else:
            raise ValueError(f'{norm} normalization not known.')

        self.pooling = pooling
        self.pool_size = pool_size
        self.return_pool_data = return_pool_data

    def forward(self, x, pool_indices=None, out_shape=None):
        in_shape = x.shape[2:]
        x = self.unpool(x, pool_indices)

        y = [conv(x, None, out_shape) for conv in self.convs]
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

        y, pool_indices = self.pool(y)
        if self.return_pool_data:
            return y, pool_indices, in_shape
        return y

    def pool(self, x):
        if self.is_transpose or self.pooling is None or self.pool_size == 1:
            return x, None

        if self.is_2d:
            pool = Pool2d(
                pool_size=self.pool_size,
                pooling=self.pooling,
                padding=self.padding
            )
        else:
            pool = Pool1d(
                pool_size=self.pool_size,
                pooling=self.pooling,
                padding=self.padding
            )
        x, pool_indices = pool(x)
        return x, pool_indices

    def unpool(self, x, pool_indices=None):
        if not self.is_transpose or not self.pooling or self.pool_size == 1:
            assert pool_indices is None, (self.is_transpose, self.pooling, self.pool_size, pool_indices is None)
            return x
        if self.is_2d:
            unpool = Unpool2d(pool_size=self.pool_size)
        else:
            unpool = Unpool1d(pool_size=self.pool_size)
        x = unpool(x, indices=pool_indices)
        return x

    def get_out_shape(self, in_shape):
        out_shape = self.convs[-1].get_out_shape(in_shape)
        if self.pooling is not None:
            out_shape = out_shape / np.array(self.pool_size)
            out_shape_ = np.floor(out_shape)
            out_shape = np.where(
                [pad is None for pad in to_list(self.padding)],
                out_shape_, out_shape
            )
            out_shape = np.ceil(out_shape)
        return out_shape.astype(np.int64)


class MultiScaleConv1d(_MultiScaleConv):
    conv_cls = Conv1d


class MultiScaleConvTranspose1d(_MultiScaleConv):
    conv_cls = ConvTranspose1d


class MultiScaleConv2d(_MultiScaleConv):
    conv_cls = Conv2d


class MultiScaleConvTranspose2d(_MultiScaleConv):
    conv_cls = ConvTranspose2d


class _MultiScaleCNN(Module):
    """
    Stack of Multi Scale Convolutional Layers.
    Base Class for MultiScaleCNN(Transpose)Xd.
    """
    conv_cls = None

    def __init__(
            self, in_channels, hidden_channels, out_channels, kernel_size,
            num_layers, dropout=0., padding='both', num_scales=1,
            dilated=False, stride=1, norm=None, activation='relu', gated=False,
            residual=False, pooling='max', pool_size=1, return_pool_data=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        num_hidden_layers = num_layers - int(out_channels is not None)
        self.hidden_channels = to_list(
            hidden_channels, num_hidden_layers
        )
        self.kernel_sizes = to_list(kernel_size, num_layers)
        self.num_scales = to_list(num_scales, num_layers)
        self.paddings = to_list(padding, num_layers)
        self.strides = to_list(stride, num_layers)
        self.poolings = to_list(pooling, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.out_channels = out_channels
        self.return_pool_data = return_pool_data

        convs = list()
        for i in range(num_hidden_layers):
            hidden_channels = self.hidden_channels[i]
            convs.append(self.conv_cls(
                in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=hidden_channels, kernel_size=self.kernel_sizes[i],
                num_scales=self.num_scales[i], dilated=dilated,
                stride=self.strides[i], padding=self.paddings[i], norm=norm,
                dropout=dropout, activation=activation, gated=gated,
                residual=residual, pooling=self.poolings[i],
                pool_size=self.pool_sizes[i], return_pool_data=return_pool_data
            ))
            in_channels = hidden_channels
        if self.out_channels is not None:
            convs.append(self.conv_cls(
                in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, kernel_size=self.kernel_sizes[-1],
                num_scales=self.num_scales[-1], dilated=dilated,
                stride=self.strides[-1], padding=self.paddings[-1], norm=None,
                dropout=dropout, activation=activation, gated=gated,
                residual=False, pooling=self.poolings[-1],
                pool_size=self.pool_sizes[-1],
                return_pool_data=return_pool_data
            ))

        self.convs = nn.ModuleList(convs)

    def forward(self, x, pool_indices=None, out_shapes=None):
        pool_indices = to_list(copy(pool_indices), self.num_layers)[::-1]
        shapes = to_list(copy(out_shapes), self.num_layers)[::-1]
        for i, conv in enumerate(self.convs):
            x = conv(x, pool_indices[i], shapes[i])
            if isinstance(x, tuple):
                x, pool_indices[i], shapes[i] = x
        if self.return_pool_data:
            return x, pool_indices, shapes
        return x

    @classmethod
    def get_transpose_config(cls, config, transpose_config=None):
        assert config['factory'] == cls
        if transpose_config is None:
            transpose_config = dict()
        if config['factory'] == MultiScaleCNN1d:
            transpose_config['factory'] = MultiScaleCNNTranspose1d
        if config['factory'] == MultiScaleCNNTranspose1d:
            transpose_config['factory'] = MultiScaleCNN1d
        if config['factory'] == MultiScaleCNN2d:
            transpose_config['factory'] = MultiScaleCNNTranspose2d
        if config['factory'] == MultiScaleCNNTranspose2d:
            transpose_config['factory'] = MultiScaleCNN2d

        transpose_config['in_channels'] = config['out_channels']
        transpose_config['out_channels'] = config['in_channels']

        for kw in [
            'hidden_channels', 'kernel_size', 'num_scales', 'num_layers',
            'dropout', 'padding', 'dilated', 'stride', 'norm', 'activation',
            'gated', 'pooling', 'pool_size', 'residual'
        ]:
            if kw not in config.keys():
                continue
            if isinstance(config[kw], list):
                transpose_config[kw] = config[kw][::-1]
            else:
                transpose_config[kw] = config[kw]
        return transpose_config

    def get_out_shape(self, in_shape):
        out_shape = in_shape
        for conv in self.convs:
            out_shape = conv.get_out_shape(out_shape)
        return out_shape


class MultiScaleCNN1d(_MultiScaleCNN):
    conv_cls = MultiScaleConv1d


class MultiScaleCNNTranspose1d(_MultiScaleCNN):
    conv_cls = MultiScaleConvTranspose1d


class MultiScaleCNN2d(_MultiScaleCNN):
    conv_cls = MultiScaleConv2d


class MultiScaleCNNTranspose2d(_MultiScaleCNN):
    conv_cls = MultiScaleConvTranspose2d


class HybridCNN(Module):
    """
    Combines (MultiScale)CNN2d and (MultiScale)CNN1d sequentially.
    """
    def __init__(
            self, cnn_2d: CNN2d, cnn_1d: CNN1d, input_size=None,
            return_pool_data=False
    ):
        super().__init__()
        assert cnn_2d.return_pool_data == cnn_1d.return_pool_data == return_pool_data, (
                cnn_2d.return_pool_data, cnn_1d.return_pool_data, return_pool_data
        )
        self.cnn_2d = cnn_2d
        self.cnn_1d = cnn_1d
        self.input_size = input_size
        self.return_pool_data = return_pool_data

    def forward(self, x):
        x = self.cnn_2d(x)
        if self.return_pool_data:
            x, pool_indices_2d, shapes_2d = x
        x = x.view((x.shape[0], -1, x.shape[-1]))
        x = self.cnn_1d(x)
        if self.return_pool_data:
            x, pool_indices_1d, shapes_1d = x
            return x, (pool_indices_2d, pool_indices_1d), (shapes_2d, shapes_1d)
        return x

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['cnn_2d'] = {
            'factory': CNN2d,
            'return_pool_data': config['return_pool_data']
        }
        config['cnn_1d'] = {
            'factory': CNN1d,
            'return_pool_data': config['return_pool_data']
        }
        if config['input_size'] is not None:
            cnn_2d = config['cnn_2d']['factory'].from_config(config['cnn_2d'])
            output_size = cnn_2d.get_out_shape((config['input_size'], 1000))[0]
            out_channels = cnn_2d.out_channels \
                if cnn_2d.out_channels is not None \
                else cnn_2d.hidden_channels[-1]
            in_channels = out_channels * output_size
            config['cnn_1d']['in_channels'] = in_channels

    @classmethod
    def get_transpose_config(cls, config, transpose_config=None):
        assert config['factory'] == cls, (config['factory'], cls)
        if transpose_config is None:
            transpose_config = dict()
        transpose_config['factory'] = HybridCNNTranspose
        transpose_config['cnn_transpose_1d'] = config['cnn_1d']['factory'].get_transpose_config(config['cnn_1d'])
        transpose_config['cnn_transpose_2d'] = config['cnn_2d']['factory'].get_transpose_config(config['cnn_2d'])
        return transpose_config


class HybridCNNTranspose(Module):
    """
    Combines (MultiScale)CNNTranspose1d and (MultiScale)CNNTranspose2d sequentially.
    """
    def __init__(
            self, cnn_transpose_1d: CNNTranspose1d,
            cnn_transpose_2d: CNNTranspose2d
    ):
        super().__init__()
        self.cnn_transpose_1d = cnn_transpose_1d
        self.cnn_transpose_2d = cnn_transpose_2d

    def forward(self, x, pool_indices=(None, None), out_shapes=(None, None)):
        pool_indices_2d, pool_indices_1d = pool_indices
        shapes_2d, shapes_1d = out_shapes
        x = self.cnn_transpose_1d(x, pool_indices_1d, shapes_1d)
        x = x.view(
            (x.shape[0], self.cnn_transpose_2d.in_channels, -1, x.shape[-1])
        )
        x = self.cnn_transpose_2d(x, pool_indices_2d, shapes_2d)
        return x

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['cnn_transpose_1d']['factory'] = CNNTranspose1d
        config['cnn_transpose_2d']['factory'] = CNNTranspose2d

    @classmethod
    def get_transpose_config(cls, config, transpose_config=None):
        assert config['factory'] == cls
        if transpose_config is None:
            transpose_config = dict()
        transpose_config['factory'] = HybridCNN
        transpose_config['cnn_2d'] = config['cnn_transpose_2d']['factory'].get_transpose_config(config['cnn_transpose_2d'])
        transpose_config['cnn_1d'] = config['cnn_transpose_1d']['factory'].get_transpose_config(config['cnn_transpose_1d'])
        return transpose_config
