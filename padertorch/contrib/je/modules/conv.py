import math

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from padertorch.contrib.je.modules.norm import Norm
from torch import nn
from copy import copy
from collections import defaultdict
from einops import rearrange
from padertorch.contrib.je.modules.global_pooling import compute_mask


def to_pair(x):
    return tuple(to_list(x, 2))


class Pad(Module):
    """
    Adds padding of a certain size either to front, end or both.
    """
    def __init__(self, side='both', mode='constant'):
        super().__init__()
        self.side = side
        self.mode = mode

    def forward(self, x, size):
        """

        Args:
            x: input tensor of shape b,c,(f,)t
            size: size to pad to dims (f,)t

        Returns:

        """
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        pad = []
        for side, size in list(zip(sides, sizes))[::-1]:
            if side is None or size < 1:
                assert size == 0, sizes
                pad.extend([0, 0])
            elif side == 'front':
                pad.extend([size, 0])
            elif side == 'both':
                # if size is odd: end is padded more than front
                pad.extend([size // 2, math.ceil(size / 2)])
            elif side == 'end':
                pad.extend([0, size])
            else:
                raise ValueError(f'pad side {side} unknown')

        x = F.pad(x, pad, mode=self.mode)
        return x


class Trim(Module):
    """
    Removes a certain number of values either from front, end or both.
    (Counter part to Pad)
    """
    def __init__(self, side='both'):
        super().__init__()
        self.side = side

    def forward(self, x, size):
        """

        Args:
            x: input tensor of shape b,c,(f,)t
            size: size to trim at dims (f,)t

        Returns:

        """
        sides = to_list(self.side, x.dim() - 2)
        sizes = to_list(size, x.dim() - 2)
        slc = [slice(None)] * x.dim()
        for i, (side, size) in enumerate(zip(sides, sizes)):
            idx = 2 + i
            if side is None or size < 1:
                assert size == 0, sizes
                continue
            elif side == 'front':
                slc[idx] = slice(size, x.shape[idx])
            elif side == 'both':
                # if size is odd: end is trimmed more than front
                slc[idx] = slice(size//2, -math.ceil(size / 2))
            elif side == 'end':
                slc[idx] = slice(0, -size)
            else:
                raise ValueError
        x = x[tuple(slc)]
        return x


class _Conv(Module):
    """
    Wrapper for torch.nn.ConvXd and torch.nn.ConvTransoseXd for X in {1,2}
    including additional options of applying a (gated) activation or
    normalizing the network output (or input if pre_activation).
    Base Class for Conv(Transpose)Xd.
    """
    conv_cls = None

    @classmethod
    def is_transpose(cls):
        return cls.conv_cls in [nn.ConvTranspose1d, nn.ConvTranspose2d]

    @classmethod
    def is_2d(cls):
        return cls.conv_cls in [nn.Conv2d, nn.ConvTranspose2d]

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dropout=0.,
            pad_side='both',
            dilation=1,
            stride=1,
            bias=True,
            norm=None,
            norm_kwargs={},
            activation_fn='relu',
            pre_activation=False,
            gated=False,
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
            activation_fn:
            pre_activation:
            gated:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.is_2d():
            pad_side = to_pair(pad_side)
            kernel_size = to_pair(kernel_size)
            dilation = to_pair(dilation)
            stride = to_pair(stride)
        self.dropout = dropout
        self.pad_side = pad_side
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation_fn = ACTIVATION_FN_MAP[activation_fn]()
        self.pre_activation = pre_activation
        self.gated = gated

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
        else:
            num_channels = in_channels if pre_activation else out_channels
            if self.is_2d():
                norm_kwargs = {
                    "data_format": 'bcft',
                    "shape": (None, num_channels, None, None),
                    **norm_kwargs
                }
            else:
                norm_kwargs = {
                    "data_format": 'bct',
                    "shape": (None, num_channels, None),
                    **norm_kwargs
                }
            if norm == 'batch':
                norm_kwargs["statistics_axis"] = 'btf' if self.is_2d() else 'bt'
            elif norm == 'sequence':
                norm_kwargs["statistics_axis"] = 't'
            else:
                raise ValueError(f'{norm} normalization not known.')
            self.norm = Norm(**norm_kwargs)

        if self.gated:
            self.gate_conv = self.conv_cls(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias)
            torch.nn.init.xavier_uniform_(self.gate_conv.weight)
            if bias:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def forward(
            self, x, seq_len=None, out_shape=None, seq_len_out=None,
            norm_kwargs={}
    ):
        """

        Args:
            x: input tensor of shape b,c,(f,)t
            seq_len: input sequence lengths for each sequence in the mini-batch
            out_shape: desired output shape relevant for transposed convs
            seq_len_out: desired output sequence lengths relevant for transposed convs
            norm_kwargs:

        Returns:

        """
        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)

        if self.pre_activation:
            if self.norm is not None:
                x = self.norm(x, seq_len=seq_len, **norm_kwargs)
            x = self.activation_fn(x)

        if not self.is_transpose():
            x = self.pad_or_trim(x)

        y = self.conv(x)
        if seq_len_out is not None:
            seq_len = seq_len_out
        elif seq_len is not None:
            seq_len = self.get_seq_len_out(seq_len)

        if not self.pre_activation:
            if self.norm is not None:
                y = self.norm(y, seq_len=seq_len, **norm_kwargs)
            y = self.activation_fn(y)
        if self.gated:
            g = self.gate_conv(x)
            y = y * torch.sigmoid(g)

        if out_shape is not None:
            y = self.trim_padded_or_pad_trimmed(y, out_shape)
        return y, seq_len

    def pad_or_trim(self, x):
        """
        Either adds padding or trims parts of the signal not being processed (due to striding).
        Args:
            x: input tensor of shape b,c,(f,)t

        Returns:

        """
        assert not self.is_transpose()
        pad_dims = np.array([side is not None for side in to_list(self.pad_side)])
        if any(pad_dims):
            max_pad_size = np.array(self.dilation) * (np.array(self.kernel_size) - 1)
            # prevent excessive padding, e.g., padding at the front while
            # striding results in the end of the signal not being processed.
            excess_pad_size = (np.array(x.shape[2:]) - 1) % np.array(self.stride)
            pad_size = max_pad_size - excess_pad_size
            pad_size *= pad_dims
            x = Pad(side=self.pad_side)(x, size=pad_size.tolist())
        if not all(pad_dims):
            # for those axis not being padded: trim size (at both sides) which is not going to be processed anyway
            trim_size = (
                (np.array(x.shape[2:]) - np.array(self.kernel_size))
                % np.array(self.stride)
            )
            trim_size *= (1 - pad_dims)
            x = Trim(side='both')(x, trim_size.tolist())
        return x

    def trim_padded_or_pad_trimmed(self, x, out_shape=None):
        """
        counter part to pad_or_trim used in transposed convolutions.
        Only implemented if out_shape is not None!
        Args:
            x: input tensor of shape b,c,(f,)t
            out_shape: target output shape

        Returns:

        """
        assert self.is_transpose()
        if out_shape is not None:
            # assert matching batch and channels dims
            assert x.shape[:2] == tuple(out_shape)[:2], (x.shape, out_shape)
            pad_side = [
                'both' if side is None else side  # if no padding has been used both sides have been trimmed
                for side in to_list(self.pad_side)
            ]
            size = np.array(x.shape[2:]) - np.array(out_shape[2:])
            if any(size > 0):
                x = Trim(side=pad_side)(x, size=size*(size > 0))
            if any(size < 0):
                x = Pad(side=pad_side, mode='constant')(x, size=-size*(size < 0))
        elif any([side is not None for side in to_list(self.pad_side)]):
            # pad_size can not be inferred
            raise NotImplementedError
        return x

    def get_out_shape(self, in_shape):
        """
        compute output shape given input shape

        Args:
            in_shape: input shape

        Returns:

        """
        out_shape = np.array(in_shape)
        assert len(out_shape) == 3 + self.is_2d(), (
            len(out_shape), self.is_2d()
        )
        assert in_shape[1] == self.in_channels, (
            in_shape[1], self.in_channels
        )
        out_shape[1] = self.out_channels
        if self.is_transpose():
            raise NotImplementedError
        else:
            out_shape_ = out_shape[2:] - (
                np.array(self.dilation) * (np.array(self.kernel_size) - 1)
            )
            out_shape[2:] = np.where(
                [pad is None for pad in to_list(self.pad_side)],
                out_shape_, out_shape[2:]
            )
            out_shape[2:] = np.ceil(out_shape[2:]/np.array(self.stride))
        return out_shape.astype(np.int64)

    def get_seq_len_out(self, seq_len_in):
        """
        Compute output sequence lengths for each sequence in the mini-batch

        L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding}
                    + \text{kernel\_size} + \text{output\_padding}

        Args:
            seq_len_in: array/list of sequence lengths

        Returns:

        """
        seq_len_out = np.array(seq_len_in)
        assert seq_len_out.ndim == 1, seq_len_out.ndim
        if self.is_transpose():
            raise NotImplementedError
        else:
            if to_list(self.pad_side)[-1] is None:
                seq_len_out = seq_len_out - (
                    to_list(self.dilation)[-1]
                    * (to_list(self.kernel_size)[-1] - 1)
                )
            seq_len_out = np.ceil(seq_len_out/to_list(self.stride)[-1])
        return seq_len_out.astype(np.int64)


class Conv1d(_Conv):
    conv_cls = nn.Conv1d


class ConvTranspose1d(_Conv):
    conv_cls = nn.ConvTranspose1d


class Conv2d(_Conv):
    conv_cls = nn.Conv2d


class ConvTranspose2d(_Conv):
    conv_cls = nn.ConvTranspose2d


class Pool1d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool1d including padding
    """
    def __init__(self, pool_type, pool_size, pad_side='both'):
        super().__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pad_side = pad_side

    def forward(self, x, seq_len=None):
        if self.pool_size < 2:
            return x, seq_len, None
        if self.pad_side is not None:
            pad_size = self.pool_size - 1 - ((x.shape[-1] - 1) % self.pool_size)
            x = Pad(side=self.pad_side)(x, size=pad_size)
        x = Trim(side='both')(x, size=x.shape[2] % self.pool_size)
        if self.pool_type == 'max':
            x, pool_indices = nn.MaxPool1d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pool_type == 'avg':
            x = nn.AvgPool1d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pool_type} pooling unknown.')

        if seq_len is not None:
            seq_len = seq_len / self.pool_size
            if self.pad_side is None:
                seq_len = np.floor(seq_len).astype(np.int)
            else:
                seq_len = np.ceil(seq_len).astype(np.int)
        return x, seq_len, pool_indices


class Unpool1d(Module):
    """
    1d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x, seq_len=None, indices=None):
        if self.pool_size < 2:
            return x, seq_len
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool1d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        if seq_len is not None:
            seq_len = seq_len * self.pool_size
            seq_len = np.maximum(seq_len, x.shape[-1])
        return x, seq_len


class Pool2d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool2d including padding
    """
    def __init__(self, pool_type, pool_size, pad_side='both'):
        super().__init__()
        self.pool_type = pool_type
        self.pool_size = to_pair(pool_size)
        self.pad_side = to_pair(pad_side)

    def forward(self, x, seq_len=None):
        if all(np.array(self.pool_size) < 2):
            return x, seq_len, None
        pad_size = (
            self.pool_size[0] - 1 - ((x.shape[-2] - 1) % self.pool_size[0]),
            self.pool_size[1] - 1 - ((x.shape[-1] - 1) % self.pool_size[1])
        )
        pad_size = np.where([pad is None for pad in self.pad_side], 0, pad_size)
        if any(pad_size > 0):
            x = Pad(side=self.pad_side)(x, size=pad_size)
        x = Trim(side='both')(x, size=np.array(x.shape[2:]) % self.pool_size)
        if self.pool_type == 'max':
            x, pool_indices = nn.MaxPool2d(
                kernel_size=self.pool_size, return_indices=True
            )(x)
        elif self.pool_type == 'avg':
            x = nn.AvgPool2d(kernel_size=self.pool_size)(x)
            pool_indices = None
        else:
            raise ValueError(f'{self.pool_type} pooling unknown.')

        if seq_len is not None:
            seq_len = seq_len / self.pool_size[-1]
            if self.pad_side[-1] is None:
                seq_len = np.floor(seq_len).astype(np.int)
            else:
                seq_len = np.ceil(seq_len).astype(np.int)
        return x, seq_len, pool_indices


class Unpool2d(Module):
    """
    2d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = to_pair(pool_size)

    def forward(self, x, seq_len=None, indices=None):
        if all(np.array(self.pool_size) < 2):
            return x, seq_len
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool2d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        if seq_len is not None:
            seq_len = seq_len * self.pool_size[-1]
            seq_len = np.maximum(seq_len, x.shape[-1])
        return x, seq_len


class _CNN(Module):
    """
    Stack of Convolutional Layers. Base Class for CNN(Transpose)Xd.
    """
    conv_cls = None
    conv_transpose_cls = None

    @classmethod
    def is_transpose(cls):
        return cls.conv_cls.is_transpose()

    @classmethod
    def is_2d(cls):
        return cls.conv_cls.is_2d()

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            input_layer=True,
            output_layer=True,
            residual_connections=None,
            dense_connections=None,
            dropout=0.,
            pad_side='both',
            dilation=1,
            stride=1,
            norm=None,
            norm_kwargs={},
            activation_fn='relu',
            pre_activation=False,
            gated=False,
            pool_type='max',
            pool_size=1,
            return_pool_indices=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = copy(out_channels)
        num_layers = len(out_channels)
        assert num_layers >= input_layer + output_layer, (num_layers, input_layer, output_layer)
        self.num_layers = num_layers
        self.kernel_sizes = to_list(kernel_size, num_layers)
        residual_connections = to_list(residual_connections, num_layers)
        self.residual_connections = [
            None if destination_idx is None else to_list(destination_idx)
            for destination_idx in residual_connections
        ]
        dense_connections = to_list(dense_connections, num_layers)
        self.dense_connections = [
            None if destination_idx is None else to_list(destination_idx)
            for destination_idx in dense_connections
        ]
        self.pad_sides = to_list(pad_side, num_layers)
        self.dilations = to_list(dilation, num_layers)
        self.strides = to_list(stride, num_layers)
        self.pool_types = to_list(pool_type, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.return_pool_indices = return_pool_indices
        self.activation_fn = to_list(activation_fn, num_layers)
        self.norm = to_list(norm, num_layers)
        self.gated = to_list(gated, num_layers)

        if input_layer:
            self.gated[0] = False
            if pre_activation:
                self.activation_fn[0] = 'identity'
                self.norm[0] = None
        if output_layer:
            self.gated[-1] = False
            if not pre_activation:
                self.activation_fn[-1] = 'identity'
                self.norm[-1] = None

        convs = list()
        layer_in_channels = [in_channels] + copy(out_channels)
        for i in range(num_layers):
            if self.dense_connections[i] is not None:
                for dst_idx in self.dense_connections[i]:
                    assert dst_idx > i, (i, dst_idx)
                    layer_in_channels[dst_idx] += layer_in_channels[i]
            convs.append(self.conv_cls(
                in_channels=layer_in_channels[i],
                out_channels=self.out_channels[i],
                kernel_size=self.kernel_sizes[i],
                dropout=dropout,
                dilation=self.dilations[i],
                stride=self.strides[i],
                pad_side=self.pad_sides[i],
                norm=self.norm[i],
                norm_kwargs=norm_kwargs,
                activation_fn=self.activation_fn[i],
                pre_activation=pre_activation,
                gated=self.gated[i],
            ))
        self.convs = nn.ModuleList(convs)

        residual_convs = dict()
        for source_idx, destination_indices in enumerate(self.residual_connections):
            if destination_indices is None:
                continue
            assert len(set(destination_indices)) == len(destination_indices), destination_indices
            for dst_idx in destination_indices:
                assert dst_idx > source_idx, (source_idx, dst_idx)
                if layer_in_channels[dst_idx] != layer_in_channels[source_idx]:
                    residual_convs[f'{source_idx}->{dst_idx}'] = self.conv_cls(
                        in_channels=layer_in_channels[source_idx],
                        out_channels=layer_in_channels[dst_idx],
                        kernel_size=1,
                        dropout=dropout,
                        dilation=1,
                        stride=1,
                        pad_side=None,
                        norm=None,
                        activation_fn='identity',
                        pre_activation=False,
                        gated=False,
                    )
        self.residual_convs = nn.ModuleDict(residual_convs)
        self.layer_in_channels = layer_in_channels

    def forward(
            self, x, seq_len=None,
            out_shapes=None, seq_lens_out=None, pool_indices=None,
            norm_kwargs={}
    ):
        """

        Args:
            x:
            seq_len:
            out_shapes:
            seq_lens_out:
            pool_indices:
            norm_kwargs:

        Returns:

        """
        assert x.dim() == (3 + self.is_2d()), (x.shape, self.is_2d())
        if not self.is_transpose():
            assert out_shapes is None, out_shapes
            assert seq_lens_out is None, seq_lens_out
            assert pool_indices is None, pool_indices.shape
        out_shapes = to_list(copy(out_shapes), self.num_layers)
        seq_lens_out = to_list(copy(seq_lens_out), self.num_layers)
        pool_indices = to_list(copy(pool_indices), self.num_layers)
        residual_skip_signals = defaultdict(list)
        dense_skip_signals = defaultdict(list)
        for i, conv in enumerate(self.convs):
            x, seq_len = self.maybe_unpool(
                x,
                pool_type=self.pool_types[i],
                pool_size=self.pool_sizes[i],
                seq_len=seq_len,
                pool_indices=pool_indices[i],
            )
            if self.residual_connections[i] is not None:
                for dst_idx in self.residual_connections[i]:
                    residual_skip_signals[dst_idx].append((i, x))
            if self.dense_connections[i] is not None:
                assert not self.is_transpose()
                for dst_idx in sorted(self.dense_connections[i]):
                    dense_skip_signals[dst_idx].append((i, x))
            x, seq_len = conv(
                x,
                seq_len=seq_len, out_shape=out_shapes[i], seq_len_out=seq_lens_out[i],
                norm_kwargs=norm_kwargs
            )
            for src_idx, x_ in dense_skip_signals[i + 1]:
                x_ = F.interpolate(x_, size=x.shape[2:])
                x = torch.cat((x, x_), dim=1)
            for src_idx, x_ in residual_skip_signals[i + 1]:
                x_ = F.interpolate(x_, size=x.shape[2:])
                if f'{src_idx}->{i+1}' in self.residual_convs:
                    x_, _ = self.residual_convs[f'{src_idx}->{i + 1}'](x_)
                x = x + x_
            x, seq_len, pool_indices[i] = self.maybe_pool(
                x,
                pool_type=self.pool_types[i],
                pool_size=self.pool_sizes[i],
                pad_side=self.pad_sides[i],
                seq_len=seq_len
            )
        if self.return_pool_indices:
            return x, seq_len, pool_indices
        return x, seq_len

    def maybe_pool(self, x, pool_type, pool_size, pad_side, seq_len=None):
        if self.is_transpose() or pool_type is None or pool_size == 1:
            return x, seq_len, None

        pool_cls = Pool2d if self.is_2d() else Pool1d
        x, seq_len, pool_indices = pool_cls(
            pool_type=pool_type,
            pool_size=pool_size,
            pad_side=pad_side,
        )(x, seq_len=seq_len)
        return x, seq_len, pool_indices

    def maybe_unpool(self, x, pool_type, pool_size, seq_len=None, pool_indices=None):
        if not self.is_transpose() or not pool_type or pool_size == 1:
            assert pool_indices is None, (
                self.is_transpose(), pool_type, pool_size, pool_indices is None
            )
            return x, seq_len
        unpool_cls = Unpool2d if self.is_2d() else Unpool1d
        x, seq_len = unpool_cls(pool_size=pool_size)(
            x, seq_len=seq_len, indices=pool_indices
        )
        return x, seq_len

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

        channels = [config['in_channels']] + config['out_channels']
        num_layers = len(config['out_channels'])
        if 'residual_connections' in config.keys():
            if config['residual_connections'] is not None:
                skip_connections = defaultdict(list)
                for src_idx, dst_indices in enumerate(
                        to_list(config['residual_connections'], num_layers)
                ):
                    for dst_idx in to_list(dst_indices):
                        if dst_idx is not None:
                            skip_connections[num_layers - dst_idx].append(
                                num_layers - src_idx
                            )
                transpose_config['residual_connections'] = [
                    None if i not in skip_connections
                    else skip_connections[i][0] if len(skip_connections) == 1
                    else skip_connections[i]
                    for i in range(num_layers)
                ]
            else:
                transpose_config['residual_connections'] = None
        if 'dense_connections' in config.keys() \
                and config['dense_connections'] is not None:
            raise ValueError('dense connections cannot be transposed.')

        transpose_config['in_channels'] = channels[-1]
        transpose_config['out_channels'] = channels[:-1][::-1]
        for kw in [
            'kernel_size', 'pad_side', 'dilation', 'stride', 'pool_type', 'pool_size', 'norm'
        ]:
            if kw not in config.keys():
                continue
            if isinstance(config[kw], list):
                transpose_config[kw] = config[kw][::-1]
            else:
                transpose_config[kw] = config[kw]
        for kw in [
            'activation_fn', 'pre_activation', 'dropout', 'gated', 'norm_kwargs'
        ]:
            if kw not in config.keys():
                continue
            transpose_config[kw] = config[kw]
        return transpose_config

    def get_shapes(self, in_shape):
        assert in_shape[1] == self.in_channels, (in_shape[1], self.in_channels)
        out_shape = in_shape
        shapes = [in_shape]
        for i, conv in enumerate(self.convs):
            out_shape = conv.get_out_shape(out_shape)
            out_shape[1] = self.layer_in_channels[i + 1]  # has to be adjusted with dense skip connections
            if self.pool_types[i] is not None:
                if self.is_transpose():
                    raise NotImplementedError
                else:
                    out_shape_ = out_shape[2:] / np.array(self.pool_sizes[i])
                    out_shape[2:] = np.where(
                        [pad is None for pad in to_list(self.pad_sides[i])],
                        np.floor(out_shape_), np.ceil(out_shape_)
                    )
            shapes.append(out_shape)
        return shapes

    def get_transposed_out_shapes(self, in_shape):
        return self.get_shapes(in_shape)[::-1][1:]

    def get_seq_lens(self, seq_len_in):
        seq_len_out = seq_len_in
        seq_lens = [seq_len_in]
        for i, conv in enumerate(self.convs):
            seq_len_out = conv.get_seq_len_out(seq_len_out)
            if self.pool_types[i] is not None:
                if self.is_transpose():
                    raise NotImplementedError
                else:
                    seq_len_out = seq_len_out / to_list(self.pool_sizes[i])[-1]
                    if to_list(self.pad_sides[i])[-1] is None:
                        seq_len_out = np.floor(seq_len_out)
                    else:
                        seq_len_out = np.ceil(seq_len_out)
            seq_lens.append(seq_len_out)
        return seq_lens

    def get_transposed_seq_lens_out(self, seq_len_in):
        return self.get_seq_lens(seq_len_in)[::-1][1:]

    def get_receptive_field(self):
        receptive_field = np.ones(1+self.is_2d()).astype(np.int)
        for i in reversed(range(self.num_layers)):
            receptive_field *= np.array(self.strides[i])*np.array(self.pool_sizes[i])
            receptive_field += np.array(self.kernel_sizes[i]) - np.array(self.strides[i])
        return receptive_field


class CNN1d(_CNN):
    conv_cls = Conv1d


class CNNTranspose1d(_CNN):
    conv_cls = ConvTranspose1d


class CNN2d(_CNN):
    conv_cls = Conv2d


class CNNTranspose2d(_CNN):
    conv_cls = ConvTranspose2d
