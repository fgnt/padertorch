import math

import numpy as np
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.utils import to_list
from torch import nn


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
            x padded by size in the last (two) dimension(s) ?

        """
        assert x.dim() in [3, 4], x.shape
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
            x shortened by size in the last (two) dimension(s)

        """
        assert x.dim() in [3, 4], x.shape
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


class Pool1d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool1d including padding
    ToDo: move down or to separate file
    """
    def __init__(self, pool_type, pool_size, pad_side='both'):
        super().__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pad_side = pad_side

    def forward(self, x, sequence_lengths=None):
        if self.pool_size < 2:
            assert self.pool_size == 1, self.pool_size
            return x, sequence_lengths, None
        assert self.pool_type is not None, (
            'pool_size > 1 not allowed when pool_type is None'
        )
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

        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths / self.pool_size
            if self.pad_side is None:
                sequence_lengths = np.floor(sequence_lengths).astype(np.int)
            else:
                sequence_lengths = np.ceil(sequence_lengths).astype(np.int)
        return x, sequence_lengths, pool_indices


class Unpool1d(Module):
    """
    1d MaxUnpooling if indices are provided else upsampling
    ToDo: move down or to separate file
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x, sequence_lengths=None, indices=None):
        if self.pool_size < 2:
            return x, sequence_lengths
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool1d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths * self.pool_size
            sequence_lengths = np.maximum(sequence_lengths, x.shape[-1])
        return x, sequence_lengths


class Pool2d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool2d including padding
    ToDo: move down or to separate file
    """
    def __init__(self, pool_type, pool_size, pad_side='both'):
        super().__init__()
        self.pool_type = pool_type
        self.pool_size = to_pair(pool_size)
        self.pad_side = to_pair(pad_side)

    def forward(self, x, sequence_lengths=None):
        if all(np.array(self.pool_size) < 2):
            return x, sequence_lengths, None
        assert self.pool_type is not None, (
            'pool_size > 1 not allowed when pool_type is None'
        )
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

        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths / self.pool_size[-1]
            if self.pad_side[-1] is None:
                sequence_lengths = np.floor(sequence_lengths).astype(np.int)
            else:
                sequence_lengths = np.ceil(sequence_lengths).astype(np.int)
        return x, sequence_lengths, pool_indices


class Unpool2d(Module):
    """
    2d MaxUnpooling if indices are provided else upsampling
    ToDo: move down or to separate file
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = to_pair(pool_size)

    def forward(self, x, sequence_lengths=None, indices=None):
        if all(np.array(self.pool_size) < 2):
            return x, sequence_lengths
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool2d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths * self.pool_size[-1]
            sequence_lengths = np.maximum(sequence_lengths, x.shape[-1])
        return x, sequence_lengths


def to_pair(x):
    return tuple(to_list(x, 2))


def _finalize_norm_kwargs(norm_kwargs, norm, num_channels, is_2d):
    assert (
        "data_format" not in norm_kwargs
        and "shape" not in norm_kwargs
        and "statistics_axis" not in norm_kwargs
        and "batch_axis" not in norm_kwargs
        and "sequence_axis" not in norm_kwargs
    ), norm_kwargs
    if is_2d:
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
        norm_kwargs["statistics_axis"] = 'btf' if is_2d else 'bt'
    elif norm == 'sequence':
        norm_kwargs["statistics_axis"] = 't'
    else:
        raise ValueError(f'{norm} normalization not known.')
    return norm_kwargs


def map_activation_fn(activation_fn):
    if activation_fn in ['linear', None]:
        activation_fn = 'identity'
    if isinstance(activation_fn, str):
        activation_fn = ACTIVATION_FN_MAP[activation_fn]()
    elif not callable(activation_fn):
        raise ValueError(
            f'Type {type(activation_fn)} not supported for activation_fn'
        )
    return activation_fn


def compute_conv_output_shape(
        input_shape, out_channels, kernel_size, dilation, pad_side, stride
):
    input_shape = np.array(input_shape)
    output_shape = np.zeros_like(input_shape)
    output_shape[0] = input_shape[0]
    output_shape[1] = out_channels
    channel_shape_without_pad = input_shape[2:] - (
        np.array(dilation) * (np.array(kernel_size) - 1)
    )
    channel_shape = np.where(
        [pad is None for pad in to_list(pad_side)],
        channel_shape_without_pad, input_shape[2:]
    )
    output_shape[2:] = np.ceil(channel_shape/np.array(stride))
    return output_shape.astype(np.int64)


def compute_conv_output_sequence_lengths(
        input_sequence_lengths, kernel_size, dilation, pad_side, stride
):
    seq_len_out = np.array(input_sequence_lengths)
    assert seq_len_out.ndim == 1, seq_len_out.ndim
    if to_list(pad_side)[-1] is None:
        seq_len_out = seq_len_out - (
            to_list(dilation)[-1] * (to_list(kernel_size)[-1] - 1)
        )
    return np.ceil(seq_len_out / to_list(stride)[-1]).astype(np.int64)


def compute_pool_output_shape(input_shape, pool_type, pool_size, pad_side):
    output_shape = np.array(input_shape)
    if pool_type is not None:
        channel_shape = output_shape[2:] / np.array(pool_size)
        output_shape[2:] = np.where(
            [pad is None for pad in to_list(pad_side)],
            np.floor(channel_shape), np.ceil(channel_shape)
        )
    return output_shape


def compute_pool_output_sequence_lengths(
        input_sequence_lengths, pool_type, pool_size, pad_side
):
    output_sequence_lengths = np.array(input_sequence_lengths)
    if pool_type is not None:
        output_sequence_lengths = output_sequence_lengths / to_list(pool_size)[-1]
        if to_list(pad_side)[-1] is None:
            output_sequence_lengths = np.floor(output_sequence_lengths)
        else:
            output_sequence_lengths = np.ceil(output_sequence_lengths)
    return output_sequence_lengths
