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

        if not any(np.array(sizes)):
            return x

        pad = []
        for side, size in reversed(list(zip(sides, sizes))):
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
    """
    def __init__(self, pool_type, pool_size, pad_type=None):
        super().__init__()
        assert np.isscalar(pool_size)
        assert pad_type is None or np.isscalar(pad_type)
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pad_type = pad_type

    def forward(self, x, sequence_lengths=None):
        if self.pool_size < 2:
            assert self.pool_size == 1, self.pool_size
            return x, sequence_lengths, None
        assert self.pool_type is not None, (
            'pool_size > 1 not allowed when pool_type is None'
        )
        front_pad, end_pad = compute_pad_size(self.pool_size, 1, self.pool_size, self.pad_type)
        if front_pad > 0:
            x = Pad(side='front')(x, size=front_pad)
        if end_pad > 0:
            x = Pad(side='end')(x, size=end_pad)
        x = Trim(side='end')(x, size=np.array(x.shape[2:]) % self.pool_size)
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
            sequence_lengths = _compute_conv_out_size(sequence_lengths, self.pool_size, 1, self.pool_size, self.pad_type)
            assert all(sequence_lengths > 0), sequence_lengths
        return x, sequence_lengths, pool_indices


class Unpool1d(Module):
    """
    1d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size, pad_type=None):
        super().__init__()
        assert np.isscalar(pool_size)
        assert pad_type is None or np.isscalar(pad_type)
        self.pool_size = pool_size
        self.pad_type = pad_type

    def forward(self, x, sequence_lengths=None, indices=None):
        if self.pool_size < 2:
            return x, sequence_lengths
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool1d(kernel_size=self.pool_size)(
                x, indices=indices
            )
        front_pad, end_pad = compute_pad_size(self.pool_size, 1, self.pool_size, self.pad_type)
        if front_pad > 0:
            x = Trim(side='front')(x, size=front_pad)
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths * self.pool_size - front_pad
            # sequence_lengths = np.maximum(sequence_lengths, x.shape[-1])
        return x, sequence_lengths


class Pool2d(Module):
    """
    Wrapper for nn.{Max,Avg}Pool2d including padding
    """
    def __init__(self, pool_type, pool_size, pad_type=None):
        super().__init__()
        self.pool_type = pool_type
        self.pool_size = to_pair(pool_size)
        self.pad_type = to_pair(pad_type)

    def forward(self, x, sequence_lengths=None):
        if all(np.array(self.pool_size) < 2):
            return x, sequence_lengths, None
        assert self.pool_type is not None, (
            'pool_size > 1 not allowed when pool_type is None'
        )
        front_pad, end_pad = list(zip(*[
            compute_pad_size(k, 1, k, t)
            for k, t in zip(self.pool_size, self.pad_type)
        ]))
        if any(np.array(front_pad) > 0):
            x = Pad(side='front')(x, size=front_pad)
        if any(np.array(end_pad) > 0):
            x = Pad(side='end')(x, size=end_pad)
        x = Trim(side='end')(x, size=np.array(x.shape[2:]) % self.pool_size)
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
            sequence_lengths = _compute_conv_out_size(
                sequence_lengths, self.pool_size[-1], 1, self.pool_size[-1],
                self.pad_type[-1]
            )
            assert all(sequence_lengths > 0), sequence_lengths
        return x, sequence_lengths, pool_indices


class Unpool2d(Module):
    """
    2d MaxUnpooling if indices are provided else upsampling
    """
    def __init__(self, pool_size, pad_type=None):
        super().__init__()
        self.pool_size = to_pair(pool_size)
        self.pad_type = to_pair(pad_type)

    def forward(self, x, sequence_lengths=None, indices=None):
        if all(np.array(self.pool_size) < 2):
            return x, sequence_lengths
        if indices is None:
            x = F.interpolate(x, scale_factor=self.pool_size)
        else:
            x = nn.MaxUnpool2d(kernel_size=self.pool_size)(x, indices=indices)

        front_pad, end_pad = list(zip(*[
            compute_pad_size(k, 1, k, t)
            for k,t in zip(self.pool_size, self.pad_type)
        ]))
        if any(np.array(front_pad) > 0):
            x = Trim(side='front')(x, size=front_pad)

        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths * self.pool_size[-1] - front_pad[-1]
            # sequence_lengths = np.maximum(sequence_lengths, x.shape[-1])
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


def compute_pad_size(kernel_size, dilation, stride, pad_type):
    ks = 1 + dilation * (kernel_size - 1)
    if pad_type is None:
        return 0, 0
    if pad_type == 'front':
        return max(ks - stride, 0), min(stride - 1, ks - 1)
    if pad_type == 'both':
        return max(ks - stride, 0) // 2, min(stride - 1, ks - 1) + math.ceil(max(ks - stride, 0)/2)
    if pad_type == 'end':
        return 0,  ks - 1


def _compute_conv_out_size(in_size, kernel_size, dilation, stride, pad_type):
    pad_size = sum(compute_pad_size(kernel_size, dilation, stride, pad_type))
    ks = 1 + dilation * (kernel_size - 1)
    out_size = in_size - (ks - 1) + pad_size
    return 1 + (out_size-1) // stride


def _compute_transpose_out_size(in_size, kernel_size, dilation, stride, pad_type):
    out_size = (1 + (in_size - 1) * stride) + (dilation * (kernel_size - 1))
    front_pad, end_pad = compute_pad_size(kernel_size, dilation, stride, pad_type)
    end_pad = max(end_pad-stride+1, 0)
    return out_size.astype(np.int64) - front_pad - end_pad


def compute_conv_output_shape(
        input_shape, out_channels, kernel_size, dilation, stride, pad_type, transpose=False
):
    input_shape = np.array(input_shape)
    output_shape = np.zeros_like(input_shape)
    output_shape[0] = input_shape[0]
    output_shape[1] = out_channels
    kernel_size = to_list(kernel_size, len(input_shape)-2)
    dilation = to_list(dilation, len(input_shape)-2)
    stride = to_list(stride, len(input_shape)-2)
    pad_type = to_list(pad_type, len(input_shape)-2)
    for d in range(len(kernel_size)):
        if transpose:
            output_shape[2+d] = _compute_transpose_out_size(
                input_shape[2+d], kernel_size[d], dilation[d], stride[d], pad_type[d]
            )
        else:
            output_shape[2+d] = _compute_conv_out_size(
                input_shape[2+d], kernel_size[d], dilation[d], stride[d], pad_type[d]
            )
    assert all(output_shape > 0), output_shape
    return output_shape.astype(np.int64)


def compute_conv_output_sequence_lengths(
        input_sequence_lengths, kernel_size, dilation, pad_type, stride, transpose=False
):
    kernel_size = to_list(kernel_size)
    dilation = to_list(dilation)
    stride = to_list(stride)
    pad_type = to_list(pad_type)
    if transpose:
        seq_len_out = _compute_transpose_out_size(
            input_sequence_lengths, kernel_size[-1], dilation[-1], stride[-1], pad_type[-1]
        )
    else:
        seq_len_out = _compute_conv_out_size(
            input_sequence_lengths, kernel_size[-1], dilation[-1], stride[-1], pad_type[-1]
        )
    assert all(seq_len_out > 0), seq_len_out
    return seq_len_out.astype(np.int64)
