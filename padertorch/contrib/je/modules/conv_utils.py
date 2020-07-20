import math

import numpy as np
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.utils import to_list
from torch import nn


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
    ToDo: move down or to separate file
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
    ToDo: move down or to separate file
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
    ToDo: move down or to separate file
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
