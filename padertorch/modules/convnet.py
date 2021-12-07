"""
This code is inspired by https://github.com/funcwj/conv-tasnet
"""

import padertorch as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from padertorch.utils import to_list
from padertorch.contrib.je.modules.conv import Pad, compute_pad_size
from padertorch.contrib.jensheit.norm import build_norm #ToDo move to norm
from typing import Optional


class Conv1d(pt.Module):
    """
    simplified version of padertorch.contrib.je.modules.Conv1d
    #ToDo: replace with JE version when published
    """
    conv_cls = nn.Conv1d

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dropout=0.,
            pad_type='both',
            groups=1,
            dilation=1,
            stride=1,
            bias=True,
            norm=None,
            activation_fn='relu',
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
            norm:
            activation_fn:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout = dropout
        self.pad_type = pad_type
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation_fn = pt.ops.mappings.ACTIVATION_FN_MAP[activation_fn]()
        if norm is not None:
            assert callable(norm), norm
        self.norm = norm
        self.conv = self.conv_cls(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias, groups=groups
        )

    def forward(self, x):
        """

        Args:
            x: input tensor of shape b,c,t

        Returns:

        """

        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)
        if self.norm is not None:
            x = self.norm(x)
        x = self.pad(x)
        y = self.conv(x)
        y = self.activation_fn(y)
        return y

    def pad(self, x):
        """
        adds padding
        Args:
            x: input tensor of shape b,c,(f,)t

        Returns:

        """
        front_pad, end_pad = list(zip(*[
            compute_pad_size(k, d, s, t)
            for k, d, s, t in zip(
                to_list(self.kernel_size, 1),
                to_list(self.dilation, 1),
                to_list(self.stride, 1),
                to_list(self.pad_type, 1),
            )
        ]))
        if any(np.array(front_pad) > 0):
            x = Pad(side='front')(x, size=front_pad)
        if any(np.array(end_pad) > 0):
            x = Pad(side='end')(x, size=end_pad)
        return x


class _Conv1DBlock(pt.Module):
    """

    1D convolutional block:
        consists of Conv1D - PReLU - Norm - Conv1D - PReLU - Norm -  Conv1D
    input: B x F x T

    >>> conv = _Conv1DBlock()
    >>> conv(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    >>> conv = _Conv1DBlock(norm='gLN')
    >>> conv(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    """
    def __init__(self,
                 in_channels=256,
                 hidden_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN"):
        super().__init__()

        # ToDo: this can be replaced by a CNN1D from JE

        self.input_norm = build_norm(norm, in_channels)
        self.input_conv = Conv1d(in_channels, hidden_channels, 1, pad_type=None,
                                 norm=self.input_norm, activation_fn='prelu')

        self.conv = Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            groups=hidden_channels,
            activation_fn='prelu',
            pad_type='both',
            dilation=dilation
        )

        self.norm = build_norm(norm, hidden_channels)
        self.output_conv = Conv1d(hidden_channels, in_channels, 1,
                                  norm=self.norm, activation_fn='identity')

    def forward(self, x):
        y = self.input_conv(x)
        y = self.conv(y)
        y = self.output_conv(y)
        x = x + y
        return x


class ConvNet(pt.Module):
    """
    Convolutional separator module presented in:
        TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation
        https://arxiv.org/abs/1809.07454

    >>> module = ConvNet()
    >>> module(torch.rand(4, 323, 256), None).shape
    torch.Size([4, 323, 256])
    """

    def __init__(
            self,
            input_size=256,
            num_blocks=8,
            num_repeats=4,
            hidden_channels=512,
            kernel_size=3,
            norm="gLN",
    ):
        """

        Args:
            input_size:
            num_blocks: number of _Conv1DBlock with dilation between
                        1 and 2**(num_blocks-1) per repitition
            num_repeats: number of repitition of num_blocks
            input_size:
            hidden_channels:
            kernel_size:
            norm:
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = input_size

        # repeat blocks
        self.conv_blocks = self._build_repeats(
            num_repeats,
            num_blocks,
            in_channels=input_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            norm=norm)

    def _build_blocks(self, num_blocks, **block_kwargs):
        blocks = [
            _Conv1DBlock(**block_kwargs, dilation=(2 ** b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for _ in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
                self,
                sequence: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """

            Args:
                sequence (B, L, N):
                sequence_lengths:

            Returns:

        """
        x = rearrange(sequence, 'b l n -> b n l')
        y = self.conv_blocks(x)

        return rearrange(y, 'b n l -> b l n')
