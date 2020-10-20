import padertorch as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from padertorch.contrib.je.modules.conv import Pad, compute_pad_size, to_list
from enh_plath.module.norm import build_norm
from typing import Optional

class Conv1d(pt.Module):
    """
    simplified version of padertorch.contrib.je.modules.Conv1d
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



class Conv1DBlock(pt.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    input: N x F x T

    >>> conv = Conv1DBlock()
    >>> conv(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    >>> conv = Conv1DBlock(norm='gLN')
    >>> conv(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    """
    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN"):
        super().__init__()
        # 1x1 conv

        input_lnorm = build_norm(norm, in_channels)
        self.input_conv = Conv1d(in_channels, out_channels, 1, pad_type=None,
                                 norm=input_lnorm, activation_fn='prelu')
        # depthwise conv

        self.lnorm2 = build_norm(norm, out_channels)
        self.dconv = Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            activation_fn='prelu',
            pad_type='both',
            dilation=dilation,
            bias=True)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)

    def forward(self, x):
        y = self.input_conv(x)
        y = self.dconv(y)
        y = self.sconv(y)
        x = x + y
        return x


class ConvNet(pt.Module):
    """
    Convolutional separator module presented in:
        TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation
        https://arxiv.org/abs/1809.07454

    >>> module = ConvNet()
    >>> module(torch.rand(4, 3, 323, 256), None).shape
    torch.Size([4, 3, 2, 323, 256])
    >>> module = ConvNet(514)
    >>> tensor = [torch.zeros((1, 2500, 514)), torch.zeros((1, 3501, 514))]
    >>> module(tensor, None).shape
    torch.Size([2, 1, 2, 3501, 514])
    """

    def __init__(
            self,
            input_size=256,
            X=8,
            R=4,
            B=256,
            H=512,
            P=3,
            norm="gLN",
            activation="relu",
    ):
        """

        Args:
            feat_size:
            X:
            R:
            B:
            H:
            P:
            norm:
            num_spks:
            activation:
        """
        super().__init__()
        self.input_size = input_size
        self.activation = pt.mappings.ACTIVATION_FN_MAP[activation]()
        # before repeat blocks, always cLN
        self.layer_norm = build_norm('cLN', input_size)
        # input convolution could be replaced by feed forward
        # n x N x T => n x B x T
        self.projection = Conv1d(input_size, B, 1, pad_type=None)
        # repeat blocks
        # n x B x T => n x B x T
        self.conv_blocks = self._build_repeats(
            R,
            X,
            in_channels=B,
            out_channels=H,
            kernel_size=P,
            norm=norm, )
        self.hidden_size = B

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2 ** b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
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
