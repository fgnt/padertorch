from collections import defaultdict
from copy import copy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from padertorch.base import Module
from padertorch.contrib.je.modules.conv_utils import (
    to_pair, _finalize_norm_kwargs, Pad, Trim,
    Pool1d, Pool2d, Unpool1d, Unpool2d, map_activation_fn,
    compute_conv_output_shape, compute_conv_output_sequence_lengths,
    compute_pad_size
)
from padertorch.modules.normalization import Normalization
from padertorch.utils import to_list
from torch import nn


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
            pad_type='both',
            dilation=1,
            stride=1,
            bias=True,
            norm=None,
            norm_kwargs=None,
            activation_fn='relu',
            pre_activation=False,
            gated=False,
            return_state=False,
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
            norm: may be None, 'batch' or 'sequence'
            activation_fn:
            pre_activation: If True normalization and activation is applied to
                the input rather than to the output. This is important when
                skip connections are used. See, e.g., https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e .
            gated:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.is_2d():
            pad_type = to_pair(pad_type)
            kernel_size = to_pair(kernel_size)
            dilation = to_pair(dilation)
            stride = to_pair(stride)
        self.bias = bias
        self.dropout = dropout
        self.pad_type = pad_type
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.activation_fn = map_activation_fn(activation_fn)
        self.pre_activation = pre_activation
        self.gated = gated
        self.return_state = return_state

        self.conv = self.conv_cls(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, stride=stride,
            bias=bias
        )

        if self.gated:
            self.gate_conv = self.conv_cls(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, stride=stride,
                bias=bias
            )

        if norm is None:
            self.norm = None
            assert norm_kwargs is None, norm_kwargs
        else:
            norm_kwargs = {} if norm_kwargs is None else norm_kwargs
            num_channels = self.in_channels if self.pre_activation \
                else self.out_channels
            norm_kwargs = _finalize_norm_kwargs(
                norm_kwargs, norm, num_channels, self.is_2d()
            )
            self.norm = Normalization(**norm_kwargs)

    def reset_parameters(self, output_activation_fn):
        if isinstance(self.activation_fn, torch.nn.PReLU):
            with torch.no_grad():
                self.activation_fn.weight.fill_(.25)
        output_activation_fn = map_activation_fn(output_activation_fn)
        if (
            isinstance(output_activation_fn, torch.nn.ReLU)
            or isinstance(output_activation_fn, torch.nn.ELU)
        ):
            torch.nn.init.kaiming_uniform_(
                self.conv.weight, nonlinearity='relu'
            )
        elif isinstance(output_activation_fn, torch.nn.LeakyReLU):
            torch.nn.init.kaiming_uniform_(
                self.conv.weight, nonlinearity='leaky_relu',
                a=output_activation_fn.negative_slope
            )
        elif isinstance(output_activation_fn, torch.nn.PReLU):
            torch.nn.init.kaiming_uniform_(
                self.conv.weight, nonlinearity='leaky_relu', a=.25
            )
        else:
            if isinstance(output_activation_fn, torch.nn.Tanh):
                torch.nn.init.xavier_uniform_(self.conv.weight, gain=5/3)
            else:
                torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.)
        if self.gated:
            torch.nn.init.xavier_uniform_(self.gate_conv.weight, gain=1.)
        if self.bias:
            torch.nn.init.zeros_(self.conv.bias)
            if self.gated:
                torch.nn.init.zeros_(self.gate_conv.bias)

    def freeze(self, freeze_norm_stats=True):
        for param in self.parameters():
            param.requires_grad = False
        if self.norm is not None:
            self.norm.freeze(freeze_stats=freeze_norm_stats)

    def forward(self, x, sequence_lengths=None, state=None):
        """

        Args:
            x: input tensor of shape b,c,(f,)t
            sequence_lengths: input sequence lengths for each sequence in the mini-batch
            state:

        Returns:

        """

        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)

        if self.pre_activation:
            if self.norm is not None:
                x = self.norm(x, sequence_lengths=sequence_lengths)
            x = self.activation_fn(x)

        if self.is_transpose():
            assert state is None
            assert self.return_state is False
        else:
            x, state = self.pad(x, state)

        y = self.conv(x)
        if sequence_lengths is not None:
            sequence_lengths = self.get_output_sequence_lengths(sequence_lengths, state)

        if not self.pre_activation:
            if self.norm is not None:
                y = self.norm(y, sequence_lengths=sequence_lengths)
            y = self.activation_fn(y)
        if self.gated:
            g = self.gate_conv(x)
            y = y * torch.sigmoid(g)

        if self.is_transpose():
            y = self.trim_padding(y)

        if self.return_state:
            return y, sequence_lengths, state
        return y, sequence_lengths

    def pad(self, x, state=None):
        """
        adds padding
        Args:
            x: input tensor of shape b,c,(f,)t
            state: optional input tensor of shape b,c,(f,)k_t-s_t

        Returns:

        """
        assert not self.is_transpose()
        kernel_size = to_list(self.kernel_size, 1+self.is_2d())
        dilation = to_list(self.dilation, 1+self.is_2d())
        stride = to_list(self.stride, 1+self.is_2d())
        pad_type = list(to_list(self.pad_type, 1+self.is_2d()))
        state_len = 1 + (kernel_size[-1] - 1) * dilation[-1] - stride[-1]
        if state is not None:
            *bcf, t = x.shape
            expected_shape = (*bcf, state_len)
            assert state.shape == expected_shape, (state.shape, expected_shape)
            if state_len > 0:
                x = torch.cat((state, x), dim=-1)
            pad_type[-1] = None
        pad_type = tuple(pad_type)
        if self.return_state:
            state = x[..., x.shape[-1]-state_len:]
        else:
            state = None
        front_pad, end_pad = list(zip(*[
            compute_pad_size(k, d, s, t)
            for k, d, s, t in zip(
                kernel_size, dilation, stride, pad_type,
            )
        ]))
        if any(np.array(front_pad) > 0):
            x = Pad(side='front')(x, size=front_pad)
        if any(np.array(end_pad) > 0):
            x = Pad(side='end')(x, size=end_pad)
        return x, state

    def trim_padding(self, x):
        """
        counter part to pad_or_trim used in transposed convolutions.
        Only implemented if out_shape is not None!
        Args:
            x: input tensor of shape b,c,(f,)t

        Returns:

        """
        assert self.is_transpose()
        front_pad, end_pad = list(zip(*[
            compute_pad_size(k, d, s, t)
            for k, d, s, t in zip(
                to_list(self.kernel_size, 1+self.is_2d()),
                to_list(self.dilation, 1+self.is_2d()),
                to_list(self.stride, 1+self.is_2d()),
                to_list(self.pad_type, 1+self.is_2d()),
            )
        ]))
        end_pad = np.maximum(np.array(end_pad)-np.array(self.stride)+1, 0)

        if any(front_pad):
            x = Trim(side='front')(x, size=front_pad)
        if any(end_pad):
            x = Trim(side='end')(x, size=end_pad)
        return x

    def get_output_shape(self, input_shape):
        """
        compute output shape given input shape

        Args:
            input_shape: input shape

        Returns:

        >>> cnn = Conv1d(4, 20, 5, stride=2, pad_type=None)
        >>> signal = torch.rand((5, 4, 103))
        >>> out, seq_len = cnn(signal)
        >>> cnn.get_output_shape(signal.shape)
        array([ 5, 20, 50])
        >>> cnn = Conv1d(4, 20, 5, stride=2, pad_type='both')
        >>> signal = torch.rand((5, 4, 103))
        >>> out, seq_len = cnn(signal)
        >>> cnn.get_output_shape(signal.shape)
        array([ 5, 20, 52])
        """
        assert len(input_shape) == 3 + self.is_2d(), (
            len(input_shape), self.is_2d()
        )
        assert input_shape[1] == self.in_channels, (
            input_shape[1], self.in_channels
        )
        return compute_conv_output_shape(
            input_shape,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size, dilation=self.dilation,
            pad_type=self.pad_type, stride=self.stride,
            transpose=self.is_transpose()
        )

    def get_input_shape(self, output_shape):
        """
        compute input shape given output shape

        Args:
            output_shape: input shape

        Returns:

        >>> cnn = ConvTranspose1d(20, 4, 5, stride=2, pad_type=None)
        >>> output_shape = (5, 4, 103)
        >>> cnn.get_input_shape(output_shape)
        array([ 5, 20, 50])
        >>> cnn = ConvTranspose1d(20, 4, 5, stride=2, pad_type='both')
        >>> output_shape = (5, 4, 103)
        >>> cnn.get_input_shape(output_shape)
        array([ 5, 20, 52])
        """
        assert len(output_shape) == 3 + self.is_2d(), (
            len(output_shape), self.is_2d()
        )
        assert output_shape[1] == self.out_channels, (
            output_shape[1], self.out_channels
        )
        return compute_conv_output_shape(
            output_shape,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size, dilation=self.dilation,
            pad_type=self.pad_type, stride=self.stride,
            transpose=not self.is_transpose()
        )

    def get_output_sequence_lengths(self, input_sequence_lengths, state):
        """
        Compute output sequence lengths for each sequence in the mini-batch

        Args:
            input_sequence_lengths: array/list of sequence lengths

        Returns:

        """
        input_sequence_lengths = np.array(input_sequence_lengths)
        assert input_sequence_lengths.ndim == 1, input_sequence_lengths.ndim
        pad_type = list(to_list(self.pad_type, 1+self.is_2d()))
        if state is not None:
            pad_type[-1] = 'front'
        return compute_conv_output_sequence_lengths(
            input_sequence_lengths,
            kernel_size=self.kernel_size, dilation=self.dilation,
            stride=self.stride, pad_type=pad_type,
            transpose=self.is_transpose(),
        )

    def get_input_sequence_lengths(self, output_sequence_lengths):
        """
        Compute input sequence lengths for each sequence in the mini-batch

        Args:
            output_sequence_lengths: array/list of sequence lengths

        Returns:

        """
        output_sequence_lengths = np.array(output_sequence_lengths)
        assert output_sequence_lengths.ndim == 1, output_sequence_lengths.ndim
        return compute_conv_output_sequence_lengths(
            output_sequence_lengths,
            kernel_size=self.kernel_size, dilation=self.dilation,
            stride=self.stride, pad_type=self.pad_type,
            transpose=not self.is_transpose()
        )


class Conv1d(_Conv):
    conv_cls = nn.Conv1d


class ConvTranspose1d(_Conv):
    conv_cls = nn.ConvTranspose1d


class Conv2d(_Conv):
    conv_cls = nn.Conv2d


class ConvTranspose2d(_Conv):
    conv_cls = nn.ConvTranspose2d


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

    @classmethod
    def pool_cls(cls, *args, **kwargs):
        return Pool2d(*args, **kwargs) if cls.is_2d() else Pool1d(*args, **kwargs)

    @classmethod
    def unpool_cls(cls, *args, **kwargs):
        return Unpool2d(*args, **kwargs) if cls.is_2d() else Unpool1d(*args, **kwargs)

    def __init__(
            self,
            in_channels,
            out_channels: List[int],
            kernel_size,
            input_layer=True,
            output_layer=True,
            residual_connections=None,
            dense_connections=None,
            dropout=0.,
            pad_type='both',
            dilation=1,
            stride=1,
            norm=None,
            norm_kwargs=None,
            activation_fn='relu',
            pre_activation=False,
            gated=False,
            pool_type='max',
            pool_size=1,
            pool_stride=None,
            return_pool_indices=False,
            return_state=False,
            normalize_skip_convs=False,
    ):
        """

        Args:
            in_channels: number of input channels
            out_channels: list of number of output channels for each layer
            kernel_size:
            input_layer: if True neither normalization nor
                activation_fn is applied to input.
            output_layer: if True neither normalization nor
                activation_fn is applied to output and last layer doesn't use
                gating.
            residual_connections: None or list of None/int/List[int] such that
                if residual_connections[src_idx] == dst_idx or dst_idx in
                residual_connections[src_idx] a residual connection is added
                from the input of the src_idx-th layer to the input of the
                dst_idx-th layer. If, e.g.,
                residual_connections = [None, 3, None, None] there is a single
                residual connection skipping the second and third layer.
            dense_connections: None or list of None/int/List[int] such that
                if dense_connections[src_idx] == dst_idx or dst_idx in
                dense_connections[src_idx] a dense connection is added
                from the input of the src_idx-th layer to the input of the
                dst_idx-th layer. If, e.g.,
                dense_connections = [None, 3, None, None] there is a single
                dense_connection connection skipping the second and third layer.
            dropout:
            pad_type:
            dilation:
            stride:
            norm:
            norm_kwargs:
            activation_fn:
            pre_activation:
            gated:
            pool_type:
            pool_size:
            return_pool_indices:
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = copy(out_channels)
        num_layers = len(out_channels)
        # assert num_layers >= input_layer + output_layer, (num_layers, input_layer, output_layer)
        self.num_layers = num_layers
        self.kernel_sizes = to_list(kernel_size, num_layers)
        residual_connections = to_list(residual_connections, num_layers)
        self.residual_connections = [
            [] if destination_idx is None else to_list(destination_idx)
            for destination_idx in residual_connections
        ]
        dense_connections = to_list(dense_connections, num_layers)
        self.dense_connections = [
            [] if destination_idx is None else to_list(destination_idx)
            for destination_idx in dense_connections
        ]
        self.pad_types = to_list(pad_type, num_layers)
        self.dilations = to_list(dilation, num_layers)
        self.strides = to_list(stride, num_layers)
        self.pool_types = to_list(pool_type, num_layers)
        self.pool_sizes = to_list(pool_size, num_layers)
        self.pool_strides = self.pool_sizes if pool_stride is None else to_list(pool_stride, num_layers)
        self.return_pool_indices = return_pool_indices
        self.return_state = return_state
        self.activation_fn = list(to_list(activation_fn, num_layers+1))
        self.norm = list(to_list(norm, num_layers+1))
        self.gated = to_list(gated, num_layers)
        self.pre_activation = pre_activation

        if input_layer:
            assert (
                not isinstance(activation_fn, (list, tuple))
                or activation_fn[0] == 'identity'
            )
            self.activation_fn[0] = 'identity'
            assert (
                not isinstance(norm, (list, tuple))
                or norm[0] is None
            )
            self.norm[0] = None
        if output_layer:
            assert not isinstance(gated, (list, tuple)) or not gated[-1]
            self.gated[-1] = False
            assert (
                not isinstance(activation_fn, (list, tuple))
                or activation_fn[-1] == 'identity'
            )
            self.activation_fn[-1] = 'identity'
            assert (
                not isinstance(norm, (list, tuple))
                or norm[-1] is None
            )
            self.norm[-1] = None

        convs = list()
        layer_in_channels = [in_channels] + copy(out_channels)
        for i in range(num_layers):
            norm = self.norm[i + (not pre_activation)]
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
                pad_type=self.pad_types[i],
                norm=norm,
                norm_kwargs=None if norm is None else norm_kwargs,
                activation_fn=self.activation_fn[i + (not pre_activation)],
                pre_activation=pre_activation,
                gated=self.gated[i],
                return_state=return_state,
            ))
        self.convs = nn.ModuleList(convs)

        residual_skip_convs = dict()
        for src_idx, destination_indices in enumerate(self.residual_connections):
            if destination_indices is None:
                continue
            assert len(set(destination_indices)) == len(destination_indices), destination_indices
            for dst_idx in destination_indices:
                assert dst_idx > src_idx, (src_idx, dst_idx)
                if layer_in_channels[dst_idx] != layer_in_channels[src_idx]:
                    skip_norm = self.norm[src_idx if pre_activation else dst_idx] if normalize_skip_convs else None
                    residual_skip_convs[f'{src_idx}->{dst_idx}'] = self.conv_cls(
                        in_channels=layer_in_channels[src_idx],
                        out_channels=layer_in_channels[dst_idx],
                        kernel_size=1,
                        dropout=dropout,
                        dilation=1,
                        stride=1,
                        pad_type=None,
                        norm=skip_norm,
                        norm_kwargs=None if skip_norm is None else norm_kwargs,
                        activation_fn='identity',
                        pre_activation=pre_activation,
                        gated=False,
                    )
        self.residual_skip_convs = nn.ModuleDict(residual_skip_convs)

        self.layer_in_channels = layer_in_channels

        if self.norm[0] is not None and not pre_activation:
            norm_kwargs = {} if norm_kwargs is None else norm_kwargs
            norm_kwargs = _finalize_norm_kwargs(
                norm_kwargs, self.norm[0], self.in_channels, self.is_2d()
            )
            self.input_norm = Normalization(**norm_kwargs)
            self.input_activation_fn = map_activation_fn(self.activation_fn[0])
        else:
            self.input_norm = None
            self.input_activation_fn = None
        if self.norm[-1] is not None and pre_activation:
            norm_kwargs = {} if norm_kwargs is None else norm_kwargs
            norm_kwargs = _finalize_norm_kwargs(
                norm_kwargs, self.norm[-1], layer_in_channels[-1], self.is_2d()
            )
            self.output_norm = Normalization(**norm_kwargs)
            self.output_activation_fn = map_activation_fn(self.activation_fn[-1])
        else:
            self.output_norm = None
            self.output_activation_fn = None
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.input_activation_fn, torch.nn.PReLU):
            with torch.no_grad():
                self.input_activation_fn.weight.fill_(0.25)
        if isinstance(self.output_activation_fn, torch.nn.PReLU):
            with torch.no_grad():
                self.output_activation_fn.weight.fill_(0.25)
        for i in range(self.num_layers):
            output_activation_fn = self.output_activation_fn \
                if i == self.num_layers - 1 and self.pre_activation \
                else self.convs[i+self.pre_activation].activation_fn
            self.convs[i].reset_parameters(output_activation_fn)
        for conv in self.residual_skip_convs.values():
            conv.reset_parameters('linear')

    def freeze(self, num_layers=None, freeze_norm_stats=True):
        num_layers = len(self.convs) if num_layers is None else min(num_layers, len(self.convs))
        if num_layers == 0:
            return
        assert num_layers > 0, num_layers
        if self.input_norm is not None:
            self.input_norm.freeze(freeze_stats=freeze_norm_stats)
        if isinstance(self.input_activation_fn, torch.nn.PReLU):
            self.input_activation_fn.weight.requires_grad = False
        for i in range(num_layers):
            self.convs[i].freeze(freeze_norm_stats=freeze_norm_stats)
        for key, conv in self.residual_skip_convs.items():
            dst_idx = int(key.split('->')[1])
            if dst_idx <= num_layers:
                conv.freeze(freeze_norm_stats=freeze_norm_stats)
        if num_layers >= len(self.convs):
            if self.output_norm is not None:
                self.output_norm.freeze(freeze_stats=freeze_norm_stats)
            if isinstance(self.output_activation_fn, torch.nn.PReLU):
                self.output_activation_fn.weight.requires_grad = False

    def forward(
            self, x, sequence_lengths=None,
            target_shape=None, target_sequence_lengths=None, pool_indices=None,
            state=None,
    ):
        """

        Args:
            x:
            sequence_lengths:
            target_shape:
            target_sequence_lengths:
            pool_indices:
            state:

        Returns:

        """
        assert x.dim() == (3 + self.is_2d()), (x.shape, self.is_2d())
        if state is None:
            state = len(self.convs)*[None]
        else:
            assert len(state) == len(self.convs)
        if not self.is_transpose():
            assert target_shape is None, target_shape
            assert target_sequence_lengths is None, target_sequence_lengths
            assert pool_indices is None, pool_indices.shape
        if target_shape is not None:
            expected_input_shape, *output_shapes = self.get_shapes(target_shape=target_shape)
            assert (np.array(x.shape) == expected_input_shape).all(), (x.shape, expected_input_shape)
        else:
            output_shapes = None
        if target_sequence_lengths is not None:
            expected_sequence_lengths, *output_sequence_lengths = self.get_sequence_lengths(
                target_sequence_lengths=target_sequence_lengths
            )
            assert sequence_lengths is not None and (
                np.array(sequence_lengths) == np.array(expected_sequence_lengths)
            ).all(), (sequence_lengths, expected_sequence_lengths)
        else:
            output_sequence_lengths = None
        pool_indices = to_list(copy(pool_indices), self.num_layers)[::-1]
        skip_signals = []
        for i, conv in enumerate(self.convs):
            if self.is_transpose() and any(np.array(to_list(self.pool_sizes[i])) > 1):
                x, sequence_lengths = self.unpool_cls(
                    pool_size=self.pool_sizes[i],
                    stride=self.pool_strides[i],
                    pad_type=self.pad_types[i]
                )(
                    x, sequence_lengths=sequence_lengths, indices=pool_indices[i]
                )

                for src_idx, x_ in enumerate(skip_signals):
                    if x_ is not None:
                        skip_signals[src_idx], _ = self.unpool_cls(
                            pool_size=self.pool_strides[i],
                            pad_type=self.pad_types[i],
                        )(x_)

            if i == 0 and self.input_norm is not None:
                x = self.input_norm(x, sequence_lengths=sequence_lengths)
                x = self.input_activation_fn(x)
            skip_signals.append(None if not (self.residual_connections[i]+self.dense_connections[i]) else x)

            if conv.return_state:
                x, sequence_lengths, state[i] = conv(x, sequence_lengths=sequence_lengths, state=state[i])
            else:
                x, sequence_lengths = conv(x, sequence_lengths=sequence_lengths, state=state[i])
            for src_idx, x_ in enumerate(skip_signals):
                if x_ is None:
                    continue
                if any(np.array(to_list(self.strides[i])) > 1):
                    if self.is_transpose():
                        x_, _ = self.unpool_cls(
                            pool_size=self.strides[i],
                            pad_type=self.pad_types[i],
                        )(x_)
                    else:
                        x_, *_ = self.pool_cls(
                            pool_type='avg',
                            pool_size=self.strides[i],
                            pad_type=self.pad_types[i],
                        )(x_)
                    skip_signals[src_idx] = x_

                if i+1 in self.dense_connections[src_idx]:
                    size = np.array(x_.shape[2:]) - np.array(x.shape[2:])
                    assert all(size >= 0), size
                    if any(size > 0):
                        x_ = Trim(side='end')(x_, size=size)
                    x = torch.cat((x, x_), dim=1)
            for src_idx, x_ in enumerate(skip_signals):
                if x_ is not None and i+1 in self.residual_connections[src_idx]:
                    size = np.array(x_.shape[2:]) - np.array(x.shape[2:])
                    assert all(size >= 0), size
                    if any(size > 0):
                        assert self.is_transpose() and output_shapes is not None
                        x_ = Trim(side='end')(x_, size=size)
                    if f'{src_idx}->{i+1}' in self.residual_skip_convs:
                        x_, _ = self.residual_skip_convs[f'{src_idx}->{i + 1}'](x_)
                    x = x + x_
                destinations = self.dense_connections[src_idx] + self.residual_connections[src_idx]
                if destinations and max(destinations) <= i+1:
                    skip_signals[src_idx] = None

            if i == self.num_layers - 1 and self.output_norm is not None:
                x = self.output_norm(x, sequence_lengths=sequence_lengths)
                x = self.output_activation_fn(x)

            if not self.is_transpose() and any(np.array(to_list(self.pool_sizes[i])) > 1):
                assert self.pool_types[i] is not None
                x, sequence_lengths, pool_indices[i] = self.pool_cls(
                    pool_type=self.pool_types[i],
                    pool_size=self.pool_sizes[i],
                    stride=self.pool_strides[i],
                    pad_type=self.pad_types[i],
                )(x, sequence_lengths=sequence_lengths)

                for src_idx, x_ in enumerate(skip_signals):
                    if x_ is not None:
                        skip_signals[src_idx], *_ = self.pool_cls(
                            pool_type='avg',
                            pool_size=self.pool_strides[i],
                            pad_type=self.pad_types[i],
                        )(x_)

            if output_shapes is not None:
                assert self.is_transpose()
                # assert matching batch and channels dims
                assert x.shape[:2] == tuple(output_shapes[i])[:2], (x.shape, output_shapes[i])
                size = np.array(x.shape[2:]) - np.array(output_shapes[i][2:])
                assert all((size >= 0)), (
                    f'Desired output_shape {np.abs(size)} bins greater than actual output shape. '
                    f'Maybe you did not use padding.'
                )
                if any(size > 0):
                    x = Trim(side='end')(x, size=size*(size > 0))

            if output_sequence_lengths is not None:
                assert self.is_transpose()
                sequence_lengths = output_sequence_lengths[i]

        outputs = [x, sequence_lengths]
        if self.return_pool_indices:
            outputs.append(pool_indices)
        if self.return_state:
            outputs.append(state)
        return tuple(outputs)

    @classmethod
    def get_transpose_config(cls, config, transpose_config=None):
        """
        generates config of a symmetric transposed CNN. Useful with autoencoders.

        Args:
            config:
            transpose_config:

        Returns:

        """
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
        if 'input_layer' in config.keys():
            transpose_config['output_layer'] = config['input_layer']
        if 'output_layer' in config.keys():
            transpose_config['input_layer'] = config['output_layer']
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
                    else sorted(skip_connections[i])
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
            'kernel_size', 'pad_type', 'dilation', 'stride', 'pool_type', 'pool_size', 'pool_stride', 'norm'
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

    def get_shapes(self, input_shape=None, target_shape=None):
        assert (input_shape is None) ^ (target_shape is None), (input_shape, target_shape)
        if input_shape is not None:
            shapes = [input_shape]
            cur_shape = input_shape
            for i, conv in enumerate(self.convs):
                cur_shape = conv.get_output_shape(cur_shape)
                cur_shape[1] = self.layer_in_channels[i + 1]  # layer channels differ from conv channels with dense skip connections
                if self.pool_types[i] is not None:
                    cur_shape = compute_conv_output_shape(
                        cur_shape,
                        out_channels=cur_shape[1],
                        kernel_size=self.pool_sizes[i],
                        dilation=1,
                        stride=self.pool_strides[i],
                        pad_type=self.pad_types[i],
                        transpose=self.is_transpose()
                    )
                shapes.append(cur_shape)
        else:
            shapes = [target_shape]
            cur_shape = target_shape
            for i, conv in reversed(list(enumerate(self.convs))):
                cur_shape = np.copy(cur_shape)
                cur_shape[1] = conv.out_channels  # conv channels differ from layer channels with dense skip connections
                cur_shape = conv.get_input_shape(cur_shape)
                if self.pool_types[i] is not None:
                    cur_shape = compute_conv_output_shape(
                        cur_shape,
                        out_channels=cur_shape[1],
                        kernel_size=self.pool_sizes[i],
                        dilation=1,
                        stride=self.pool_strides[i],
                        pad_type=self.pad_types[i],
                        transpose=not self.is_transpose()
                    )
                shapes.append(cur_shape)
            shapes = shapes[::-1]
        return shapes

    def get_sequence_lengths(
            self, input_sequence_lengths=None, target_sequence_lengths=None
    ):
        assert (input_sequence_lengths is None) ^ (target_sequence_lengths is None), (
            input_sequence_lengths, target_sequence_lengths
        )
        if input_sequence_lengths is not None:
            seq_lens = [input_sequence_lengths]
            cur_seq_len = input_sequence_lengths
            for i, conv in enumerate(self.convs):
                cur_seq_len = conv.get_output_sequence_lengths(cur_seq_len)
                if self.pool_types[i] is not None:
                    cur_seq_len = compute_conv_output_sequence_lengths(
                        cur_seq_len,
                        kernel_size=self.pool_sizes[i],
                        dilation=1,
                        stride=self.pool_strides[i],
                        pad_type=self.pad_types[i],
                        transpose=self.is_transpose()
                    )
                seq_lens.append(cur_seq_len)
        else:
            seq_lens = [target_sequence_lengths]
            cur_seq_len = target_sequence_lengths
            for i, conv in reversed(list(enumerate(self.convs))):
                cur_seq_len = conv.get_input_sequence_lengths(cur_seq_len)
                if self.pool_types[i] is not None:
                    cur_seq_len = compute_conv_output_sequence_lengths(
                        cur_seq_len,
                        kernel_size=self.pool_sizes[i],
                        dilation=1,
                        stride=self.pool_strides[i],
                        pad_type=self.pad_types[i],
                        transpose=not self.is_transpose()
                    )
                seq_lens.append(cur_seq_len)
            seq_lens = seq_lens[::-1]
        return seq_lens

    def get_receptive_field(self):
        receptive_field = np.ones(1+self.is_2d()).astype(np.int)
        for i in reversed(range(self.num_layers)):
            receptive_field *= np.array(self.pool_strides[i])
            receptive_field += np.array(self.pool_sizes[i]) - np.array(self.pool_strides[i])
            receptive_field *= np.array(self.strides[i])
            receptive_field += 1 + (np.array(self.kernel_sizes[i])-1) * self.dilations[i] - np.array(self.strides[i])
        return receptive_field


class CNN1d(_CNN):
    conv_cls = Conv1d


class CNNTranspose1d(_CNN):
    conv_cls = ConvTranspose1d


class CNN2d(_CNN):
    conv_cls = Conv2d


class CNNTranspose2d(_CNN):
    conv_cls = ConvTranspose2d


def resnet50(in_channels, out_channels, out_pool_size=1, activation_fn='relu', pre_activation=False, norm='batch'):
    """

    Args:
        in_channels:
        out_channels:
        activation_fn:
        pre_activation:
        norm:

    Returns:

    >>> resnet = resnet50(3,1000, out_pool_size=8)
    >>> y, _ = resnet(torch.randn(1, 3, 256, 256))
    >>> y.shape

    """
    out_channels = [64] + 3*3*[64] + 4*3*[128] + 6*3*[256] + 3*3*[512] + [out_channels]
    assert len(out_channels) == 50
    for i in range(3, 50, 3):
        out_channels[i] *= 4
    kernel_size = [7]+49*[1]
    for i in range(2, 50, 3):
        kernel_size[i] *= 3
    stride = [2] + 3*3*[1] + [2] + (4*3-1)*[1] + [2] + (6*3-1)*[1] + [2] + 3*3*[1]
    pool_size = [3] + 47*[1] + [out_pool_size] + [1]
    pool_stride = [2] + 47*[1] + [out_pool_size] + [1]
    pool_type = ['max'] + 47 * [None] + ['avg'] + [None]
    residual_connections = 50*[None]
    for i in range(1, 48, 3):
        residual_connections[i] = i+3
    return CNN2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        pool_size=pool_size,
        pool_stride=pool_stride,
        pool_type=pool_type,
        residual_connections=residual_connections,
        activation_fn=activation_fn,
        pre_activation=pre_activation,
        norm=norm,
    )
