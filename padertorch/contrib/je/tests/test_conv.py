from copy import copy

import numpy as np
import torch
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.conv import CNN2d, CNNTranspose2d
from padertorch.contrib.je.modules.conv import Conv1d, ConvTranspose1d
from padertorch.contrib.je.modules.conv import Conv2d, ConvTranspose2d
from padertorch.contrib.je.modules.hybrid import HybridCNN, HybridCNNTranspose


def get_input_1d(num_frames=129):

    batch_size = 8
    in_channels = 40

    shape = (batch_size, in_channels, num_frames)
    x = torch.ones(shape)
    return x


def get_input_2d(num_frames=129, num_features=140):
    batch_size = 8
    in_channels = 3

    shape = (batch_size, in_channels, num_features, num_frames)
    x = torch.ones(shape)
    return x


def sweep(kwargs_sweep):
    kwargs_sweep = copy(kwargs_sweep)
    key, values = kwargs_sweep.pop(0)
    for value in values:
        kwargs = {key: value}
        if len(kwargs_sweep) > 0:
            for d in sweep(kwargs_sweep):
                kwargs = copy(kwargs)
                kwargs.update(d)
                yield kwargs
        else:
            yield kwargs


def run_conv_sweep(x, enc_cls, dec_cls, kwargs_sweep):
    in_channels = x.shape[1]
    for kwargs in sweep(kwargs_sweep):
        out_channels = kwargs.pop('out_channels')
        enc = enc_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            norm='batch',
            **kwargs
        )
        seq_len_in = x.shape[0]*[x.shape[-1]]
        z, seq_len = enc(x, sequence_lengths=seq_len_in)
        seq_lens_out = z.shape[0]*[z.shape[-1]]
        # print(z.shape)
        assert all(z.shape == enc.get_output_shape(x.shape)), (
            z.shape, enc.get_shapes(x.shape), kwargs
        )
        assert all(seq_lens_out == enc.get_output_sequence_lengths(seq_len_in)), (
            seq_lens_out, enc.get_output_sequence_lengths(seq_len_in), kwargs
        )
        dec = dec_cls(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=3,
            norm='batch',
            **kwargs
        )
        x_hat, seq_len = dec(
            z, sequence_lengths=seq_len,
            output_shape=x.shape, output_sequence_lengths=seq_len_in
        )
        assert x_hat.shape == x.shape, (x_hat.shape, x.shape)


def test_conv_1d():
    for num_frames in [129, 140]:
        run_conv_sweep(
            get_input_1d(num_frames),
            Conv1d,
            ConvTranspose1d,
            [
                ('stride', [1, 2]),
                ('pad_side', ['both', None]),
                ('out_channels', [10]),
            ]
        )


def test_conv_2d():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        run_conv_sweep(
            get_input_2d(num_frames, num_features),
            Conv2d,
            ConvTranspose2d,
            [
                ('stride', [1, 2]),
                ('pad_side', ['both', None, (None, 'both')]),
                ('out_channels', [10]),
            ]
        )


def run_cnn_sweep(x, enc_cls, kwargs_sweep, *, decode=True):
    x.requires_grad = True
    for kwargs in sweep(kwargs_sweep):
        enc = enc_cls(
            return_pool_indices=True,
            **kwargs
        )
        seq_len_in = x.shape[0]*[x.shape[-1]]
        if x.grad is not None:
            x.grad.zero_()
        z, seq_len, pool_indices = enc(x, sequence_lengths=seq_len_in)
        shapes_enc = enc.get_shapes(input_shape=x.shape)
        seq_lens_enc = enc.get_sequence_lengths(input_sequence_lengths=seq_len_in)
        expected_seq_len = z.shape[0]*[z.shape[-1]]
        if kwargs['norm'] is None and kwargs['pool_type'] == 'avg':
            if enc.is_2d():
                z[..., z.shape[-2]//2, z.shape[-1]//2].sum().backward()
            else:
                z[..., z.shape[-1]//2].sum().backward()
            expected_rf = np.abs(x.grad.data.numpy().sum((0,1))) > 1e-6
            expected_rf = [expected_rf.sum(ax).max() for ax in range(1+enc.is_2d())]
            rf = enc.get_receptive_field()
            assert all(rf == expected_rf), (rf, expected_rf, kwargs)
        # print(z.shape)
        assert all(z.shape == shapes_enc[-1]), (z.shape, shapes_enc, kwargs)
        assert all(expected_seq_len == seq_lens_enc[-1]), (expected_seq_len, seq_lens_enc, kwargs)
        if decode:
            kwargs = copy(kwargs)
            kwargs['factory'] = enc_cls
            transpose_kwargs = enc.get_transpose_config(kwargs)
            dec_cls = transpose_kwargs.pop('factory')
            dec = dec_cls(**transpose_kwargs)
            shapes_dec = dec.get_shapes(target_shape=x.shape)
            assert (np.array(shapes_dec) == np.array(shapes_enc)[::-1]).all(), (shapes_dec, shapes_enc)
            seq_lens_dec = dec.get_sequence_lengths(target_sequence_lengths=seq_len_in)
            assert (np.array(seq_lens_dec) == np.array(seq_lens_enc)[::-1]).all(), (seq_lens_enc, seq_lens_dec)
            x_hat, seq_len = dec(
                z, sequence_lengths=seq_len,
                target_shape=x.shape,
                target_sequence_lengths=seq_len_in,
                pool_indices=pool_indices
            )
            assert x_hat.shape == x.shape, (x_hat.shape, x.shape, kwargs)
            transpose_kwargs = copy(transpose_kwargs)
            transpose_kwargs['factory'] = dec_cls
            transpose_transpose_kwargs = dec.get_transpose_config(transpose_kwargs)
            assert transpose_transpose_kwargs == kwargs, (
                kwargs, transpose_kwargs, transpose_transpose_kwargs
            )


def test_cnn_1d():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('activation_fn', ['relu', 'prelu']),
                ('pre_activation', [False, True]),
                # ('input_layer', [False]),
                # ('output_layer', [False]),
            ]
        )


def test_cnn_2d():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        x = get_input_2d(num_frames, num_features)
        run_cnn_sweep(
            x,
            CNN2d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('activation_fn', ['relu', 'prelu']),
                ('pre_activation', [False, True]),
                # ('input_layer', [False]),
                # ('output_layer', [False]),
            ]
        )


def test_resnet_1d():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[[1], [2], None]]),
            ]
        )


def test_resnet_2d():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        x = get_input_2d(num_frames, num_features)
        run_cnn_sweep(
            x,
            CNN2d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[[1], [2], None]]),
            ]
        )


def test_densenet_1d():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('dense_connections', [[[1], [2], None]]),
            ],
            decode=False
        )


def test_densenet_2d():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        x = get_input_2d(num_frames, num_features)
        run_cnn_sweep(
            x,
            CNN2d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('dense_connections', [[[1], [2], None]]),
            ],
            decode=False
        )


def test_denseresnet_1d():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[[1], [2], None]]),
                ('dense_connections', [[[1], [2], None]]),
            ],
            decode=False
        )


def test_denseresnet_2d():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        x = get_input_2d(num_frames, num_features)
        run_cnn_sweep(
            x,
            CNN2d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch', None]),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[[1], [2], None]]),
                ('dense_connections', [[[1], [2], None]]),
            ],
            decode=False
        )


def test_get_transpose_config():
    for cls, cls_transpose in zip(
            [CNN1d, CNN2d],
            [CNNTranspose1d, CNNTranspose2d]
    ):
        config = {
            'factory': cls,
            'in_channels': 3,
            'out_channels': [1, 2, 3, 4, 10],
            'kernel_size': 3,
        }
        expected_transpose_config = {
            'factory': cls_transpose,
            'in_channels': 10,
            'out_channels': [4, 3, 2, 1, 3],
            'kernel_size': 3,
        }
        transpose_config = cls.get_transpose_config(config)
        assert transpose_config == expected_transpose_config
        transpose_transpose_config = cls_transpose.get_transpose_config(transpose_config)
        assert transpose_transpose_config == config

    config = dict(
        factory=HybridCNN,
        cnn_2d={
            'factory': CNN2d,
            'in_channels': 3,
            'out_channels': [1, 2, 3, 4, 10],
            'kernel_size': 3,
        },
        cnn_1d={
            'factory': CNN1d,
            'in_channels': 3,
            'out_channels': [1, 2, 3, 4, 10],
            'kernel_size': 3,
        }
    )
    expected_transpose_config = dict(
        factory=HybridCNNTranspose,
        cnn_transpose_2d={
            'factory': CNNTranspose2d,
            'in_channels': 10,
            'out_channels': [4, 3, 2, 1, 3],
            'kernel_size': 3,
        },
        cnn_transpose_1d={
            'factory': CNNTranspose1d,
            'in_channels': 10,
            'out_channels': [4, 3, 2, 1, 3],
            'kernel_size': 3,
        }
    )
    transpose_config = HybridCNN.get_transpose_config(config)
    assert transpose_config == expected_transpose_config
    transpose_transpose_config = HybridCNNTranspose.get_transpose_config(transpose_config)
    assert transpose_transpose_config == config
