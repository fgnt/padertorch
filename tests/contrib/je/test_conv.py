import torch
from padertorch.contrib.je.modules.conv import Conv1d, ConvTranspose1d
from padertorch.contrib.je.modules.conv import Conv2d, ConvTranspose2d
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.conv import CNN2d, CNNTranspose2d
from padertorch.contrib.je.modules.conv import HybridCNN, HybridCNNTranspose
from copy import copy


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
        in_lengths = x.shape[0]*[x.shape[-1]]
        z, seq_len = enc(x, seq_len=in_lengths)
        out_lengths = z.shape[0]*[z.shape[-1]]
        # print(z.shape)
        assert all(z.shape == enc.get_out_shape(x.shape)), (
            z.shape, enc.get_out_shape(x.shape)
        )
        assert all(out_lengths == enc.get_out_lengths(in_lengths)), (
            out_lengths, enc.get_out_lengths(in_lengths)
        )
        dec = dec_cls(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=3,
            norm='batch',
            **kwargs
        )
        x_hat, seq_len = dec(
            z, seq_len=seq_len, out_shape=x.shape, out_lengths=in_lengths
        )
        assert x_hat.shape == x.shape, (x_hat.shape, x.shape)


def test_conv_1d_shapes():
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


def test_conv_2d_shapes():
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


def run_cnn_sweep(x, enc_cls, kwargs_sweep):

    for kwargs in sweep(kwargs_sweep):
        enc = enc_cls(
            return_pool_data=True,
            **kwargs
        )
        in_lengths = x.shape[0]*[x.shape[-1]]
        z, seq_len, shapes, lengths, pool_indices = enc(x, seq_len=in_lengths)
        out_lengths = z.shape[0]*[z.shape[-1]]
        # print(z.shape)
        assert all(z.shape == enc.get_out_shape(x.shape)), (
            z.shape, enc.get_out_shape(x.shape)
        )
        assert all(out_lengths == enc.get_out_lengths(in_lengths)), (
            out_lengths, enc.get_out_lengths(in_lengths)
        )
        kwargs = copy(kwargs)
        kwargs['factory'] = enc_cls
        transpose_kwargs = enc.get_transpose_config(kwargs)
        dec_cls = transpose_kwargs.pop('factory')
        dec = dec_cls(**transpose_kwargs)
        x_hat, seq_len = dec(
            z, seq_len=seq_len,
            out_shapes=shapes, out_lengths=lengths, pool_indices=pool_indices
        )
        assert x_hat.shape == x.shape, (x_hat.shape, x.shape)
        transpose_kwargs = copy(transpose_kwargs)
        transpose_kwargs['factory'] = dec_cls
        transpose_transpose_kwargs = dec.get_transpose_config(transpose_kwargs)
        # ToDo: compare transpose_transpose_kwargs to kwargs
        pass


def test_cnn_1d_shapes():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
            ]
        )


def test_cnn_2d_shapes():
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
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
            ]
        )


def test_resnet_1d_shapes():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[1, 2, None]]),
            ]
        )


def test_resnet_2d_shapes():
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
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[1, 2, None]]),
            ]
        )


def test_densenet_1d_shapes():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('dense_connections', [[1, 2, None]]),
            ]
        )


def test_densenet_2d_shapes():
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
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('dense_connections', [[1, 2, None]]),
            ]
        )


def test_denseresnet_1d_shapes():
    for num_frames in [129, 140]:
        x = get_input_1d(num_frames)
        run_cnn_sweep(
            x,
            CNN1d,
            [
                ('in_channels', [x.shape[1]]),
                ('out_channels', [2*[16] + [10]]),
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[1, 2, None]]),
                ('dense_connections', [[1, 2, None]]),
            ]
        )


def test_denseresnet_2d_shapes():
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
                ('norm', ['batch']),
                ('kernel_size', [3]),
                ('stride', [1, 2]),
                ('pool_type', ['avg', 'max']),
                ('pool_size', [1, 2]),
                ('pad_side', ['both', None, 3*[(None, 'both')]]),
                ('pre_activation', [False, True]),
                ('residual_connections', [[1, 2, None]]),
                ('dense_connections', [[1, 2, None]]),
            ]
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
