import torch
from padertorch.contrib.je.modules.conv import Conv1d, ConvTranspose1d
from padertorch.contrib.je.modules.conv import Conv2d, ConvTranspose2d
from padertorch.contrib.je.modules.conv import MultiScaleConv1d, MultiScaleConvTranspose1d
from padertorch.contrib.je.modules.conv import MultiScaleConv2d, MultiScaleConvTranspose2d
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.conv import CNN2d, CNNTranspose2d
from padertorch.contrib.je.modules.conv import MultiScaleCNN1d, MultiScaleCNNTranspose1d
from padertorch.contrib.je.modules.conv import MultiScaleCNN2d, MultiScaleCNNTranspose2d
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
    in_channels = 1

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


def run(x, enc_cls, dec_cls, kwargs_sweep):

    in_channels = x.shape[1]
    out_channels = 10

    for kwargs in sweep(kwargs_sweep):
        enc = enc_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            norm='batch',
            return_pool_data=True,
            **kwargs
        )
        z, pool_indices, shapes = enc(x)
        # print(z.shape)
        assert all(z.shape[2:] == enc.get_out_shape(x.shape[2:])), (z.shape[2:], enc.get_out_shape(x.shape[2:]))
        deconv = dec_cls(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=3,
            norm='batch',
            **kwargs
        )
        if isinstance(pool_indices, list):
            pool_indices = pool_indices[::-1]
        if isinstance(shapes, list):
            shapes = shapes[::-1]
        x_hat = deconv(z, pool_indices, shapes)
        assert x_hat.shape == x.shape, (x_hat.shape, x.shape)


def test_conv_1d_shapes():
    for num_frames in [129, 140]:
        run(
            get_input_1d(num_frames),
            Conv1d,
            ConvTranspose1d,
            [
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None])
            ]
        )


def test_conv_2d_shapes():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        run(
            get_input_2d(num_frames, num_features),
            Conv2d,
            ConvTranspose2d,
            [
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None, (None, 'both')])
            ]
        )


def test_msconv_1d_shapes():
    for num_frames in [129, 140]:
        run(
            get_input_1d(num_frames),
            MultiScaleConv1d,
            MultiScaleConvTranspose1d,
            [
                ('num_scales', [1, 2]),
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None]),
                ('hidden_channels', [16])
            ]
        )


def test_msconv_2d_shapes():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        run(
            get_input_2d(num_frames, num_features),
            MultiScaleConv2d,
            MultiScaleConvTranspose2d,
            [
                ('num_scales', [1, 2]),
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None, (None, 'both')]),
                ('hidden_channels', [16])
            ]
        )


def test_cnn_1d_shapes():
    for num_frames in [129, 140]:
        run(
            get_input_1d(num_frames),
            CNN1d,
            CNNTranspose1d,
            [
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None]),
                ('num_layers', [3]),
                ('hidden_channels', [16])
            ]
        )


def test_cnn_2d_shapes():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        run(
            get_input_2d(num_frames, num_features),
            CNN2d,
            CNNTranspose2d,
            [
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None, 3*[(None, 'both')]]),
                ('num_layers', [3]),
                ('hidden_channels', [16])
            ]
        )


def test_mscnn_1d_shapes():
    for num_frames in [129, 140]:
        run(
            get_input_1d(num_frames),
            MultiScaleCNN1d,
            MultiScaleCNNTranspose1d,
            [
                ('num_scales', [1, 2]),
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None]),
                ('num_layers', [3]),
                ('hidden_channels', [16])
            ]
        )


def test_mscnn_2d_shapes():
    for num_frames, num_features in zip(
            [129, 140],
            [140, 129]
    ):
        run(
            get_input_2d(num_frames, num_features),
            MultiScaleCNN2d,
            MultiScaleCNNTranspose2d,
            [
                ('num_scales', [1, 2]),
                ('stride', [1, 2]),
                ('pooling', ['max', 'avg']),
                ('pool_size', [1, 2]),
                ('padding', ['both', None, 3*[(None, 'both')]]),
                ('num_layers', [3]),
                ('hidden_channels', [16])
            ]
        )


def test_get_transpose_config():
    for cls, cls_transpose in zip(
            [CNN1d, MultiScaleCNN1d, CNN2d, MultiScaleCNN2d],
            [CNNTranspose1d, MultiScaleCNNTranspose1d,
             CNNTranspose2d, MultiScaleCNNTranspose2d]
    ):
        config = {
            'factory': cls,
            'in_channels': 3,
            'out_channels': 10,
            'kernel_size': 3,
            'hidden_channels': [1, 2, 3, 4]
        }
        expected_transpose_config = {
            'factory': cls_transpose,
            'in_channels': 10,
            'out_channels': 3,
            'kernel_size': 3,
            'hidden_channels': [4, 3, 2, 1]
        }
        transpose_config = cls.get_transpose_config(config)
        assert transpose_config == expected_transpose_config
        transpose_transpose_config = cls_transpose.get_transpose_config(transpose_config)
        assert transpose_transpose_config == config

    for cls_2d, cls_1d, cls_transpose_1d, cls_transpose_2d in zip(
            [CNN2d, MultiScaleCNN2d],
            [CNN1d, MultiScaleCNN1d],
            [CNNTranspose1d, MultiScaleCNNTranspose1d],
            [CNNTranspose2d, MultiScaleCNNTranspose2d]
    ):
        config = dict(
            factory=HybridCNN,
            cnn_2d={
                'factory': cls_2d,
                'in_channels': 3,
                'out_channels': 10,
                'kernel_size': 3,
                'hidden_channels': [1, 2, 3, 4]
            },
            cnn_1d={
                'factory': cls_1d,
                'in_channels': 3,
                'out_channels': 10,
                'kernel_size': 3,
                'hidden_channels': [1, 2, 3, 4]
            }
        )
        expected_transpose_config = dict(
            factory=HybridCNNTranspose,
            cnn_transpose_2d={
                'factory': cls_transpose_2d,
                'in_channels': 10,
                'out_channels': 3,
                'kernel_size': 3,
                'hidden_channels': [4, 3, 2, 1]
            },
            cnn_transpose_1d={
                'factory': cls_transpose_1d,
                'in_channels': 10,
                'out_channels': 3,
                'kernel_size': 3,
                'hidden_channels': [4, 3, 2, 1]
            }
        )
        transpose_config = HybridCNN.get_transpose_config(config)
        assert transpose_config == expected_transpose_config
        transpose_transpose_config = HybridCNNTranspose.get_transpose_config(transpose_config)
        assert transpose_transpose_config == config
