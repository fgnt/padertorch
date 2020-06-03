import os
from pathlib import Path

import numpy as np
from padertorch.contrib.examples.audio_tagging.data import get_datasets
from padertorch.contrib.examples.audio_tagging.models import CRNN
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.modules.augment import (
    MelWarping, LogTruncNormalSampler, TruncExponentialSampler
)
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

nickname = 'audio_tagging'
ex = Exp(nickname)
storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / nickname / timeStamped('')[1:]
)
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


@ex.config
def config():
    resume = False

    # Data configuration
    audio_reader = {
        'source_sample_rate': None,
        'target_sample_rate': 44100,
    }
    stft = {
        'shift': 882,
        'window_length': 2*882,
        'size': 2048,
        'fading': None,
        'pad': False,
    }

    batch_size = 16
    num_workers = 8
    prefetch_buffer = 10 * batch_size
    max_total_size = None
    max_padding_rate = 0.1
    bucket_expiration = 1000 * batch_size

    # Trainer configuration
    trainer = {
        'model': {
            'factory':  CRNN,
            'feature_extractor': {
                'sample_rate': audio_reader['target_sample_rate'],
                'fft_length': stft['size'],
                # 'stft_scale_window': 100,
                # 'stft_scale_eps': .1,
                'scale_sigma': .8,
                'mixup_prob': .5,
                'n_mels': 128,
                'warping_fn': {
                    'factory': MelWarping,
                    'alpha_sampling_fn': {
                        'factory': LogTruncNormalSampler,
                        'scale': .07,
                        'truncation': np.log(1.3),
                    },
                    'fhi_sampling_fn': {
                        'factory': TruncExponentialSampler,
                        'scale': .5,
                        'truncation': 5.,
                    },
                    # 'fhi_sampling_fn': {
                    #     'factory': UniformSampler,
                    #     'scale': (1. + fhi_excess_rate),
                    #     'center': (1. + fhi_excess_rate) / 2,
                    # },
                },
                'max_resample_rate': 1.,
                'n_time_masks': 1,
                'max_masked_time_steps': 70,
                'max_masked_time_rate': .2,
                'n_mel_masks': 1,
                'max_masked_mel_steps': 16,
                'max_masked_mel_rate': .2,
                'max_noise_scale': .0,
            },
            'cnn_2d': {
                'out_channels': [16, 16, 32, 32, 64, 64, 128, 128, 256],
                'pool_size': [1, 2, 1, 2, 1, 2, 1, (2, 1), (2, 1)],
                # 'residual_connections': [None, 3, None, 5, None, 7, None],
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                # 'norm_kwargs': {
                #     'interpolation_factor': 1.,
                # },
                'activation_fn': 'relu',
                # 'pre_activation': True,
                'dropout': .0,
            },
            'cnn_1d': {
                'out_channels': 3*[256],
                # 'residual_connections': [None, 3, None],
                'input_layer': False,
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                # 'norm_kwargs': {
                #     'interpolation_factor': 1.,
                # },
                'activation_fn': 'relu',
                # 'pre_activation': True,
                'dropout': .0,
            },
            'rnn_fwd': {
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': .0,
            },
            'fcn_fwd': {
                'hidden_size': 256,
                'output_size': 527,
                'activation': 'relu',
                'dropout': .0,
            },
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
            'gradient_clipping': 10.,
            'weight_decay': 1e-6,
        },
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'stop_trigger': (100000, 'iteration')
    }
    Trainer.get_config(trainer)


@ex.automain
def train(
        _run,
        audio_reader, stft,
        num_workers, batch_size, max_padding_rate,
        trainer, resume,
):

    print_config(_run)
    trainer = Trainer.from_config(trainer)

    train_iter, validation_iter = get_datasets(
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers,
        batch_size=batch_size,
        max_padding_rate=max_padding_rate,
        storage_dir=trainer.storage_dir
    )
    trainer.test_run(train_iter, validation_iter)

    trainer.register_validation_hook(
        validation_iter, metric='fscore', maximize=True
    )

    trainer.train(train_iter, resume=resume)
