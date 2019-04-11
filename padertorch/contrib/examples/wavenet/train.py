"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from pathlib import Path

from paderbox.utils.nested import deflatten
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data.data_provider import DataProvider
from padertorch.contrib.je.data.transforms import (
    ReadAudio, STFT, Spectrogram, MelTransform, GlobalNormalize, SegmentAxis,
    Declutter, Reshape, Fragmenter
)
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

nickname = 'wavenet-training'
ex = Exp(nickname)
storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / nickname / timeStamped('')[1:]
)
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


@ex.config
def config():
    # Data configuration
    data_config = deflatten({
        'database_name': 'timit',
        'training_set_names': 'train',
        'validation_set_names': 'test_core',
        'transforms': deflatten({
            '0.factory': ReadAudio,
            '0.input_sample_rate': 16000,
            '0.target_sample_rate': 16000,
            '1.factory': STFT,
            '1.frame_step': 160,
            '1.frame_length': 400,
            '1.fft_length': 512,
            '1.keep_input': True,
            '2.factory': Spectrogram,
            '3.factory': MelTransform,
            '3.n_mels': 80,
            '4.factory': GlobalNormalize,
            '4.center_axes.spectrogram': 1,
            '4.scale_axes.spectrogram': (1, 2),
            '5.factory': SegmentAxis,
            '5.axis': 1,
            '5.segment_steps.audio_data': 16000,
            '5.segment_steps.spectrogram': 100,
            '5.segment_lengths.audio_data': 16000,
            '5.segment_lengths.spectrogram': 102,
            '5.pad': True,
            '6.factory': Declutter,
            '6.required_keys': ['audio_data', 'spectrogram'],
            '6.dtypes.audio_data': 'float32',
            '6.dtypes.spectrogram': 'float32',
            '7.factory': Reshape,
            '7.permutations.spectrogram': (0, 1, 3, 2),
            '8.factory': Fragmenter,
            '8.split_axes.audio_data': (0, 1),
            '8.split_axes.spectrogram': (0, 1),
            '8.squeeze': True
        }),
        'fragment': True,
        'max_workers': 16,
        'prefetch_buffer': 10,
        'shuffle_buffer': 1000,
        'batch_size': 3
    })
    data_config['transforms']['3']['sample_rate'] = \
        data_config['transforms']['0']['target_sample_rate']
    data_config['transforms']['3']['fft_length'] = \
        data_config['transforms']['1']['fft_length']
    DataProvider.get_config(data_config)

    # Trainer configuration
    train_config = deflatten({
        'model.factory':  WaveNet,
        'model.audio_key': 'audio_data',
        'model.feature_key': 'spectrogram',
        'model.wavenet.n_cond_channels': data_config['transforms']['3'][
            'n_mels'],
        'model.wavenet.upsamp_window': data_config['transforms']['1'][
            'frame_length'],
        'model.wavenet.upsamp_stride': data_config['transforms']['1'][
            'frame_step'],
        'model.sample_rate': data_config['transforms']['0'][
            'target_sample_rate'],
        'optimizer.factory': Adam,
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (100, 'epoch')
    })
    Trainer.get_config(train_config)


@ex.automain
def train(data_config, train_config):
    data_provider = DataProvider.from_config(data_config)

    train_iter = data_provider.get_train_iterator()
    validation_iter = data_provider.get_validation_iterator()

    trainer = Trainer.from_config(train_config)
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
