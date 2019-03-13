"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from collections import OrderedDict
from pathlib import Path

from paderbox.utils.nested import deflatten
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data import DataProvider
from padertorch.contrib.je.transforms import ReadAudio, STFT, Spectrogram, \
    MelTransform, Declutter, GlobalNormalize, SegmentAxis, Fragmenter
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
        'test_set_names': 'test_core',
        'transforms.0.factory': ReadAudio,
        'transforms.0.input_sample_rate': 16000,
        'transforms.0.target_sample_rate': 16000,
        'transforms.1.factory': STFT,
        'transforms.1.frame_step': 160,
        'transforms.1.frame_length': 400,
        'transforms.1.fft_length': 512,
        'transforms.1.keep_input': True,
        'transforms.2.factory': Spectrogram,
        'transforms.3.factory': MelTransform,
        'transforms.3.n_mels': 80,
        'transforms.4.factory': Declutter,
        'transforms.4.required_keys': ['audio_data', 'spectrogram'],
        'transforms.4.dtypes.audio_data': 'float32',
        'transforms.4.dtypes.spectrogram': 'float32',
        'transforms.4.permutations.spectrogram': (0, 2, 1),
        'transforms.5.factory': SegmentAxis,
        'transforms.5.axis': -1,
        'transforms.5.segment_steps.audio_data': 16000,
        'transforms.5.segment_steps.spectrogram': 100,
        'transforms.5.segment_lengths.audio_data': 16000,
        'transforms.5.segment_lengths.spectrogram': 102,
        'transforms.5.pad': False,
        'normalizer.factory': GlobalNormalize,
        'normalizer.axes.spectrogram': (2, 3),
        'normalizer.std_reduce_axes.spectrogram': 1,
        'subset_size': 1000,
        'storage_dir': storage_dir,
        'fragmenter.factory': Fragmenter,
        'fragmenter.split_axes.audio_data': (0, 1),
        'fragmenter.split_axes.spectrogram': (0, 2),
        'fragmenter.squeeze': True,
        'max_workers': 16,
        'prefetch_buffer': 10,
        'shuffle_buffer': 1000,
        'batch_size': 4
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
