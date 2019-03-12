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
        'transforms.reader.factory': ReadAudio,
        'transforms.reader.input_sample_rate': 16000,
        'transforms.reader.target_sample_rate': 16000,
        'transforms.stft.factory': STFT,
        'transforms.stft.frame_step': 160,
        'transforms.stft.frame_length': 400,
        'transforms.stft.fft_length': 512,
        'transforms.stft.keep_input': True,
        'transforms.spectrogram.factory': Spectrogram,
        'transforms.mel_transform.factory': MelTransform,
        'transforms.mel_transform.n_mels': 80,
        'transforms.declutter.factory': Declutter,
        'transforms.declutter.required_keys': ['audio_data', 'spectrogram'],
        'transforms.declutter.dtypes.audio_data': 'float32',
        'transforms.declutter.dtypes.spectrogram': 'float32',
        'transforms.declutter.permutations.spectrogram': (0, 2, 1),
        'normalizer.factory': GlobalNormalize,
        'normalizer.axes': {'spectrogram': 2},
        'normalizer.std_reduce_axes': {'spectrogram': 1},
        'subset_size': 1000,
        'storage_dir': storage_dir,
        'segmenters.factory': SegmentAxis,
        'segmenters.axis': -1,
        'segmenters.segment_steps.audio_data': 16000,
        'segmenters.segment_steps.spectrogram': 100,
        'segmenters.segment_lengths.audio_data': 16000,
        'segmenters.segment_lengths.spectrogram': 102,
        'segmenters.pad': False,
        'fragmenter.factory': Fragmenter,
        'fragmenter.split_axes.audio_data': (0, 1),
        'fragmenter.split_axes.spectrogram': (0, 2),
        'fragmenter.squeeze': True,
        'max_workers': 16,
        'prefetch_buffer': 10,
        'shuffle_buffer': 1000,
        'batch_size': 4
    })
    data_config['transforms']['mel_transform']['sample_rate'] = \
        data_config['transforms']['reader']['target_sample_rate']
    data_config['transforms']['mel_transform']['fft_length'] = \
        data_config['transforms']['stft']['fft_length']
    DataProvider.get_config(data_config)

    # Trainer configuration
    train_config = deflatten({
        'model.factory':  WaveNet,
        'model.audio_key': 'audio_data',
        'model.feature_key': 'spectrogram',
        'model.wavenet.n_cond_channels': data_config['transforms'][
            'mel_transform']['n_mels'],
        'model.wavenet.upsamp_window': data_config['transforms'][
            'stft']['frame_length'],
        'model.wavenet.upsamp_stride': data_config['transforms'][
            'stft']['frame_step'],
        'model.sample_rate': data_config['transforms']['reader']['target_sample_rate'],
        'optimizer.factory': Adam,
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (100, 'epoch')
    })
    Trainer.get_config(train_config)


@ex.automain
def train(data_config, train_config):
    data_config['transforms'] = OrderedDict(
        reader=data_config['transforms']['reader'],
        stft=data_config['transforms']['stft'],
        spectrogram=data_config['transforms']['spectrogram'],
        mel_transform=data_config['transforms']['mel_transform'],
        declutter=data_config['transforms']['declutter']
    )
    data_provider = DataProvider.from_config(data_config)

    train_iter = data_provider.get_train_iterator()
    validation_iter = data_provider.get_validation_iterator()

    trainer = Trainer.from_config(train_config)
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
