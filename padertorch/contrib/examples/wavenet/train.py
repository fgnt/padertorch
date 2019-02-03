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
from padertorch.data.fragmenter import Fragmenter
from padertorch.data.transforms import ReadAudio, Spectrogram
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

nickname = 'wavenet'
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
        'transforms.reader.factory': ReadAudio,
        'transforms.spectrogram.factory': Spectrogram,
        'max_workers': 16,
        'normalize_features': ['spectrogram'],
        'subset_size': 1000,
        'storage_dir': storage_dir,
        'fragmenters.factory': Fragmenter,
        'fragmenters.fragment_steps.audio_data': 16000,
        'fragmenters.fragment_steps.spectrogram': 100,
        'fragmenters.drop_last': True,
        'fragmenters.axis': -1,
        'shuffle_buffer_size': 1000,
        'batch_size': 4
    })
    DataProvider.get_config(data_config)
    data_config['transforms']['spectrogram']['sample_rate'] = \
        data_config['transforms']['reader']['sample_rate']

    # Trainer configuration
    train_config = deflatten({
        'model.factory':  WaveNet,
        'model.wavenet.n_cond_channels': data_config['transforms'][
            'spectrogram']['n_mels'],
        'model.wavenet.upsamp_window': data_config['transforms'][
            'spectrogram']['frame_length'],
        'model.wavenet.upsamp_stride': data_config['transforms'][
            'spectrogram']['frame_step'],
        'model.sample_rate': data_config['transforms']['reader']['sample_rate'],
        'optimizer.factory': Adam,
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (20, 'epoch')
    })
    Trainer.get_config(train_config)


@ex.automain
def train(data_config, train_config):
    data_config['transforms'] = OrderedDict(
        reader=data_config['transforms']['reader'],
        spectrogram=data_config['transforms']['spectrogram']
    )
    data_provider = DataProvider.from_config(data_config)

    def to_tuple(example):
        return example["spectrogram"][0], example["audio_data"]

    train_iter = data_provider.get_train_iterator().map(to_tuple)
    validation_iter = data_provider.get_validation_iterator().map(to_tuple)

    trainer = Trainer.from_config(train_config)
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
