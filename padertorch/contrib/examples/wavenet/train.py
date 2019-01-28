import os
from collections import OrderedDict
from pathlib import Path

from paderbox.utils.nested import deflatten
from padertorch.configurable import Configurable, config_to_instance
from padertorch.contrib.je.data import datasets
from padertorch.data.fragmenter import Fragmenter
from padertorch.data.transforms import ReadAudio, Spectrogram
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from torch.utils.data.dataloader import default_collate

ex = Exp('wavenet')
storage_dir = str(Path(os.environ['STORAGE_DIR']))
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


@ex.config
def config():
    # Data configuration
    data_config = dict(cls=datasets)
    Configurable.get_config(
        out_config=data_config,
        updates={
            'database_name': 'timit',
            'training_set_names': 'train',
            'validation_set_names': 'test_core',
            'transforms': {
                'reader': {'cls': ReadAudio},
                'spectrogram': {'cls': Spectrogram}
            },
            'max_workers': 16,
            'normalize_features': ['spectrogram'],
            'subset_size': 1000,
            'storage_dir': storage_dir,
            'fragmentations': {
                'cls': Fragmenter,
                'kwargs': {
                    'fragment_steps': {
                        'spectrogram': 100, 'audio_data': 16000
                    },
                    'drop_last': True,
                    'axis': -1
                }
            },
            'shuffle_buffer_size': 1000,
            'batch_size': 8,
        }
    )
    data_config['kwargs']['transforms']['spectrogram']['kwargs']['sample_rate'] = \
        data_config['kwargs']['transforms']['reader']['kwargs']['sample_rate']

    # Model configuration
    model_config = dict(cls=WaveNet)
    Configurable.get_config(
        out_config=model_config,
        updates=deflatten(
            {
                'wavenet.kwargs': {
                    'n_cond_channels': data_config['kwargs']['transforms'][
                        'spectrogram']['kwargs']['n_mels'],
                    'upsamp_window': data_config['kwargs']['transforms'][
                        'spectrogram']['kwargs']['frame_length'],
                    'upsamp_stride': data_config['kwargs']['transforms'][
                        'spectrogram']['kwargs']['frame_step']
                },
                'sample_rate': data_config['kwargs']['transforms']['reader'][
                    'kwargs']['sample_rate']
            }
        )
    )

    # Optimizer configuration
    optimizer_config = dict(cls=Adam)
    Configurable.get_config(
        out_config=optimizer_config
    )

    # Trainer configuration
    # TODO: LD beantworten, warum train_config anders als model_config.
    train_config = Trainer.get_signature()
    del train_config['optimizer']
    train_config.update(
        dict(
            summary_trigger=(1000, 'iteration'),
            checkpoint_trigger=(1, 'epoch'),
            max_trigger=(20, 'epoch')
        )
    )


@ex.automain
def train(
        _config, data_config, model_config, optimizer_config, train_config
):

    data_config['kwargs']['transforms'] = OrderedDict(
        reader=data_config['kwargs']['transforms']['reader'],
        spectrogram=data_config['kwargs']['transforms']['spectrogram']
    )
    train_iter, validation_iter = config_to_instance(data_config)

    def to_tuple(example):
        return example["spectrogram"][0], example["audio_data"]

    train_iter = train_iter.map(default_collate).map(to_tuple)
    validation_iter = validation_iter.map(default_collate).map(to_tuple)

    model = config_to_instance(model_config)
    optimizer = config_to_instance(optimizer_config)
    trainer = Trainer(
        model=model,
        storage_dir=storage_dir,
        optimizer=optimizer,
        **train_config
    )
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
