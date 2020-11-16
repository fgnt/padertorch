"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from pathlib import Path

from lazy_dataset.database import JsonDatabase
from padertorch.contrib.examples.audio_synthesis.wavenet.data import \
    prepare_dataset
from padertorch.contrib.examples.audio_synthesis.wavenet.model import WaveNet
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver

ex = Experiment('wavenet')


@ex.config
def config():
    database_json = (
        str((Path(os.environ['NT_DATABASE_JSONS_DIR']) / 'librispeech.json').expanduser())
        if 'NT_DATABASE_JSONS_DIR' in os.environ else None
    )
    assert database_json is not None, (
        'database_json cannot be None.\n'
        'Either start the training with "python -m padertorch.contrib.examples.'
        'audio_synthesis.wavenet.train with database_json=</path/to/json>" '
        'or make sure there is an environment variable "NT_DATABASE_JSONS_DIR"'
        'pointing to a directory with a "librispeech.json" in it (see README '
        'for the JSON format).'
    )
    training_sets = ['train_clean_100', 'train_clean_360', 'train_other_500']
    validation_sets = ['dev_clean', 'dev_other']
    audio_reader = {
        'source_sample_rate': 16000,
        'target_sample_rate': 16000,
    }
    stft = {
        'shift': 200,
        'window_length': 800,
        'size': 1024,
        'fading': 'full',
        'pad': True,
    }
    max_length = 1.
    batch_size = 3
    n_mels = 80
    trainer = {
        'model': {
            'factory': WaveNet,
            'wavenet': {
                'n_cond_channels': n_mels,
                'upsamp_window': stft['window_length'],
                'upsamp_stride': stft['shift'],
                'fading': stft['fading'],
            },
            'sample_rate': audio_reader['target_sample_rate'],
            'fft_length': stft['size'],
            'n_mels': n_mels,
            'fmin': 50
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
        },
        'storage_dir': get_new_storage_dir(
            'wavenet', id_naming='time', mkdir=False
        ),
        'summary_trigger': (1_000, 'iteration'),
        'checkpoint_trigger': (10_000, 'iteration'),
        'stop_trigger': (200_000, 'iteration'),
    }
    trainer = Trainer.get_config(trainer)
    resume = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def main(
        _run, _log, trainer, database_json, training_sets, validation_sets,
        audio_reader, stft, max_length, batch_size, resume
):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(
        _run.config, _log, config_filename=str(storage_dir / 'config.json')
    )

    db = JsonDatabase(database_json)
    training_data = db.get_dataset(training_sets)
    validation_data = db.get_dataset(validation_sets)
    training_data = prepare_dataset(
        training_data, audio_reader=audio_reader, stft=stft,
        max_length=max_length, batch_size=batch_size, shuffle=True
    )
    validation_data = prepare_dataset(
        validation_data, audio_reader=audio_reader, stft=stft,
        max_length=max_length, batch_size=batch_size, shuffle=False
    )

    trainer.test_run(training_data, validation_data)
    trainer.register_validation_hook(validation_data)
    trainer.train(training_data, resume=resume)
