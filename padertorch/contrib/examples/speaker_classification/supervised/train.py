"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.speaker_classification.supervised.train with database_json=</path/to/json>
"""
import os
from pathlib import Path

from torch.nn import GRU
from sacred import Experiment, commands

from padertorch.io import get_new_storage_dir
from padertorch import Trainer
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.train.optimizer import Adam
from padertorch.modules.normalization import Normalization
from .model import SpeakerClf
from .data import get_datasets


ex = Experiment('speaker_clf')


@ex.config
def defaults():
    database_json = (
        str(Path(os.environ['NT_DATABASE_JSONS_DIR']) / 'librispeech.json')
        if 'NT_DATABASE_JSONS_DIR' in os.environ else None
    )
    assert database_json is not None, (
        'database_json cannot be None.\n'
        'Either start the training with "python -m padertorch.contrib.examples.'
        'speaker_classification.train with database_json=</path/to/json>" '
        'or export "NT_DATABASE_JSONS_DIR" which points to a directory with a '
        '"librispeech.json" prior to training start (see README for the '
        'JSON format).'
    )
    dataset = 'train_clean_100'
    batch_size = 16
    num_speakers = 251
    trainer = {
        'model': {
            'factory': SpeakerClf,
            'feature_extractor': {
                'factory': Normalization,
                'data_format': 'bft',
                'shape': (None, 64, None),
                'statistics_axis': 'bt',
                'independent_axis': None
            },
            'cnn': {
                'factory': CNN1d,
                'in_channels': 64,
                'out_channels': 4 * [512],
                'output_layer': False,
                'kernel_size': 5,
                'norm': 'batch'
            },
            'enc': {
                'factory': GRU,
                'input_size': 512,
                'hidden_size': 256,
                'num_layers': 2,
                'batch_first': True
            },
            'fcn': {
                'factory': fully_connected_stack,
                'input_size': 256,
                'hidden_size': [256],
                'output_size': num_speakers,
                'dropout': 0.
            }
        },
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
        },
        'storage_dir': get_new_storage_dir(
            # do not create when performing test_run
            'speaker_clf', id_naming='time', mkdir=False
        ),
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'stop_trigger': (100_000, 'iteration'),
    }
    trainer = Trainer.get_config(trainer)


@ex.command
def test_run(trainer, database_json, dataset, batch_size, num_speakers):
    # Perform a few training and validation steps to test whether data
    # preperation and the model are working

    trainer = Trainer.from_config(trainer)
    train_set, validate_set, _ = get_datasets(
        None, database_json, dataset, batch_size
    )
    trainer.test_run(train_set, validate_set)


@ex.automain
def main(_run, _log, trainer, database_json, dataset, batch_size):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(
        _run.config, _log, config_filename=str(storage_dir / 'config.json')
    )

    train_set, validate_set, _ = get_datasets(
        storage_dir, database_json, dataset, batch_size
    )

    # Early stopping if loss is not decreasing after three consecutive validation
    # runs. Typically around 20k iterations (13 epochs) with an accuracy >98%
    # on the test set.
    trainer.register_validation_hook(validate_set, early_stopping_patience=3)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)
