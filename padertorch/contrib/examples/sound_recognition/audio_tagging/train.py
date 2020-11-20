"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.sound_recognition.audio_tagging.train
"""
import os
from pathlib import Path

from padertorch import Trainer
from padertorch.contrib.examples.sound_recognition.audio_tagging.data import \
    get_datasets
from padertorch.contrib.examples.sound_recognition.audio_tagging.model import \
    WALNet
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import Adam
from sacred import Experiment, commands

ex = Experiment('audio_tagging')


@ex.config
def config():
    database_json = (
        str((Path(os.environ['NT_DATABASE_JSONS_DIR']) / 'audio_set.json').expanduser())
        if 'NT_DATABASE_JSONS_DIR' in os.environ else None
    )
    assert database_json is not None, (
        'database_json cannot be None.\n'
        'Either start the training with "python -m padertorch.contrib.examples.'
        'audio_synthesis.wavenet.train with database_json=</path/to/json>" '
        'or make sure there is an environment variable "NT_DATABASE_JSONS_DIR"'
        'pointing to a directory with a "audio_set.json" in it (see README '
        'for the JSON format).'
    )
    training_set = 'balanced_train'
    audio_reader = {
        'source_sample_rate': 44_100,
        'target_sample_rate': 44_100,
    }
    stft = {
        'shift': 882,
        'window_length': 2*882,
        'size': 2048,
        'fading': None,
        'pad': False,
    }
    num_workers = 8
    batch_size = 24
    max_padding_rate = .05
    trainer = {
        'model': {
            'factory': WALNet,
            'sample_rate': audio_reader['target_sample_rate'],
            'fft_length': stft['size'],
            'output_size': 527,
        },
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
            'gradient_clipping': 60.,
        },
        'storage_dir': get_new_storage_dir(
            'audio_tagging', id_naming='time', mkdir=False
        ),
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1_000, 'iteration'),
        'stop_trigger': (50_000, 'iteration'),
    }
    trainer = Trainer.get_config(trainer)
    validation_metric = 'map'
    maximize_metric = True
    resume = False


@ex.automain
def main(
        _run, _log, trainer, database_json, training_set,
        validation_metric, maximize_metric,
        audio_reader, stft, num_workers, batch_size, max_padding_rate, resume
):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(
        _run.config, _log, config_filename=str(storage_dir / 'config.json')
    )

    training_data, validation_data, _ = get_datasets(
        database_json=database_json, min_signal_length=1.5,
        audio_reader=audio_reader, stft=stft, num_workers=num_workers,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        training_set=training_set, storage_dir=storage_dir,
        mixup_probs=(1/2, 1/2), max_mixup_length=12., min_mixup_overlap=.8,
    )

    trainer.test_run(training_data, validation_data)
    trainer.register_validation_hook(
        validation_data, metric=validation_metric, maximize=maximize_metric
    )
    trainer.train(training_data, resume=resume)
