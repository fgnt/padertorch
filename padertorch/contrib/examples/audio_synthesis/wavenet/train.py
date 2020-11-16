"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train
"""
from pathlib import Path

from padercontrib.database.librispeech import LibriSpeech
from padertorch.contrib.examples.audio_synthesis.wavenet.data import prepare_dataset
from padertorch.contrib.examples.audio_synthesis.wavenet.model import WaveNet
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands

ex = Experiment('wavenet')


@ex.config
def config():
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


@ex.automain
def main(
        _run, _log, trainer, training_sets, validation_sets,
        audio_reader, stft, max_length, batch_size, resume
):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(
        _run.config, _log, config_filename=str(storage_dir / 'config.json')
    )

    db = LibriSpeech()
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
