"""
Example call on NT infrastructure:

export STORAGE_ROOT=<your desired storage root>
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m padertorch.contrib.examples.source_separation.tasnet.train with database_json=${paths to your JSON}
"""
import os

import numpy as np
import paderbox as pb
import sacred.commands
import torch
from lazy_dataset.database import JsonDatabase
from pathlib import Path
from sacred import Experiment
from sacred.observers.file_storage import FileStorageObserver
from sacred.utils import InvalidConfigError, MissingConfigError

import padertorch as pt
import padertorch.contrib.examples.source_separation.tasnet.model
from padertorch.contrib.neumann.chunking import RandomChunkSingle

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
experiment_name = "tasnet"
ex = Experiment(experiment_name)

JSON_BASE = os.environ.get('NT_DATABASE_JSONS_DIR', None)


@ex.config
def config():
    debug = False
    batch_size = 4  # Runs on 4GB GPU mem. Can safely be set to 12 on 12 GB (e.g., GTX1080)
    chunk_size = 32000  # 4s chunks @8kHz

    train_dataset = "mix_2_spk_min_tr"
    validate_dataset = "mix_2_spk_min_cv"
    target = 'speech_source'
    lr_scheduler_step = 2
    lr_scheduler_gamma = 0.98
    load_model_from = None
    database_json = None
    if database_json is None and JSON_BASE:
        database_json = Path(JSON_BASE) / 'wsj0_2mix_8k.json'

    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')

    feat_size = 64
    encoder_window_size = 16
    trainer = {
        "model": {
            "factory": padertorch.contrib.examples.source_separation.tasnet.TasNet,
            'encoder': {
                'factory': padertorch.contrib.examples.source_separation.tasnet.tas_coders.TasEncoder,
                'window_length': encoder_window_size,
                'feature_size': feat_size,
            },
            'decoder': {
                'factory': padertorch.contrib.examples.source_separation.tasnet.tas_coders.TasDecoder,
                'window_length': encoder_window_size,
                'feature_size': feat_size,
            },
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (100, "epoch"),
        "loss_weights": {
            "si-sdr": 1.0,
            "log-mse": 0.0,
            "log1p-mse": 0.0,
        }
    }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = pt.io.get_new_storage_dir(experiment_name)

    ex.observers.append(FileStorageObserver(
        Path(trainer['storage_dir']) / 'sacred')
    )


@ex.named_config
def win2():
    """
    This is the configuration for the best performing model from the DPRNN
    paper. Training takes very long time with this configuration.
    """
    # The model becomes very memory consuming with this small window size.
    # You might have to reduce the chunk size as well.
    batch_size = 1

    trainer = {
        'model': {
            'encoder': {
                'window_length': 2
            },
            'separator': {
                'window_length': 250,
                'hop_size': 125,  # Half of window length
            },
            'decoder': {
                'window_length': 2
            }
        }
    }


@ex.named_config
def stft():
    """
    Use the STFT and iSTFT as encoder and decoder instead of a learned
    transformation
    """
    trainer = {
        'model': {
            'encoder': {
                'factory': 'padertorch.contrib.examples.source_separation.tasnet.tas_coders.StftEncoder'
            },
            'decoder': {
                'factory': 'padertorch.contrib.examples.source_separation.tasnet.tas_coders.IstftDecoder'
            },
        }
    }


@ex.named_config
def dprnn():
    trainer = {'model': {'separator': {
        'factory': pt.modules.dual_path_rnn.DPRNN,
        'input_size': 64,
        'rnn_size': 128,
        'window_length': 100,
        'hop_size': 50,
        'num_blocks': 6,
    }}}


@ex.named_config
def convnet():
    feat_size = 256
    trainer = {'model': {'separator': dict(
        factory='padertorch.contrib.jensheit.convnet.ConvNet',
        input_size=feat_size,
        X=8, R=4, B=256, H=512, P=3,
        norm="gLN", activation="relu",
    )}}


@ex.named_config
def log_mse():
    """
    Use the log_mse loss
    """
    trainer = {
        'loss_weights': {
            'si-sdr': 0.0,
            'log-mse': 1.0,
        }
    }


@ex.named_config
def log1p_mse():
    """
    Use the log1p_mse loss
    """
    trainer = {
        'loss_weights': {
            'si-sdr': 0.0,
            'log1p-mse': 1.0,
        }
    }


@ex.named_config
def on_wsj0_2mix_max():
    chunk_size = -1
    train_dataset = "mix_2_spk_max_tr"
    validate_dataset = "mix_2_spk_max_cv"


@ex.capture
def pre_batch_transform(inputs):
    return {
        's': np.ascontiguousarray([
            pb.io.load_audio(p)
            for p in inputs['audio_path']['speech_source']
        ], np.float32),
        'y': np.ascontiguousarray(
            pb.io.load_audio(inputs['audio_path']['observation']), np.float32),
        'num_samples': inputs['num_samples'],
        'example_id': inputs['example_id'],
        'audio_path': inputs['audio_path'],
    }


def prepare_iterable(
        db, dataset: str, batch_size, chunk_size, prefetch=True,
        iterator_slice=None
):
    """
    This is re-used in the evaluate script
    """
    iterator = db.get_dataset(dataset)

    if iterator_slice is not None:
        iterator = iterator[iterator_slice]

    chunker = RandomChunkSingle(chunk_size, chunk_keys=('y', 's'), axis=-1)
    iterator = (
        iterator
            .map(pre_batch_transform)
            .map(chunker)
            .shuffle(reshuffle=True)
            .batch(batch_size)
            .map(pt.data.batch.Sorter('num_samples'))
            .map(pt.data.utils.collate_fn)
    )

    # FilterExceptions are only raised inside the chunking code if the
    # example is too short. If min_length <= 0 or chunk_size == -1, no filter
    # exception is raised.
    catch_exception = chunker.chunk_size != -1 and chunker.min_length > 0
    if prefetch:
        iterator = iterator.prefetch(
            8, 16, catch_filter_exception=catch_exception)
    elif catch_exception:
        iterator = iterator.catch()

    return iterator


@ex.capture
def prepare_iterable_captured(
        database_obj, dataset, batch_size, debug, chunk_size,
        iterator_slice=None,
):
    if iterator_slice is None:
        if debug:
            iterator_slice = slice(0, 100, 1)

    return prepare_iterable(
        database_obj, dataset, batch_size, chunk_size,
        prefetch=not debug,
        iterator_slice=iterator_slice,
    )


@ex.capture
def dump_config_and_makefile(_config):
    """
    Dumps the configuration into the experiment dir and creates a Makefile
    next to it. If a Makefile already exists, it does not do anything.
    """
    experiment_dir = Path(_config['trainer']['storage_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"

    if not makefile_path.exists():
        from padertorch.contrib.examples.source_separation.tasnet.templates import \
            MAKEFILE_TEMPLATE_TRAIN

        config_path = experiment_dir / "config.json"
        pb.io.dump_json(_config, config_path)

        makefile_path.write_text(
            MAKEFILE_TEMPLATE_TRAIN.format(
                main_python_path=pt.configurable.resolve_main_python_path(),
                experiment_name=experiment_name,
                eval_python_path=('.'.join(
                    pt.configurable.resolve_main_python_path().split('.')[:-1]
                ) + '.evaluate')
            )
        )


@ex.command(unobserved=True)
def init(_config, _run):
    """Create a storage dir, write Makefile. Do not start any training."""
    sacred.commands.print_config(_run)
    dump_config_and_makefile()

    print()
    print('Initialized storage dir. Now run these commands:')
    print(f"cd {_config['trainer']['storage_dir']}")
    print(f"make train")
    print()
    print('or')
    print()
    print(f"cd {_config['trainer']['storage_dir']}")
    print('make ccsalloc')


@ex.capture
def prepare_and_train(_run, _log, trainer, train_dataset, validate_dataset,
                      lr_scheduler_step, lr_scheduler_gamma,
                      load_model_from, database_json):
    trainer = get_trainer(trainer, load_model_from)

    db = JsonDatabase(database_json)

    # Perform a test run to check if everything works
    trainer.test_run(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
    )

    # Register hooks and start the actual training

    # Learning rate scheduler
    if lr_scheduler_step:
        trainer.register_hook(pt.train.hooks.LRSchedulerHook(
            torch.optim.lr_scheduler.StepLR(
                trainer.optimizer.optimizer,
                step_size=lr_scheduler_step,
                gamma=lr_scheduler_gamma,
            )
        ))

        # Don't use LR back-off
        trainer.register_validation_hook(
            prepare_iterable_captured(db, validate_dataset),
        )
    else:
        # Use LR back-off
        trainer.register_validation_hook(
            prepare_iterable_captured(db, validate_dataset),
            n_back_off=5, back_off_patience=3
        )

    trainer.train(
        prepare_iterable_captured(db, train_dataset),
        resume=trainer.checkpoint_dir.exists()
    )


def get_trainer(trainer_config, load_model_from):
    trainer = pt.Trainer.from_config(trainer_config)

    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'
    if load_model_from is not None and not checkpoint_path.is_file():
        _log.info(f'Loading model weights from {load_model_from}')
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    return trainer


@ex.command
def test_run(_run, _log, trainer, train_dataset, validate_dataset,
                      load_model_from, database_json):
    trainer = get_trainer(trainer, load_model_from)

    db = JsonDatabase(database_json)

    # Perform a test run to check if everything works
    trainer.test_run(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
    )


@ex.main
def main(_config, _run):
    """Main does resume directly.

    It also writes the `Makefile` and `config.json` again, even when you are
    resuming from an initialized storage dir. This way, the `config.json` is
    always up to date. Historic configuration can be found in Sacred's folder.
    """
    sacred.commands.print_config(_run)
    dump_config_and_makefile()
    prepare_and_train()


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
