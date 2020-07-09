"""
Example call on NT infrastructure:

export STORAGE=<your desired storage root>
mkdir -p $STORAGE/pth_models/dprnn
python -m padertorch.contrib.examples.tasnet.train print_config
python -m padertorch.contrib.examples.tasnet.train
"""
import torch
from paderbox.io import load_audio

from sacred import Experiment
import sacred.commands

# sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False

import os
from pathlib import Path

from sacred.utils import InvalidConfigError, MissingConfigError

import padertorch as pt
import padertorch.contrib.examples.tasnet.tasnet
import paderbox as pb
import numpy as np

from sacred.observers.file_storage import FileStorageObserver
from lazy_dataset.database import JsonDatabase

from padertorch.contrib.neumann.chunking import RandomChunkSingle
from padertorch.contrib.ldrude.utils import get_new_folder

nickname = "tasnet"
ex = Experiment(nickname)


def get_storage_dir():
    # Sacred should not add path_template to the config
    # -> move this few lines to a function
    path_template = Path(os.environ["STORAGE"]) / 'pth_models' / nickname
    path_template.mkdir(exist_ok=True, parents=True)
    return get_new_folder(path_template, mkdir=False)


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

    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')

    # Start with an empty dict to allow tracking by Sacred
    feat_size = 64
    encoder_window_size = 16
    trainer = {
        "model": {
            "factory": pt.contrib.examples.tasnet.tasnet.ModularTasNet,
            'encoder': {
                'factory': pt.contrib.examples.tasnet.tas_coders.TasEncoder,
                'window_size': encoder_window_size,
                'feature_size': feat_size,
            },
            'separator': {
                'factory': pt.modules.dual_path_rnn.DPRNN,
                'feat_size': 64,
                'rnn_size': 128,
                'window_length': 100,
                'hop_size': 50,
                'num_blocks': 6,
            },
            'decoder': {
                'factory': pt.contrib.examples.tasnet.tas_coders.TasDecoder,
                'window_size': encoder_window_size,
                'feature_size': feat_size,
            },
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (100_000, "iteration"),
        "loss_weights": {
            "si-sdr": 1.0,
            "log-mse": 0.0,
            "si-sdr-grad-stop": 0.0,
        }
    }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = get_storage_dir()

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
                'window_size': 2
            },
            'separator': {
                'window_length': 250,
                'dprnn_hop_size': 125,  # Half of window length
            }
        }
    }

@ex.named_config
def stft():
    trainer = {
        'model': {
            'encoder': {
                'factory': 'padertorch.contrib.examples.tasnet.tas_coders.StftEncoder'
            },
            'decoder': {
                'factory': 'padertorch.contrib.examples.tasnet.tas_coders.IstftDecoder'
            },
        }
    }


@ex.named_config
def log_mse():
    trainer = {
        'loss_weights': {
            'si-sdr': 0.0,
            'log-mse': 1.0,
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
            load_audio(p)
            for p in inputs['audio_path']['speech_source']
        ], np.float32),
        'y': np.ascontiguousarray(
            load_audio(inputs['audio_path']['observation']), np.float32),
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

    iterator = (
        iterator
            .map(pre_batch_transform)
            .map(RandomChunkSingle(chunk_size, chunk_keys=('y', 's'), axis=-1))
            .shuffle(reshuffle=True)
            .batch(batch_size)
            .map(lambda batch: sorted(
            batch,
            key=lambda example: example['num_samples'],
            reverse=True,
        ))
            .map(pt.data.utils.collate_fn)
    )

    if prefetch:
        iterator = iterator.prefetch(8, 16, catch_filter_exception=True)
    elif chunk_size > 0:
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
        config_path = experiment_dir / "config.json"

        pb.io.dump_json(_config, config_path)

        main_python_path = pt.configurable.resolve_main_python_path()
        eval_python_path = '.'.join(
            pt.configurable.resolve_main_python_path().split('.')[:-1]
        ) + '.evaluate'
        makefile_path.write_text(
            f"SHELL := /bin/bash\n"
            f"MODEL_PATH := $(shell pwd)\n"
            f"\n"
            f"export OMP_NUM_THREADS=1\n"
            f"export MKL_NUM_THREADS=1\n"
            f"\n"
            f"train:\n"
            f"\tpython -m {main_python_path} with config.json\n"
            f"\n"
            f"ccsalloc:\n"
            f"\tccsalloc \\\n"
            f"\t\t--notifyuser=awe \\\n"
            f"\t\t--res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\\n"
            f"\t\t--time=100h \\\n"
            f"\t\t--join \\\n"
            f"\t\t--stdout=stdout \\\n"
            f"\t\t--tracefile=%x.%reqid.trace \\\n"
            f"\t\t-N train_{nickname} \\\n"
            f"\t\tpython -m {main_python_path} with config.json\n"
            f"\n"
            f"evaluate:\n"
            f"\tpython -m {eval_python_path} init with model_path=$(MODEL_PATH)\n"
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
    trainer = pt.Trainer.from_config(trainer)
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    if load_model_from is not None and not checkpoint_path.is_file():
        _log.info(f'Loading model weights from {load_model_from}')
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

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
        resume=checkpoint_path.is_file()
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
