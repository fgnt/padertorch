"""
Example call on NT infrastructure:

export STORAGE=<your desired storage root>
mkdir -p $STORAGE/pth_models/dprnn
python -m padertorch.contrib.neumann.dual_path_rnn.train print_config
python -m padertorch.contrib.neumann.dual_path_rnn.train
"""
import torch

from functools import partial

from sacred import Experiment
import sacred.commands
from sacred.utils import ConfigError

from padercontrib.database.wsj0_mix import WSJ0_2Mix_8k
import os
from pathlib import Path
import padertorch as pt
import paderbox as pb
from padercontrib.database.keys import *
import numpy as np

from sacred.observers.file_storage import FileStorageObserver

from padertorch.contrib.neumann.chunking import RandomChunkSingle
from padertorch.contrib.ldrude.utils import get_new_folder
import padercontrib

MAKEFILE_TEMPLATE = """
SHELL := /bin/bash
MODEL_PATH := $(shell pwd)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

train:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--notifyuser=awe \\
\t\t--res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\
\t\t--time=100h \\
\t\t--join \\
\t\t--stdout=stdout \\
\t\t--tracefile=%x.%reqid.trace \\
\t\t-N train_{nickname} \\
\t\tpython -m {main_python_path} with config.json

evaluate:
\tpython -m {eval_python_path} init with model_path=$(MODEL_PATH)
"""

nickname = "dprnn"
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

    database = 'wsj0-2mix'
    train_dataset = "mix_2_spk_min_tr"
    validate_dataset = "mix_2_spk_min_cv"
    target = 'speech_source'
    lr_scheduler_step = 2
    lr_scheduler_gamma = 0.98
    load_model_from = None

    # Start with an empty dict to allow tracking by Sacred
    trainer = {
        "model": {
            "factory": 'padertorch.contrib.examples.tasnet.tasnet.TasNet',
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (600_000, "iteration"),
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
    This is the configuration for the best performing model from the DPRNN paper. Training takes very long time
    with this configuration.
    """
    # The model becomes very memory consuming with this small window size. You might have to reduce the chunk size as
    # well.
    batch_size = 1

    trainer = {
        'model': {
            'encoder_block_size': 2,
            'dprnn_window_length': 250,
            'dprnn_hop_size': 125,  # Half of window length
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
        's': np.ascontiguousarray(
            inputs['audio_data']['speech_source'], np.float32),
        'y': np.ascontiguousarray(
            inputs['audio_data']['observation'], np.float32),
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

    # Read audio data keys 'observation' and 'speech_source'
    audio_reader = padercontrib.database.iterator.AudioReader(
        audio_keys=[OBSERVATION, SPEECH_SOURCE], read_fn=db.read_fn)
    iterator = iterator.map(audio_reader)

    iterator = (
        iterator
        .map(pre_batch_transform)
        .map(RandomChunkSingle(chunk_size, chunk_keys=('y', 's'), axis=-1))
        .shuffle(reshuffle=False)
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
        database_obj, dataset, batch_size, debug, chunk_size
):
    return prepare_iterable(
        database_obj, dataset, batch_size, chunk_size,
        prefetch=not debug,
        iterator_slice=slice(0, 100, 1) if debug else None
    )


@ex.command
def init(_config, _run):
    """Create a storage dir, write Makefile. Do not start any training."""
    experiment_dir = Path(_config['trainer']['storage_dir'])
    config_path = experiment_dir / "config.json"
    pb.io.dump_json(_config, config_path)

    makefile_path = Path(experiment_dir) / "Makefile"
    makefile_path.write_text(MAKEFILE_TEMPLATE.format(
        main_python_path=pt.configurable.resolve_main_python_path(),
        experiment_dir=experiment_dir,
        nickname=nickname,
        eval_python_path='.'.join(
            pt.configurable.resolve_main_python_path().split('.')[:-1]
        ) + '.evaluate',
        model_path=experiment_dir
    ))

    sacred.commands.print_config(_run)
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
                      load_model_from):
    sacred.commands.print_config(_run)
    trainer = pt.Trainer.from_config(trainer)
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    if load_model_from is not None and not checkpoint_path.is_file():
        _log.info(f'Loading model weights from {load_model_from}')
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    db = WSJ0_2Mix_8k()

    # Perform a test run to check if everything works
    trainer.test_run(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
    )

    # Register hooks and start the actual training
    trainer.register_validation_hook(
        prepare_iterable_captured(db, validate_dataset)
    )

    # Learning rate scheduler
    trainer.register_hook(pt.train.hooks.LRSchedulerHook(
        torch.optim.lr_scheduler.StepLR(
            trainer.optimizer.optimizer,
            step_size=lr_scheduler_step,
            gamma=lr_scheduler_gamma,
        )
    ))

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
    experiment_dir = Path(_config['trainer']['storage_dir'])
    config_path = experiment_dir / "config.json"
    pb.io.dump_json(_config, config_path)

    makefile_path = Path(experiment_dir) / "Makefile"
    makefile_path.write_text(MAKEFILE_TEMPLATE.format(
        main_python_path=pt.configurable.resolve_main_python_path(),
        experiment_dir=experiment_dir,
        nickname=nickname,
        eval_python_path='.'.join(
            pt.configurable.resolve_main_python_path().split('.')[:-1]
        ) + '.evaluate',
        model_path=Path(experiment_dir)
    ))

    prepare_and_train()


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
