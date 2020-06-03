"""
Example call on NT infrastructure:

export STORAGE=<your desired storage root>
mkdir -p $STORAGE/pth_models/dprnn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m padertorch.contrib.examples.or_pit.train with database_jsons=${paths to your JSONs}
"""
import lazy_dataset
import torch
from paderbox.io import load_audio

from sacred import Experiment
import sacred.commands

import os
from pathlib import Path

from sacred.utils import InvalidConfigError, MissingConfigError
from typing import List

import padertorch as pt
import paderbox as pb
import numpy as np

from sacred.observers.file_storage import FileStorageObserver
from lazy_dataset.database import JsonDatabase, DictDatabase

from padertorch.contrib.neumann.chunking import RandomChunkSingle
from padertorch.contrib.ldrude.utils import get_new_folder

nickname = "or-pit"
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

    train_datasets = ["mix_2_spk_min_tr", "mix_3_spk_min_tr"]
    validate_datasets = ["mix_2_spk_min_cv", "mix_3_spk_min_cv"]
    target = 'speech_source'
    lr_scheduler_step = 2
    lr_scheduler_gamma = 0.98
    load_model_from = None
    database_jsons = []

    # if not database_jsons:
    #     raise MissingConfigError(
    #         'You have to set the path to the database JSON!', 'database_jsons')

    # Start with an empty dict to allow tracking by Sacred
    trainer = {
        "model": {
            "factory": 'padertorch.contrib.examples.or_pit.or_pit.OneAndRestPIT',
            "separator": {
                "factory": 'padertorch.contrib.examples.tasnet.tasnet.TasNet'
            }
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "stop_trigger": (100_000, "iteration"),
        "loss_weights": {
            "si-sdr": 0.0,
            "log-mse": 1.0,
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
        's': np.ascontiguousarray([
            load_audio(p)
            for p in inputs['audio_path']['speech_source']
        ], np.float32),
        'y': np.ascontiguousarray(
            load_audio(inputs['audio_path']['observation']), np.float32),
        'num_samples': inputs['num_samples'],
        'example_id': inputs['example_id'],
        'audio_path': inputs['audio_path'],
        'num_speakers': len(inputs['speaker_id']),
    }


def prepare_iterable(
        db, datasets: List[str], batch_size, chunk_size, prefetch=True,
        iterator_slice=None, shuffle=True
):
    """
    This is re-used in the evaluate script
    """
    # Create an iterator from the datasets (a simple concatenation of the
    # single datasets) and determine the number of speakers in each example
    if isinstance(datasets, str):
        datasets = datasets.split(',')

    iterator = db.get_dataset(datasets).map(
        lambda x: (x.update(num_speakers=len(x['speaker_id'])), x)[1])

    # TODO: this does not make too much sense when we have multiple datasets
    if iterator_slice is not None:
        iterator = iterator[iterator_slice]

    # This
    iterators = list(iterator.groupby(lambda x: x['num_speakers']).values())

    iterators = [
        iterator
        .map(pre_batch_transform)
        .map(RandomChunkSingle(chunk_size, chunk_keys=('y', 's'), axis=-1))
        .shuffle(reshuffle=shuffle)
        .batch(batch_size)
        .map(lambda batch: sorted(
            batch,
            key=lambda example: example['num_samples'],
            reverse=True,
        ))
        .map(pt.data.utils.collate_fn)
        for iterator in iterators
    ]

    iterator = lazy_dataset.concatenate(*iterators).shuffle(reshuffle=shuffle)

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
            f"finetune:\n"
            f"\tpython -m {main_python_path} init with config.json trainer.model.finetune=True\n"
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

@ex.command
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
def prepare_and_train(_run, _log, trainer, train_datasets, validate_datasets,
                      lr_scheduler_step, lr_scheduler_gamma,
                      load_model_from, database_jsons):
    trainer = pt.Trainer.from_config(trainer)
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    if load_model_from is not None and not checkpoint_path.is_file():
        _log.info(f'Loading model weights from {load_model_from}')
        checkpoint = torch.load(load_model_from)
        trainer.model.load_state_dict(checkpoint['model'])

    if isinstance(database_jsons, str):
        database_jsons = database_jsons.split(',')

    db = DictDatabase(pb.utils.nested.nested_merge(
        {}, *(pb.io.load(td) for td in database_jsons)
    ))

    # Perform a test run to check if everything works
    trainer.test_run(
        prepare_iterable_captured(db, train_datasets),
        prepare_iterable_captured(db, validate_datasets),
    )

    # Register hooks and start the actual training
    trainer.register_validation_hook(
        prepare_iterable_captured(db, validate_datasets)
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
        prepare_iterable_captured(db, train_datasets),
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
