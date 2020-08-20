"""
Example call on NT infrastructure:

export STORAGE_ROOT=<your desired storage root>
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m padertorch.contrib.examples.source_separation.or_pit.train with database_jsons=${paths to your JSONs}
"""
import copy

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
from padertorch.io import get_new_storage_dir

experiment_name = "or-pit"
ex = Experiment(experiment_name)


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
            "factory": pt.contrib.examples.source_separation.or_pit.OneAndRestPIT,
            "separator": {
                "factory": pt.contrib.examples.source_separation.tasnet.TasNet,
                'encoder': {
                    'factory': pt.contrib.examples.source_separation.tasnet.tas_coders.TasEncoder,
                    'window_length': 16,
                    'feature_size': 64,
                },
                'separator': {
                    'factory': pt.modules.dual_path_rnn.DPRNN,
                    'input_size': 64,
                    'rnn_size': 128,
                    'window_length': 100,
                    'hop_size': 50,
                    'num_blocks': 6,
                },
                'decoder': {
                    'factory': pt.contrib.examples.source_separation.tasnet.tas_coders.TasDecoder,
                    'window_length': 16,
                    'feature_size': 64,
                },
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
        trainer['storage_dir'] = get_new_storage_dir(experiment_name)

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
            'separator': {
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
    # single datasets)
    if isinstance(datasets, str):
        datasets = datasets.split(',')
    iterator = db.get_dataset(datasets)

    # TODO: this does not make too much sense when we have multiple datasets
    if iterator_slice is not None:
        iterator = iterator[iterator_slice]

    # Determine the number of speakers in each example
    def add_num_speakers(example):
        example.update(num_speakers=len(example['speaker_id']))
        return example
    iterator = iterator.map(add_num_speakers)

    # Group iterators by number of speakers so that all examples in a batch
    # have the same number of speakers
    iterators = list(iterator.groupby(lambda x: x['num_speakers']).values())

    chunker = RandomChunkSingle(chunk_size, chunk_keys=('y', 's'), axis=-1)
    iterators = [
        iterator
            .map(pre_batch_transform)
            .map(chunker)
            .shuffle(reshuffle=shuffle)
            .batch(batch_size)
            .map(pt.data.batch.Sorter('num_samples'))
            .map(pt.data.utils.collate_fn)
        for iterator in iterators
    ]

    iterator = lazy_dataset.concatenate(*iterators).shuffle(reshuffle=shuffle)

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
        # Dump config
        config_path = experiment_dir / "config.json"
        pb.io.dump_json(_config, config_path)

        # Dump makefile
        from .templates import MAKEFILE_TEMPLATE_TRAIN
        makefile_path.write_text(MAKEFILE_TEMPLATE_TRAIN.format(
            main_python_path=pt.configurable.resolve_main_python_path(),
            eval_python_path='.'.join(
                pt.configurable.resolve_main_python_path().split('.')[:-1]
                + ['evaluate']
            ),
            experiment_name=experiment_name,
            model_path=experiment_dir,
        ))


@ex.command
def init(_config, _run):
    """Create a storage dir, write Makefile. Do not start any training."""
    sacred.commands.print_config(_run)
    dump_config_and_makefile(_config)

    print()
    print('Initialized storage dir. Now run these commands:')
    print(f"cd {_config['trainer']['storage_dir']}")
    print(f"make train")
    print()
    print('or')
    print()
    print(f"cd {_config['trainer']['storage_dir']}")
    print('make ccsalloc')


@ex.command
def init_with_new_storage_dir(_config, _run):
    """Like init, but ignores the set storage dir. Can be used to continue
    training with a modified configuration"""
    # Create a mutable copy of the config and overwrite the storage dir
    _config = copy.deepcopy(_config)
    _config['trainer']['storage_dir'] = get_storage_dir()
    _run.config = _config
    init(_config)


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
