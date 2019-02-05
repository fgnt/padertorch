"""

# Start command:
# - Change STORAGE_ROOT
export STORAGE_ROOT=/net/vol/$USER/sacred/torch/examples
mkdir -p $STORAGE_ROOT/acoustic_model
python -m padertorch.contrib.examples.acoustic_model.train

# One liner:
export STORAGE_ROOT=/net/vol/$USER/sacred/torch/examples && mkdir -p $STORAGE_ROOT/acoustic_model &&
python -m padertorch.contrib.examples.acoustic_model.train

"""
# ToDo: why export STORAGE_ROOT if in one liner?
import os
import datetime
from pathlib import Path

import sacred
import sacred.commands

import paderbox as pb
import padertorch as pt

from padertorch.contrib.ldrude.utils import (
    decorator_append_file_storage_observer_with_lazy_basedir
)
from padertorch.contrib.cb import (
    get_new_folder,
    write_makefile_and_config,
)

from padertorch.contrib.examples.acoustic_model.model import AcousticExperiment

ex = sacred.Experiment('AM')


def get_basedir():
    # ToDo: Improve error message if no STORAGE_ROOT is set
    basedir = (Path(
        os.environ['STORAGE_ROOT']
    ) / 'acoustic_model').expanduser().resolve()
    if not basedir.is_dir():
        raise FileNotFoundError(
            f'No such directory: {basedir}\n'
            f'run `mkdir -p {basedir}`'
        )
    return basedir


@ex.config
def config():
    dataset_train = 'tr05_simu'
    dataset_dev = 'dt05_simu'

    trainer = pb.utils.nested.deflatten({
        'model.factory': AcousticExperiment,
        'model.db': 'Chime4',
        'optimizer.factory': pt.optimizer.Adam,
        'summary_trigger': (1000, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (50, 'epoch'),
        'loss_weights': None,
        'keep_all_checkpoints': True,
        'storage_dir': None,
    })
    pt.Trainer.get_config(trainer)

    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = get_new_folder(get_basedir(), mkdir=False)


@decorator_append_file_storage_observer_with_lazy_basedir(ex)
def basedir(_config):
    return Path(_config['trainer']['storage_dir']) / 'sacred'


@ex.capture
def prepare_and_train(
        _config,
        dataset_train,
        dataset_dev,
        _run,
        resume=False
):
    print('Start time:', str(datetime.datetime.now()))
    storage_dir = Path(_config['trainer']['storage_dir'])
    try:
        sacred.commands.print_config(_run)
        storage_dir = Path(storage_dir)
        print('Storage dir:', storage_dir)

        trainer = pt.Trainer.from_config(_config['trainer'])

        model: AcousticExperiment = trainer.model
        print('model:')
        print(model)

        it_tr = model.get_iterable(dataset_train).shuffle(reshuffle=False)
        it_tr = it_tr.map(model.transform)
        it_dt = model.get_iterable(dataset_dev)
        it_dt = it_dt.map(model.transform)

        print('it_tr:')
        print(it_tr)
        print('it_dt:')
        print(it_dt)

        print('Storage dir:', storage_dir)

        trainer.test_run(it_tr.catch(), it_dt.catch())
        trainer.train(
            it_tr.prefetch(4, 8, catch_filter_exception=True),
            it_dt.prefetch(4, 8, catch_filter_exception=True),
            resume=resume,
        )
    finally:
        print('Storage dir:', storage_dir)
        print('End time:', str(datetime.datetime.now()))


@ex.command
def resume():
    return prepare_and_train(resume=True)


@ex.main
def main(_config, _run):
    # ToDo: perhaps short explanation why a makefile is written here
    write_makefile_and_config(
        _config['trainer']['storage_dir'], _config, _run,
        backend='yaml'
    )
    return prepare_and_train(resume=False)


if __name__ == '__main__':
    ex.run_commandline()
