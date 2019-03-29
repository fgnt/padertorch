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

# import pytorch_sanity.trainer
import os
import datetime
import subprocess
from pathlib import Path

import sacred
import sacred.commands

import torch

# from paderbox.database.chime import Chime4

# import cbj.pytorch.am
from padertorch.contrib.cb.hooks import CPUTimeLimitExceededHook

import paderbox as pb

import padertorch as pt

from padertorch.contrib.ldrude.utils import (
    decorator_append_file_storage_observer_with_lazy_basedir
)
from padertorch.contrib.cb import (
    get_new_folder,
    write_makefile_and_config,
)

from cbj_lib.run import get_new_folder
import cbj
import cbj.pyon

from padertorch.contrib.cb.lpit.model import Model



ex = sacred.Experiment('lpit')


states_count = 1983

@ex.config
def config():

    # storage_dir = str(Path('.').expanduser().resolve())

    # database = pt.configurable.class_to_str(Chime4)
    dataset_train = 'train_si284'
    dataset_dev = 'cv_dev93'
    batch_size = None

    trainer = pb.utils.nested.deflatten({
        'model.factory': Model,
        # 'model.db.factory': database,
        'optimizer.factory': pt.optimizer.Adam,
        'optimizer.gradient_clipping': 10,
        # 'gpu': 0 if torch.cuda.is_available() or cbj.host.on_pc2() else False,
        'summary_trigger': (500, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (50, 'epoch'),
        'loss_weights': None,
        'keep_all_checkpoints': True,
        'storage_dir': None,
    })
    pt.Trainer.get_config(trainer)

    if trainer['storage_dir'] is None:
        trainer['storage_dir'] \
            = str(get_new_folder('~/sacred/torch/lpit', mkdir=False))


@decorator_append_file_storage_observer_with_lazy_basedir(ex)
def basedir(_config):
    return Path(_config['trainer']['storage_dir']) / 'sacred'


@ex.capture
def prepare_and_train(
        _config,
        # storage_dir,
        dataset_train,
        dataset_dev,
        _run,
        seed,
        # states_count,
        batch_size,
        resume=False,
):
    print('Start time:', str(datetime.datetime.now()))
    storage_dir = Path(_config['trainer']['storage_dir'])

    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    if CUDA_VISIBLE_DEVICES is not None:
        print('CUDA_VISIBLE_DEVICES:', CUDA_VISIBLE_DEVICES)

    try:
        sacred.commands.print_config(_run)
        storage_dir = Path(storage_dir)
        print('Storage dir:', storage_dir)
        print('trainer:')
        trainer = pt.Trainer.from_config(_config['trainer'])
        print(trainer)
        print('Storage dir:', storage_dir)

        # Does not always work
        code = cbj.pyon.dumps_configurable_config(
            _config['trainer'],
            # trace=2,
        )
        Path(storage_dir / 'config.py').write_text(code)

        model: Model = trainer.model
        print('model:')
        print(model)

        it_tr = model.get_iterable(dataset_train).shuffle(reshuffle=False)
        it_tr = it_tr.map(model.transform)
        it_dt = model.get_iterable(dataset_dev)
        it_dt = it_dt.map(model.transform)

        # assert states_count == model.db.occs.size, (states_count, model.db.occs.size)
        print('it_tr:')
        print(repr(it_tr))
        print('it_dt:')
        print(repr(it_dt))

        print('Storage dir:', storage_dir)
        trainer.test_run(
            it_tr.catch(),
            it_dt.catch(),
        )

        if batch_size is not None:
            it_tr = it_tr.batch(batch_size)
            # .map(pt.data.Sorter(lambda example: example.Observation.shape[0]))

        trainer.train(
            it_tr.prefetch(4, 8, catch_filter_exception=True),
            it_dt.prefetch(4, 8, catch_filter_exception=True),
            resume=resume,
            hooks=[CPUTimeLimitExceededHook(), model]
        )
    finally:
        print('Storage dir:', storage_dir)
        print('End time:', str(datetime.datetime.now()))


@ex.command
def resume(_config):
    storage_dir = Path(_config['trainer']['storage_dir'])
    resume = (storage_dir / 'checkpoints').exists()
    return prepare_and_train(resume=resume)


@ex.command
def init(_config, _run):
    storage_dir = Path(_config['trainer']['storage_dir'])
    write_makefile_and_config(
        storage_dir, _config, _run,
        backend='yaml'
    )

    with open(storage_dir / 'Makefile', 'a') as fd:
        experiment_name = ex.path
        fd.write(
            '\n'
            'evalfolder := $(patsubst checkpoints/%.pth,eval/%,$(wildcard checkpoints/*.pth))\n'
            'evalfolder := $(filter-out eval/ckpt_best_loss, $(evalfolder))\n'
            'evalfolder := $(filter-out eval/ckpt_latest, $(evalfolder))\n'
            '\n'
            '$(evalfolder): eval/%: checkpoints/%.pth\n'
            '\t# mpiexec -np 8 python -m cbj.pytorch.eval_am with model_path=. checkpoint=$^\n'
            '\tmkdir -p $@_ccs\n'
            '\tccsalloc '
                '--stdout=$@_ccs/%reqid.out '
                '--stderr=$@_ccs/%reqid.err '
                '--tracefile=$@_ccs/%reqid.trace '
                '--group=hpc-prf-nt3 '
                '--res=rset=40:mem=4G:ncpus=1 '
                '-t 6h '
                f'--name={experiment_name}_{storage_dir.name}_eval '
                'ompi -- '
                'python -m padertorch.contrib.cb.lpit.eval with model_path=. checkpoint=$<'
            '\n'
            '\n'
            'allEval: $(evalfolder)'
            '\n'
        )

        fd.write(
            '\n'
            'collect_scores_write_tfevents:\n'
            '\trm -i events.out.tfevents.*.fe2.scores || true'
            '\tpython -m padertorch.contrib.cb.lpit.collect_scores_write_tfevents'
            '\n'
        )
    print('Storage dir:', storage_dir)


@ex.command
def pc2(_config, _run):
    init()
    storage_dir = Path(_config['trainer']['storage_dir'])

    experiment_name = ex.path

    options = ' '.join([
        '--stdout=ccs/%reqid.out',
        '--stderr=ccs/%reqid.err',
        '--tracefile=ccs/%reqid.trace',

        # ccsinfo name
        f'--name={experiment_name}_{storage_dir.name}',

        # Sends the signal SIGXCPU 60 minutes before the resource is released.
        # This signal is handled from CPUTimeLimitExceededHook
        '--notifyjob=XCPU,60m',

        '--group=hpc-prf-nt3',
        '--res=rset=1:ncpus=8:gtx1080=1',
    ])
    (storage_dir / 'ccs').mkdir(exist_ok=True)
    with open(Path(storage_dir) / 'Makefile', 'a') as fd:
        fd.write(
            '\n'
            'pc2resume: ccs\n'
            f'\tccsalloc {options} -t 3d make resume'
            '\n'
        )
        fd.write(
            '\n'
            'tail:\n'
            # '\ttail -F ccs/*'
            '''\tfind ccs -iname "*.*" | sort | tail -n 3 | xargs -n3 --delimiter='\\n' tail -F'''
            '\n'
        )

    subprocess.run(
        ['make', 'pc2resume'],
        cwd=str(storage_dir)
    )


@ex.main
def main():
    init()
    # # Add to Makefile
    # evalfolder := $(patsubst checkpoints/%.pth,eval/%,$(wildcard checkpoints/*.pth))
    # $(evalfolder): eval/%: checkpoints/%.pth
    #   mpiexec -np 8 python -m cbj.pytorch.eval_am with model_path=. checkpoint=$^
    return prepare_and_train(resume=False)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(RuntimeError):
        ex.run_commandline()
