"""
Example call on NT infrastructure:

export STORAGE=<your desired storage root>
mkdir -p $STORAGE/pth_models/pit
python -m padertorch.contrib.examples.pit.train print_config
python -m padertorch.contrib.examples.pit.train


Example call on PC2 infrastructure:

export STORAGE=<your desired storage root>
mkdir -p $STORAGE/pth_models/pit
python -m padertorch.contrib.examples.pit.train init
make ccsalloc


TODO: Enable shuffle
TODO: Change to sacred IDs again, otherwise I can not apply `unique` to `_id`.
"""
from sacred import Experiment
import sacred.commands
from paderbox.database.merl_mixtures import MerlMixtures
import os
from pathlib import Path
import padertorch as pt
import paderbox as pb

from padertorch.contrib.ldrude.data import prepare_iterable
from padertorch.contrib.ldrude.utils import (
    decorator_append_file_storage_observer_with_lazy_basedir,
    get_new_folder
)


MAKEFILE_TEMPLATE = """
SHELL := /bin/bash

train:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--notifyuser=awe \\
\t\t--res=rset=1:ncpus=4:gtx1080=1:vmem=50g:ompthreads=1 \\
\t\t--time=100h \\
\t\t--join \\
\t\t--stdout=stdout \\
\t\t--tracefile=trace_%reqid.trace \\
\t\t-N train_{nickname} \\
\t\tpython -m {main_python_path} with config.json
"""


nickname = "pit"
ex = Experiment(nickname)

path_template = Path(os.environ["STORAGE"]) / "pth_models" / nickname


@ex.config
def config():
    debug = False
    batch_size = 6

    train_dataset = "mix_2_spk_min_tr"
    validate_dataset = "mix_2_spk_min_cv"

    # Start with an empty dict to allow tracking by Sacred
    trainer = {
        "model": {
            "factory": pt.models.bss.PermutationInvariantTrainingModel,
            "dropout_input": 0.,
            "dropout_hidden": 0.,
            "dropout_linear": 0.
        },
        "storage_dir": None,
        "optimizer": {
            "factory": pt.optimizer.Adam,
            "gradient_clipping": 1
        },
        "summary_trigger": (1000, "iteration"),
        "max_trigger": (350_000, "iteration"),
        "loss_weights": {
            "pit_ips_loss": 1.0,
            "pit_mse_loss": 0.0,
        }
    }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = get_new_folder(path_template, mkdir=False)


@decorator_append_file_storage_observer_with_lazy_basedir(ex)
def basedir(_config):
    return Path(_config['trainer']['storage_dir']) / 'sacred'


@ex.capture
def prepare_iterable_captured(
        database, dataset, batch_size, debug
):
    return_keys = 'X_abs Y_abs cos_phase_difference num_frames'.split()
    return prepare_iterable(
        database, dataset, batch_size, return_keys,
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
        nickname=nickname
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
def prepare_and_train(_config, _run, train_dataset, validate_dataset):
    sacred.commands.print_config(_run)
    trainer = pt.Trainer.from_config(_config["trainer"])
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    db = MerlMixtures()
    trainer.test_run(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
    )
    trainer.train(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
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
        nickname=nickname
    ))

    prepare_and_train()


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
