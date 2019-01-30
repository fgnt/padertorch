"""
Example call:

python -m padertorch.contrib.ldrude.train_pit print_config
python -m padertorch.contrib.ldrude.train_pit

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
    write_makefile_and_config_json,
    get_new_folder
)


nickname = "pit"
ex = Experiment(nickname)

path_template = Path(os.environ["STORAGE"]) / "pth_models" / nickname


@ex.config
def config():
    debug = False
    batch_size = 4

    train_dataset = "mix_2_spk_min_tr"
    validate_dataset = "mix_2_spk_min_cv"

    # Start with an empty dict to allow tracking by Sacred
    trainer = pb.utils.nested.deflatten(
        {
            "model.factory": pt.models.bss.PermutationInvariantTrainingModel,
            "storage_dir": None,
            "optimizer.factory": pt.optimizer.Adam,
            "summary_trigger": (1000, "iteration"),
            "max_trigger": (500_000, "iteration"),
            "loss_weights.pit_ips_loss": 0.0,
            "loss_weights.pit_mse_loss": 1.0,
        },
        sep=".",
    )
    pt.Trainer.get_config(
        trainer,
    )
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] \
            = get_new_folder(path_template, mkdir=False)


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


@ex.capture
def prepare_and_train(
        _config, _run, train_dataset, validate_dataset, resume=False
):
    sacred.commands.print_config(_run)
    trainer = pt.Trainer.from_config(_config["trainer"])

    db = MerlMixtures()
    trainer.train(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
        resume=resume
    )


@ex.main
def main(_config, _run):
    write_makefile_and_config_json(
        _config['trainer']['storage_dir'], _config, _run
    )
    prepare_and_train()


@ex.command
def resume():
    return prepare_and_train(resume=True)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
