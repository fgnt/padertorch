"""
Example call on NT infrastructure:

python -m padertorch.contrib.examples.source_separation.pit.train print_config
python -m padertorch.contrib.examples.source_separation.pit.train with database_json=<path/to/database.json>


Example call on PC2 infrastructure (only relevant for Paderborn University usage):

python -m padertorch.contrib.examples.pit.source_separation.train init with database_json=<database_json>
make ccsalloc


"""
from sacred import Experiment
import sacred.commands
import os
from pathlib import Path
import padertorch as pt
import paderbox as pb
from lazy_dataset.database import JsonDatabase

from sacred.observers.file_storage import FileStorageObserver
from sacred.utils import InvalidConfigError, MissingConfigError

from padertorch.contrib.examples.source_separation.pit.data import prepare_iterable
from padertorch.contrib.examples.source_separation.pit.templates import MAKEFILE_TEMPLATE_TRAIN as MAKEFILE_TEMPLATE

experiment_name = "pit"
ex = Experiment(experiment_name)

@ex.config
def config():
    debug = False
    batch_size = 6
    database_json = None  # Path to WSJ0_2mix .json
    if "WSJ0_2MIX" in os.environ:
        database_json = os.environ.get("WSJ0_2MIX")
    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')
    train_dataset = "mix_2_spk_min_tr"
    validate_dataset = "mix_2_spk_min_cv"

    # Dict describing the model parameters, to allow changing the parameters from the command line.
    # Configurable automatically inserts the default values of not mentioned parameters to the config.json
    trainer = {
        "model": {
            "factory": pt.contrib.examples.source_separation.pit.model.PermutationInvariantTrainingModel,
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
        "stop_trigger": (300_000, "iteration"),
        "loss_weights": {
            "pit_ips_loss": 1.0,
            "pit_mse_loss": 0.0,
        }
    }
    pt.Trainer.get_config(trainer)
    if trainer['storage_dir'] is None:
        trainer['storage_dir'] = pt.io.get_new_storage_dir(experiment_name)

    ex.observers.append(FileStorageObserver(
        Path(trainer['storage_dir']) / 'sacred')
    )


@ex.capture
def prepare_iterable_captured(
        database, dataset, batch_size, debug
):
    return_keys = 'X_abs Y_abs cos_phase_difference num_frames'.split()
    return prepare_iterable(
        database, dataset, batch_size, return_keys,
        prefetch=not debug,
    )


@ex.command
def init(_config, _run):
    """ Creates a storage dir, writes Makefile. Does not start any training."""
    experiment_dir = Path(_config['trainer']['storage_dir'])
    config_path = experiment_dir / "config.json"
    pb.io.dump_json(_config, config_path)

    makefile_path = Path(experiment_dir) / "Makefile"
    makefile_path.write_text(MAKEFILE_TEMPLATE.format(
        main_python_path=pt.configurable.resolve_main_python_path(),
        experiment_dir=experiment_dir,
        experiment_name=experiment_name
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
def prepare_and_train(_config, _run, train_dataset, validate_dataset, database_json):
    """ Prepares the train and validation dataset from the database object """

    sacred.commands.print_config(_run)
    trainer = pt.Trainer.from_config(_config["trainer"])
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    db = JsonDatabase(json_path=database_json)
    print(repr(train_dataset), repr(validate_dataset))

    trainer.test_run(
        prepare_iterable_captured(db, train_dataset),
        prepare_iterable_captured(db, validate_dataset),
    )
    trainer.register_validation_hook(
        prepare_iterable_captured(db, validate_dataset)
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
    experiment_dir = Path(_config['trainer']['storage_dir'])
    config_path = experiment_dir / "config.json"
    pb.io.dump_json(_config, config_path)
    makefile_path = Path(experiment_dir) / "Makefile"
    if not makefile_path.exists():

        makefile_path.write_text(MAKEFILE_TEMPLATE.format(
            main_python_path=pt.configurable.resolve_main_python_path(),
            experiment_dir=experiment_dir,
            experiment_name=experiment_name
        ))

    prepare_and_train()


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
