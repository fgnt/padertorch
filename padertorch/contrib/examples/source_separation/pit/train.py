"""
Example call on NT infrastructure:

python -m padertorch.contrib.examples.source_separation.pit.train print_config
python -m padertorch.contrib.examples.source_separation.pit.train with database_json=<path/to/database.json>


Example call on PC2 infrastructure (only relevant for Paderborn University usage):

python -m padertorch.contrib.examples.pit.source_separation.train init with database_json=<database_json>
make ccsalloc

Required environment variables: STORAGE_ROOT, NT_DATABASE_JSONS_DIR
or with sacred: trainer.storage_dir, database_json
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

from padertorch.contrib.examples.source_separation.pit.data import prepare_dataset
from padertorch.contrib.examples.source_separation.pit.templates import MAKEFILE_TEMPLATE_TRAIN as MAKEFILE_TEMPLATE

experiment_name = 'pit'
ex = Experiment(experiment_name)

JSON_BASE = os.environ.get('NT_DATABASE_JSONS_DIR', None)


@ex.config
def config():
    """
    Configuration dict for the training process of the pit example.
    config is used by sacred for the configuration of the 'pit' experiment.
    """
    debug = False
    batch_size = 6
    database_json = None  # Path to WSJ0_2mix .json
    if database_json is None and JSON_BASE:
        database_json = Path(JSON_BASE) / 'wsj0_2mix_8k.json'

    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')
    train_dataset_name = "mix_2_spk_min_tr"
    validate_dataset_name = "mix_2_spk_min_cv"

    # Dict describing the model parameters, to allow changing the parameters from the command line.
    # Configurable automatically inserts the default values of not mentioned parameters to the config.json
    trainer = {
        "model": {
            "factory": pt.contrib.examples.source_separation.pit.model.PermutationInvariantTrainingModel,
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


@ex.command
def init(_config, _run):
    """
    Creates/updates config file (config.json) with current configurations of the pit experiment.
    Writes Makefile if no Makefile currently exists. (e.g. when train.py is called the first time.)

    Args:
        _config: Configuration dict of the experiment
        _run: Run object of the current run of the experiment
    """

    # Creates/Updates the config.json
    experiment_dir = Path(_config['trainer']['storage_dir'])
    config_path = experiment_dir / "config.json"
    pb.io.dump_json(_config, config_path)

    # Creates the Makefile if necessary
    makefile_path = Path(experiment_dir) / "Makefile"
    if not makefile_path.exists():
        makefile_path.write_text(MAKEFILE_TEMPLATE.format(
            main_python_path=pt.configurable.resolve_main_python_path(),
            experiment_dir=experiment_dir,
            experiment_name=experiment_name
        ))

    # Print the config of the current run to the console
    sacred.commands.print_config(_run)


def prepare(_config):
    """
    Preparation of the train and validation datasets for the training and initialization of the padertorch trainer,
    using the configuration dict.
    Args:
        _config: Configuration dict of the experiment

    Returns:
    3-Tuple of the prepared datasets and the trainer.
        trainer: padertorch trainer
        train_dataset: training_dataset
        validate_dataset: dataset for validation
    """

    # Extraction needed strings from the config dict
    train_dataset_name = _config['train_dataset_name']
    validate_dataset_name = _config['validate_dataset_name']
    database_json = _config['database_json']

    # Initialization of the trainer
    trainer = pt.Trainer.from_config(_config["trainer"])
    db = JsonDatabase(json_path=database_json)

    # Preparation of the datasets
    train_dataset = prepare_dataset(db, train_dataset_name,
                                    _config['batch_size'],
                                    prefetch = not _config['debug'])

    validate_dataset = prepare_dataset(db, validate_dataset_name,
                                       _config['batch_size'],
                                       prefetch = not _config['debug'])

    # Print the representations of the two datasets to the console.
    print(repr(train_dataset_name), repr(validate_dataset_name))

    return (trainer, train_dataset, validate_dataset)


def train(_config, trainer, train_dataset, validate_dataset):
    """
    Starts the training, with the passed trainer and the passed datasets.
    Creates checkpoints of the training, according to the trainer configuration.

    Args:
        _config: Configuration dict of the experiment
        trainer: padertorch trainer
        train_dataset: training_dataset
        validate_dataset: dataset for validation

    Returns:
        None
    """

    # Test run to detects possible errors in the trainer/datasets
    trainer.test_run(train_dataset, validate_dataset)

    # Path where the checkpoints of the training are stored
    checkpoint_path = trainer.checkpoint_dir / 'ckpt_latest.pth'

    # Start of the training
    trainer.register_validation_hook(validate_dataset, metric = 'loss', maximize=False, max_checkpoints=1)
    trainer.train(train_dataset, resume=checkpoint_path.is_file())

@ex.command
def test_run(_config, _run):
    """
    Test run, used for automatic testing

    Args:
        _config: Configuration dict of the experiment
        _run: Run object of the current run of the experiment

    Returns:
        None
    """
    init(_config, _run)
    trainer, train_dataset, validate_dataset = prepare(_config)

    trainer.test_run(train_dataset, validate_dataset)

@ex.main
def main(_config, _run):
    """
    Starts the training with the given configuration dict.
    Arguments are automatically filled by Scared.
    Also writes the `Makefile` and `config.json` again, even when you are
    resuming from an initialized storage dir. This way, the `config.json` is
    always up to date. Historic configuration can be found in Sacred's folder.

    Args:
        _config: Configuration dict of the experiment
        _run: Run object of the current run of the experiment

    Returns:
        None
    """
    init(_config, _run)

    (trainer, train_dataset, validate_dataset) = prepare(_config)

    train(_config, trainer, train_dataset, validate_dataset)

if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()

