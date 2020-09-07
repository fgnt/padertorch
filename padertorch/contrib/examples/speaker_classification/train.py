"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.speaker_classification.train with database_json=</path/to/json>
"""
from torch.nn import GRU
from sacred import Experiment, commands

from padertorch.io import get_new_storage_dir
from padertorch import Trainer
from padertorch.contrib.examples.speaker_classification.model import SpeakerClf
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.train.optimizer import Adam
from padertorch.modules.normalization import Normalization
from padertorch.contrib.examples.speaker_classification.data import get_datasets


ex = Experiment('speaker_clf')


@ex.config
def defaults():
    database_json = None
    assert database_json is not None, (
        'database_json cannot be None.\n'
        'Start the training with "python -m padertorch.contrib.examples.'
        'speaker_classification.train with database_json=</path/to/json>"'
    )
    dataset = 'train_clean_100'
    batch_size = 16
    num_speakers = 251


def get_model(num_speakers):
    feature_extractor = Normalization(
        data_format='bft',
        shape=(None, 64, None),
        statistics_axis='bt',
        independent_axis=None,
    )
    cnn = CNN1d(
        in_channels=64,
        out_channels=4*[512],
        output_layer=False,
        kernel_size=5,
        norm='batch'
    )
    gru = GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
    fcn = fully_connected_stack(
        256, hidden_size=[256], output_size=num_speakers, dropout=0.
    )

    speaker_clf = SpeakerClf(feature_extractor, cnn, gru, fcn)
    return speaker_clf


@ex.capture
def train(speaker_clf, storage_dir, database_json, dataset, batch_size):
    train_set, validate_set, _ = get_datasets(
        storage_dir, database_json, dataset, batch_size
    )

    trainer = Trainer(
        model=speaker_clf,
        optimizer=Adam(lr=3e-4),
        storage_dir=str(storage_dir),
        summary_trigger=(100, 'iteration'),
        checkpoint_trigger=(1000, 'iteration'),
        stop_trigger=(100000, 'iteration')
    )
    # Early stopping if loss is not decreasing after three consecutive validation
    # runs. Typically around 20k iterations (13 epochs) with an accuracy >98%
    # on the test set.
    trainer.register_validation_hook(validate_set, early_stopping_patience=3)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)


@ex.command
def test_run(database_json, dataset, batch_size, num_speakers):
    # Perform a few training and validation steps to test whether data
    # preperation and the model are working

    train_set, validate_set, _ = get_datasets(
        None, database_json, dataset, batch_size
    )
    trainer = Trainer(
        model=get_model(num_speakers),
        optimizer=Adam(lr=3e-4),
        storage_dir='/tmp/',  # not used during test run
        summary_trigger=(100, 'iteration'),
        checkpoint_trigger=(1000, 'iteration'),
        stop_trigger=(100000, 'iteration')
    )
    trainer.test_run(train_set, validate_set)


@ex.automain
def main(_run, _log, num_speakers):
    storage_dir = get_new_storage_dir(
        'speaker_clf', id_naming='time', mkdir=True
    )
    commands.save_config(
        _run.config, _log, config_filename=str(storage_dir / 'config.json')
    )

    model = get_model(num_speakers)
    train(model, storage_dir)
