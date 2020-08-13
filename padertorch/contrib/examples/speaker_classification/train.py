"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.speaker_classification.train
"""
import os
from pathlib import Path

from paderbox.utils.timer import timeStamped
from padertorch import Trainer
from padertorch.contrib.examples.speaker_classification.model import SpeakerClf
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.train.optimizer import Adam
from padertorch.modules.normalization import Normalization
from torch.nn import GRU
from padertorch.contrib.examples.speaker_classification.data import get_datasets


def get_model():
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
        256, hidden_size=[256], output_size=251, dropout=0.
    )

    speaker_clf = SpeakerClf(feature_extractor, cnn, gru, fcn)
    return speaker_clf


def train(speaker_clf, storage_dir):
    train_set, validate_set, _ = get_datasets(storage_dir)

    trainer = Trainer(
        model=speaker_clf,
        optimizer=Adam(lr=3e-4),
        storage_dir=str(storage_dir),
        summary_trigger=(100, 'iteration'),
        checkpoint_trigger=(1000, 'iteration'),
        stop_trigger=(100000, 'iteration')
    )
    trainer.register_validation_hook(validate_set)
    trainer.test_run(train_set, validate_set)
    trainer.train(train_set)


if __name__ == '__main__':
    storage_dir = str(
        Path(os.environ['STORAGE_ROOT']) / 'speaker_clf' / timeStamped('')[1:]
    )
    os.makedirs(storage_dir, exist_ok=True)

    model = get_model()
    train(model, storage_dir)
