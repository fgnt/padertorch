"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.audio_tagging.simple
"""
import os
from pathlib import Path

from paderbox.utils.timer import timeStamped
from padertorch import Trainer, optimizer
from padertorch.contrib.examples.audio_tagging.data import get_datasets
from padertorch.contrib.examples.audio_tagging.models import WALNet

storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'audio_tagging' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)


def main():
    model = WALNet(44100, 2048, 527)
    trainer = Trainer(
        model=model,
        optimizer=optimizer.Adam(lr=3e-4, gradient_clipping=60.),
        storage_dir=storage_dir,
        summary_trigger=(100, 'iteration'),
        stop_trigger=(50000, 'iteration'),
        checkpoint_trigger=(1000, 'iteration')
    )
    training_data, validation_data = get_datasets(
        audio_reader=dict(source_sample_rate=44100, target_sample_rate=44100),
        stft=dict(shift=882, window_length=2*882, size=2048, fading=None, pad=False),
        num_workers=8, batch_size=24, max_padding_rate=.1,
        storage_dir=storage_dir
    )
    trainer.register_validation_hook(
        validation_data, metric='macro_fscore', maximize=True
    )

    trainer.test_run(training_data, validation_data)
    trainer.train(training_data)


if __name__ == '__main__':
    main()
