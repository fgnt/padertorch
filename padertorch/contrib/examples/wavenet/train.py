"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from pathlib import Path
import numpy as np

from paderbox.database.timit import Timit
from paderbox.utils.nested import deflatten
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data.data_provider import DataProvider
from padertorch.contrib.je.data.transforms import Transform as BaseTransform
from padertorch.contrib.je.data.transforms import segment_axis
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

nickname = 'wavenet-training'
ex = Exp(nickname)
storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / nickname / timeStamped('')[1:]
)
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


class Transform(BaseTransform):
    def __init__(
            self,
            input_sample_rate=16000,
            target_sample_rate=16000,
            frame_step=160,
            frame_length=400,
            fft_length=512,
            n_mels=64,
            fmin=50,
            storage_dir=None,
            segment_length=16000
    ):
        super().__init__(
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            frame_step=frame_step,
            frame_length=frame_length,
            fft_length=fft_length,
            fading=True,
            pad=True,
            n_mels=n_mels,
            fmin=fmin,
            label_key=None,
            storage_dir=storage_dir
        )
        self.segment_length = segment_length
        assert segment_length % self.frame_step == 0

    def extract_features(self, example, training=False):
        tail = example['audio_data'].shape[-1] % self.frame_step
        if tail > 0:
            pad_width = example['audio_data'].ndim * [(0, 0)]
            pad_width[-1] = (0, int(self.frame_step - tail))
            example['audio_data'] = np.pad(
                example['audio_data'], pad_width, mode='constant'
            )
        return super().extract_features(example)

    def finalize(self, example, training=False):
        # split channels
        audio, log_mel = [example['audio_data'], example['log_mel']]
        if self.segment_length is not None:
            audio_segment_step = audio_segment_length = self.segment_length
            mel_segment_step = audio_segment_step // self.frame_step
            mel_segment_length = self.stft.samples2frames(self.segment_length)
            audio, log_mel = segment_axis(
                [audio, log_mel], axis=1,
                segment_step=[audio_segment_step, mel_segment_step],
                segment_length=[audio_segment_length, mel_segment_length],
                pad=True
            )
            audio = audio.reshape((-1, audio_segment_length))
            log_mel = log_mel.reshape((-1, mel_segment_length, self.n_mels))

        fragments = []
        audio = np.split(audio, audio.shape[0], axis=0)
        log_mel = np.split(log_mel, log_mel.shape[0], axis=0)
        assert len(audio) == len(log_mel)
        for i in range(len(log_mel)):
            fragments.append(
                {
                    'example_id': example['example_id'],
                    'audio_data': audio[i].squeeze(0).astype(np.float32),
                    'log_mel': np.moveaxis(
                        log_mel[i].squeeze(0), 0, 1
                    ).astype(np.float32),
                }
            )
        return fragments


@ex.config
def config():
    # Data configuration

    data_provider = {
        'transform': {
            'factory': Transform
        },
        'max_workers': 8,
        'shuffle_buffer': 1000,
        'batch_size': 3
    }
    data_provider['prefetch_buffer'] = 4 * data_provider['batch_size']
    DataProvider.get_config(data_provider)

    # Trainer configuration
    trainer = deflatten({
        'model.factory':  WaveNet,
        'model.audio_key': 'audio_data',
        'model.feature_key': 'log_mel',
        'model.wavenet.n_cond_channels': data_provider['transform']['n_mels'],
        'model.wavenet.upsamp_window':
            data_provider['transform']['frame_length'],
        'model.wavenet.upsamp_stride':
            data_provider['transform']['frame_step'],
        'model.sample_rate':
            data_provider['transform']['target_sample_rate'],
        'optimizer.factory': Adam,
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1, 'epoch'),
        'max_trigger': (100, 'epoch')
    })
    Trainer.get_config(trainer)


@ex.capture
def get_datasets(data_provider):
    db = Timit()
    training_set = db.get_dataset('train')
    validation_set = db.get_dataset('test_core')

    dp = DataProvider.from_config(data_provider)
    dp.transform.initialize_norm(
        dataset=training_set, max_workers=dp.num_workers
    )
    return (
        dp.prepare_iterable(training_set, fragment=True, training=True),
        dp.prepare_iterable(validation_set, fragment=True)
    )


@ex.automain
def train(trainer):
    train_iter, validation_iter = get_datasets()

    trainer = Trainer.from_config(trainer)
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
