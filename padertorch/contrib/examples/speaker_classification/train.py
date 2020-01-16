"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.speaker_classification.train
"""
import os
from pathlib import Path

import numpy as np
from padercontrib.database.librispeech import LibriSpeech
from paderbox.utils.timer import timeStamped
from padertorch import Trainer
from padertorch.contrib.examples.speaker_classification.model import SpeakerClf
from padertorch.contrib.je.data.transforms import LabelEncoder, AudioReader, \
    STFT, MelTransform, Normalizer, Collate
from padertorch.contrib.je.data.utils import split_dataset
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.train.optimizer import Adam
from torch.nn import GRU

storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'speaker_clf' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)


def get_datasets():
    db = LibriSpeech()
    train_clean_100 = db.get_dataset('train_clean_100')

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['speaker_id'] = example['speaker_id'].split('-')[0]
        return example

    train_clean_100 = train_clean_100.map(prepare_example)

    train_set, validate_set = split_dataset(train_clean_100, fold=0)
    training_data = prepare_dataset(train_set, training=True)
    validation_data = prepare_dataset(validate_set, training=False)
    return training_data, validation_data


def prepare_dataset(dataset, training=False):
    batch_size = 16
    label_encoder = LabelEncoder(
        label_key='speaker_id', storage_dir=storage_dir
    )
    label_encoder.initialize_labels(dataset, verbose=True)
    dataset = dataset.map(label_encoder)
    audio_reader = AudioReader(
        source_sample_rate=16000, target_sample_rate=16000
    )
    dataset = dataset.map(audio_reader)
    stft = STFT(
        shift=160, window_length=400, size=512, fading=None, pad=False
    )
    dataset = dataset.map(stft)
    mel_transform = MelTransform(
        sample_rate=16000, fft_length=512, n_mels=64, fmin=50
    )
    dataset = dataset.map(mel_transform)
    normalizer = Normalizer(
        key='mel_transform', center_axis=(1,), scale_axis=(1, 2),
        storage_dir=storage_dir
    )
    normalizer.initialize_moments(
        dataset.shuffle()[:10000].prefetch(num_workers=8, buffer_size=16),
        verbose=True
    )
    dataset = dataset.map(normalizer)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': np.moveaxis(example['mel_transform'], 1, 2).astype(np.float32),
            'seq_len': example['mel_transform'].shape[-2],
            'speaker_id': example['speaker_id'].astype(np.int64)
        }

    dataset = dataset.map(finalize)

    if training:
        dataset = dataset.shuffle(reshuffle=True)
    return dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    ).batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key='seq_len', max_padding_rate=0.1,
        expiration=1000*batch_size, drop_incomplete=training,
        sort_key='seq_len', reverse_sort=True
    ).map(Collate())


def get_model():
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

    speaker_clf = SpeakerClf(cnn, gru, fcn)
    return speaker_clf


def train(speaker_clf):
    train_set, validate_set = get_datasets()

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
    model = get_model()
    train(model)
