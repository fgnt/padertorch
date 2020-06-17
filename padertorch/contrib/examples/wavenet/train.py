"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from pathlib import Path

import numpy as np
from padercontrib.database.librispeech import LibriSpeech
from paderbox.utils.timer import timeStamped
from padertorch import modules
from padertorch.contrib.examples.wavenet.wavenet import WaveNet
from padertorch.contrib.je.data.transforms import AudioReader, STFT, \
    fragment_signal, Collate
from padertorch.contrib.je.data.utils import split_dataset
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer


def get_datasets(max_length=1., batch_size=3):
    db = LibriSpeech()
    train_clean_100 = db.get_dataset('train_clean_100')

    train_set, validate_set = split_dataset(train_clean_100, fold=0)
    test_set = db.get_dataset('test_clean')
    training_data = prepare_dataset(train_set, max_length=max_length, batch_size=batch_size, training=True)
    validation_data = prepare_dataset(validate_set, max_length=max_length, batch_size=batch_size, training=False)
    test_data = prepare_dataset(test_set, max_length=max_length, batch_size=batch_size, training=False)
    return training_data, validation_data, test_data


def prepare_dataset(dataset, max_length=1., batch_size=3, training=False):
    dataset = dataset.filter(lambda ex: ex['num_samples'] > 16000, lazy=False)
    stft_shift = 160
    window_length = 480
    target_sample_rate = 16000

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['speaker_id'] = example['speaker_id'].split('-')[0]
        return example

    dataset = dataset.map(prepare_example)

    audio_reader = AudioReader(
        source_sample_rate=16000, target_sample_rate=target_sample_rate
    )
    dataset = dataset.map(audio_reader)

    stft = STFT(
        shift=stft_shift, window_length=window_length, size=512, fading='full',
        pad=True
    )
    dataset = dataset.map(stft)

    def fragment(example):
        audio, features = example['audio_data'], example['stft']
        pad_width = window_length - stft_shift
        assert pad_width > 0, pad_width
        audio = np.pad(
            audio, (audio.ndim-1)*[(0, 0)] + [(pad_width, window_length - 1)],
            mode='constant')
        fragment_step = int(max_length*target_sample_rate)
        fragment_length = fragment_step + 2*pad_width
        stft_fragment_step = fragment_step / stft_shift
        stft_fragment_length = stft.samples_to_frames(fragment_step)
        fragments = []
        for audio, features in zip(*fragment_signal(
            audio, features, axis=1,
            step=[fragment_step, stft_fragment_step],
            max_length=[fragment_length, stft_fragment_length],
            min_length=[fragment_length, stft_fragment_length],
            random_start=training
        )):
            fragments.append({
                'example_id': example['example_id'],
                'audio_data': audio[..., pad_width:-pad_width].astype(np.float32),
                'stft': features.astype(np.float32),
                'seq_len': features.shape[1],
            })
        return fragments

    dataset = dataset.map(fragment)

    if training:
        dataset = dataset.shuffle(reshuffle=True)
    return dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    ).unbatch().shuffle(reshuffle=True, buffer_size=10*batch_size).batch(
        batch_size=batch_size
    ).map(Collate())


def get_model():
    wavenet = modules.wavenet.WaveNet(
        n_cond_channels=64, upsamp_window=400, upsamp_stride=160, fading='full'
    )
    model = WaveNet(
        wavenet=wavenet, sample_rate=16000, fft_length=512, n_mels=64, fmin=50
    )
    return model


def train(model, storage_dir):
    train_set, validate_set, _ = get_datasets()

    trainer = Trainer(
        model=model,
        optimizer=Adam(lr=5e-4),
        storage_dir=str(storage_dir),
        summary_trigger=(1000, 'iteration'),
        checkpoint_trigger=(10000, 'iteration'),
        stop_trigger=(100000, 'iteration')
    )

    trainer.test_run(train_set, validate_set)
    trainer.register_validation_hook(validate_set)
    trainer.train(train_set)


if __name__ == '__main__':
    storage_dir = str(
        Path(os.environ['STORAGE_ROOT']) / 'wavenet' / timeStamped('')[1:]
    )
    os.makedirs(storage_dir, exist_ok=True)
    model = get_model()
    train(model, storage_dir)
