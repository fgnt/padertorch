"""
Very simple training script for a mask estimator.
Saves checkpoints and summaries to $STORAGE_ROOT/speech_enhancement/simple_mask_estimator_{id}
may be called with:
python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.train
"""

import numpy as np
import paderbox as pb
import padercontrib.database.keys as K
import padertorch as pt
from sacred import Experiment
from padercontrib.database import JsonAudioDatabase
from padercontrib.database.chime import Chime3
from padercontrib.database.iterator import AudioReader
from pb_bss.extraction.mask_module import biased_binary_mask
from padertorch.contrib.examples.speech_enhancement.simple_mask_estimator import SimpleMaskEstimator

ex = Experiment('Simple Mask Estimator')


def change_example_structure(example):
    stft = pb.transform.stft
    audio_data = example[K.AUDIO_DATA]
    net_input = dict()
    net_input['observation_stft'] = stft(
        audio_data[K.OBSERVATION]).astype(np.complex64)
    net_input['observation_abs'] = np.abs(
        net_input['observation_stft']).astype(np.float32)
    speech_image = stft(audio_data[K.SPEECH_IMAGE])
    noise_image = stft(audio_data[K.NOISE_IMAGE])
    target_mask, noise_mask = biased_binary_mask(
        np.stack([speech_image, noise_image], axis=0)
    )
    net_input['speech_mask_target'] = target_mask.astype(np.float32)
    net_input['noise_mask_target'] = noise_mask.astype(np.float32)
    return net_input


def get_train_dataset(database: JsonAudioDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    audio_reader = AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    train_ds = database.get_dataset_train()
    return (train_ds
            .map(audio_reader)
            .map(change_example_structure)
            .prefetch(num_workers=4, buffer_size=4))


def get_validation_dataset(database: JsonAudioDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    audio_reader = AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    val_iterator = database.get_dataset_validation()
    return val_iterator.map(audio_reader) \
        .map(change_example_structure) \
        .prefetch(num_workers=4, buffer_size=4)


@ex.command
def test_run():
    model = SimpleMaskEstimator(513)
    print(f'Simple training for the following model: {model}')
    database = Chime3()
    train_dataset = get_train_dataset(database)
    validation_dataset = get_validation_dataset(database)
    trainer = pt.train.trainer.Trainer(
        model, '/this/is/no/path', optimizer=pt.train.optimizer.Adam(),
        stop_trigger=(int(1e5), 'iteration')
    )
    trainer.test_run(train_dataset, validation_dataset)


@ex.automain
def train():
    storage_dir = pt.io.get_new_storage_dir(
        'speech_enhancement', prefix='simple_mask_estimator')
    model = SimpleMaskEstimator(513)
    print(f'Simple training for the following model: {model}')
    database = Chime3()
    train_dataset = get_train_dataset(database)
    validation_dataset = get_validation_dataset(database)
    trainer = pt.Trainer(model, storage_dir,
                         optimizer=pt.train.optimizer.Adam(),
                         stop_trigger=(int(1e5), 'iteration'))
    trainer.test_run(train_dataset, validation_dataset)
    trainer.register_validation_hook(
        validation_dataset, n_back_off=5, lr_update_factor=1 / 10,
        back_off_patience=1, early_stopping_patience=None)
    trainer.train(train_dataset)
