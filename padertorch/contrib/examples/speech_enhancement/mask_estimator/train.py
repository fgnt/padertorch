"""
Very simple training script for a mask estimator.
Saves checkpoints and summaries to $STORAGE_ROOT/speech_enhancement/simple_mask_estimator_{id}
may be called with:
python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.train with database_json=/path/to/json
"""

from pathlib import Path

import os
import numpy as np
import paderbox as pb
import padertorch as pt
from lazy_dataset.database import JsonDatabase
from pb_bss.extraction.mask_module import biased_binary_mask
from sacred import Experiment, observers

from .model import SimpleMaskEstimator

ex = Experiment('Train Simple Mask Estimator')


@ex.config
def config():
    storage_dir = None
    if storage_dir is None:
        storage_dir = pt.io.get_new_storage_dir(
            'speech_enhancement', prefix='simple_mask_estimator')
    database_json = None
    if database_json is None:
        if 'NT_DATABASE_JSONS_DIR' in os.environ:
            database_json = Path(
                os.environ['NT_DATABASE_JSONS_DIR']) / 'chime.json'
    assert database_json is not None, (
        'You have to specify a path to a json describing your database,'
        'use "with database_json=/Path/To/Json" as suffix to your call'
    )
    assert Path(database_json).exists(), database_json
    ex.observers.append(observers.FileStorageObserver(
        Path(storage_dir).expanduser().resolve() / 'sacred')
    )


def prepare_data(example):
    stft = pb.transform.STFT(shift=256, size=1024)
    net_input = dict()
    audio_data = dict()
    for key in ['observation', 'speech_image', 'noise_image']:
        audio_data[key] = stft(np.array([
            pb.io.load_audio(audio) for audio in example['audio_path'][key]]))
    net_input['observation_abs'] = np.abs(
        audio_data['observation']).astype(np.float32)
    target_mask, noise_mask = biased_binary_mask(np.stack(
        [audio_data['speech_image'], audio_data['noise_image']], axis=0
    ))
    net_input['speech_mask_target'] = target_mask.astype(np.float32)
    net_input['noise_mask_target'] = noise_mask.astype(np.float32)
    return net_input


def get_train_dataset(database: JsonDatabase):
    train_ds = database.get_dataset('tr05_simu')
    return (train_ds
            .map(prepare_data)
            .prefetch(num_workers=4, buffer_size=4))


def get_validation_dataset(database: JsonDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    val_iterator = database.get_dataset('dt05_simu')
    return val_iterator.map(prepare_data) \
        .prefetch(num_workers=4, buffer_size=4)


@ex.command
def test_run(storage_dir, database_json):
    model = SimpleMaskEstimator(513)
    print(f'Simple training for the following model: {model}')
    database = JsonDatabase(database_json)
    train_dataset = get_train_dataset(database)
    validation_dataset = get_validation_dataset(database)
    trainer = pt.train.trainer.Trainer(
        model, storage_dir, optimizer=pt.train.optimizer.Adam(),
        stop_trigger=(int(1e5), 'iteration')
    )
    trainer.test_run(train_dataset, validation_dataset)


@ex.automain
def train(storage_dir, database_json):
    model = SimpleMaskEstimator(513)
    print(f'Simple training for the following model: {model}')
    database = JsonDatabase(database_json)
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
