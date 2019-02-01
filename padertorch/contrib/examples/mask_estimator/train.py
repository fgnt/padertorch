"""
Very simple training script for a mask estimator.
Saves checkpoints and summaries to STORAGE_ROOT / 'simple_mask_estimator'
may be called with:
python -m padertorch.contrib.examples.mask_estimator.train.py
"""
import os
from pathlib import Path

import numpy as np
import torch

import paderbox as pb
import paderbox.database.keys as K
import padertorch as pt

STORAGE_ROOT = Path(os.environ['STORAGE_ROOT'])
assert STORAGE_ROOT.exists(), 'You have to specify an existing STORAGE_ROOT' \
                              'environmental variable see geting_started'


class SmallExampleModel(pt.Model):
    def __init__(self, num_features, num_units=1024, dropout=0.5,
                 activation='elu'):
        """
        :param num_features: number of input features
        :param num_units: number of units in linear layern
        :param dropout: dropout forget ratio
        :param activation:
        """
        super().__init__()
        self.num_features = num_features
        self.net = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_features, num_units),
            pt.mappings.ACTIVATION_FN_MAP[activation](),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_units, num_units),
            pt.mappings.ACTIVATION_FN_MAP[activation](),
            torch.nn.Linear(num_units, 2 * num_features),
            # twice num_features for speech and noise_mask
            torch.nn.Sigmoid()
            # Output activation to force outputs between 0 and 1
        )

    def forward(self, batch):
        x = batch['observation_abs']
        out = self.net(x)
        return dict(
            speech_mask_pred=out[..., :self.num_features],
            noise_mask_pred=out[..., self.num_features:],
        )

    def review(self, batch, output):
        noise_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['speech_mask_pred'], batch['speech_mask_target']
        )
        speech_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['noise_mask_pred'], batch['noise_mask_target']
        )
        return dict(loss=noise_mask_loss + speech_mask_loss)


def change_example_structure(example):
    stft = pb.transform.stft
    audio_data = example[K.AUDIO_DATA]
    net_input = dict()
    net_input['observation_abs'] = np.abs(stft(
        audio_data[K.OBSERVATION])).astype(np.float32)
    speech_image = stft(audio_data[K.SPEECH_IMAGE])
    noise_image = stft(audio_data[K.NOISE_IMAGE])
    target_mask, noise_mask = pb.speech_enhancement.biased_binary_mask(
        np.stack([speech_image, noise_image], axis=0)
    )
    net_input['speech_mask_target'] = target_mask.astype(np.float32)
    net_input['noise_mask_target'] = noise_mask.astype(np.float32)
    return net_input


def get_train_iterator(database: pb.database.JsonDatabase):
    audio_reader = pb.database.iterator.AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    train_iterator = database.get_iterator_by_names(database.datasets_train)
    return train_iterator.map(audio_reader).map(change_example_structure)


def get_validation_iterator(database: pb.database.JsonDatabase):
    audio_reader = pb.database.iterator.AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    train_iterator = database.get_iterator_by_names(database.datasets_eval)
    return train_iterator.map(audio_reader).map(change_example_structure)


def train():
    model = SmallExampleModel(513)
    database = pb.database.chime.Chime3()
    train_iterator = get_train_iterator(database)
    validation_iterator = get_validation_iterator(database)[:500]
    trainer = pt.Trainer(model, STORAGE_ROOT / 'simple_mask_estimator',
                         optimizer=pt.train.optimizer.Adam(),
                         max_trigger=(int(1e5), 'iteration'))
    trainer.test_run(train_iterator, validation_iterator)
    trainer.train(train_iterator, validation_iterator)


if __name__ == '__main__':
    train()
