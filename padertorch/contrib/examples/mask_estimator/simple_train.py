"""
Very simple training script for a mask estimator.
Saves checkpoints and summaries to $STORAGE_ROOT/simple_mask_estimator
may be called with:
python -m padertorch.contrib.examples.mask_estimator.simple_train
"""
import os
from pathlib import Path

import numpy as np
import torch

import paderbox as pb
import paderbox.database.keys as K
import padertorch as pt
from padertorch.summary import mask_to_image, stft_to_image


class SimpleMaskEstimator(pt.Model):
    def __init__(self, num_features, num_units=1024, dropout=0.5,
                 activation='elu'):
        """

        :param num_features: number of input features
        :param num_units: number of units in linear layern
        :param dropout: dropout forget ratio
        :param activation:

        >>> SimpleMaskEstimator(513)
        SmallExampleModel(
          (net): Sequential(
            (0): Dropout(p=0.5)
            (1): Linear(in_features=513, out_features=1024, bias=True)
            (2): ELU(alpha=1.0)
            (3): Dropout(p=0.5)
            (4): Linear(in_features=1024, out_features=1024, bias=True)
            (5): ELU(alpha=1.0)
            (6): Linear(in_features=1024, out_features=1026, bias=True)
            (7): Sigmoid()
          )
        )
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
            # twice num_features for speech and noise_mask
            torch.nn.Linear(num_units, 2 * num_features),
            # Output activation to force outputs between 0 and 1
            torch.nn.Sigmoid()
        )

    def forward(self, batch):

        x = batch['observation_abs']
        out = self.net(x)
        return dict(
            speech_mask_prediction=out[..., :self.num_features],
            noise_mask_prediction=out[..., self.num_features:],
        )

    def review(self, batch, output):
        noise_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['noise_mask_prediction'], batch['noise_mask_target']
        )
        speech_mask_loss = torch.nn.functional.binary_cross_entropy(
            output['speech_mask_prediction'], batch['speech_mask_target']
        )
        return dict(loss=noise_mask_loss + speech_mask_loss,
                    images=self.add_images(batch, output))

    def add_images(self, batch, output):
        speech_mask = output['speech_mask_prediction']
        observation = batch['observation_abs']
        images = dict()
        images['speech_mask'] = mask_to_image(speech_mask, True)
        images['observed_stft'] = stft_to_image(observation, True)

        if 'noise_mask_prediction' in output:
            noise_mask = output['noise_mask_prediction']
            images['noise_mask'] = mask_to_image(noise_mask, True)
        if batch is not None and 'speech_mask_prediction' in batch:
            images['speech_mask_target'] = mask_to_image(
                batch['speech_mask_target'], True)
            if 'speech_mask_target' in batch:
                images['noise_mask_target'] = mask_to_image(
                    batch['noise_mask_target'], True)
        return images


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
    target_mask, noise_mask = pb.speech_enhancement.biased_binary_mask(
        np.stack([speech_image, noise_image], axis=0)
    )
    net_input['speech_mask_target'] = target_mask.astype(np.float32)
    net_input['noise_mask_target'] = noise_mask.astype(np.float32)
    return net_input


def get_train_iterator(database: pb.database.JsonDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    audio_reader = pb.database.iterator.AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    train_iterator = database.get_dataset(database.datasets_train)
    return train_iterator.map(audio_reader)\
        .map(change_example_structure)\
        .prefetch(num_workers=4, buffer_size=4)


def get_validation_iterator(database: pb.database.JsonDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    audio_reader = pb.database.iterator.AudioReader(audio_keys=[
        K.OBSERVATION, K.NOISE_IMAGE, K.SPEECH_IMAGE
    ])
    val_iterator = database.get_iterator_by_names(database.datasets_eval)
    return val_iterator.map(audio_reader)\
        .map(change_example_structure)\
        .prefetch(num_workers=4, buffer_size=4)


def train():
    model = SimpleMaskEstimator(513)
    print(f'Simple training for the following model: {model}')
    database = pb.database.chime.Chime3()
    train_iterator = get_train_iterator(database)
    validation_iterator = get_validation_iterator(database)
    trainer = pt.Trainer(model, STORAGE_ROOT / 'simple_mask_estimator',
                         optimizer=pt.train.optimizer.Adam(),
                         stop_trigger=(int(1e5), 'iteration'))
    trainer.test_run(train_iterator, validation_iterator)
    trainer.register_validation_hook(validation_iterator)
    trainer.train(train_iterator)


if __name__ == '__main__':
    STORAGE_ROOT = os.environ.get('STORAGE_ROOT')
    if STORAGE_ROOT is None:
        raise EnvironmentError(
            'You have to specify an STORAGE_ROOT '
            'environmental variable see getting_started'
        )
    elif not Path(STORAGE_ROOT).exists():
        raise FileNotFoundError(
            'You have to specify an existing STORAGE_ROOT '
            'environmental variable see getting_started.\n'
            f'Got: {STORAGE_ROOT}'
        )
    else:
        STORAGE_ROOT = Path(STORAGE_ROOT)
    train()
