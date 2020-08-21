"""
Very simple training script for a mask estimator.
Saves checkpoints and summaries to $STORAGE_ROOT/speech_enhancement/simple_mask_estimator_{id}
may be called with:
python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.train
"""
import os
from pathlib import Path

import numpy as np
import torch

import paderbox as pb
import padercontrib.database.keys as K
from padercontrib.database import JsonAudioDatabase
from padercontrib.database.iterator import AudioReader
from padercontrib.database.chime import Chime3
from pb_bss.extraction.mask_module import biased_binary_mask
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
            pt.modules.Normalization(
                'btf', (1, 1, num_features), statistics_axis='t',
                independent_axis='f', batch_axis='b', sequence_axis='t'
            ),
            pt.modules.StatefulLSTM(
                num_features, num_units // 4,
                bidirectional=True, batch_first=True
            ),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((num_units // 4) * 2, num_units),
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
    return val_iterator.map(audio_reader)\
        .map(change_example_structure)\
        .prefetch(num_workers=4, buffer_size=4)


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


if __name__ == '__main__':
    train()
