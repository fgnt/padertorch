"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.audio_tagging.simple_train
"""
import os
from pathlib import Path

import numpy as np
import torch
from paderbox.database.audio_set import AudioSet
from paderbox.utils.timer import timeStamped
from padertorch import Model, Trainer, optimizer
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MelTransform, Normalizer, LabelEncoder, Collate
)
from padertorch.contrib.je.modules.conv import CNN2d
from torch import nn

storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'audio_tagging' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)


class MultiHotLabelEncoder(LabelEncoder):
    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        nhot_encoding = np.zeros(527).astype(np.float32)
        nhot_encoding[labels] = 1
        example[self.label_key] = nhot_encoding
        return example


class WALNet(Model):
    """
    >>> from paderbox.utils.nested import deflatten
    >>> tagger = WALNet(output_size=10)
    >>> inputs = {'log_mel': torch.zeros(4,1,128,128), 'events': torch.zeros((4,10))}
    >>> outputs = tagger(inputs)
    >>> outputs.shape
    torch.Size([4, 10, 1])
    >>> review = tagger.review(inputs, outputs)
    """

    def __init__(self, output_size):
        super().__init__()
        input_channels = 1
        self.cnn = CNN2d(
            in_channels=input_channels,
            out_channels=output_size,
            hidden_channels=[
                16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024
            ],
            kernel_size=11 * [3] + [2, 1],
            num_layers=13,
            padding=11 * ['both'] + 2 * [None],
            pool_size=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1],
            norm='batch',
            activation='relu'
        )

    def forward(self, inputs):
        y = self.cnn(inputs['features']).squeeze(2)
        return nn.Sigmoid()(y)

    def review(self, inputs, outputs):
        # compute loss
        x = inputs['features']
        targets = inputs['events']
        frame_probs = outputs
        sequence_probs = frame_probs.mean(-1)
        bce = nn.BCELoss(reduction='none')(sequence_probs, targets).sum(-1)

        # create review including metrics and visualizations
        review = dict(
            loss=bce.mean(),
            scalars=dict(),
            histograms=dict(
                sequence_probs=sequence_probs.flatten()
            ),
            images=dict(
                input=x[:3].flip(1).view((-1, x.shape[-1]))[None]
            )
        )
        for thres in [0.3, 0.5]:
            decision = (sequence_probs.detach() > thres).float()
            true_pos = (decision * targets).sum()
            false_pos = (decision * (1.-targets)).sum()
            false_neg = ((1.-decision) * targets).sum()
            review['scalars'].update({
                f'true_pos_{thres}': true_pos,
                f'false_pos_{thres}': false_pos,
                f'false_neg_{thres}': false_neg
            })
        return review

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        # compute precision, recall and fscore for each decision threshold
        for thres in [0.3, 0.5]:
            true_pos_key=f'true_pos_{thres}'
            false_pos_key=f'false_pos_{thres}'
            false_neg_key=f'false_neg_{thres}'
            if all([
                key in summary['scalars']
                for key in [true_pos_key, false_pos_key, false_neg_key]
            ]):
                tp = np.sum(summary['scalars'].pop(true_pos_key))
                fp = np.sum(summary['scalars'].pop(false_pos_key))
                fn = np.sum(summary['scalars'].pop(false_neg_key))
                p = tp/(tp+fp)
                r = tp/(tp+fn)
                summary['scalars'][f'precision_{thres}'] = p
                summary['scalars'][f'recall_{thres}'] = r
                summary['scalars'][f'f1_{thres}'] = 2*(p*r)/(p+r)

        # normalize images
        for image in summary['images'].values():
            image -= image.min()
            image /= image.max()
        return summary


def get_datasets():
    db = AudioSet()
    validation_data = prepare_dataset(db.get_dataset('validate'), training=False)
    training_data = prepare_dataset(db.get_dataset('balanced_train'), training=True)
    return training_data, validation_data


def prepare_dataset(dataset, training=False):
    dataset = dataset.filter(lambda ex: ex['audio_length'] > 1.3, lazy=False)
    batch_size = 48
    label_encoder = MultiHotLabelEncoder(
        label_key='events', storage_dir=storage_dir
    )
    label_encoder.initialize_labels(dataset, verbose=True)
    dataset = dataset.map(label_encoder)
    audio_reader = AudioReader(
        source_sample_rate=44100, target_sample_rate=44100
    )
    dataset = dataset.map(audio_reader)
    stft = STFT(
        shift=882, window_length=1764, size=2048, fading=None, pad=False
    )
    dataset = dataset.map(stft)
    mel_transform = MelTransform(
        sample_rate=44100, fft_length=2048, n_mels=128, fmin=50
    )
    dataset = dataset.map(mel_transform)
    normalizer = Normalizer(
        key='mel_transform', center_axis=(1,), scale_axis=(1, 2),
        storage_dir=storage_dir
    )
    normalizer.initialize_moments(
        dataset.shuffle()[:2000].prefetch(num_workers=8, buffer_size=16),
        verbose=True
    )
    dataset = dataset.map(normalizer)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'features': np.moveaxis(example['mel_transform'].mean(0, keepdims=True), 1, 2).astype(np.float32),
            'seq_len': example['mel_transform'].shape[-2],
            'events': example['events'].astype(np.float32)
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


def main():
    model = WALNet(527)
    trainer = Trainer(
        model=model,
        optimizer=optimizer.Adam(lr=3e-5, gradient_clipping=30.),
        storage_dir=storage_dir,
        summary_trigger=(100, 'iteration'),
        stop_trigger=(20000, 'iteration'),
        checkpoint_trigger=(1000, 'iteration')
    )
    training_data, validation_data = get_datasets()
    trainer.register_validation_hook(validation_data)

    trainer.test_run(training_data, validation_data)
    trainer.train(training_data)


if __name__ == '__main__':
    main()
