"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.audio_tagging.train print_config
python -m padertorch.contrib.examples.audio_tagging.train
"""
import torch
from torch import nn
import os
from pathlib import Path
import numpy as np

from paderbox.utils.timer import timeStamped
from paderbox.io.audioread import load_audio
from paderbox.transform.module_stft import STFT
from paderbox.transform.module_fbank import MelTransform
from paderbox.database.audio_set import AudioSet
from padertorch import Model, Trainer, optimizer
from padertorch.contrib.je.modules.conv import CNN2d
from padertorch.contrib.je.data.transforms import GlobalNormalize, LabelEncoder, Collate


storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'audio_tagging' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)


class WALNet(Model):
    """
    >>> from paderbox.utils.nested import deflatten
    >>> tagger = WALNet(output_size=10)
    >>> inputs = {'mel_spectrogram': torch.zeros(4,1,128,128), 'events': torch.zeros((4,10))}
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
        y = self.cnn(inputs['mel_spectrogram']).squeeze(2)
        return nn.Sigmoid()(y)

    def review(self, inputs, outputs):
        # compute loss
        x = inputs['mel_spectrogram']
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
    validation_data = prepare_dataset(db.get_dataset('validate'))
    training_data = prepare_dataset(db.get_dataset('balanced_train'))
    return training_data, validation_data


def prepare_dataset(dataset):
    batch_size = 48

    return normalize_features(extract_features(read_audio(
        encode_events(dataset)
    ))).prefetch(
        num_workers=8, buffer_size=2*batch_size
    ).batch(
        batch_size=batch_size
    ).map(Collate())


def encode_events(dataset):
    label_encoder = LabelEncoder(input_path="events")
    # Will save label mapping after first call
    label_encoder.init_params(storage_dir=storage_dir, dataset=dataset)

    def encode(example):
        nhot_encoding = np.zeros(527).astype(np.float32)
        events = label_encoder.encode(example["events"])
        nhot_encoding[events] = 1
        example["events"] = nhot_encoding
        return example

    return dataset.map(encode)


def read_audio(dataset):

    def read(example):
        example['audio_data'] = load_audio(example['audio_path'])
        return example

    return dataset.map(read)


def extract_features(dataset):

    stft = STFT(
        frame_step=882,
        frame_length=1764,
        fft_length=2048,
        pad=False,
        fading=False,
        always3d=True
    )
    mel_transform = MelTransform(
        sample_rate=44100, fft_length=2048, n_mels=128, fmin=50
    )

    def extraction(example):
        x = stft(example.pop("audio_data"))
        x = x.real**2 + x.imag**2
        x = mel_transform(x)
        example["mel_spectrogram"] = x.mean(0, keepdims=True).transpose(
            (0, 2, 1)).astype(np.float32)
        return example

    return dataset.map(extraction)


def normalize_features(dataset):

    normalize = GlobalNormalize(
        'mel_spectrogram', center_axis=-1, scale_axis=(-2, -1)
    )
    # Will save moments after first call
    normalize.init_params(storage_dir=storage_dir, dataset=dataset)

    return dataset.map(normalize)


def main():
    model = WALNet(527)
    trainer = Trainer(
        model=model,
        optimizer=optimizer.Adam(lr=3e-5, gradient_clipping=30.),
        storage_dir=storage_dir,
        summary_trigger=(100, 'iteration'),
        max_trigger=(20000, 'iteration'),
        checkpoint_trigger=(1000, 'iteration')
    )
    training_data, validation_data = get_datasets()
    trainer.train(training_data, validation_data)


if __name__ == '__main__':
    main()
