from padertorch import Trainer
from padertorch.train.optimizer import Adam
from padertorch.contrib.je.data.transforms import Transform, Collate
from padertorch.contrib.je.data.data_provider import split_dataset
from padertorch.contrib.je.modules.conv import CNN1d
from padertorch.modules.fully_connected import fully_connected_stack
from padertorch.contrib.examples.speaker_classification.model import SpeakerClf
from paderbox.database.librispeech import LibriSpeech
from functools import partial
from torch.nn import GRU
from pathlib import Path
from paderbox.utils.timer import timeStamped
import os


storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / 'speaker_clf' / timeStamped('')[1:]
)
os.makedirs(storage_dir, exist_ok=True)


def prepare_dataset(
        dataset, transform, batch_size, training=False, num_workers=8
):
    dataset = dataset.map(partial(transform, training=training))
    if training:
        dataset = dataset.shuffle(reshuffle=True)

    dataset = dataset.prefetch(
        num_workers, 10*batch_size, catch_filter_exception=True
    )

    dataset = dataset.batch_bucket_dynamic(
        batch_size=batch_size,
        key='seq_len',
        max_padding_rate=0.2,
        expiration=1000,
        drop_incomplete=training,
        sort_by_key=True
    )
    dataset = dataset.map(Collate())
    return dataset


def get_datasets():
    db = LibriSpeech()
    train_clean_100 = db.get_dataset('train_clean_100')

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['speaker_id'] = example['speaker_id'].split('-')[0]
        return example

    train_clean_100 = train_clean_100.map(prepare_example)

    train_portion, validate_portion = split_dataset(train_clean_100, fold=0)

    transform = Transform(
        input_sample_rate=16000,
        target_sample_rate=16000,
        frame_step=160,
        frame_length=400,
        fft_length=512,
        n_mels=64,
        fmin=50,
        storage_dir=storage_dir,
        label_key='speaker_id'
    )
    transform.initialize_norm(train_portion.shuffle()[:10000], max_workers=8)
    transform.initialize_labels(train_portion)
    print(len(transform.label_mapping))

    return (
        prepare_dataset(
            train_portion, transform, batch_size=16, training=True
        ),
        prepare_dataset(
            validate_portion, transform, batch_size=16, training=False
        )
    )


def get_model():
    cnn = CNN1d(
        in_channels=64,
        hidden_channels=512,
        num_layers=4,
        out_channels=None,
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
        max_trigger=(100000, 'iteration')
    )

    trainer.train(train_set, validate_set)


if __name__ == '__main__':
    model = get_model()
    train(model)
