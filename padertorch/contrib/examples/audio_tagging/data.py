import numpy as np
from padercontrib.database.audio_set import AudioSet
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MultiHotLabelEncoder, Collate
)
from padertorch.contrib.je.data.mixup import MixUpDataset, SuperposeEvents
from padertorch.contrib.je.modules.augment import LogTruncNormalSampler


def get_datasets(
        audio_reader, stft,
        num_workers, batch_size, max_padding_rate,
        storage_dir
):

    db = AudioSet()
    training_set = db.get_dataset('balanced_train')

    event_encoder = MultiHotLabelEncoder(
        label_key='events', storage_dir=storage_dir,
    )
    event_encoder.initialize_labels(dataset=training_set, verbose=True)

    kwargs = dict(
        audio_reader=audio_reader, stft=stft, event_encoder=event_encoder,
        num_workers=num_workers, batch_size=batch_size,
        max_padding_rate=max_padding_rate,
    )

    return (
        prepare_dataset(training_set, training=True, **kwargs),
        prepare_dataset(db.get_dataset('validate'), **kwargs),
    )


def prepare_dataset(
        dataset,
        audio_reader, stft, event_encoder,
        num_workers, batch_size, max_padding_rate,
        training=False,
):

    dataset = dataset.filter(lambda ex: 10.1 > ex['audio_length'] > 1.3, lazy=False)

    audio_reader = AudioReader(**audio_reader)

    def normalize(example):
        example['audio_data'] -= example['audio_data'].mean(-1, keepdims=True)
        example['audio_data'] = example['audio_data'].mean(0, keepdims=True)
        example['audio_data'] /= np.abs(example['audio_data']).max() + 1e-3
        return example
    dataset = dataset.map(audio_reader).map(normalize)

    if training:
        def random_scale(example):
            c = example['audio_data'].shape[0]
            scales = LogTruncNormalSampler(scale=1., truncation=3.)(c)[:, None]
            example['audio_data'] *= scales
            return example
        dataset = dataset.map(random_scale)
        dataset = MixUpDataset(dataset, dataset, p=[1/2, 1/2]).map(
            SuperposeEvents(
                audio_reader.target_sample_rate, min_overlap=.8, max_length=12.
            )
        )
        dataset = dataset.shuffle(reshuffle=True)

    stft = STFT(**stft)
    dataset = dataset.map(stft).map(event_encoder)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'stft': example['stft'].astype(np.float32),
            'seq_len': example['stft'].shape[1],
            'events': example['events'].astype(np.float32),
        }

    dataset = dataset.map(finalize)\
        .prefetch(num_workers, 10*batch_size, catch_filter_exception=True)

    if training:
        dataset = dataset.shuffle(
            reshuffle=True, buffer_size=min(100 * batch_size, 1000)
        )
    return dataset.batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key="seq_len",
        max_padding_rate=max_padding_rate, expiration=1000*batch_size,
        drop_incomplete=training, sort_key="seq_len", reverse_sort=True
    ).map(Collate())
