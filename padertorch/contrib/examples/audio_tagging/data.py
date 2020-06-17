import numpy as np
from padercontrib.database.audio_set import AudioSet
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MultiHotLabelEncoder, Collate
)


def get_datasets(
        audio_reader, stft,
        num_workers, batch_size, max_padding_rate,
        storage_dir
):

    db = AudioSet()
    training_set = db.get_dataset('balanced_train')

    event_encoder = MultiHotLabelEncoder(
        label_key='events', storage_dir=storage_dir, to_array=True,
    )
    event_encoder.initialize_labels(dataset=training_set, verbose=True)
    training_set = training_set.map(event_encoder)

    validation_set = db.get_dataset('validate').map(event_encoder)
    kwargs = dict(
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers, batch_size=batch_size,
        max_padding_rate=max_padding_rate,
    )

    return (
        prepare_dataset(training_set, training=True, **kwargs),
        prepare_dataset(validation_set, **kwargs),
    )


def prepare_dataset(
        dataset,
        audio_reader, stft,
        num_workers, batch_size, max_padding_rate,
        training=False,
):

    dataset = dataset.filter(lambda ex: 10.1 > ex['audio_length'] > 1.3, lazy=False)

    if training:
        dataset = dataset.shuffle(reshuffle=True)
    audio_reader = AudioReader(**audio_reader)
    stft = STFT(**stft)
    dataset = dataset.map(audio_reader).map(stft)

    def finalize(example):
        return [
            {
                'example_id': example['example_id'],
                'stft': features[None].astype(np.float32),
                'seq_len': features.shape[0],
                'events': example['events'].astype(np.float32),
            }
            for features in example['stft']
        ]

    dataset = dataset.map(finalize)\
        .prefetch(num_workers, 10*batch_size, catch_filter_exception=True)\
        .unbatch()

    if training:
        dataset = dataset.shuffle(
            reshuffle=True, buffer_size=min(100 * batch_size, 1000)
        )
    return dataset.batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key="seq_len",
        max_padding_rate=max_padding_rate, expiration=1000*batch_size,
        drop_incomplete=training, sort_key="seq_len", reverse_sort=True
    ).map(Collate())
