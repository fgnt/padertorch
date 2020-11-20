import numpy as np
from lazy_dataset.database import JsonDatabase
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MultiHotLabelEncoder, Collate
)
from padertorch.contrib.je.data.mixup import MixUpDataset, \
    SampleMixupComponents, SuperposeEvents
from padertorch.contrib.je.modules.augment import LogTruncNormalSampler


def get_datasets(
        database_json, audio_reader, stft, batch_size, storage_dir,
        num_workers=8, max_padding_rate=.05,
        min_signal_length=None, max_signal_length=None,
        mixup_probs=(1,), min_mixup_overlap=0., max_mixup_length=None,
        training_set='balanced_train',
):

    db = JsonDatabase(database_json)
    training_set = db.get_dataset(training_set)

    event_encoder = MultiHotLabelEncoder(
        label_key='events', storage_dir=storage_dir,
    )
    event_encoder.initialize_labels(dataset=training_set, verbose=True)

    kwargs = dict(
        audio_reader=audio_reader, stft=stft, event_encoder=event_encoder,
        num_workers=num_workers,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        min_signal_length=min_signal_length, max_signal_length=max_signal_length,
        mixup_probs=mixup_probs,
        min_mixup_overlap=min_mixup_overlap, max_mixup_length=max_mixup_length
    )

    return (
        prepare_dataset(training_set, training=True, **kwargs),
        prepare_dataset(db.get_dataset('validate'), **kwargs),
        prepare_dataset(db.get_dataset('eval'), **kwargs),
    )


def prepare_dataset(
        dataset,
        audio_reader, stft, event_encoder,
        num_workers, batch_size, max_padding_rate,
        min_signal_length=None, max_signal_length=None,
        mixup_probs=(1,), min_mixup_overlap=0., max_mixup_length=None,
        training=False,
):
    assert np.sum(mixup_probs) == 1., mixup_probs
    if min_signal_length is not None or max_signal_length is not None:
        dataset = dataset.filter(
            lambda ex: (
                (max_signal_length is None or ex['audio_length'] <= max_signal_length)
                and
                (min_signal_length is None or ex['audio_length'] >= min_signal_length)
            ),
            lazy=False
        )

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
        dataset = dataset.shuffle(reshuffle=True)

    stft = STFT(**stft)
    dataset = dataset.map(stft).map(event_encoder)

    def finalize(example):
        return {
            'dataset': example['dataset'],
            'example_id': example['example_id'],
            'stft': example['stft'].astype(np.float32),
            'seq_len': example['stft'].shape[1],
            'events': example['events'].astype(np.float32),
        }

    dataset = dataset.map(finalize)\
        .prefetch(num_workers, 10*batch_size, catch_filter_exception=True)

    if training and mixup_probs[0] < 1.:
        print('Mixup')
        dataset = MixUpDataset(
            dataset,
            sample_fn=SampleMixupComponents(mixup_probs),
            mixup_fn=SuperposeEvents(
                min_overlap=min_mixup_overlap,
                max_length=stft.samples_to_frames(
                    max_mixup_length*audio_reader.target_sample_rate
                )
            ),
            buffer_size=80*batch_size,
        )

    return dataset.batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key="seq_len",
        max_padding_rate=max_padding_rate, expiration=1000*batch_size,
        drop_incomplete=training, sort_key="seq_len", reverse_sort=True
    ).map(Collate())
