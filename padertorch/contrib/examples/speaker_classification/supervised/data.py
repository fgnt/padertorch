from functools import partial

import numpy as np
import torch

from lazy_dataset.database import JsonDatabase

import paderbox as pb
import padertorch as pt
from padertorch.ops.sequence.pack_module import pad_sequence
from padertorch.contrib.je.data.transforms import LabelEncoder


def get_datasets(
    storage_dir, database_json, dataset, batch_size=16, return_indexable=False
):
    db = JsonDatabase(database_json)
    ds = db.get_dataset(dataset)

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['speaker_id'] = example['speaker_id'].split('-')[0]
        return example

    ds = ds.map(prepare_example)

    speaker_encoder = LabelEncoder(
        label_key='speaker_id', storage_dir=storage_dir, to_array=True
    )
    speaker_encoder.initialize_labels(dataset=ds, verbose=True)
    ds = ds.map(speaker_encoder)

    # LibriSpeech (the default database) does not share speakers across
    # different datasets, i.e., the datasets, e.g. clean_100 and dev_clean, have
    # a different set of non-overlapping speakers. To guarantee the same set of
    # speakers during training, validation and evaluation, we perform a split of
    # the train set, e.g., clean_100 or clean_360.
    train_set, validate_set, test_set = train_test_split(ds)

    training_data = prepare_dataset(train_set, batch_size, training=True)
    validation_data = prepare_dataset(validate_set, batch_size, training=False)
    test_data = prepare_dataset(
        test_set, batch_size, training=False, return_indexable=return_indexable
    )
    return training_data, validation_data, test_data


def train_test_split(dataset, dev_split=0.1,  test_split=0.1, seed=0):
    r = np.random.RandomState(seed)

    try:
        num_examples = len(dataset)
    except TypeError:
        raise RuntimeError('dataset must be indexable!')

    indices = np.arange(num_examples)

    dev_size = int(num_examples * dev_split)
    test_size = int(num_examples * test_split)

    test_candidates = r.choice(indices, size=test_size, replace=False)
    indices = np.delete(indices, test_candidates)
    dev_candidates = r.choice(indices, size=dev_size, replace=False)
    train_candidates = np.delete(indices, dev_candidates)

    return (
        dataset[train_candidates], dataset[dev_candidates],
        dataset[test_candidates]
    )


def _prepare_features(example, training=False):
    audio_data = pb.io.load_audio(
        example['audio_path'], expected_sample_rate=16000
    )
    stft_transform = pb.transform.STFT(
        shift=160, window_length=400, size=512, fading=None, pad=False
    )
    stft = stft_transform(audio_data)
    mel_transform = pb.transform.module_fbank.MelTransform(
        sample_rate=16000, stft_size=512, number_of_filters=64, lowest_frequency=50
    )
    mel_spec = mel_transform(np.abs(stft) ** 2)

    _example = {
        'example_id': example['example_id'],
        'features': torch.from_numpy(
            mel_spec.astype(np.float32)
        ),
        'seq_len': mel_spec.shape[-2],
        'speaker_id': example['speaker_id'].astype(np.int)
    }
    if not training:
        _example['audio_path'] = example['audio_path']
    return _example


def _collate_example(example):
    example = pt.data.utils.collate_fn(example)
    example['features'] = pad_sequence(example['features'], batch_first=True)\
        .transpose(-1, -2)
    example['speaker_id'] = np.stack(example['speaker_id'])
    return example


def prepare_dataset(
    dataset, batch_size=16, training=False, return_indexable=False
):
    dataset = dataset.map(partial(_prepare_features, training=training))

    if training:
        dataset = dataset.shuffle(reshuffle=True)
    if return_indexable:
        return dataset.batch(batch_size).map(_collate_example)
    dataset = dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    )
    return dataset.batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key='seq_len', max_padding_rate=0.1,
        expiration=1000*batch_size, drop_incomplete=training,
        sort_key='seq_len', reverse_sort=True
    ).map(_collate_example)
