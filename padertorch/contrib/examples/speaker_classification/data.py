import numpy as np

from padercontrib.database.librispeech import LibriSpeech

from padertorch.contrib.je.data.transforms import (
    LabelEncoder, AudioReader, STFT, MelTransform, Collate
)


def get_datasets(storage_dir, batch_size=16):
    db = LibriSpeech()
    train_clean_100 = db.get_dataset('train_clean_100')

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        example['speaker_id'] = example['speaker_id'].split('-')[0]
        return example

    train_clean_100 = train_clean_100.map(prepare_example)

    speaker_encoder = LabelEncoder(
        label_key='speaker_id', storage_dir=storage_dir, to_array=True
    )
    speaker_encoder.initialize_labels(dataset=train_clean_100, verbose=True)
    train_clean_100 = train_clean_100.map(speaker_encoder)

    train_set, validate_set, test_set = train_test_split(train_clean_100)
    training_data = prepare_dataset(train_set, batch_size, training=True)
    validation_data = prepare_dataset(validate_set, batch_size, training=False)
    test_data = prepare_dataset(test_set, batch_size, training=False)
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


def prepare_dataset(dataset, batch_size=16, training=False):
    audio_reader = AudioReader(
        source_sample_rate=16000, target_sample_rate=16000
    )
    dataset = dataset.map(audio_reader)
    stft = STFT(
        shift=160, window_length=400, size=512, fading=None, pad=False
    )
    dataset = dataset.map(stft)
    mel_transform = MelTransform(
        sample_rate=16000, fft_length=512, n_mels=64, fmin=50
    )
    dataset = dataset.map(mel_transform)

    def finalize(example):
        _example = {
            'example_id': example['example_id'],
            'features': np.moveaxis(example['mel_transform'], 1, 2).astype(np.float32),
            'seq_len': example['mel_transform'].shape[-2],
            'speaker_id': example['speaker_id'].astype(np.int)
        }
        if not training:
            _example['audio_path'] = example['audio_path']
        return _example

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
