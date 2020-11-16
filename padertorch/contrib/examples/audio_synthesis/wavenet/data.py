import numpy as np
from padercontrib.database.librispeech import LibriSpeech
from padertorch.contrib.je.data.transforms import AudioReader, STFT, \
    fragment_signal, Collate


def get_datasets(audio_reader, stft, max_length, batch_size, shuffle_test=False):
    db = LibriSpeech()
    train_clean_100 = db.get_dataset('train_clean_100')
    dev_clean = db.get_dataset('dev_clean')
    test_set = db.get_dataset('test_clean')

    training_data = prepare_dataset(
        train_clean_100, audio_reader=audio_reader, stft=stft,
        max_length=max_length, batch_size=batch_size, shuffle=True
    )
    validation_data = prepare_dataset(
        dev_clean, audio_reader=audio_reader, stft=stft,
        max_length=max_length, batch_size=batch_size, shuffle=shuffle_test
    )
    test_data = prepare_dataset(
        test_set, audio_reader=audio_reader, stft=stft,
        max_length=max_length, batch_size=batch_size, shuffle=shuffle_test
    )
    return training_data, validation_data, test_data


def prepare_dataset(
        dataset, audio_reader, stft, max_length=1., batch_size=3, shuffle=False
):

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        return example

    dataset = dataset.map(prepare_example)

    audio_reader = AudioReader(**audio_reader)
    dataset = dataset.map(audio_reader)

    stft = STFT(**stft)
    dataset = dataset.map(stft)

    def fragment(example):
        num_samples, audio, features = example['num_samples'], example['audio_data'], example['stft']
        audio_len = num_samples/audio_reader.target_sample_rate
        pad_width = stft.window_length - stft.shift
        assert pad_width > 0, pad_width
        audio = np.pad(
            audio, (audio.ndim-1)*[(0, 0)] + [(pad_width, stft.window_length - 1)],
            mode='constant'
        )
        n = 1 if max_length is None else int(np.ceil(audio_len / max_length))
        fragment_len = audio_len / n
        sample_fragment_step = int(audio_reader.target_sample_rate * fragment_len)
        stft_fragment_step = sample_fragment_step // stft.shift
        sample_fragment_step = stft_fragment_step * stft.shift
        stft_fragment_len = stft.samples_to_frames(sample_fragment_step)
        sample_fragment_len = sample_fragment_step + 2*pad_width
        fragments = []
        for audio, features in zip(*fragment_signal(
            audio, features, axis=1,
            step=[sample_fragment_step, stft_fragment_step],
            fragment_length=[sample_fragment_len, stft_fragment_len],
            onset_mode='random' if shuffle else 'center'
        )):
            fragments.append({
                'example_id': example['example_id'],
                'audio_data': audio[..., pad_width:-pad_width].astype(np.float32),
                'stft': features.astype(np.float32),
                'seq_len': features.shape[1],
            })
        return fragments

    dataset = dataset.map(fragment)

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)
    dataset = dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    ).unbatch()
    if shuffle:
        dataset = dataset.shuffle(
            reshuffle=True, buffer_size=10*batch_size
        )
    return dataset.batch_dynamic_time_series_bucket(
        batch_size=batch_size, len_key='seq_len', max_padding_rate=0.05,
        expiration=1000*batch_size, drop_incomplete=shuffle,
        sort_key='seq_len', reverse_sort=True
    ).map(Collate())
