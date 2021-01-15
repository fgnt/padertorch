import numpy as np
from padertorch.contrib.je.data.transforms import AudioReader, STFT, Collate
from padertorch.data.segment import Segmenter


def prepare_dataset(
        dataset, audio_reader, stft, max_length_in_sec=1., batch_size=3,
        is_train_set=False, shuffle=False
):

    def prepare_example(example):
        example['audio_path'] = example['audio_path']['observation']
        return example

    dataset = dataset.map(prepare_example)

    audio_reader = AudioReader(**audio_reader)
    dataset = dataset.map(audio_reader)

    anchor = 'random' if is_train_set else 'centered_cutout'
    if max_length_in_sec is None:
        dataset = dataset.map(lambda ex: [ex])
    else:
        segmenter = Segmenter(
            length=int(max_length_in_sec*audio_reader.target_sample_rate),
            include_keys=('audio_data',),  mode='max', anchor=anchor
        )
        dataset = dataset.map(segmenter)

    stft = STFT(**stft)
    dataset = dataset.batch_map(stft)

    def finalize(example):
        return {
            'example_id': example['example_id'],
            'audio_data': example['audio_data'].astype(np.float32),
            'stft': example['stft'].astype(np.float32),
            'seq_len': example['stft'].shape[1],
        }
    dataset = dataset.batch_map(finalize)

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
