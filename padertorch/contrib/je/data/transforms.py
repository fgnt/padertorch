import json
from pathlib import Path

import numpy as np
import samplerate
import soundfile
import torch
from paderbox.transform.module_fbank import MelTransform
from paderbox.transform.module_stft import STFT
from paderbox.utils.nested import nested_op
from paderbox.utils.numpy_utils import segment_axis_v2
from padertorch.utils import to_list
from tqdm import tqdm


class Transform:
    def __init__(
            self,
            target_sample_rate,
            frame_step,
            frame_length,
            fft_length,
            n_mels,
            input_sample_rate=None,
            fading=False,
            pad=False,
            fmin=50,
            label_key=None,
            storage_dir=None
    ):
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.stft = STFT(
            frame_step=frame_step, frame_length=frame_length,
            fft_length=fft_length,
            fading=fading, pad=pad, always3d=True
        )
        self.mel_transform = MelTransform(
            sample_rate=self.target_sample_rate, fft_length=fft_length,
            n_mels=n_mels, fmin=fmin, log=True
        )
        self.moments = None

        self.label_key = label_key
        self.label_mapping = None
        self.inverse_label_mapping = None

        self.storage_dir = storage_dir

    def __call__(self, example, training=False):
        example = self.read_audio(example)
        example = self.extract_features(example, training=training)
        example = self.normalize(example)
        example = self.encode_labels(example)
        return self.finalize(example, training)

    def read_audio(self, example):
        # time.sleep(0.01)

        def read_from_file(filepath, start=0, stop=None):
            x, sr = soundfile.read(
                filepath, start=start, stop=stop, always_2d=True
            )
            if self.input_sample_rate is not None:
                assert sr == self.input_sample_rate
            if self.target_sample_rate != sr:
                x = samplerate.resample(
                    x, self.target_sample_rate / sr, "sinc_fastest"
                )
            return x.T

        audio_path = example["audio_path"]
        start_sample = 0
        if "audio_start_samples" in example:
            start_sample = example["audio_start_samples"]
        stop_sample = None
        if "audio_stop_samples" in example:
            stop_sample = example["audio_stop_samples"]

        audio = nested_op(
            read_from_file, audio_path, start_sample, stop_sample,
            broadcast=True
        )
        audio = nested_op(
            lambda x: np.concatenate(x, axis=-1)
            if isinstance(x, (list, tuple)) else x,
            audio, sequence_type=()
        )
        example["audio_data"] = audio
        return example

    def extract_features(self, example, training=False):
        audio = example["audio_data"]
        if isinstance(audio, dict):
            audio = audio['observation']
        stft = self.stft(audio)
        spec = stft.real**2 + stft.imag**2
        example["features"] = self.mel_transform(spec)
        example["seq_len"] = example["features"].shape[1]
        return example

    def normalize(self, example):
        assert self.moments is not None
        feature_key = 'features'
        mean, scale = self.moments
        example[feature_key] -= mean
        example[feature_key] /= (scale + 1e-18)
        return example

    def initialize_norm(self, dataset=None, max_workers=0):
        filename = "moments.json"
        filepath = None if self.storage_dir is None \
            else (Path(self.storage_dir) / filename).expanduser().absolute()
        if dataset is not None:
            dataset = dataset.map(self.read_audio).map(self.extract_features)
            if max_workers > 0:
                dataset = dataset.prefetch(max_workers, 2 * max_workers)
        self.moments = read_moments(
            dataset, "features", center_axis=1, scale_axis=(1, 2),
            filepath=filepath, verbose=True
        )

    def encode_labels(self, example):
        if self.label_key is None:
            return example

        def encode(labels):
            if isinstance(labels, (list, tuple)):
                return [self.label_mapping[label] for label in labels]
            return self.label_mapping[labels]
        example[self.label_key] = np.array(encode(example[self.label_key]))
        return example

    def initialize_labels(self, dataset=None):
        if self.label_key is None:
            return

        filename = f"labels.json"
        filepath = None if self.storage_dir is None \
            else (Path(self.storage_dir) / filename).expanduser().absolute()
        labels = read_labels(dataset, self.label_key, filepath, verbose=True)
        self.label_mapping = {
            label: i for i, label in enumerate(labels)
        }
        self.inverse_label_mapping = {
            i: label for label, i in self.label_mapping.items()
        }

    def finalize(self, example, training=False):
        return {
            'example_id': example['example_id'],
            'features': np.moveaxis(example['features'], 1, 2).astype(np.float32),
            'seq_len': example['seq_len'],
            self.label_key: example[self.label_key]
        }


def read_moments(
        dataset=None, key=None, center_axis=None, scale_axis=None,
        filepath=None, verbose=False
):
    if filepath and Path(filepath).exists():
        with filepath.open() as fid:
            mean, scale = json.load(fid)
        if verbose:
            print(f'Restored moments from {filepath}')
    else:
        assert dataset is not None
        mean = 0
        mean_count = 0
        energy = 0
        energy_count = 0
        for example in tqdm(dataset, disable=not verbose):
            x = example[key]
            if center_axis is not None:
                if not mean_count:
                    mean = np.sum(x, axis=center_axis, keepdims=True)
                else:
                    mean += np.sum(x, axis=center_axis, keepdims=True)
                mean_count += np.prod(
                    np.array(x.shape)[np.array(center_axis)]
                )
            if scale_axis is not None:
                if not energy_count:
                    energy = np.sum(x**2, axis=scale_axis, keepdims=True)
                else:
                    energy += np.sum(x**2, axis=scale_axis, keepdims=True)
                energy_count += np.prod(
                    np.array(x.shape)[np.array(scale_axis)]
                )
        if center_axis is not None:
            mean /= mean_count
        if scale_axis is not None:
            energy /= energy_count
            scale = np.sqrt(np.mean(
                energy - mean ** 2, axis=scale_axis, keepdims=True
            ))
        else:
            scale = np.array(1.)

        if filepath:
            with filepath.open('w') as fid:
                json.dump(
                    (mean.tolist(), scale.tolist()), fid,
                    sort_keys=True, indent=4
                )
            if verbose:
                print(f'Saved moments to {filepath}')
    return np.array(mean), np.array(scale)


def read_labels(dataset=None, key=None, filepath=None, verbose=False):
    if filepath and Path(filepath).exists():
        with filepath.open() as fid:
           labels = json.load(fid)
        if verbose:
            print(f'Restored labels from {filepath}')
    else:
        labels = set()
        for example in dataset:
            labels.update(to_list(example[key]))
        labels = sorted(labels)
        if filepath:
            with filepath.open('w') as fid:
                json.dump(
                    labels, fid,
                    sort_keys=True, indent=4
                )
            if verbose:
                print(f'Saved labels to {filepath}')
    return labels


def segment_axis(
        signals, axis, segment_step, segment_length=None, *,
        random_start=False, pad=False
):
    """

    Args:
        signals:
        axis:
        segment_step:
        segment_length:
        random_start:
        pad:

    Returns:

    >>> signals = [np.arange(10).reshape((2, 5)), np.array([[1,2]])]
    >>> from pprint import pprint
    >>> pprint(segment_axis(signals, axis=1, segment_step=[2, 1]))
    [array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[5, 6],
            [7, 8]]]),
     array([[[1],
            [2]]])]
    """
    axis = to_list(axis, len(signals))
    segment_step = to_list(segment_step, len(signals))
    segment_length = to_list(segment_length, len(signals))
    segment_length = [
        segment_step[i] if seg_len is None else seg_len
        for i, seg_len in enumerate(segment_length)
    ]

    # get random start
    if random_start:
        start = np.random.rand()

        # find max start such that at least one segment is obtained
        max_start = 1.
        for i in range(len(signals)):
            # get nested structure and cast to dict
            max_start = min(
                max_start,
                (signals[i].shape[axis[i]] - segment_length[i]) / segment_step[i]
            )
        start *= max_start

        # adjust start to match an integer index for all keys
        for step in segment_step:
            start = int(start*step) / step
    else:
        start = 0.

    for i in range(len(signals)):
        x = signals[i]
        ax = axis[i]
        step = segment_step[i]
        length = segment_length[i]
        start_idx = int(start * step)
        if start_idx > 0:
            slc = [slice(None)] * len(x.shape)
            slc[ax] = slice(int(start_idx), x.shape[ax])
            x = x[tuple(slc)]
        elif start_idx < 0 and pad:
            # pad front if start_idx < 0 (this means input length is
            # shorter than segment length and training is True).
            pad_width = x.ndim * [(0, 0)]
            pad_width[ax] = (-start_idx, 0)
            x = np.pad(x, pad_width=pad_width, mode='constant')
        assert ax < x.ndim, (ax, x.ndim)

        signals[i] = segment_axis_v2(
            x, length, step, axis=ax, end='pad' if pad else 'cut'
        )
        assert x.shape[i]
    return signals


def fragment_parallel_signals(
        signals, axis, step, max_length, min_length=1, *,
        random_start=False
):
    """

    Args:
        signals:
        axis:
        step:
        max_length:
        min_length:
        random_start:

    Returns:

    >>> signals = [np.arange(20).reshape((2, 10)), np.arange(10).reshape((2, 5))]
    >>> from pprint import pprint
    >>> pprint(fragment_parallel_signals(signals, axis=1, step=[4, 2], max_length=[4, 2]))
    [[array([[ 0,  1,  2,  3],
           [10, 11, 12, 13]]),
      array([[ 4,  5,  6,  7],
           [14, 15, 16, 17]]),
      array([[ 8,  9],
           [18, 19]])],
     [array([[0, 1],
           [5, 6]]),
      array([[2, 3],
           [7, 8]]),
      array([[4],
           [9]])]]
    >>> pprint(fragment_parallel_signals(\
        signals, axis=1, step=[4, 2], max_length=[4, 2], min_length=[4, 2]\
    ))
    [[array([[ 0,  1,  2,  3],
           [10, 11, 12, 13]]),
      array([[ 4,  5,  6,  7],
           [14, 15, 16, 17]])],
     [array([[0, 1],
           [5, 6]]), array([[2, 3],
           [7, 8]])]]
    """
    axis = to_list(axis, len(signals))
    step = to_list(step, len(signals))
    max_length = to_list(max_length, len(signals))
    min_length = to_list(min_length, len(signals))

    # get random start
    if random_start:
        start = np.random.rand()

        # find max start such that at least one segment is obtained
        max_start = 1.
        for i in range(len(signals)):
            # get nested structure and cast to dict
            max_start = max(
                min(
                    max_start,
                    (signals[i].shape[axis[i]] - max_length[i]) / step[i]
                ),
                0.
            )
        start *= max_start

        # adjust start to match an integer index for all keys
        for i in range(len(signals)):
            start = int(start*step[i]) / step[i]
    else:
        start = 0.

    fragmented_signals = []
    for i in range(len(signals)):
        x = signals[i]
        ax = axis[i]
        assert ax < x.ndim, (ax, x.ndim)
        min_len = min_length[i]
        max_len = max_length[i]
        assert max_len >= min_len

        def get_slice(start, stop):
            slc = [slice(None)] * x.ndim
            slc[ax] = slice(int(start), int(stop))
            return tuple(slc)

        start_idx = round(start * step[i])
        assert abs(start_idx - start * step[i]) < 1e-6, (start_idx, start*step[i])
        fragments = [x[get_slice(0, start_idx)]] if start_idx >= min_len \
            else []
        fragments += [
            x[get_slice(idx, idx+max_len)]
            for idx in np.arange(
                start_idx, x.shape[ax] - min_len + 1, step[i]
            )
        ]
        fragmented_signals.append(fragments)
    assert len(set([len(sig) for sig in fragmented_signals])) == 1
    return fragmented_signals


class Collate:
    """

    >>> batch = [{'a': np.ones((5,2)), 'b': '0'}, {'a': np.ones((3,2)), 'b': '1'}]
    >>> Collate()(batch)
    {'a': tensor([[[1., 1.],
             [1., 1.],
             [1., 1.],
             [1., 1.],
             [1., 1.]],
    <BLANKLINE>
            [[1., 1.],
             [1., 1.],
             [1., 1.],
             [0., 0.],
             [0., 0.]]]), 'b': ['0', '1']}
    """
    def __init__(self, to_tensor=True):
        self.to_tensor = to_tensor

    def __call__(self, example, training=False):
        example = nested_op(self.collate, *example, sequence_type=())
        return example

    def collate(self, *batch):
        batch = list(batch)
        if isinstance(batch[0], np.ndarray):
            max_len = np.zeros_like(batch[0].shape)
            for array in batch:
                max_len = np.maximum(max_len, array.shape)
            for i, array in enumerate(batch):
                pad = max_len - array.shape
                if np.any(pad):
                    assert np.sum(pad) == np.max(pad), (
                        'arrays are only allowed to differ in one dim',
                    )
                    pad = [(0, n) for n in pad]
                    batch[i] = np.pad(array, pad_width=pad, mode='constant')
            batch = np.array(batch).astype(batch[0].dtype)
            if self.to_tensor:
                batch = torch.from_numpy(batch)
        return batch


class SlidingNormalize:
    def __init__(
            self, slide_axis, window_length, center_axis=None, scale_axis=None
    ):
        assert window_length % 2 == 1
        self.slide_axis = slide_axis
        self.window_length = window_length
        self.center_axis = None if center_axis is None \
            else tuple(to_list(center_axis))
        self.scale_axis = None if scale_axis is None \
            else tuple(to_list(scale_axis))

    def __call__(self, x):
        pad_width = x.ndim * [(0, 0)]
        pad_width[self.slide_axis] = (
            self.window_length//2, self.window_length//2
        )
        x_ = np.pad(x, pad_width=pad_width, mode='symmetric')
        x_ = segment_axis_v2(
            x_, self.window_length, 1, self.slide_axis, end='cut'
        )
        x_ = np.moveaxis(x_, self.slide_axis, -1)
        if self.center_axis is not None:
            m = np.mean(x_, axis=self.center_axis, keepdims=True)
            m = np.moveaxis(m.squeeze(self.slide_axis), -1, self.slide_axis)
            x -= m
        if self.scale_axis is not None:
            s = np.sqrt(
                np.mean(x_ ** 2, axis=self.center_axis, keepdims=True)
            ) + 1e-18
            s = np.moveaxis(
                s.squeeze(self.slide_axis), -1, self.slide_axis
            ).astype(x.dtype)
            x /= s
        return x


class LocalQuantileNormalize:
    def __init__(
            self, axis, lower_quantile=0.02, upper_quantile=0.98
    ):
        self.axis = axis
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def __call__(self, x):
        x -= np.quantile(x, self.lower_quantile, axis=self.axis, keepdims=True)
        x /= (
            np.quantile(x, self.upper_quantile, axis=self.axis, keepdims=True)
            + 1e-18
        )
        x -= 0.5
        return x
