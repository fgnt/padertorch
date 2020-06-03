import json
from pathlib import Path

import numpy as np
import samplerate
import soundfile
import torch
from paderbox.transform.module_fbank import MelTransform as BaseMelTransform
from paderbox.transform.module_stft import STFT as BaseSTFT
from paderbox.utils.nested import nested_op
from padertorch.utils import to_list
from tqdm import tqdm


class AudioReader:
    def __init__(self, source_sample_rate=16000, target_sample_rate=16000):
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate

    def read_file(self, filepath, start_sample=0, stop_sample=None):
        if isinstance(filepath, (list, tuple)):
            start_sample = start_sample \
                if isinstance(start_sample, (list, tuple)) \
                else len(filepath) * [start_sample]
            stop_sample = stop_sample \
                if isinstance(stop_sample, (list, tuple)) \
                else len(filepath) * [stop_sample]
            return np.concatenate([
                self.read_file(filepath_, start_, stop_)
                for filepath_, start_, stop_ in zip(
                    filepath, start_sample, stop_sample
                )
            ], axis=-1)

        filepath = str(filepath)
        x, sr = soundfile.read(
            filepath, start=start_sample, stop=stop_sample, always_2d=True
        )
        if self.source_sample_rate is not None:
            assert sr == self.source_sample_rate, (self.source_sample_rate, sr)
        if self.target_sample_rate != sr:
            x = samplerate.resample(
                x, self.target_sample_rate / sr, "sinc_fastest"
            )
        return x.T

    def __call__(self, example):
        audio_path = example["audio_path"]
        start_samples = 0
        if "audio_start_samples" in example:
            start_samples = example["audio_start_samples"]
        stop_samples = None
        if "audio_stop_samples" in example:
            stop_samples = example["audio_stop_samples"]

        audio = self.read_file(audio_path, start_samples, stop_samples)
        example["audio_data"] = audio
        return example


class STFT(BaseSTFT):
    def __call__(self, example):
        if isinstance(example, dict):
            audio = example["audio_data"]
            x = super().__call__(audio)
            example["stft"] = np.stack([x.real, x.imag], axis=-1)
        else:
            example = super().__call__(example)
        return example


class MelTransform(BaseMelTransform):
    def __call__(self, example):
        if isinstance(example, dict):
            x = np.sum(example["stft"] ** 2, axis=-1)
            example["mel_transform"] = super().__call__(x)
        else:
            example = super().__call__(example)
        return example


class Normalizer:
    def __init__(
            self, key, center_axis=None, scale_axis=None, storage_dir=None,
            name=None
    ):
        self.key = key
        self.center_axis = None if center_axis is None else tuple(center_axis)
        self.scale_axis = None if scale_axis is None else tuple(scale_axis)
        self.storage_dir = None if storage_dir is None else Path(storage_dir)
        self.name = name
        self.moments = None

    def normalize(self, x):
        assert self.moments is not None
        mean, scale = self.moments
        x -= mean
        x /= (scale + 1e-18)
        return x

    def __call__(self, example):
        example[self.key] = self.normalize(example[self.key])
        return example

    def initialize_moments(self, dataset=None, verbose=False):
        """
        Loads or computes the global mean (center) and scale over a dataset.

        Args:
            dataset: lazy dataset providing example dicts
            verbose:

        Returns:

        """
        filepath = None if self.storage_dir is None \
            else self.storage_dir / f"{self.key}_moments_{self.name}.json" \
            if self.name else self.storage_dir / f"{self.key}_moments.json"
        if filepath is not None and Path(filepath).exists():
            with filepath.open() as fid:
                mean, scale = json.load(fid)
            if verbose:
                print(f'Restored moments from {filepath}')
        else:
            assert dataset is not None
            mean = 0.
            mean_count = 0
            energy = 0.
            energy_count = 0
            for example in tqdm(dataset, disable=not verbose):
                x = example[self.key]
                if self.center_axis is not None:
                    if not mean_count:
                        mean = np.sum(x, axis=self.center_axis, keepdims=True)
                    else:
                        mean += np.sum(x, axis=self.center_axis, keepdims=True)
                    mean_count += np.prod(
                        np.array(x.shape)[np.array(self.center_axis)]
                    )
                if self.scale_axis is not None:
                    if not energy_count:
                        energy = np.sum(x**2, axis=self.scale_axis, keepdims=True)
                    else:
                        energy += np.sum(x**2, axis=self.scale_axis, keepdims=True)
                    energy_count += np.prod(
                        np.array(x.shape)[np.array(self.scale_axis)]
                    )
            if self.center_axis is not None:
                mean /= mean_count
            if self.scale_axis is not None:
                energy /= energy_count
                scale = np.sqrt(np.mean(
                    energy - mean ** 2, axis=self.scale_axis, keepdims=True
                ))
            else:
                scale = np.array(1.)

            if filepath is not None:
                with filepath.open('w') as fid:
                    json.dump(
                        (mean.tolist(), scale.tolist()), fid,
                        sort_keys=True, indent=4
                    )
                if verbose:
                    print(f'Saved moments to {filepath}')
        self.moments = np.array(mean), np.array(scale)


class LabelEncoder:
    def __init__(self, label_key, storage_dir=None, to_array=False):
        self.label_key = label_key
        self.storage_dir = None if storage_dir is None else Path(storage_dir)
        self.to_array = to_array

        self.label_mapping = None
        self.inverse_label_mapping = None

    def __call__(self, example):
        def encode(labels):
            if isinstance(labels, (list, tuple)):
                return [self.label_mapping[label] for label in labels]
            return self.label_mapping[labels]
        y = encode(example[self.label_key])
        if self.to_array:
            example[self.label_key] = np.array(y)
        else:
            example[self.label_key] = y
        return example

    def initialize_labels(
            self, labels=None, dataset=None, dataset_name=None, verbose=False
    ):
        filename = f"{self.label_key}.json" if dataset_name is None \
            else f"{self.label_key}_{dataset_name}.json"
        filepath = None if self.storage_dir is None \
            else (self.storage_dir / filename).expanduser().absolute()

        if filepath and Path(filepath).exists():
            with filepath.open() as fid:
                labels_ = json.load(fid)
            if verbose:
                print(f'Restored labels from {filepath}')
            if labels is not None:
                assert labels_ == labels
            labels = labels_
        else:
            if labels is None:
                labels = set()
                for example in dataset:
                    labels.update(to_list(example[self.label_key]))
                labels = sorted(labels)
            if filepath:
                with filepath.open('w') as fid:
                    json.dump(labels, fid, indent=4)
                if verbose:
                    print(f'Saved labels to {filepath}')

        self.label_mapping = {
            label: i for i, label in enumerate(labels)
        }
        self.inverse_label_mapping = {
            i: label for label, i in self.label_mapping.items()
        }


class MultiHotLabelEncoder(LabelEncoder):
    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        nhot_encoding = np.zeros(len(self.label_mapping)).astype(np.float32)
        if len(labels) > 0:
            nhot_encoding[labels] = 1
        example[self.label_key] = nhot_encoding
        return example


class Collate:
    """
    >>> batch = [{'a': np.ones((5,2)), 'b': '0'}, {'a': np.ones((3,2)), 'b': '1'}]
    >>> Collate(to_tensor=True)(batch)
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
             [0., 0.]]], dtype=torch.float64), 'b': ['0', '1']}

    >>> Collate(cut_end=True, to_tensor=True)(batch)
    {'a': tensor([[[1., 1.],
             [1., 1.],
             [1., 1.]],
    <BLANKLINE>
            [[1., 1.],
             [1., 1.],
             [1., 1.]]], dtype=torch.float64), 'b': ['0', '1']}
    """
    def __init__(self, stack_arrays=True, cut_end=False, to_tensor=False):
        self.stack_arrays = stack_arrays
        self.cut_end = cut_end
        self.to_tensor = to_tensor

    def __call__(self, example):
        example = nested_op(self.collate, *example, sequence_type=())
        return example

    def collate(self, *batch):
        batch = list(batch)
        if self.stack_arrays and isinstance(batch[0], np.ndarray):
            shapes = [array.shape for array in batch]
            if self.cut_end:
                target_shape = np.min(shapes, axis=0)
            else:
                target_shape = np.max(shapes, axis=0)
            for i, array in enumerate(batch):
                diff = target_shape - array.shape
                assert np.argwhere(diff != 0).size <= 1, (
                    'arrays are only allowed to differ in one dim',
                    array.shape, target_shape,
                )
                if np.any(diff > 0):
                    pad = [(0, n) for n in diff]
                    batch[i] = np.pad(array, pad_width=pad, mode='constant')
                elif np.any(diff < 0):
                    sliceing = [slice(None) if n >= 0 else slice(n) for n in diff]
                    batch[i] = array[tuple(sliceing)]
            batch = np.array(batch).astype(batch[0].dtype)
            if self.to_tensor:
                batch = torch.from_numpy(batch)
        return batch


def fragment_signal(
        *signals, axis, step, max_length, min_length=1, random_start=False
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
    >>> pprint(fragment_signal(signals, axis=1, step=[4, 2], max_length=[4, 2]))
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
    >>> pprint(fragment_signal(\
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
        fragments = [
            x[get_slice(idx, idx + max_len)]
            for idx in np.arange(
                start_idx, x.shape[ax] - min_len + 1, step[i]
            )
        ]
        fragmented_signals.append(fragments)
    if len(signals) == 1:
        return signals[0]
    assert len(set([len(sig) for sig in fragmented_signals])) == 1, (
        [sig.shape for sig in signals], [len(sig) for sig in fragmented_signals]
    )
    return (*fragmented_signals, )
