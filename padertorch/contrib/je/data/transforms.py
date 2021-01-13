import json
from math import ceil
from pathlib import Path

import dataclasses
import numpy as np
import samplerate
import soundfile
from paderbox.transform.module_fbank import MelTransform as BaseMelTransform
from paderbox.transform.module_stft import STFT as BaseSTFT
from paderbox.utils.nested import nested_op
from padertorch.utils import to_list


@dataclasses.dataclass
class AudioReader:
    """
    >>> audio_reader = AudioReader(roll_prob=1., alignment_keys=['labels'])
    >>> example = {'audio_data': np.arange(16000.)[None], 'labels': ['a', 'b', 'c'], 'labels_start_samples': [0, 4000, 12000], 'labels_stop_samples': [4000, 10000, 16000]}
    >>> audio_reader.maybe_roll(example)
    """
    source_sample_rate: int = 16000
    target_sample_rate: int = 16000
    concat_axis: int = None
    alignment_keys: list = None
    roll_prob: float = 0.

    def _read_source(self, filepath, start_sample=0, stop_sample=None):
        if isinstance(filepath, (list, tuple)):
            if start_sample == 0:
                start_sample = len(filepath) * [start_sample]
            if stop_sample is None:
                stop_sample = len(filepath) * [stop_sample]
            assert isinstance(start_sample, (list, tuple))
            assert isinstance(stop_sample, (list, tuple))
            audio = [
                self._read_source(filepath_, start_, stop_)
                for filepath_, start_, stop_ in zip(
                    filepath, start_sample, stop_sample
                )
            ]
            assert self.concat_axis is not None
            audio = np.concatenate(audio, axis=self.concat_axis)
            return audio

        filepath = str(filepath)
        x, sr = soundfile.read(
            filepath, start=start_sample, stop=stop_sample, always_2d=True
        )
        if self.source_sample_rate is not None:
            assert sr == self.source_sample_rate, (self.source_sample_rate, sr)
        return x.T, sr

    def load(self, filepath, start_sample=0, stop_sample=None):
        x, sr = self._read_source(filepath, start_sample, stop_sample)
        if self.target_sample_rate != sr:
            x = samplerate.resample(
                x.T, self.target_sample_rate / sr, "sinc_fastest"
            ).T
        return x

    def add_start_stop_samples(self, example):
        if self.alignment_keys is not None:
            for ali_key in self.alignment_keys:
                if f'{ali_key}_start_times' in example or f'{ali_key}_stop_times' in example:
                    assert f'{ali_key}_start_times' in example and f'{ali_key}_stop_times' in example, example.keys()
                    example[f'{ali_key}_start_samples'] = [
                        int(self.target_sample_rate*t)
                        for t in example[f'{ali_key}_start_times']
                    ]
                    example[f'{ali_key}_stop_samples'] = [
                        int(self.target_sample_rate*t)
                        for t in example[f'{ali_key}_stop_times']
                    ]
        return example

    def maybe_roll(self, example):
        n_samples = example['audio_data'].shape[-1]
        if np.random.rand() < self.roll_prob:
            n_roll = int(np.random.rand()*n_samples)
            if n_roll != 0:
                fade_len = self.target_sample_rate//100
                assert example['audio_data'].shape[-1] > 2*fade_len
                example['audio_data'][..., :fade_len] *= 1/2-np.cos(np.pi*np.arange(fade_len)/fade_len)/2
                example['audio_data'][..., -fade_len:] *= 1/2+np.cos(np.pi*np.arange(fade_len)/fade_len)/2
                example['audio_data'] = np.roll(example['audio_data'], n_roll, axis=-1)
                if self.alignment_keys is not None:
                    for ali_key in self.alignment_keys:
                        labels = example.pop(ali_key)
                        start_samples = example.pop(f'{ali_key}_start_samples')
                        stop_samples = example.pop(f'{ali_key}_stop_samples')
                        example[ali_key] = []
                        example[f'{ali_key}_start_samples'] = []
                        example[f'{ali_key}_stop_samples'] = []
                        for label, n_start, n_stop in zip(labels, start_samples, stop_samples):
                            assert n_stop >= n_start, (n_start, n_stop)
                            n_start = (n_start + n_roll) % n_samples
                            n_stop = (n_stop + n_roll) % n_samples
                            if n_stop < n_start:
                                example[ali_key].append(label)
                                example[f'{ali_key}_start_samples'].append(n_start)
                                example[f'{ali_key}_stop_samples'].append(n_samples)
                                n_start = 0
                            example[ali_key].append(label)
                            example[f'{ali_key}_start_samples'].append(n_start)
                            example[f'{ali_key}_stop_samples'].append(n_stop)
                        sort_idx = np.argsort(example[f'{ali_key}_start_samples']).flatten().tolist()
                        example[ali_key] = [example[ali_key][i] for i in sort_idx]
                        example[f'{ali_key}_start_samples'] = [
                            example[f'{ali_key}_start_samples'][i]
                            for i in sort_idx
                        ]
                        example[f'{ali_key}_stop_samples'] = [
                            example[f'{ali_key}_stop_samples'][i]
                            for i in sort_idx
                        ]
        return example

    def __call__(self, example):
        audio_path = example["audio_path"]
        start_samples = 0
        if "audio_start_samples" in example:
            start_samples = example["audio_start_samples"]
        stop_samples = None
        if "audio_stop_samples" in example:
            stop_samples = example["audio_stop_samples"]

        example['audio_data'] = self.load(audio_path, start_samples, stop_samples)
        self.add_start_stop_samples(example)
        self.maybe_roll(example)
        return example


@dataclasses.dataclass
class STFT(BaseSTFT):
    """
    >>> stft = STFT(200, 801, alignment_keys=['labels'], pad=False, fading='half')
    >>> out = stft({'audio_data': np.random.rand(32000), 'labels': ['a','b','c'], 'labels_start_samples': [99, 12000, 24000], 'labels_stop_samples': [10000, 16000, 32000]})
    >>> out['stft'].shape
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    """
    alignment_keys: list = None

    def __call__(self, example):
        if isinstance(example, dict):
            audio = example["audio_data"]
            x = super().__call__(audio)
            example["stft"] = np.stack([x.real, x.imag], axis=-1)
            self.add_start_stop_frames(example)
        else:
            example = super().__call__(example)
        return example

    def add_start_stop_frames(self, example):
        if self.alignment_keys is not None:
            for ali_key in self.alignment_keys:
                if f'{ali_key}_start_samples' in example or f'{ali_key}_stop_samples' in example:
                    assert f'{ali_key}_start_samples' in example and f'{ali_key}_stop_samples' in example, example.keys()
                    example[f'{ali_key}_start_frames'] = [
                        self.sample_index_to_frame_index(int(n)+self.shift//2)
                        for n in example[f'{ali_key}_start_samples']
                    ]
                    example[f'{ali_key}_stop_frames'] = [
                        self.sample_index_to_frame_index(int(n)+self.shift//2)
                        for n in example[f'{ali_key}_stop_samples']
                    ]


@dataclasses.dataclass
class PiecewiseSTFT:
    """
    >>> stft = STFT(200, 800, alignment_keys=['labels'], pad=False, fading='full')
    >>> out = stft({'audio_data': np.random.rand(80000)[None], 'labels': ['a','b','c'], 'labels_start_samples': [100, 12000, 24000], 'labels_stop_samples': [10000, 16000, 32000]})
    >>> out['stft'].shape
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    >>> piecewise_stft = PiecewiseSTFT(stft, lambda: (np.random.choice([0.75, 1.25])), 8000)
    >>> out = piecewise_stft({'audio_data': np.random.rand(80000)[None], 'labels': ['a','b','c'], 'labels_start_samples': [100, 12000, 24000], 'labels_stop_samples': [10000, 16000, 32000]})
    >>> out['stft'].shape
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    >>> piecewise_stft = PiecewiseSTFT(stft, lambda: (np.random.choice([0.75, 1.25])), 8000, shuffle_prob=1.)
    >>> out = piecewise_stft({'audio_data': np.random.rand(80000)[None], 'labels': ['a','b','c'], 'labels_start_samples': [100, 12000, 24000], 'labels_stop_samples': [10000, 16000, 32000]})
    >>> out['stft'].shape
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    """
    base_stft: STFT
    stretch_factor_sampling_fn: callable
    segment_length: int = None
    warping_prob: float = 1.
    shuffle_prob: float = 0.

    def __call__(self, example):
        if np.random.rand() > self.warping_prob:
            return self.base_stft(example)

        audio = example["audio_data"]
        pad_widths = [0, 0]
        if self.base_stft.pad:
            pad_widths[-1] = self.base_stft.shift - 1
        if self.base_stft.fading == "full":
            pad_widths[0] += self.base_stft.window_length - self.base_stft.shift
            pad_widths[-1] += self.base_stft.window_length - self.base_stft.shift
        elif self.base_stft.fading == "half":
            pad_widths[0] += (self.base_stft.window_length - self.base_stft.shift)//2
            pad_widths[-1] += ceil((self.base_stft.window_length - self.base_stft.shift)/2)
        elif self.base_stft.fading is not None:
            raise ValueError(f'Invalid fading {self.base_stft.fading}.')
        if sum(pad_widths) > 0:
            audio = np.pad(audio, [[0, 0], pad_widths], mode='constant')
        num_samples = audio.shape[-1]
        segments = []
        cur_onset = 0
        while cur_onset <= (num_samples - self.base_stft.window_length):
            shift = int(
                self.base_stft.shift / self.stretch_factor_sampling_fn()
            )
            segment_len = num_samples if self.segment_length is None else int((self.segment_length // shift) * shift)
            assert segment_len > 0, segment_len
            if (cur_onset + segment_len) > (num_samples - self.base_stft.window_length):
                cur_offset = num_samples
            else:
                cur_offset = cur_onset + segment_len + (self.base_stft.window_length-shift)
            segments.append({
                'audio_data': audio[..., cur_onset:cur_offset],
            })
            if self.base_stft.alignment_keys is not None:
                for ali_key in self.base_stft.alignment_keys:
                    if f'{ali_key}_start_samples' in example or f'{ali_key}_stop_samples' in example:
                        assert f'{ali_key}_start_samples' in example and f'{ali_key}_stop_samples' in example, example.keys()
                        segments[-1].update({
                            ali_key: [],
                            f'{ali_key}_start_samples': [],
                            f'{ali_key}_stop_samples': [],
                        })
                        for label, start_sample, stop_sample in zip(
                            example[ali_key],
                            example[f'{ali_key}_start_samples'],
                            example[f'{ali_key}_stop_samples'],
                        ):
                            if (
                                stop_sample > (cur_onset + self.base_stft.window_length//2)
                                and start_sample < (cur_offset - self.base_stft.window_length//2)
                            ):
                                segments[-1][ali_key].append(label)
                                segments[-1][f'{ali_key}_start_samples'].append(max(start_sample+pad_widths[0]-cur_onset, 0))
                                segments[-1][f'{ali_key}_stop_samples'].append(min(stop_sample+pad_widths[0], cur_offset)-cur_onset)
            stft = STFT(
                shift=shift,
                size=self.base_stft.size,
                window_length=self.base_stft.window_length,
                window=self.base_stft.window,
                symmetric_window=self.base_stft.symmetric_window,
                pad=False,
                fading=None,
                alignment_keys=self.base_stft.alignment_keys,
            )
            segments[-1] = stft(segments[-1])
            cur_onset += segment_len
        if np.random.rand() < self.shuffle_prob:
            np.random.shuffle(segments)
        example['stft'] = np.concatenate([segment['stft'] for segment in segments], axis=1)
        n_frames = [segment['stft'].shape[1] for segment in segments]
        frame_onsets = np.cumsum(n_frames)-n_frames
        if self.base_stft.alignment_keys is not None:
            for ali_key in self.base_stft.alignment_keys:
                if f'{ali_key}_start_samples' in example or f'{ali_key}_stop_samples' in example:
                    example[f'{ali_key}_start_frames'] = []
                    example[f'{ali_key}_stop_frames'] = []
                    for segment, seg_onset in zip(segments, frame_onsets):
                        for start, stop in zip(
                            segment[f'{ali_key}_start_frames'],
                            segment[f'{ali_key}_stop_frames'],
                        ):
                            start += seg_onset
                            stop += seg_onset
                            if len(example[f'{ali_key}_stop_frames']) > 0 and example[f'{ali_key}_stop_frames'][-1] >= start:
                                example[f'{ali_key}_stop_frames'][-1] = stop
                            else:
                                example[f'{ali_key}_start_frames'].append(start)
                                example[f'{ali_key}_stop_frames'].append(stop)
        return example


class MelTransform(BaseMelTransform):
    def __call__(self, example):
        if isinstance(example, dict):
            x = np.sum(example["stft"] ** 2, axis=-1)
            example["mel_transform"] = super().__call__(x)
        else:
            example = super().__call__(example)
        return example


@dataclasses.dataclass
class LabelEncoder:
    label_key: str
    storage_dir: str = None
    to_array: bool = False

    def __post_init__(self):
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
            else (Path(self.storage_dir) / filename).expanduser().absolute()

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


@dataclasses.dataclass
class MultiHotEncoder(LabelEncoder):
    to_array: bool = False

    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        nhot_encoding = np.zeros(len(self.label_mapping)).astype(np.float32)
        if len(labels) > 0:
            nhot_encoding[labels] = 1
        example[self.label_key] = nhot_encoding
        return example


@dataclasses.dataclass
class AlignmentEncoder(LabelEncoder):
    to_array: bool = False

    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        n_frames = example['stft'].shape[1]
        ali = np.zeros(n_frames)
        assert f'{self.label_key}_start_frames' in example, (example['dataset'], example.keys())
        for label, onset, offset in zip(
                labels,
                example[f'{self.label_key}_start_frames'],
                example[f'{self.label_key}_stop_frames']
        ):
            ali[onset:offset] = label
        example[self.label_key] = ali
        return example


@dataclasses.dataclass
class MultiHotAlignmentEncoder(LabelEncoder):
    to_array: bool = False

    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        n_frames = example['stft'].shape[1]
        ali = np.zeros((n_frames, len(self.label_mapping)))
        assert f'{self.label_key}_start_frames' in example, (example['dataset'], example.keys())
        for label, onset, offset in zip(
                labels,
                example[f'{self.label_key}_start_frames'],
                example[f'{self.label_key}_stop_frames']
        ):
            ali[onset:offset, label] = 1
        example[self.label_key] = ali
        return example


@dataclasses.dataclass
class StackArrays:
    axis: int = 0
    cut_end: bool = False

    def __call__(self, example):
        if isinstance(example, dict):
            example = nested_op(self.stack, example, sequence_type=())
        elif isinstance(example, (list, tuple)):
            example = self.stack(example)
        return example

    def stack(self, batch):
        if isinstance(batch, list) and isinstance(batch[0], np.ndarray):
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
            batch = np.stack(batch, axis=self.axis).astype(batch[0].dtype)
        return batch


@dataclasses.dataclass
class ConcatenateArrays:
    axis: int

    def __call__(self, example):
        if isinstance(example, dict):
            example = nested_op(self.concatenate, example, sequence_type=())
        elif isinstance(example, (list, tuple)):
            example = self.concatenate(example)
        return example

    def concatenate(self, batch):
        if isinstance(batch, list) and isinstance(batch[0], np.ndarray):
            batch = np.concatenate(batch, axis=self.axis).astype(batch[0].dtype)
        return batch


@dataclasses.dataclass
class Collate:
    """
    >>> batch = [{'a': np.ones((5,2)), 'b': '0'}, {'a': np.ones((3,2)), 'b': '1'}]
    >>> Collate()(batch)
    {'a': array([[[1., 1.],
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

    >>> Collate(leaf_op=StackArrays(cut_end=True))(batch)
    {'a': array([[[1., 1.],
            [1., 1.],
            [1., 1.]],
    <BLANKLINE>
           [[1., 1.],
            [1., 1.],
            [1., 1.]]]), 'b': ['0', '1']}

    >>> Collate(leaf_op=ConcatenateArrays(axis=0))(batch)
    {'a': array([[1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.]]), 'b': ['0', '1']}
    """
    leaf_op: callable = StackArrays()

    def __call__(self, example):
        example = nested_op(self.collate, *example, sequence_type=())
        return example

    def collate(self, *batch):
        batch = list(batch)
        if self.leaf_op is not None:
            batch = self.leaf_op(batch)
        return batch
