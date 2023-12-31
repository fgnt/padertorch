import json
from math import ceil
from pathlib import Path
from typing import Callable
from functools import partial

import dataclasses
import numpy as np
import samplerate
import soundfile
from paderbox.transform.module_fbank import MelTransform as BaseMelTransform
from paderbox.transform.module_stft import STFT as BaseSTFT
from paderbox.utils.nested import nested_op
from padertorch.utils import to_list
from tqdm import tqdm
from collections import defaultdict
from paderbox.transform.module_filter import preemphasis_with_offset_compensation


@dataclasses.dataclass
class AudioReader:
    """
    >>> audio_reader = AudioReader(roll_prob=1., alignment_keys=['labels'])
    >>> example = {'audio_data': np.arange(16000.)[None], 'labels': ['a', 'b', 'c'], 'labels_start_samples': [0, 4000, 12000], 'labels_stop_samples': [4000, 10000, 16000]}
    """
    source_sample_rate: int = 16000
    target_sample_rate: int = 16000
    concat_axis: int = None
    average_channels: bool = False
    normalization_type: str = None  # max, power
    normalization_domain: str = None  # instance, dataset, global
    channelwise_norm: bool = False
    eps: float = 1e-3
    storage_dir: str = None
    preemphasis_factor: float = 0.
    alignment_keys: list = None
    cutoff_length: int = None

    def __post_init__(self):
        self.norm = None

    def _load_source(self, filepath, start_sample=0, stop_sample=None):
        if isinstance(filepath, (list, tuple)):
            assert self.concat_axis is not None
            if start_sample == 0:
                start_sample = len(filepath) * [start_sample]
            if stop_sample is None:
                stop_sample = len(filepath) * [stop_sample]
            assert isinstance(start_sample, (list, tuple))
            assert isinstance(stop_sample, (list, tuple))
            audio_sr = [
                self._load_source(filepath_, start_, stop_)
                for filepath_, start_, stop_ in zip(
                    filepath, start_sample, stop_sample
                )
            ]
            audio, sr = list(zip(*audio_sr))
            assert len(set(sr)) == 1, sr
            return np.concatenate(audio, axis=self.concat_axis), sr[0]

        filepath = str(filepath)
        x, sr = soundfile.read(
            filepath, start=start_sample, stop=stop_sample, always_2d=True
        )
        if self.source_sample_rate is not None:
            assert sr == self.source_sample_rate, (self.source_sample_rate, sr)
        return x.T, sr

    def load(self, filepath, start_sample=0, stop_sample=None):
        x, sr = self._load_source(filepath, start_sample, stop_sample)
        # print(start_sample, stop_sample, len(x[0]))
        if self.target_sample_rate != sr:
            x = samplerate.resample(
                x.T, self.target_sample_rate / sr, "sinc_fastest"
            ).T
        if self.cutoff_length is not None:
            x = x[..., :int(self.cutoff_length*self.target_sample_rate)]
            # print(x.shape)
        return x

    def _prenormalize(self, audio):
        audio -= audio.mean(-1, keepdims=True)
        if self.average_channels:
            audio = audio.mean(0, keepdims=True)
        if self.preemphasis_factor > 0.:
            audio = preemphasis_with_offset_compensation(audio, self.preemphasis_factor)
        return audio

    def normalize(self, example):
        example['audio_data'] = self._prenormalize(example['audio_data'])
        if self.normalization_domain is None:
            assert self.normalization_type is None, self.normalization_type
            return example
        elif self.normalization_domain == 'global':
            assert isinstance(self.norm, dict), type(self.norm)
            norm = np.array(self.norm['global_norm'])
        elif self.normalization_domain == 'dataset':
            assert isinstance(self.norm, dict), type(self.norm)
            norm = np.array(self.norm[example['dataset']])
        elif self.normalization_domain == 'instance':
            norm = self._get_audio_norm(example['audio_data'])
        else:
            raise ValueError(f'Invalid normalization domain {self.normalization_domain}')
        example['audio_data'] /= norm + self.eps
        return example

    def _get_audio_norm(self, audio):
        if self.normalization_type is None:
            return 1
        elif self.normalization_type == "max":
            if self.channelwise_norm:
                return np.abs(audio).max(-1, keepdims=True)
            else:
                return np.abs(audio).max()
        elif self.normalization_type == "power":
            if self.channelwise_norm:
                return audio.std(-1, keepdims=True)
            else:
                return audio.std()
        else:
            raise ValueError(f'Invalid normalization {self.normalization_type}')

    def initialize_norm(self, dataset=None):
        if self.normalization_domain is None \
                or self.normalization_domain == 'instance' \
                or self.normalization_type is None:
            print('No audio norm initialization required')
            return
        filepath = None if self.storage_dir is None \
            else Path(self.storage_dir) / f"audio_norm.json"
        if filepath is not None and Path(filepath).exists():
            with filepath.open() as fid:
                self.norm = {
                    key: np.array(norm) for key, norm in json.load(fid).items()
                }
            print(f'Restored audio norm from {filepath}')
        else:
            print(f'Initialize audio norm')
            assert dataset is not None
            count = 0
            for example in tqdm(dataset):
                dataset_name = example['dataset']
                assert dataset_name != 'global_norm'
                audio_path = example["audio_path"]
                start_samples = example.get("audio_start_samples", 0)
                stop_samples = example.get("audio_stop_samples", None)
                audio = self.load(audio_path, start_samples, stop_samples)
                audio = self._prenormalize(audio)
                audio_norm = self._get_audio_norm(audio)
                n_samples = audio.shape[-1] if self.channelwise_norm else np.prod(audio.shape)
                if count == 0:
                    self.norm = defaultdict(lambda: np.zeros_like(audio_norm))
                    count = defaultdict(lambda: 0)
                for key in [dataset_name, 'global_norm']:
                    if self.normalization_type == "power":
                        self.norm[key] *= count[key] / (count[key] + n_samples)
                        self.norm[key] += n_samples / (count[key] + n_samples) * audio_norm**2
                    elif self.normalization_type == "max":
                        self.norm[key] = np.maximum(self.norm[key], audio_norm)
                    count[key] += n_samples
            if self.normalization_type == "power":
                self.norm = {key: np.sqrt(power) for key, power in self.norm.items()}
            elif self.normalization_type == "max":
                self.norm = {key: norm for key, norm in self.norm.items()}
            if filepath is not None:
                with filepath.open('w') as fid:
                    json.dump(
                        {key: norm.tolist() for key, norm in self.norm.items()},
                        fid, sort_keys=True, indent=4
                    )
                print(f'Saved audio norm to {filepath}')

    def add_start_stop_samples(self, example):
        if self.alignment_keys is not None:
            for ali_key in self.alignment_keys:
                if f'{ali_key}_start_times' in example or f'{ali_key}_stop_times' in example:
                    example[f'{ali_key}_start_samples'] = (np.asanyarray(example[f'{ali_key}_start_times'])*self.target_sample_rate).astype(int)
                    example[f'{ali_key}_stop_samples'] = (np.asanyarray(example[f'{ali_key}_stop_times'])*self.target_sample_rate).astype(int)
        return example

    def __call__(self, example):
        audio_path = example["audio_path"]
        start_samples = example.get("audio_start_samples", 0)
        stop_samples = example.get("audio_stop_samples", None)
        example['audio_data'] = self.load(audio_path, start_samples, stop_samples)
        example = self.normalize(example)
        self.add_start_stop_samples(example)
        return example


@dataclasses.dataclass
class STFT(BaseSTFT):
    """
    >>> stft = STFT(200, 1024, 800, alignment_keys=['labels'], pad=False, fading='half')
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
            example["stft"] = np.stack([x.real, x.imag], axis=-1).astype(np.float32)
            self.add_start_stop_frames(example)
        else:
            example = super().__call__(example)
        return example

    def add_start_stop_frames(self, example):
        if self.alignment_keys is not None:
            for ali_key in self.alignment_keys:
                if f'{ali_key}_start_samples' in example or f'{ali_key}_stop_samples' in example:
                    assert ali_key in example and f'{ali_key}_start_samples' in example and f'{ali_key}_stop_samples' in example, example.keys()
                    example[f'{ali_key}_start_frames'] = [
                        self.sample_index_to_frame_index(int(n)+self.shift//2)
                        for n in example[f'{ali_key}_start_samples']
                    ]
                    example[f'{ali_key}_stop_frames'] = [
                        self.sample_index_to_frame_index(int(n)+self.shift//2)
                        for n in example[f'{ali_key}_stop_samples']
                    ]


@dataclasses.dataclass
class TimeWarpedSTFT:
    """
    >>> stft = STFT(200, 1024, 800, alignment_keys=['labels'], pad=True, fading='full')
    >>> out = stft({'audio_data': np.random.rand(80000)[None], 'labels': ['a','b','c'], 'labels_start_samples': [100, 12000, 24000], 'labels_stop_samples': [40000, 60000, 80000]})
    >>> out['stft'].shape
    >>> out['labels']
    >>> out['labels_start_samples']
    >>> out['labels_stop_samples']
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    >>> time_warped_stft = TimeWarpedSTFT(stft, lambda: (np.random.choice([.5,])), lambda: (np.random.choice([0.1,])))
    >>> out = time_warped_stft({'audio_data': np.random.rand(80000)[None], 'labels': ['a','b','c'], 'labels_start_samples': [100, 12000, 24000], 'labels_stop_samples': [40000, 60000, 80000]})
    >>> out['stft'].shape
    >>> out['labels']
    >>> out['labels_start_samples']
    >>> out['labels_stop_samples']
    >>> out['labels_start_frames']
    >>> out['labels_stop_frames']
    """
    base_stft: STFT
    anchor_sampling_fn: Callable
    anchor_shift_sampling_fn: Callable

    def __call__(self, example):
        assert callable(self.anchor_sampling_fn), type(self.anchor_sampling_fn)
        assert callable(self.anchor_shift_sampling_fn), type(self.anchor_shift_sampling_fn)
        anchor = self.anchor_sampling_fn()
        shift = self.anchor_shift_sampling_fn()
        warp_factor = (anchor + shift) / anchor
        overlap = self.base_stft.window_length - self.base_stft.shift
        audio = self.pad_audio(example["audio_data"])
        num_samples = audio.shape[-1]
        shifts = [
            round(self.base_stft.shift / warp_factor),
            round(self.base_stft.shift * (1 - anchor)/(1 - anchor*warp_factor))
        ]
        warp_factor = self.base_stft.shift / shifts[0]
        boundary_sample = (num_samples - overlap) * anchor
        boundary_sample = round(boundary_sample/shifts[0]) * shifts[0] + overlap
        onsets = [0, boundary_sample - overlap]
        seg_lens = [boundary_sample, num_samples - boundary_sample + overlap]

        x = []
        for i, (onset, seg_len, shift) in enumerate(zip(onsets, seg_lens, shifts)):
            offset = onset + seg_len
            stft = STFT(
                shift=shift,
                size=self.base_stft.size,
                window_length=self.base_stft.window_length,
                window=self.base_stft.window,
                symmetric_window=self.base_stft.symmetric_window,
                pad=(i == 1) and self.base_stft.pad,
                fading=None,
                alignment_keys=self.base_stft.alignment_keys,
            )
            x.append(stft(audio[..., onset:offset]))
        x = np.concatenate(x, axis=1)
        example["stft"] = np.stack([x.real, x.imag], axis=-1).astype(np.float32)
        num_frames = example["stft"].shape[1]
        # print(num_frames, boundary_frame)
        if self.base_stft.alignment_keys is not None:
            self.base_stft.add_start_stop_frames(example)
            boundary_frame = self.base_stft.sample_index_to_frame_index(boundary_sample)
            for ali_key in self.base_stft.alignment_keys:
                if f'{ali_key}_start_frames' in example:
                    example[f'{ali_key}_start_frames'] = [
                        round(start_frame*warp_factor) if start_frame < boundary_frame
                        else round(
                            boundary_frame*warp_factor
                            + (start_frame - boundary_frame)
                            * (num_frames - boundary_frame*warp_factor)
                            / (num_frames - boundary_frame)
                        )
                        for start_frame in example[f'{ali_key}_start_frames']
                    ]
                if f'{ali_key}_stop_frames' in example:
                    example[f'{ali_key}_stop_frames'] = [
                        round(stop_frame*warp_factor) if stop_frame < boundary_frame
                        else round(
                            boundary_frame*warp_factor
                            + (stop_frame - boundary_frame)
                            * (num_frames - boundary_frame*warp_factor)
                            / (num_frames - boundary_frame)
                        )
                        for stop_frame in example[f'{ali_key}_stop_frames']
                    ]
        return example

    def pad_audio(self, audio):
        pad_widths = [0, 0]
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
        return audio


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

    def encode(self, labels):
        if isinstance(labels, (list, tuple)):
            return [self.label_mapping[label] for label in labels]
        return self.label_mapping[labels]

    def __call__(self, example):
        y = self.encode(example[self.label_key])
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
    to_array: bool = True

    def __call__(self, example):
        labels = np.array(super().__call__(example)[self.label_key], dtype=np.int)
        if labels.ndim == 0:
            labels = labels[None]
        assert labels.ndim == 1, labels.shape
        nhot_encoding = np.zeros(len(self.label_mapping), dtype=np.float32)
        if len(labels) > 0:
            nhot_encoding[labels] = 1
        if self.to_array:
            example[self.label_key] = nhot_encoding
        else:
            example[self.label_key] = nhot_encoding.tolist()
        return example


@dataclasses.dataclass
class AlignmentEncoder(LabelEncoder):
    to_array: bool = False

    def __call__(self, example):
        labels = super().__call__(example)[self.label_key]
        n_frames = example['stft'].shape[1]
        ali = np.zeros(n_frames, dtype=np.float32)
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
        assert f'{self.label_key}_start_frames' in example, (example['dataset'], example.keys())
        labels = super().__call__(example)[self.label_key]
        seq_len = example['stft'].shape[1]
        example[self.label_key] = self.encode_alignment(
            zip(
                example[f'{self.label_key}_start_frames'],
                example[f'{self.label_key}_stop_frames'],
                labels
            ),
            seq_len=seq_len,
        )
        return example

    def encode_alignment(self, onset_offset_label, seq_len):
        ali = np.zeros((seq_len, len(self.label_mapping)), dtype=np.float32)
        for onset, offset, label in onset_offset_label:
            ali[onset:offset, label] = 1
        return ali


@dataclasses.dataclass
class StackArrays:
    """
    >>> batch = [np.ones((2,7)), np.zeros((2,10))]
    >>> StackArrays()(batch)
    >>> StackArrays(axis=1)(batch)
    >>> StackArrays(axis=2)(batch)
    >>> StackArrays(cut_end=True)(batch)
    """

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
            axis = self.axis
            if axis < 0:
                axis = len(target_shape) + 1 + axis
            assert -(len(target_shape)+1) <= axis <= len(target_shape), axis
            stack_shape = [*target_shape[:axis].tolist(), len(batch), *target_shape[axis:].tolist()]
            stacked_arrays = np.zeros(stack_shape, dtype=batch[0].dtype)
            for i, array in enumerate(batch):
                diff = target_shape - array.shape
                assert np.argwhere(diff != 0).size <= 1, (
                    'arrays are only allowed to differ in one dim',
                    array.shape, target_shape,
                )
                shape = np.minimum(target_shape, array.shape)
                sliceing = tuple([slice(int(n)) for n in shape])
                array = array[sliceing]
                sliceing = tuple([*sliceing[:axis], i, *sliceing[axis:]])
                stacked_arrays[sliceing] = array
            return stacked_arrays
            #     if np.any(diff > 0):
            #         pad = [(0, n) for n in diff]
            #         batch[i] = np.pad(array, pad_width=pad, mode='constant')
            #     elif np.any(diff < 0):
            #         sliceing = [slice(None) if n >= 0 else slice(n) for n in diff]
            #         batch[i] = array[tuple(sliceing)]
            # batch = np.stack(batch, axis=self.axis).astype(batch[0].dtype)
        return batch


@dataclasses.dataclass
class ConcatenateArrays:
    axis: int = -1

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
    leaf_op: dict = "stack"

    def __call__(self, example):
        if isinstance(self.leaf_op, dict):
            example = nested_op(lambda *x: list(x), *example, sequence_type=())
            for key, leaf_op in self.leaf_op.items():
                if leaf_op is None:
                    continue
                leaf_op = self._get_leaf_op(leaf_op)
                example[key] = nested_op(leaf_op, example[key], sequence_type=())
        else:
            leaf_op = partial(self._collate, leaf_op=self._get_leaf_op(self.leaf_op))
            example = nested_op(leaf_op, *example, sequence_type=())
        return example

    def _get_leaf_op(self, leaf_op):
        if leaf_op == "stack":
            leaf_op = StackArrays()
        elif leaf_op == "concat" or leaf_op == "concatenate":
            leaf_op = ConcatenateArrays()
        return leaf_op

    def _collate(self, *batch, leaf_op):
        batch = list(batch)
        if leaf_op is not None:
            batch = leaf_op(batch)
        return batch
