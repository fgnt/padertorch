import abc
import json
from collections import OrderedDict
from pathlib import Path
from functools import partial

import numpy as np
import samplerate
import soundfile
from paderbox.transform.module_fbank import MelTransform as MelModule
from paderbox.transform.module_stft import STFT as STFTModule
from paderbox.transform.module_mfcc import delta
from paderbox.utils.nested import squeeze_nested, nested_op, nested_update, \
    flatten, deflatten
from padertorch.configurable import Configurable
from padertorch.contrib.je import Keys
from padertorch.utils import to_list
from tqdm import tqdm
from paderbox.utils.numpy_utils import segment_axis_v2
from copy import copy, deepcopy
import torch


class Transform(Configurable, abc.ABC):
    """
    Base class for callable transformations. Not intended to be instantiated.
    """
    @abc.abstractmethod
    def __call__(self, example, training=False):
        raise NotImplementedError

    def init_params(self, values=None, storage_dir=None, dataset=None):
        pass


class Compose:
    """
    Accepts an ordered iterable of Transform objects and performs a
    transformation composition by successively applying them in the given
    order. Inspired by a torchvision class with the same name.

    `Compose(operator.neg, abs)(data)` is equal to `abs(-data)`.

    See
    https://stackoverflow.com/questions/16739290/composing-functions-in-python
    for alternatives.
    """
    def __init__(self, *args):
        # Wouldn't "functions" instead of transforms not a better name
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        if len(args) == 1 and isinstance(args[0], dict):
            if not isinstance(args[0], OrderedDict):
                raise ValueError('OrderedDict required.')
            self._transforms = args[0]
        else:
            self._transforms = OrderedDict()
            for i, transform in enumerate(args):
                self._transforms[str(i)] = transform

    def __call__(self, example, training=False):
        for key, transform in self._transforms.items():
            example = transform(example, training=False)
        return example

    def __repr__(self):
        """
        >>> import operator
        >>> Compose(operator.neg, abs)
        Compose(<built-in function neg>, <built-in function abs>)
        >>> Compose(operator.neg, abs)(1)
        1
        """
        s = ', '.join([repr(t) for k, t in self._transforms.items()])
        return f'{self.__class__.__name__}({s})'


class ReadAudio(Transform):
    """
    Read audio from disk. Expects an key 'audio_path' in input dict
    and adds an entry 'audio_data' with the read audio data.

    """
    def __init__(
            self, input_sample_rate, target_sample_rate,
            converter_type="sinc_fastest"
    ):
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.converter_type = converter_type

    def read(self, audio_path, start=0, stop=None):
        x, sr = soundfile.read(
            audio_path, start=start, stop=stop, always_2d=True
        )
        assert sr == self.input_sample_rate
        if self.target_sample_rate != sr:
            x = samplerate.resample(
                x, self.target_sample_rate / sr, self.converter_type
            )
        return x.T  # (C, T)

    def __call__(self, example, training=False):
        example[Keys.AUDIO_DATA] = nested_op(
            self.read, example[Keys.AUDIO_PATH]
        )
        if (
            Keys.NUM_SAMPLES not in example
            or self.target_sample_rate != self.input_sample_rate
        ):
            example[Keys.NUM_SAMPLES] = squeeze_nested(nested_op(
                lambda x: x.shape[-1], example[Keys.AUDIO_DATA]
            ))
        for key in Keys.lable_keys():
            if f'{key}_start_times' in example and (
                    f'{key}_start_samples' not in example
                    or self.target_sample_rate != self.input_sample_rate
            ):
                example[f'{key}_start_samples'] = squeeze_nested(nested_op(
                    lambda x: int(x * self.target_sample_rate),
                    example[f'{key}_start_times']
                ))
            if f'{key}_stop_times' in example and (
                    f'{key}_stop_samples' not in example
                    or self.target_sample_rate != self.input_sample_rate
            ):
                example[f'{key}_stop_samples'] = squeeze_nested(nested_op(
                    lambda x: int(x * self.target_sample_rate),
                    example[f'{key}_stop_times']
                ))

        return example


class STFT(STFTModule, Transform):
    def __init__(
            self,
            frame_step: int,
            fft_length: int,
            frame_length: int = None,
            window: str = "blackman",
            symmetric_window: bool = False,
            fading: bool = True,
            pad: bool = True,
            keep_input = False
    ):
        """

        Args:
            frame_step:
            fft_length:
            frame_length:
            window:
            symmetric_window:
            fading:
            pad:
            keep_input:

        >>> stft = STFT(160, 512)
        >>> ex = dict(\
                audio_data=np.zeros(16000), \
                phones_start_samples=[8000, 12000], \
                phones_stop_samples=[12000, 16000])
        >>> ex = stft(ex)
        >>> print(ex["num_frames"])
        103
        >>> print(ex["phones_start_frames"])
        [52, 77]
        >>> print(ex["phones_stop_frames"])
        [78, 103]
        """
        super().__init__(
            frame_step=frame_step,
            fft_length=fft_length,
            frame_length=frame_length,
            window=window,
            symmetric_window=symmetric_window,
            fading=fading,
            pad=False,
            always3d=True
        )
        self.pad = pad
        self.keep_input = keep_input

    def prepare_audio(self, audio):
        if self.pad:
            tail = audio.shape[-1] % self.frame_step
            if tail > 0:
                pad_width = audio.ndim * [(0, 0)]
                pad_width[-1] = (0, int(self.frame_step - tail))
                audio = np.pad(audio, pad_width, mode='constant')
        else:
            noverlap = self.frame_length - self.frame_step
            tail = (audio.shape[-1] - noverlap) % self.frame_step
            if tail > 0:
                audio = audio[..., :-tail]
        return audio

    def __call__(self, example, training=False):
        example[Keys.AUDIO_DATA] = nested_op(
            self.prepare_audio, example[Keys.AUDIO_DATA]
        )
        example[Keys.STFT] = nested_op(
            super().__call__, example[Keys.AUDIO_DATA]
        )
        if not self.keep_input:
            example.pop(Keys.AUDIO_DATA)
        example[Keys.NUM_FRAMES] = squeeze_nested(nested_op(
            lambda x: x.shape[-2], example[Keys.STFT]
        ))
        for key in Keys.lable_keys():
            if f'{key}_start_samples' in example:
                example[f'{key}_start_frames'] = squeeze_nested(nested_op(
                    self.sampleid2frameid,
                    example[f'{key}_start_samples']
                ))
            if f'{key}_stop_samples' in example:
                example[f'{key}_stop_frames'] = squeeze_nested(nested_op(
                    lambda x: self.sampleid2frameid(x - 1) + 1,
                    example[f'{key}_stop_samples']
                ))
        return example

    def invert(self, example):
        raise NotImplementedError


class Spectrogram(Transform):
    def __init__(self, keep_input=False):
        self.keep_input = keep_input

    def __call__(self, example, training=False):
        example[Keys.SPECTROGRAM] = nested_op(
            lambda x: x.real**2 + x.imag**2, example[Keys.STFT]
        )
        if not self.keep_input:
            example.pop(Keys.STFT)
        return example


class MelTransform(MelModule, Transform):
    def __call__(self, example, training=False):
        example[Keys.SPECTROGRAM] = nested_op(
            super().__call__, example[Keys.SPECTROGRAM]
        )
        return example


class Mean(Transform):
    def __init__(self, axes, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def __call__(self, example, training=False):
        for key, axis in self.axes.items():
            keepdims = self.keepdims[key] \
                if isinstance(self.keepdims, dict) else self.keepdims
            example[key] = nested_op(
                lambda x: np.mean(x, axis=axis, keepdims=keepdims),
                example[key]
            )
        return example


class AddDeltas(Transform):
    def __init__(self, axes, num_deltas=1):
        """

        Args:
            axes:
            num_deltas:

        >>> deltas = AddDeltas(axes={'a': 1})
        >>> example = {'a': np.zeros((3, 16))}
        >>> deltas(example)['a_deltas'].shape
        (1, 3, 16)
        """
        self.axes = axes
        if not isinstance(num_deltas, dict):
            num_deltas = {key: num_deltas for key in axes}
        self.num_deltas = num_deltas

    def get_deltas(self, x, key):
        axis = self.axes[key]
        num_deltas = self.num_deltas[key]
        x = np.array(
            [
                delta(x, axis=axis, order=order)
                for order in range(1, num_deltas + 1)
            ]
        )
        return x

    def __call__(self, example, training=False):
        for key, axis in self.axes.items():
            example[f'{key}_deltas'] = nested_op(
                partial(self.get_deltas, key=key), example[key]
            )
        return example


class AddEnergy(Transform):
    def __init__(self, axes, keepdims=False):
        """

        Args:
            axes:

        >>> energies = AddEnergy(axes={'a': 0}, keepdims=True)
        >>> example = {'a': np.zeros((3, 16))}
        >>> energies(example)['a_energy'].shape
        (1, 16)
        """
        self.axes = axes
        self.keepdims = keepdims

    def get_energy(self, x, key):
        return np.sum(x, axis=self.axes[key], keepdims=self.keepdims)

    def __call__(self, example, training=False):
        for key, axis in self.axes.items():
            example[f'{key}_energy'] = nested_op(
                partial(self.get_energy, key=key), example[key]
            )
        return example


class SegmentAxis(Transform):
    def __init__(
            self, segment_steps, segment_lengths=None, axis=1, pad=False
    ):
        """

        Args:
            segment_steps:
            segment_lengths:
            axis:
            squeeze:
            pad:


        >>> time_segmenter = SegmentAxis({'a':2, 'b':1}, axis=1)
        >>> example = {'a': np.arange(10).reshape((2, 5)), 'b': np.array([[1,2]])}
        >>> from pprint import pprint
        >>> pprint(time_segmenter(example))
        {'a': array([[[0, 1],
                [2, 3]],
        <BLANKLINE>
               [[5, 6],
                [7, 8]]]),
         'b': array([[[1],
                [2]]])}
        >>> time_segmenter = SegmentAxis({'a':2, 'b':1}, axis=1, pad=True)
        >>> example = {'a': np.arange(10).reshape((2, 5)), 'b': np.array([[1,2]])}
        >>> from pprint import pprint
        >>> pprint(time_segmenter(example))
        {'a': array([[[0, 1],
                [2, 3],
                [4, 0]],
        <BLANKLINE>
               [[5, 6],
                [7, 8],
                [9, 0]]]),
         'b': array([[[1],
                [2]]])}
        >>> time_segmenter = SegmentAxis(\
                {'a':1, 'b':1}, {'a':2, 'b':1}, axis=1)
        >>> example = {'a': np.arange(8).reshape((2, 4)), 'b': np.array([[1,2,3]])}
        >>> pprint(time_segmenter(example))
        {'a': array([[[0, 1],
                [1, 2],
                [2, 3]],
        <BLANKLINE>
               [[4, 5],
                [5, 6],
                [6, 7]]]),
         'b': array([[[1],
                [2],
                [3]]])}
        >>> channel_segmenter = SegmentAxis({'a':1}, axis=0)
        >>> example = {'a': np.arange(8).reshape((2, 4))}
        >>> pprint(channel_segmenter(example))
        {'a': array([[[0, 1, 2, 3]],
        <BLANKLINE>
               [[4, 5, 6, 7]]])}
        >>> channel_segmenter = SegmentAxis({'a':3}, axis=0, pad=True)
        >>> example = {'a': np.arange(12).reshape((3, 4))}
        >>> pprint(channel_segmenter(example))
        {'a': array([[[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]]])}
        >>> channel_segmenter = SegmentAxis({'a':3}, axis=0, pad=True)
        >>> example = {'a': np.arange(12).reshape((3, 4))}
        >>> pprint(channel_segmenter(example, training=True))
        {'a': array([[[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]]])}
        # >>> channel_segmenter = SegmentAxis({'a': 5}, axis=0, pad=True)
        # >>> example = {'a': np.arange(12).reshape((3, 4))}
        # >>> pprint(channel_segmenter(example, training=True))
        """
        self.segment_steps = segment_steps
        self.segment_lengths = segment_lengths if segment_lengths is not None \
            else segment_steps
        self.axis = axis
        self.pad = pad

    def __call__(self, example, training=False):
        if training:
            start = np.random.rand()
            for key in self.segment_steps:
                value = example[key]
                value = flatten(value) if isinstance(value, dict) else {'': value}
                max_start = min(nested_op(
                    lambda x:
                        (x.shape[self.axis] - self.segment_lengths[key])
                        / self.segment_steps[key],
                    value
                ).values())
                start *= min(1., max_start)
            for segment_step in self.segment_steps.values():
                start = int(start*segment_step) / segment_step
        else:
            start = 0.

        def segment(x, key):
            segment_step = self.segment_steps[key]
            segment_length = self.segment_lengths[key]
            start_idx = int(start * segment_step)
            if start_idx > 0:
                slc = [slice(None)] * len(x.shape)
                slc[self.axis] = slice(
                    int(start_idx), x.shape[self.axis]
                )
                x = x[tuple(slc)]
            elif start_idx < 0 and self.pad:
                pad_width = x.ndim * [(0, 0)]
                pad_width[self.axis] = (-start_idx, 0)
                x = np.pad(x, pad_width=pad_width, mode='constant')
            assert self.axis < x.ndim, (self.axis, x.ndim)
            x = segment_axis_v2(
                x, segment_length, segment_step, axis=self.axis,
                end='pad' if self.pad else 'cut'
            )
            # x = np.split(x, x.shape[self.axis], axis=self.axis).squeeze(
            #     axis=self.axis
            # )
            # x = [np.squeeze(xi, axis=self.axis) for xi in x]
            # if self.squeeze and segment_length == 1:
            #     x = [np.squeeze(xi, axis=self.axis) for xi in x]
            return x

        for key in self.segment_steps.keys():
            example[key] = nested_op(lambda x: segment(x, key), example[key])
        return example


class Fragmenter(Transform):
    def __init__(
            self, split_axes, squeeze=True, broadcast_keys=None,
            deepcopy=False
    ):
        """

        Args:
            split_axes:
            squeeze:
            broadcast_keys:
            deepcopy:

        >>> decollate = Fragmenter(split_axes={'a': 1, 'b': 1})
        >>> example = {\
                'a': np.array([[[0, 1],[2, 3]],[[4, 5], [6, 7]]]),\
                'b': np.array([[[1],[2]]])\
            }
        >>> decollate(example)
        [{'a': array([[0, 1],
               [4, 5]]), 'b': array([[1]])}, {'a': array([[2, 3],
               [6, 7]]), 'b': array([[2]])}]
        """
        self.split_axes = split_axes
        self.squeeze = squeeze
        self.broadcast_keys = to_list(broadcast_keys) \
            if broadcast_keys is not None else []
        self.deepcopy = deepcopy

    def __call__(self, example, training=False):
        for key, axes in self.split_axes.items():
            for axis in to_list(axes):
                example[key] = nested_op(
                    lambda x: np.split(x, x.shape[axis], axis=axis)
                    if x.shape[axis] > 0 else [],
                    example[key]
                )

        def flatten_list(x, key):
            flat_list = list()
            for xi in x:
                if isinstance(xi, (list, tuple)):
                    flat_list.extend(flatten_list(xi, key))
                elif isinstance(xi, np.ndarray):
                    for axis in sorted(
                            to_list(self.split_axes[key]), reverse=True
                    ):
                        if self.squeeze:
                            xi = np.squeeze(xi, axis)
                    flat_list.append(xi)
            return flat_list

        features = flatten({
            key: nested_op(
                lambda x: flatten_list(x, key), example[key], sequence_type=()
            )
            for key in self.split_axes.keys()
        })

        num_fragments = np.array(
            [len(features[key]) for key in list(features.keys())]
        )
        assert all(num_fragments == num_fragments[0]), (list(features.keys()), num_fragments)
        fragment_template = dict()
        for key in self.broadcast_keys:
            fragment_template[key] = example[key]
        fragments = list()
        for i in range(int(num_fragments[0])):
            fragment = deepcopy(fragment_template) if self.deepcopy \
                else copy(fragment_template)
            fragment[Keys.FRAGMENT_ID] = i
            for key in features.keys():
                x = features[key][i]
                fragment[key] = deepcopy(x) if self.deepcopy else x
            fragment = deflatten(fragment)
            fragments.append(fragment)
        return fragments


class LabelEncoder(Transform):
    def __init__(self, key='scene', input_mapping=None, oov_label=None):
        assert key is not None
        self.key = key
        self.input_mapping = input_mapping
        self.oov_label = oov_label
        self.label2idx = None
        self.idx2label = None

    def init_params(self, labels=None, storage_dir=None, dataset=None):
        storage_dir = storage_dir or ""
        file = (Path(storage_dir) / f'{self.key}.json').expanduser().absolute()
        if storage_dir and file.exists():
            with file.open() as f:
                labels = json.load(f)
            print(f'Restored {self.key} from {file}')
        if labels is None:
            labels = self._read_labels_from_dataset(dataset)
        if self.input_mapping is not None:
            labels = sorted({
                self.input_mapping[label] if label in self.input_mapping
                else label for label in labels
            })
        if self.oov_label is not None and self.oov_label not in labels:
            labels.append(self.oov_label)
        if storage_dir and not file.exists():
            with file.open('w') as f:
                json.dump(labels, f, sort_keys=True, indent=4)
            print(f'Saved {self.key} to {file}')
        self.label2idx = {label: i for i, label in enumerate(labels)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}

    @property
    def num_labels(self):
        return len(self.idx2label)

    def encode(self, labels):
        def encode_label(label):
            if self.input_mapping is not None:
                label = self.input_mapping[label]
            if self.oov_label is not None and label not in self.label2idx:
                label = self.oov_label
            return self.label2idx[label]
        return nested_op(encode_label, labels)

    def __call__(self, example, training=False):
        example[self.key] = self.encode(example[self.key])
        return example

    def decode(self, labels):
        def decode_label(idx):
            return self.idx2label[idx]
        return nested_op(decode_label, labels)

    def _read_labels_from_dataset(self, dataset):
        def _read_labels(val_or_nested):
            labels = set()
            if isinstance(val_or_nested, (str, int)):
                labels.add(val_or_nested)
            elif isinstance(val_or_nested, (list, tuple)):
                for val_or_nested_ in val_or_nested:
                    labels.update(_read_labels(val_or_nested_))
            elif isinstance(val_or_nested, dict):
                for val_or_nested_ in val_or_nested.values():
                    labels.update(_read_labels(val_or_nested_))
            else:
                raise ValueError
            return labels

        labels = set()
        for example in dataset:
            labels.update(_read_labels(example[self.key]))
        return sorted(labels)


class Declutter(Transform):
    def __init__(self, required_keys, dtypes=None):
        """

        Args:
            required_keys:
            dtypes:
            permutations:

        >>> example = {'a': np.array([[1,2,3]]), 'b': 2, 'c': 3}
        >>> Declutter(['a', 'b'], {'a': 'float32'}, {'a': (1, 0)})(example)
        {'a': array([[1.],
               [2.],
               [3.]], dtype=float32), 'b': 2}
        """
        self.required_keys = to_list(required_keys)
        self.dtypes = dtypes

    def __call__(self, example, training=False):
        example = {key: example[key] for key in self.required_keys}
        if self.dtypes is not None:
            for key, dtype in self.dtypes.items():
                example[key] = np.array(example[key]).astype(
                    getattr(np, dtype)
                )
        return example


class GlobalNormalize(Transform):
    def __init__(
            self, center_axes=None, scale_axes=None, verbose=False, name=None
    ):
        """

        Args:
            center_axes:
            scale_axes:
            verbose:

        >>> norm = GlobalNormalize(center_axes={'spectrogram': 0})
        >>> ex = dict(spectrogram=2*np.ones((2, 4)))
        >>> norm.init_params(dataset=[ex])
        >>> norm.moments
        ({'spectrogram': [[2.0, 2.0, 2.0, 2.0]]}, {'spectrogram': [1.0]})
        >>> norm(ex)
        {'spectrogram': array([[0., 0., 0., 0.],
               [0., 0., 0., 0.]])}
        >>> norm = GlobalNormalize(\
            center_axes={'spectrogram': 0}, scale_axes={'spectrogram': 1})
        >>> ex = dict(spectrogram=2*np.ones((2, 4)))
        >>> norm.init_params(dataset=[ex])
        >>> norm.moments
        ({'spectrogram': [[2.0, 2.0, 2.0, 2.0]]}, {'spectrogram': [[0.0], [0.0]]})
        >>> norm(ex)
        {'spectrogram': array([[0., 0., 0., 0.],
               [0., 0., 0., 0.]])}
        >>> norm = GlobalNormalize(\
            scale_axes={'spectrogram': 1})
        >>> ex = dict(spectrogram=2*np.ones((2, 4)))
        >>> norm.init_params(dataset=[ex])
        >>> norm.moments
        ({'spectrogram': [0.0]}, {'spectrogram': [[2.0], [2.0]]})
        >>> norm(ex)
        {'spectrogram': array([[1., 1., 1., 1.],
               [1., 1., 1., 1.]])}
        """
        self.center_axes = {} if center_axes is None else {
            key: tuple(to_list(ax)) for key, ax in center_axes.items()
        }
        self.scale_axes = {} if scale_axes is None else {
            key: tuple(to_list(ax)) for key, ax in scale_axes.items()
        }
        self.verbose = verbose
        self.name = name
        self.moments = None

    def init_params(self, moments=None, dataset=None, storage_dir=None):
        self.moments = moments
        storage_dir = storage_dir or ""
        filename = "moments.json" if self.name is None else f"moments_{self.name}.json"
        file = (Path(storage_dir) / filename).expanduser().absolute()
        if self.moments is None and storage_dir and file.exists():
            with file.open() as f:
                self.moments = json.load(f)
            print(f'Restored moments from {file}')
        if self.moments is None:
            assert dataset is not None
            self.moments = self._read_moments_from_dataset(dataset)
        if storage_dir and not file.exists():
            with file.open('w') as f:
                json.dump(self.moments, f, sort_keys=True, indent=4)
            print(f'Saved moments to {file}')

    def norm(self, x, mean, scale):
        x -= mean
        x /= (np.array(scale) + 1e-18).astype(x.dtype)

    def __call__(self, example, training=False):
        means, scales = self.moments
        example_ = {key: example[key] for key in means.keys()}
        nested_op(self.norm, example_, means, scales)  # inplace
        return example

    def invert(self, example):
        example_ = {key: example[key] for key in self.center_axes}
        means, scale = self.moments
        nested_update(
            example,
            nested_op(lambda x, y, z: x*z+y, example_, means, scale)
        )
        return example

    def _read_moments_from_dataset(self, dataset):
        means = {key: 0 for key in self.center_axes}
        frame_counts_m = {key: 0 for key in self.center_axes}
        energies = {key: 0 for key in self.scale_axes}
        frame_counts_e = {key: 0 for key in self.scale_axes}
        axes = {
            key: self.center_axes[key] if key in self.center_axes
            else self.scale_axes[key]
            for key in {*self.center_axes.keys(), *self.scale_axes.keys()}
        }
        for example in tqdm(dataset, disable=not self.verbose):
            for key in axes:
                if key in means:
                    means[key] = nested_op(
                        lambda x, y:
                            y + np.sum(
                                x, axis=self.center_axes[key], keepdims=True
                            ),
                        example[key], means[key]
                    )
                    frame_counts_m[key] = nested_op(
                        lambda x, y:
                            y + np.prod(np.array(x.shape)[np.array(self.center_axes[key])]),
                        example[key], frame_counts_m[key]
                    )
                if key in energies:
                    energies[key] = nested_op(
                        lambda x, y:
                            y + np.sum(
                                x ** 2, axis=self.scale_axes[key],
                                keepdims=True
                            ),
                        example[key], energies[key]
                    )
                    frame_counts_e[key] = nested_op(
                        lambda x, y:
                            y + np.prod(np.array(x.shape)[np.array(self.scale_axes[key])]),
                        example[key], frame_counts_e[key]
                    )
        means = nested_op(lambda x, c: x / c, means, frame_counts_m)
        scales = {}
        for key in axes:
            if key not in means:
                means[key] = np.array([0.])
            if key not in energies:
                scales[key] = np.array([1.])
            else:
                scales[key] = nested_op(
                    lambda x, y, c: np.sqrt(
                        np.mean(
                            x/c - y ** 2, axis=self.scale_axes[key],
                            keepdims=True
                        )
                    ),
                    energies[key], means[key], frame_counts_e[key]
                )
        return (nested_op(lambda x: x.tolist(), means),
                nested_op(lambda x: x.tolist(), scales))


class LocalNormalize(Transform):
    def __init__(
            self, center_axes=None, scale_axes=None
    ):
        """

        Args:
            axes:
            std_reduce_axes:
            verbose:

        >>> norm = LocalNormalize(center_axes={'spectrogram': 0})
        >>> ex = dict(spectrogram=2*np.ones((2, 4)))
        >>> norm(ex)
        {'spectrogram': array([[0., 0., 0., 0.],
               [0., 0., 0., 0.]])}
        >>> norm = LocalNormalize(\
            center_axes={'spectrogram': 0}, scale_axes={'spectrogram': (0, 1)})
        >>> ex = dict(spectrogram=2*np.ones((2, 4)))
        >>> norm(ex)
        {'spectrogram': array([[0., 0., 0., 0.],
               [0., 0., 0., 0.]])}
        """
        self.keys = set()
        if center_axes is None:
            self.center_axes = None
        else:
            self.center_axes = {
                key: tuple(to_list(ax)) for key, ax in center_axes.items()
            }
            self.keys.update(self.center_axes.keys())
        if scale_axes is None:
            self.scale_axes = None
        else:
            self.scale_axes = {
                key: tuple(to_list(ax)) for key, ax in scale_axes.items()
            }
            self.keys.update(self.scale_axes.keys())

    def norm(self, x, key):
        if self.center_axes is not None and key in self.center_axes:
            x -= np.mean(x, axis=self.center_axes[key], keepdims=True)
        if self.scale_axes is not None and key in self.scale_axes:
            x /= (np.sqrt(np.mean(x ** 2, axis=self.scale_axes[key], keepdims=True)) + 1e-18).astype(x.dtype)

    def __call__(self, example, training=False):
        for key in self.keys:
            nested_op(
                partial(self.norm, key=key),
                example[key]
            )
        return example


class Reshape(Transform):
    def __init__(self, permutations=None, shapes=None):
        self.permutations = permutations
        self.shapes = shapes

    def __call__(self, example, training=False):
        if self.permutations is not None:
            for key, perm in self.permutations.items():
                if torch.is_tensor(example[key]):
                    transpose = lambda x: x.permute(perm)
                else:
                    transpose = lambda x: x.transpose(perm)
                example[key] = nested_op(lambda x: transpose(x), example[key])
        if self.shapes is not None:
            for key, shape in self.shapes.items():
                example[key] = nested_op(
                    lambda x: x.reshape(shape), example[key]
                )
        return example
