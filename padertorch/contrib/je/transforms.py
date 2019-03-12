import abc
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import samplerate
import soundfile
from paderbox.transform.module_fbank import MelTransform as MelModule
from paderbox.transform.module_stft import STFT as STFTModule
from paderbox.utils.nested import squeeze_nested, nested_op, nested_update, \
    flatten, deflatten
from padertorch.configurable import Configurable
from padertorch.contrib.je import Keys
from padertorch.utils import to_list
from tqdm import tqdm
from paderbox.utils.numpy_utils import segment_axis_v2
from copy import copy, deepcopy


class Transform(Configurable, abc.ABC):
    """
    Base class for callable transformations. Not intended to be instantiated.
    """
    @abc.abstractmethod
    def __call__(self, example, training=False):
        raise NotImplementedError


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


class ReadAudio:
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
            window: str = "hann",
            symmetric_window: bool=False,
            fading: bool=True,
            pad: bool=True,
            keep_input=False
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
        """
        self.segment_steps = segment_steps
        self.segment_lengths = segment_lengths if segment_lengths is not None \
            else segment_steps
        self.axis = axis
        self.pad = pad

    def __call__(self, example, training=False):
        if training:
            start = np.random.rand()
            for fragment_step in self.segment_steps.values():
                start = int(int(start*fragment_step) / fragment_step)
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
                x = x[slc]
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
            self, split_axes, squeeze=False, broadcast_keys=None,
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
                        if self.squeeze and xi.shape[axis+1] == 1:
                            xi = np.squeeze(xi, axis+1)
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

    def init_labels(self, labels=None, storage_dir=None, dataset=None):
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


class GlobalNormalize(Transform):
    """
    >>> norm = GlobalNormalize(time_axis=0)
    >>> ex = dict(spectrogram=2*np.ones(10))
    >>> norm.init_moments([ex])
    >>> norm(ex)
    {'spectrogram': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
    """
    def __init__(self, keys=Keys.SPECTROGRAM, time_axis=1, verbose=False):
        self.keys = to_list(keys)
        self.time_axis = time_axis
        self.verbose = verbose
        self.moments = None

    def init_moments(self, dataset=None, storage_dir=None):
        storage_dir = storage_dir or ""
        file = (Path(storage_dir) / "moments.json").expanduser().absolute()
        if storage_dir and file.exists():
            with file.open() as f:
                self.moments = json.load(f)
            print(f'Restored moments from {file}')
        if self.moments is None:
            self.moments = self._read_moments_from_dataset(dataset)
            if storage_dir and not file.exists():
                with file.open('w') as f:
                    json.dump(self.moments, f, sort_keys=True, indent=4)
                print(f'Saved moments to {file}')

    def __call__(self, example, training=False):
        example_ = {key: example[key] for key in self.keys}
        means, std = self.moments
        nested_update(
            example,
            nested_op(
                lambda x, y, z: ((x-y)/(z+1e-18)).astype(x.dtype),
                example_, means, std
            )
        )
        return example

    def invert(self, example):
        example_ = {key: example[key] for key in self.keys}
        means, std = self.moments
        nested_update(
            example,
            nested_op(lambda x, y, z: x*z+y, example_, means, std)
        )
        return example

    def _read_moments_from_dataset(self, dataset):
        means = {key: 0 for key in self.keys}
        energies = {key: 0 for key in self.keys}
        counts = {key: 0 for key in self.keys}
        for example in tqdm(dataset, disable=not self.verbose):
            example = {key: example[key] for key in self.keys}
            means = nested_op(
                lambda x, y: y + np.sum(x, axis=self.time_axis, keepdims=True),
                example, means
            )
            energies = nested_op(
                lambda x, y: y + np.sum(x**2, axis=self.time_axis, keepdims=True),
                example, energies
            )
            counts = nested_op(
                lambda x, y: y + x.shape[self.time_axis],
                example, counts
            )
        means = nested_op(lambda x, c: x / c, means, counts)
        std = nested_op(
            lambda x, y, c: np.sqrt(np.mean(x/c - y ** 2)),
            energies, means, counts
        )
        return (nested_op(lambda x: x.tolist(), means),
                nested_op(lambda x: x.tolist(), std))
