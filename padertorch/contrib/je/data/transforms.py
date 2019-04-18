import abc
import json
from collections import OrderedDict, Mapping
from pathlib import Path
import numbers

import numpy as np
import samplerate
import soundfile
from paderbox.transform.module_fbank import MelTransform as MelModule
from paderbox.transform.module_stft import STFT as STFTModule
from paderbox.transform.module_mfcc import delta
from paderbox.utils.nested import flatten, deflatten, nested_op, nested_merge
from padertorch.configurable import Configurable
from padertorch.contrib.je import Keys
from padertorch.utils import to_list
from tqdm import tqdm
from paderbox.utils.numpy_utils import segment_axis_v2
from copy import copy, deepcopy
import torch


def nested_transform(
        transform_fn, input_path, input_dict, *args,
        sequence_type=(tuple, list), output_path=None
):
    """

    Args:
        transform_fn:
        input_path:
        input_dict:
        *args:
        sequence_type:
        output_path:

    Returns:

    """
    if not isinstance(input_path, (list, tuple)):
        input_path = input_path.split('/')
    input_dict_ = input_dict
    for key in input_path[:-1]:
        input_dict_ = input_dict_[key]
    key = input_path[-1]
    arg1 = input_dict_[key]
    transformed = nested_op(
        transform_fn, arg1, *args, sequence_type=sequence_type
    )
    if output_path is not None:
        if output_path == input_path:
            input_dict_[key] = transformed
        else:
            nested_merge(
                input_dict, deflatten({output_path: transformed}, sep='/'),
                inplace=True
            )
    return transformed


class Transform(Configurable, abc.ABC):
    """
    Base class for callable transformations. Not intended to be instantiated.
    """

    def init_params(self, values=None, storage_dir=None, dataset=None):
        pass

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
        s = ', '.join([repr(t) for k, t in self._transforms.items()])
        return f'{self.__class__.__name__}({s})'


class ReadAudio(Transform):
    """
    Read audio from disk. Expects an key 'audio_path' in input dict
    and adds an entry 'audio_data' with the read audio data.

    """
    def __init__(
            self, input_sample_rate, target_sample_rate,
            converter_type="sinc_fastest",
            input_path=Keys.AUDIO_PATH,
            output_path=Keys.AUDIO_DATA
    ):
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.converter_type = converter_type
        self.input_path = input_path
        self.ouput_path = output_path

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
        nested_transform(
            self.read, self.input_path, example, output_path=self.ouput_path
        )
        if (
            Keys.NUM_SAMPLES not in example
            or self.target_sample_rate != self.input_sample_rate
        ):
            nested_transform(
                lambda x: x.shape[-1], Keys.AUDIO_DATA, example,
                output_path=Keys.NUM_SAMPLES
            )
        for key in Keys.lable_keys():
            if f'{key}_start_times' in example and (
                    f'{key}_start_samples' not in example
                    or self.target_sample_rate != self.input_sample_rate
            ):
                nested_transform(
                    lambda x: int(x * self.target_sample_rate),
                    f'{key}_start_times', example,
                    output_path=f'{key}_start_samples'
                )
            if f'{key}_stop_times' in example and (
                    f'{key}_stop_samples' not in example
                    or self.target_sample_rate != self.input_sample_rate
            ):
                nested_transform(
                    lambda x: int(x * self.target_sample_rate),
                    f'{key}_stop_times', example,
                    output_path=f'{key}_stop_samples'
                )
        return example


class STFT(STFTModule, Transform):
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
    def __init__(
            self,
            frame_step: int,
            fft_length: int,
            frame_length: int = None,
            window: str = "blackman",
            symmetric_window: bool = False,
            fading: bool = True,
            pad: bool = True,
            input_path=Keys.AUDIO_DATA,
            output_path=Keys.STFT
    ):
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
        self.input_path = input_path
        self.ouput_path = output_path

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
        nested_transform(
            self.prepare_audio, self.input_path, example,
            output_path=self.input_path
        )
        nested_transform(
            super().__call__, self.input_path, example,
            output_path=self.ouput_path
        )
        nested_transform(
            lambda x: x.shape[-2], Keys.STFT, example,
            output_path=Keys.NUM_FRAMES
        )
        for key in Keys.lable_keys():
            if f'{key}_start_samples' in example:
                nested_transform(
                    self.sampleid2frameid, f'{key}_start_samples', example,
                    output_path=f'{key}_start_frames'
                )
            if f'{key}_stop_samples' in example:
                nested_transform(
                    lambda x: self.sampleid2frameid(x - 1) + 1,
                    f'{key}_stop_samples', example,
                    output_path=f'{key}_stop_frames'
                )
        return example

    def invert(self, example):
        raise NotImplementedError


class Spectrogram(Transform):
    def __init__(self, input_path=Keys.STFT):
        self.input_path = input_path

    def __call__(self, example, training=False):
        nested_transform(
            lambda x: x.real**2 + x.imag**2, self.input_path, example,
            output_path=Keys.SPECTROGRAM
        )
        return example


class MelTransform(MelModule, Transform):
    def __init__(
            self,
            sample_rate: int,
            fft_length: int,
            n_mels: int = 40,
            fmin: int = 20,
            fmax: int = None,
            log: bool = True,
            always3d: bool = False,
            input_path=Keys.SPECTROGRAM,
            output_path=Keys.MEL_SPECTROGRAM
    ):
        super().__init__(
            sample_rate=sample_rate,
            fft_length=fft_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            log=log,
            always3d=always3d
        )
        self.input_path = input_path
        self.output_path = output_path

    def __call__(self, example, training=False):
        nested_transform(
            super().__call__, self.input_path, example,
            output_path=self.output_path
        )
        return example


class AddDeltas(Transform):
    def __init__(self, input_path, axis, num_deltas=1):
        """

        Args:
            input_path:
            axis:
            num_deltas:

        >>> deltas = AddDeltas('a', 1)
        >>> example = {'a': np.zeros((3, 16))}
        >>> deltas(example)['deltas'].shape
        (1, 3, 16)
        """
        self.input_path = input_path
        self.axis = axis
        self.num_deltas = num_deltas

    def __call__(self, example, training=False):
        nested_transform(
            lambda x: self.get_deltas(
                x, axis=self.axis, num_deltas=self.num_deltas
            ),
            self.input_path, example, output_path=Keys.DELTAS
        )
        return example

    @staticmethod
    def get_deltas(x, axis, num_deltas):
        x = np.array(
            [
                delta(x, axis=axis, order=order)
                for order in range(1, num_deltas + 1)
            ]
        )
        return x


class AddEnergy(Transform):
    def __init__(self, input_path, axis=-1, keepdims=False):
        """

        Args:
            input_path:
            axis:

        >>> energies = AddEnergy(input_path='a', axis=0, keepdims=True)
        >>> example = {'a': np.zeros((3, 16))}
        >>> energies(example)['energy'].shape
        (1, 16)
        """
        self.input_path = input_path
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, example, training=False):
        nested_transform(
            lambda x: np.sum(x, axis=self.axis, keepdims=self.keepdims),
            self.input_path, example, output_path=Keys.ENERGY
        )
        return example


class Mean(Transform):
    def __init__(self, input_path, axis, keepdims=False):
        self.input_path = input_path
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, example, training=False):
        nested_transform(
            lambda x: np.mean(x, axis=self.axis, keepdims=self.keepdims),
            self.input_path, example, output_path=self.input_path
        )
        return example


class Log(Transform):
    def __init__(self, input_path):
        self.input_path = input_path

    def __call__(self, example, training=False):
        nested_transform(
            lambda x: np.log(x + 1e-15), self.input_path, example,
            output_path=self.input_path
        )
        return example


class GlobalNormalize(Transform):
    """

    Args:
        input_path:
        center_axis:
        scale_axis:
        verbose:

    >>> norm = GlobalNormalize(input_path='spectrogram', center_axis=0)
    >>> ex = dict(spectrogram=2*np.ones((2, 4)))
    >>> norm.init_params(dataset=[ex])
    >>> norm.moments
    ([[2.0, 2.0, 2.0, 2.0]], 1.0)
    >>> norm(ex)
    {'spectrogram': array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])}
    >>> norm = GlobalNormalize(\
        input_path='spectrogram', center_axis=0, scale_axis=1)
    >>> ex = dict(spectrogram=2*np.ones((2, 4)))
    >>> norm.init_params(dataset=[ex])
    >>> norm.moments
    ([[2.0, 2.0, 2.0, 2.0]], [[0.0], [0.0]])
    >>> norm(ex)
    {'spectrogram': array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])}
    >>> norm = GlobalNormalize(\
        input_path='spectrogram', scale_axis=1)
    >>> ex = dict(spectrogram=2*np.ones((2, 4)))
    >>> norm.init_params(dataset=[ex])
    >>> norm.moments
    (0.0, [[2.0], [2.0]])
    >>> norm(ex)
    {'spectrogram': array([[1., 1., 1., 1.],
           [1., 1., 1., 1.]])}
    """
    def __init__(
            self, input_path, center_axis=None, scale_axis=None, verbose=False,
            name=None
    ):
        self.input_path = input_path
        self.center_axis = None if center_axis is None \
            else tuple(to_list(center_axis))
        self.scale_axis = None if scale_axis is None \
            else tuple(to_list(scale_axis))
        self.verbose = verbose
        self.name = name
        self.moments = None

    def init_params(self, moments=None, storage_dir=None, dataset=None):
        self.moments = moments
        storage_dir = storage_dir or ""
        filename = "moments.json" if self.name is None \
            else f"{self.name}_moments.json"
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
        return x

    def __call__(self, example, training=False):
        means, scales = self.moments
        nested_transform(
            self.norm, self.input_path, example, means, scales,
            output_path=self.input_path
        )
        return example

    def _read_moments_from_dataset(self, dataset):
        means = np.array(0.)
        frame_counts_m = 0
        energies = np.array(0.)
        frame_counts_e = 0
        for example in tqdm(dataset, disable=not self.verbose):
            if self.center_axis is not None:
                means = nested_transform(
                    lambda x, y:
                        y + np.sum(x, axis=self.center_axis, keepdims=True),
                    self.input_path, example, means
                )
                frame_counts_m = nested_transform(
                    lambda x, y:
                        y + np.prod(np.array(x.shape)[np.array(self.center_axis)]),
                    self.input_path, example, frame_counts_m
                )
            if self.scale_axis is not None:
                energies = nested_transform(
                    lambda x, y:
                        y + np.sum(
                            np.abs(x) ** 2, axis=self.scale_axis,
                            keepdims=True
                        ),
                    self.input_path, example, energies
                )
                frame_counts_e = nested_transform(
                    lambda x, y:
                        y + np.prod(np.array(x.shape)[np.array(self.scale_axis)]),
                    self.input_path, example, frame_counts_e
                )
        if self.center_axis is not None:
            means = nested_op(lambda x, c: x / c, means, frame_counts_m)
        if self.scale_axis is not None:
            scales = nested_op(
                lambda x, y, c: np.sqrt(
                    np.mean(
                        x/c - y ** 2, axis=self.scale_axis,
                        keepdims=True
                    )
                ),
                energies, means, frame_counts_e
            )
        else:
            scales = np.array(1.)
        return (
            nested_op(lambda x: x.tolist(), means),
            nested_op(lambda x: x.tolist(), scales)
        )


class LocalNormalize(Transform):
    """

    Args:
        input_path:
        center_axis:
        scale_axis:

    >>> norm = LocalNormalize('spectrogram', center_axis=0)
    >>> ex = dict(spectrogram=2*np.ones((2, 4)))
    >>> norm(ex)
    {'spectrogram': array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])}
    >>> norm = LocalNormalize(\
            'spectrogram', center_axis=0, scale_axis=(0, 1))
    >>> ex = dict(spectrogram=2*np.ones((2, 4)))
    >>> norm(ex)
    {'spectrogram': array([[0., 0., 0., 0.],
           [0., 0., 0., 0.]])}
    """
    def __init__(self, input_path, center_axis=None, scale_axis=None):
        self.input_path = input_path
        self.center_axis = center_axis
        self.scale_axis = scale_axis

    def norm(self, x):
        if self.center_axis is not None:
            x -= np.mean(x, axis=self.center_axis, keepdims=True)
        if self.scale_axis is not None:
            x /= (np.sqrt(np.mean(
                x ** 2, axis=self.scale_axis, keepdims=True
            )) + 1e-18).astype(x.dtype)
        return x

    def __call__(self, example, training=False):
        nested_transform(
            self.norm, self.input_path, example, output_path=self.input_path
        )
        return example


class SegmentAxis(Transform):
    """

    Args:
        axes:
        segment_steps:
        segment_lengths:
        pad:


    >>> time_segmenter = SegmentAxis({'a': 1, 'b': 1}, {'a': 2, 'b': 1})
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
    >>> time_segmenter = SegmentAxis({'a/b': 1}, {'a/b': 2})
    >>> example = {'a': {'b': np.arange(10).reshape((2, 5))}}
    >>> pprint(time_segmenter(example))
    {'a': {'b': array([[[0, 1],
            [2, 3]],
    <BLANKLINE>
           [[5, 6],
            [7, 8]]])}}
    >>> time_segmenter = SegmentAxis({'a': 1, 'b': 1}, {'a':2, 'b':1}, pad=True)
    >>> example = {'a': np.arange(10).reshape((2, 5)), 'b': np.array([[1,2]])}
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
            {'a':1, 'b':1}, {'a':1, 'b':1}, {'a':2, 'b':1})
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
    >>> channel_segmenter = SegmentAxis({'a': 0}, {'a': 1})
    >>> example = {'a': np.arange(8).reshape((2, 4))}
    >>> pprint(channel_segmenter(example))
    {'a': array([[[0, 1, 2, 3]],
    <BLANKLINE>
           [[4, 5, 6, 7]]])}
    >>> channel_segmenter = SegmentAxis({'a': 0}, {'a':3}, pad=True)
    >>> example = {'a': np.arange(12).reshape((3, 4))}
    >>> pprint(channel_segmenter(example))
    {'a': array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])}
    >>> channel_segmenter = SegmentAxis({'a': 0}, {'a':3}, pad=True)
    >>> example = {'a': np.arange(12).reshape((3, 4))}
    >>> pprint(channel_segmenter(example, training=True))
    {'a': array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])}
    """
    def __init__(
            self, axes, segment_steps, segment_lengths=None,
            random_start=True, pad=False,

    ):
        self.axes = axes

        if isinstance(segment_steps, Mapping):
            self.segment_steps = segment_steps
        else:
            assert isinstance(segment_steps, numbers.Integral)
            self.segment_steps = {key: segment_steps for key in axes}

        if segment_lengths is None:
            self.segment_lengths = self.segment_steps
        elif isinstance(segment_lengths, Mapping):
            self.segment_lengths = segment_lengths
        else:
            assert isinstance(segment_lengths, numbers.Integral)
            self.segment_lengths = {key: segment_lengths for key in axes}

        self.random_start = random_start
        self.pad = pad

    def __call__(self, example, training=False):

        # get random start
        if training and self.random_start:
            start = np.random.rand()

            # find max start such that at least one segment is obtained
            max_start = 1.
            for key in self.segment_steps:
                # get nested structure and cast to dict
                value = nested_transform(lambda x: x, key, example)
                value = flatten(value) if isinstance(value, dict) \
                    else {'': value}

                max_start = min(
                    max_start,
                    min(nested_op(
                        lambda x:
                            (x.shape[self.axes[key]]
                             - self.segment_lengths[key])
                            / self.segment_steps[key],
                        value
                    ).values())
                )
            start *= max_start

            # adjust start to match an integer index for all keys
            for segment_step in self.segment_steps.values():
                start = int(start*segment_step) / segment_step
        else:
            start = 0.

        def segment(x, axis, step, length):
            start_idx = int(start * step)
            if start_idx > 0:
                slc = [slice(None)] * len(x.shape)
                slc[axis] = slice(int(start_idx), x.shape[axis])
                x = x[tuple(slc)]
            elif start_idx < 0 and self.pad:
                # pad front if start_idx < 0 (this means input length is
                # shorter than segment length and training is True).
                pad_width = x.ndim * [(0, 0)]
                pad_width[axis] = (-start_idx, 0)
                x = np.pad(x, pad_width=pad_width, mode='constant')
            assert axis < x.ndim, (axis, x.ndim)
            x = segment_axis_v2(
                x, length, step, axis=axis,
                end='pad' if self.pad else 'cut'
            )
            return x

        for key in self.segment_steps.keys():
            nested_transform(
                lambda x: segment(
                    x, self.axes[key], self.segment_steps[key],
                    self.segment_lengths[key]
                ),
                key, example, output_path=key
            )
        return example


class Fragmenter(Transform):
    """

    Args:
        split_axes:
        squeeze:
        broadcast_keys:
        deepcopy:

    >>> fragment = Fragmenter(split_axes={'a': 1, 'b': 1})
    >>> example = {\
            'a': np.array([[[0, 1],[2, 3]],[[4, 5], [6, 7]]]),\
            'b': np.array([[[1],[2]]])\
        }
    >>> fragment(example)
    [{'fragment_id': 0, 'a': array([[0, 1],
           [4, 5]]), 'b': array([[1]])}, {'fragment_id': 1, 'a': array([[2, 3],
           [6, 7]]), 'b': array([[2]])}]
    >>> fragment = Fragmenter(split_axes={'a/b': 1})
    >>> example = {\
            'a': {'b': np.array([[[0, 1],[2, 3]],[[4, 5], [6, 7]]])}\
        }
    >>> fragment(example)
    [{'fragment_id': 0, 'a': {'b': array([[0, 1],
           [4, 5]])}}, {'fragment_id': 1, 'a': {'b': array([[2, 3],
           [6, 7]])}}]
    """
    def __init__(
            self, split_axes, squeeze=True, broadcast_keys=None, deepcopy=False
    ):
        self.split_axes = split_axes
        self.squeeze = squeeze
        self.broadcast_keys = to_list(broadcast_keys) \
            if broadcast_keys is not None else []
        self.deepcopy = deepcopy

    def __call__(self, example, training=False):
        for key, axes in self.split_axes.items():
            for axis in to_list(axes):
                nested_transform(
                    lambda x: (
                        np.split(x, x.shape[axis], axis=axis)
                        if x.shape[axis] > 0 else []
                    ),
                    key, example, output_path=key
                )

        def flatten_list(x, axes):
            flat_list = list()
            for xi in x:
                if isinstance(xi, (list, tuple)):
                    flat_list.extend(flatten_list(xi, axes))
                elif isinstance(xi, np.ndarray):
                    for axis in sorted(to_list(axes), reverse=True):
                        if self.squeeze:
                            xi = np.squeeze(xi, axis)
                    flat_list.append(xi)
            return flat_list

        features = flatten({
            key: nested_transform(
                lambda x: flatten_list(x, self.split_axes[key]), key, example,
                sequence_type=()
            )
            for key in self.split_axes.keys()
        }, sep='/')

        num_fragments = np.array(
            [len(features[key]) for key in features.keys()]
        )
        assert all(num_fragments == num_fragments[0]), (
            list(features.keys()), num_fragments
        )
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
            fragment = deflatten(fragment, sep='/')
            fragments.append(fragment)
        return fragments


class LabelEncoder(Transform):
    """

    Args:
        input_path:
        input_mapping:
        oov_label:
        name:

    >>> example = {'labels':['b', 'a', 'c']}
    >>> label_encoder = LabelEncoder('labels')
    >>> label_encoder.init_params(dataset=[example])
    >>> label_encoder(example)
    {'labels': [1, 0, 2]}
    >>> example = {'labels': {'subkey': ['b', 'a', 'c']}}
    >>> label_encoder = LabelEncoder('labels/subkey')
    >>> label_encoder.init_params(dataset=[example])
    >>> label_encoder(example)
    {'labels': {'subkey': [1, 0, 2]}}
    """
    def __init__(
            self, input_path, input_mapping=None, oov_label=None, name=None
    ):
        assert input_path is not None
        self.input_path = input_path
        self.input_mapping = input_mapping
        self.oov_label = oov_label
        self.label2idx = None
        self.idx2label = None
        self.name = name

    def init_params(self, labels=None, storage_dir=None, dataset=None):
        storage_dir = storage_dir or ""
        filename = "labels.json" if self.name is None \
            else f'{self.name}_labels.json'
        file = (Path(storage_dir) / filename).expanduser().absolute()
        if storage_dir and file.exists():
            with file.open() as f:
                labels = json.load(f)
            print(f'Restored {self.input_path} from {file}')
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
            print(f'Saved {self.input_path} to {file}')
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
        nested_transform(
            self.encode, self.input_path, example, output_path=self.input_path,
            sequence_type=()
        )
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
            labels.update(_read_labels(
                nested_transform(lambda x: x, self.input_path, example)
            ))
        return sorted(labels)


class Declutter(Transform):
    """

    Args:
        required_keys:
        dtypes:

    >>> example = {'a': np.array([[1,2,3]]), 'b': 2, 'c': 3}
    >>> Declutter(['a', 'b'], {'a': 'float32'})(example)
    {'a': array([[1., 2., 3.]], dtype=float32), 'b': 2}
    >>> example = {'a': {'b': np.array([[1,2,3]]), 'c':2}, 'd': 3}
    >>> Declutter(['a/b', 'd'], {'a/b': 'float32'})(example)
    {'a': {'b': array([[1., 2., 3.]], dtype=float32)}, 'd': 3}
    """
    def __init__(self, required_keys, dtypes=None):
        self.required_keys = to_list(required_keys)
        self.dtypes = dtypes

    def __call__(self, example, training=False):
        # get nested
        example = {
            key: nested_transform(lambda x: x, key, example)
            for key in self.required_keys
        }
        if self.dtypes is not None:
            for key, dtype in self.dtypes.items():
                example[key] = np.array(example[key]).astype(
                    getattr(np, dtype)
                )
        return deflatten(example, sep='/')


class Reshape(Transform):
    """

    Args:
        input_path:
        permutation:
        shape:

    >>> example = {'a': np.array([[1,2,3],[1,2,3]])}
    >>> Reshape('a', (1, 0),  (-1,))(example)
    {'a': array([1, 1, 2, 2, 3, 3])}
    >>> example = {'a': {'b': np.array([[1,2,3],[1,2,3]])}}
    >>> Reshape('a/b', (1, 0), (-1,))(example)
    {'a': {'b': array([1, 1, 2, 2, 3, 3])}}
    """
    def __init__(self, input_path, permutation=None, shape=None):
        self.input_path = input_path
        self.permutation = permutation
        self.shape = shape

    def __call__(self, example, training=False):
        if self.permutation is not None:
            nested_transform(
                lambda x: x.transpose(self.permutation), self.input_path,
                example, output_path=self.input_path
            )
        if self.shape is not None:
            nested_transform(
                lambda x: x.reshape(self.shape), self.input_path, example,
                output_path=self.input_path
            )
        return example


class Collate(Transform):
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
            batch = np.array(batch)
            if self.to_tensor:
                batch = torch.Tensor(batch)
        return batch
