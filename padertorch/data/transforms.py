import abc
import json
from pathlib import Path
from typing import Optional
from warnings import warn

import librosa
import numpy as np
from cached_property import cached_property
from scipy import signal
from tqdm import tqdm

from paderbox.database import keys as NTKeys
from paderbox.io import load_audio
from paderbox.utils.nested import squeeze_nested, nested_op, nested_update
from padertorch.configurable import Configurable
from padertorch.utils import to_list

class Keys:
    SPECTROGRAM = "spectrogram"


LABEL_KEYS = [NTKeys.PHONES, NTKeys.WORDS, NTKeys.EVENTS, NTKeys.SCENE]


class Transform(Configurable, abc.ABC):
    """
    Base class for callable transformations. Not intended to be instantiated.
    """
    @abc.abstractmethod
    def __call__(self, example):
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
    def __init__(self, *transforms):
        # Wouldn't "functions" instead of transforms not a better name=
        self.transforms = transforms

    def __call__(self, example):
        for transform in self.transforms:
            example = transform(example)
        return example

    def __repr__(self):
        """
        >>> import operator
        >>> Compose(operator.neg, abs)
        Compose(<built-in function neg>, <built-in function abs>)
        >>> Compose(operator.neg, abs)(1)
        1
        """
        s = ', '.join([repr(t) for t in self.transforms])
        return f'{self.__class__.__name__}({s})'


class ReadAudio(Transform):
    """
    Read audio from File. Expects an entry 'audio_path' in input dict and adds
    an entry 'audio_data' with the read audio data.

    """
    def __init__(self, sample_rate=16000, converter_type="sinc_fastest"):
        self.sample_rate = sample_rate
        self.converter_type = converter_type

    def read(self, audio_path):
        x, sr = load_audio(audio_path, return_sample_rate=True)
        if sr != self.sample_rate:
            import samplerate
            warn('Sample rate mismatch -> Resample.')
            x = x.T
            x = samplerate.resample(
                x, self.sample_rate / sr, self.converter_type
            )
            x = x.T
        return x

    def __call__(self, example):
        example[NTKeys.AUDIO_DATA] = nested_op(
            self.read, example[NTKeys.AUDIO_PATH]
        )
        example[NTKeys.NUM_SAMPLES] = squeeze_nested(nested_op(
            lambda x: x.shape[-1], example[NTKeys.AUDIO_DATA]
        ))
        for key in LABEL_KEYS:
            if f'{key}_start_times' in example:
                example[f'{key}_start_samples'] = squeeze_nested(nested_op(
                    lambda x: x * self.sample_rate,
                    example[f'{key}_start_times']
                ))
            if f'{key}_end_times' in example:
                example[f'{key}_end_samples'] = squeeze_nested(nested_op(
                    lambda x: x * self.sample_rate,
                    example[f'{key}_end_times']
                ))

        return example


class Spectrogram(Transform):
    """
    Transforms audio data to Log spectrogram (linear or mel spectrogram).
    Also allows to invert the transformation using Griffin Lim Algorithm.

    >>> spec = Spectrogram()
    >>> ex = dict(\
    audio_data=np.zeros(16000), \
    phones_start_samples=[8000, 12000], \
    phones_end_samples=[12000, 16000])
    >>> ex = spec(ex)
    >>> print(ex["num_frames"])
    100
    >>> print(ex["phones_start_frames"])
    [50.0, 75.0]
    >>> print(ex["phones_end_frames"])
    [75.0, 100.0]
    """
    def __init__(
            self,
            sample_rate: int = 16000,
            frame_length: int = 400,
            frame_step: int = 160,
            fft_length:int = 512,
            window: str = "hann",
            padded: bool = True,
            n_mels: Optional[int] = 40,
            fmin: Optional[int] = 20,
            fmax: Optional[int] = None,
            log: bool = True
    ):
        self.sample_rate = sample_rate
        self.padded = padded
        self.frame_length = frame_length
        self.window = window
        self.fft_length = fft_length
        self.frame_step = frame_step
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log

    @cached_property
    def fbanks(self):
        return librosa.filters.mel(
            n_mels=self.n_mels,
            n_fft=self.fft_length,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=True
        )

    @cached_property
    def ifbanks(self):
        return np.linalg.pinv(self.fbanks)

    def stft(self, x):
        # ToDo: switch to toolbox stft?
        noverlap = self.frame_length - self.frame_step
        if self.padded:
            pad_width = x.ndim * [(0, 0)]
            pad_width[-1] = (int(noverlap // 2), int(np.ceil(noverlap / 2)))
            x = np.pad(x, pad_width, mode='constant')
        return signal.stft(
            x,
            window=self.window,
            nperseg=self.frame_length,
            noverlap=noverlap,
            nfft=self.fft_length,
            padded=False,
            boundary=None)[-1]  # (C, F, T)

    def istft(self, x):
        noverlap = self.frame_length - self.frame_step
        audio = signal.istft(
            x,
            window=self.window,
            nperseg=self.frame_length,
            noverlap=noverlap,
            nfft=self.fft_length,
            boundary=None)[-1]
        if self.padded:
            return audio[int(noverlap // 2):-int(np.ceil(noverlap / 2))]
        return audio

    def energy(self, x):
        x = self.stft(x)
        return x.real ** 2 + x.imag ** 2

    def mel(self, x):
        return np.dot(self.fbanks, x).transpose((1, 0, 2))

    def imel(self, x):
        return np.maximum(np.dot(self.ifbanks, x), 0.)

    def transform(self, x):
        if x.ndim == 1:
            x = x[None]
        assert x.ndim == 2
        x = self.energy(x)
        if self.n_mels is not None:
            x = self.mel(x)
        if self.log is not None:
            x = np.log(np.maximum(x, 1e-18))
        return x.astype(np.float32)

    def transform_audio(self, audio):
        if self.padded:
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

    def sample2frame(self, sample):
        if self.padded:
            return sample / self.frame_step
        else:
            noverlap = self.frame_length - self.frame_step
            return max(sample - noverlap/2, 0.) / self.frame_step

    def __call__(self, example):
        example[NTKeys.AUDIO_DATA] = nested_op(
            self.transform_audio, example[NTKeys.AUDIO_DATA]
        )
        example[Keys.SPECTROGRAM] = nested_op(
            self.transform, example[NTKeys.AUDIO_DATA]
        )
        example[NTKeys.NUM_FRAMES] = squeeze_nested(nested_op(
            lambda x: x.shape[-1], example[Keys.SPECTROGRAM]
        ))
        for key in LABEL_KEYS:
            if f'{key}_start_samples' in example:
                example[f'{key}_start_frames'] = squeeze_nested(nested_op(
                    self.sample2frame,
                    example[f'{key}_start_samples']
                ))
            if f'{key}_end_samples' in example:
                example[f'{key}_end_frames'] = squeeze_nested(nested_op(
                    self.sample2frame,
                    example[f'{key}_end_samples']
                ))
        return example

    def griffin_lim(self, x, iterations=100, pow=.5, verbose=False):
        x = np.exp(x)
        if self.n_mels is not None:
            x = self.imel(x)
        x = np.power(x, pow)
        nframes = x.shape[-1]
        nsamples = int(nframes * self.frame_step)
        # Initialize the reconstructed signal to noise.
        audio = np.random.randn(nsamples)
        n = iterations  # number of iterations of Griffin-Lim algorithm.
        while n > 0:
            n -= 1
            reconstruction_spectrogram = self.stft(audio)
            reconstruction_magnitude = np.abs(reconstruction_spectrogram)
            reconstruction_angle = np.angle(reconstruction_spectrogram)
            # Discard magnitude part of the reconstruction and use the supplied
            # magnitude spectrogram instead.
            diff = (np.sqrt(np.mean((reconstruction_magnitude - x) ** 2)))
            proposal_spec = x * np.exp(1.0j * reconstruction_angle)
            audio = self.istft(proposal_spec)

            if verbose:
                print('Reconstruction iteration: {}/{} RMSE: {} '.format(
                    iterations - n, iterations, diff))
        return audio


class LabelEncoder(object):
    def __init__(self, key='scene', input_mapping=None):
        assert key is not None
        self.key = key
        self.input_mapping = input_mapping
        self.label2idx = None
        self.idx2label = None

    def init_labels(self, labels=None, storage_dir=None, iterator=None):
        storage_dir = storage_dir or ""
        file = (Path(storage_dir) / f'{self.key}.json').expanduser().absolute()
        if storage_dir and file.exists():
            with file.open() as f:
                labels = json.load(f)
            print(f'Restored {self.key} from {file}')
        if labels is None:
            labels = self._read_labels_from_iterator(iterator)
        if self.input_mapping is not None:
            labels = sorted({self.input_mapping[label] for label in labels})
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
            return self.label2idx[label]
        return nested_op(encode_label, labels)

    def __call__(self, example):
        example[self.key] = self.encode(example[self.key])
        return example

    def decode(self, labels):
        def decode_label(idx):
            return self.idx2label[idx]
        return nested_op(decode_label, labels)

    def _read_labels_from_iterator(self, iterator):
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
        for example in iterator:
            labels.update(_read_labels(example[self.key]))
        return sorted(labels)


class GlobalNormalize(Configurable):
    """
    >>> norm = GlobalNormalize()
    >>> ex = dict(spectrogram=2*np.ones(10))
    >>> norm.init_moments([ex])
    >>> norm(ex)
    {'spectrogram': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
    """
    def __init__(self, keys=Keys.SPECTROGRAM, show_progress=False):
        self.keys = to_list(keys)
        self.moments = None
        self._show_progress = show_progress

    def init_moments(self, iterator=None, storage_dir=None):
        storage_dir = storage_dir or ""
        file = (Path(storage_dir) / "moments.json").expanduser().absolute()
        if storage_dir and file.exists():
            with file.open() as f:
                self.moments = json.load(f)
            print(f'Restored moments from {file}')
        if self.moments is None:
            self.moments = self._read_moments_from_iterator(iterator)
            if storage_dir and not file.exists():
                with file.open('w') as f:
                    json.dump(self.moments, f, sort_keys=True, indent=4)
                print(f'Saved moments to {file}')

    def __call__(self, example):
        example_ = {key: example[key] for key in self.keys}
        means, std = self.moments
        nested_update(
            example,
            nested_op(lambda x, y, z: ((x-y)/(z+1e-18)).astype(np.float32), example_, means, std)
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

    def _read_moments_from_iterator(self, iterator):
        means = {key: 0 for key in self.keys}
        energies = {key: 0 for key in self.keys}
        counts = {key: 0 for key in self.keys}
        for example in tqdm(iterator, disable=not(self._show_progress)):
            example = {key: example[key] for key in self.keys}
            means = nested_op(
                lambda x, y: y + np.sum(x, axis=-1, keepdims=True),
                example, means
            )
            energies = nested_op(
                lambda x, y: y + np.sum(x**2, axis=-1, keepdims=True),
                example, energies
            )
            counts = nested_op(
                lambda x, y: y + x.shape[-1],
                example, counts
            )
        means = nested_op(lambda x, c: x / c, means, counts)
        std = nested_op(
            lambda x, y, c: np.sqrt(np.mean(x/c - y ** 2)),
            energies, means, counts
        )
        return (nested_op(lambda x: x.tolist(), means),
                nested_op(lambda x: x.tolist(), std))


class LocalNormalize(Configurable):
    """
    >>> norm = LocalNormalize()
    >>> ex = dict(spectrogram=2*np.ones(10))
    >>> norm(ex)
    {'spectrogram': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}"""
    def __init__(self, keys=Keys.SPECTROGRAM):
        self.keys = to_list(keys)

    def __call__(self, example):
        means = {key: 0 for key in self.keys}
        means = nested_op(
            lambda x, y: np.mean(y, axis=-1, keepdims=True),
            means, example
        )
        std = nested_op(
            lambda x, y: np.sqrt(np.std(y, axis=-1, keepdims=True)),
            means, example
        )
        nested_update(
            example,
            nested_op(lambda y, z, x: (x-y)/(z+1e-18), means, std, example)
        )
        return example
