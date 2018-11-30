import json
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
from cached_property import cached_property
from scipy import signal

from nt.database import keys as NTKeys
from nt.database.iterator import recursive_transform


class Keys:
    SPECTROGRAM = "spectrogram"
    LABELS = "labels"


class Transform(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, example):
        raise NotImplementedError


class Compose(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, example):
        for transform in self.transforms:
            example = transform(example)
        return example


class Spectrogram(Transform):
    """
    Transforms audio data to Log spectrogram (linear or mel spectrogram).
    Also allows to invert the transformation using Griffin Lim Algorithm.
    required entries in config dict: {
        frame_step: int,
        frame_length: int,
        fft_length: int,
        window: str,
        padded: bool,
        n_mels: int,
        fmin: int,
        fmax: None/int
    }
    """
    @cached_property
    def fbanks(self):
        return librosa.filters.mel(
            n_mels=self.config['n_mels'],
            n_fft=self.config['fft_length'],
            sr=self.config['sample_rate'],
            fmin=self.config['fmin'],
            fmax=self.config['fmax'],
            htk=True
        )

    @cached_property
    def ifbanks(self):
        return np.linalg.pinv(self.fbanks)

    def stft(self, x):
        noverlap = self.config['frame_length'] - self.config['frame_step']
        pad_width = x.ndim * [(0, 0)]
        pad_width[-1] = (int(noverlap // 2), int(np.ceil(noverlap / 2)))
        x = np.pad(x, pad_width, mode='constant')
        return signal.stft(
            x,
            window=self.config['window'],
            nperseg=self.config['frame_length'],
            noverlap=noverlap,
            nfft=self.config['fft_length'],
            padded=False,
            boundary=None)[-1]  # (C, F, T)

    def istft(self, x):
        noverlap = self.config['frame_length'] - self.config['frame_step']
        audio = signal.istft(
            x,
            window=self.config['window'],
            nperseg=self.config['frame_length'],
            noverlap=noverlap,
            nfft=self.config['fft_length'],
            boundary=None)[-1]
        return audio[int(noverlap // 2):-int(np.ceil(noverlap / 2))]

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
        if self.config['n_mels'] is not None:
            x = self.mel(x)
        if self.config['log'] is not None:
            x = np.log(np.maximum(x, 1e-18))
        return x.astype(np.float32)

    def adapt_sequence_length(self, sequence_length):
        noverlap = self.config['frame_length'] - self.config['frame_step']
        return (sequence_length - noverlap) // self.config['frame_step']

    def adapt_labels(self, labels):
        if not labels or not isinstance(labels[0], (list, tuple)):
            return labels
        assert len(labels[0]) == 3
        noverlap = self.config['frame_length'] - self.config['frame_step']
        return [
            (
                segment[0],
                max(0, int((segment[1] - noverlap//2)
                           // self.config['frame_step'])),
                max(0, int(np.ceil((segment[2] - noverlap//2)
                                   / self.config['frame_step'])))
            )
            for segment in labels
        ]

    def __call__(self, example):
        example = deepcopy(example)
        example[Keys.SPECTROGRAM] = recursive_transform(
            self.transform, example[NTKeys.AUDIO_DATA]
        )
        if NTKeys.NUM_SAMPLES in example:
            example[NTKeys.NUM_FRAMES] = recursive_transform(
                self.adapt_sequence_length, example[NTKeys.NUM_SAMPLES]
            )
        if Keys.LABELS in example:
            example[Keys.LABELS] = recursive_transform(
                self.adapt_labels, example[Keys.LABELS]
            )
        return example

    def griffin_lim(self, x, iterations=100, pow=.5, verbose=False):
        x = np.exp(x)
        if self.config['n_mels'] is not None:
            x = self.imel(x)
        x = np.power(x, pow)
        nframes = x.shape[-1]
        nsamples = int(nframes * self.config['frame_step'])
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
