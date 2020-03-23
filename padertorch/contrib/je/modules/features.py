from functools import partial
from typing import Optional

import numpy as np
import torch
from padertorch.base import Module
from torch import nn


class MelTransform(Module):
    def __init__(
            self,
            n_mels: int,
            sample_rate: int,
            fft_length: int,
            fmin: Optional[float] = 50.,
            fmax: Optional[float] = None,
            log: bool = True,
            eps=1e-18,
            *,
            warping_fn=None,
            **kwargs
    ):
        """
        Transforms linear spectrogram to (log) mel spectrogram.

        Args:
            sample_rate: sample rate of audio signal
            fft_length: fft_length used in stft
            n_mels: number of filters to be applied
            fmin: lowest frequency (onset of first filter)
            fmax: highest frequency (offset of last filter)
            log: apply log to mel spectrogram
            eps:

        >>> mel_transform = MelTransform(40, 16000, 512)
        >>> spec = torch.zeros((10, 1, 100, 257))
        >>> logmelspec = mel_transform(spec)
        >>> logmelspec.shape
        torch.Size([10, 1, 100, 40])
        >>> rec = mel_transform.inverse(logmelspec)
        >>> rec.shape
        torch.Size([10, 1, 100, 257])
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log
        self.eps = eps
        self.warping_fn = warping_fn
        self.kwargs = kwargs

        fbanks = get_fbanks(
            n_mels=self.n_mels,
            fft_length=self.fft_length,
            sample_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
        ).astype(np.float32)
        fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
        self._fbanks = nn.Parameter(torch.from_numpy(fbanks.T), requires_grad=False)

    def get_fbanks(self, x):
        if not self.training or self.warping_fn is None:
            fbanks = self._fbanks
        else:
            fbanks = get_fbanks(
                n_mels=self.n_mels,
                fft_length=self.fft_length,
                sample_rate=self.sample_rate,
                fmin=self.fmin,
                fmax=self.fmax,
                warping_fn=partial(
                    self.warping_fn, n=x.shape[0], **self.kwargs
                )
            ).astype(np.float32)
            fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
            fbanks = torch.from_numpy(fbanks).transpose(-2, -1).to(x.device)
            while x.dim() > fbanks.dim():
                fbanks = fbanks[:, None]
        return nn.ReLU()(fbanks)

    def forward(self, x):
        x = x @ self.get_fbanks(x)
        if self.log:
            x = torch.log(x + self.eps)
        return x

    def inverse(self, x):
        """Invert the mel-filterbank transform."""
        ifbanks = (
            self._fbanks / (self._fbanks.sum(dim=-1, keepdim=True) + 1e-6)
        ).transpose(-2, -1)
        if self.log:
            x = torch.exp(x)
        x = x @ ifbanks
        return torch.max(x, torch.zeros_like(x))


def get_fbanks(
        n_mels, sample_rate, fft_length, fmin=0., fmax=None, warping_fn=None
):
    fmax = sample_rate/2 if fmax is None else fmax
    if fmax < 0:
        fmax = fmax % sample_rate/2
    f = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels+2))
    if warping_fn is not None:
        f = warping_fn(f)
    k = hz2bin(f, sample_rate, fft_length)
    centers = k[..., 1:-1, None]
    onsets = np.minimum(k[..., :-2, None], centers - 1)
    offsets = np.maximum(k[..., 2:, None], centers + 1)
    idx = np.arange(fft_length/2+1)
    fbanks = np.maximum(
        np.minimum(
            (idx-onsets)/(centers-onsets),
            (idx-offsets)/(centers-offsets)
        ),
        0
    )
    return fbanks


def hz2mel(f):
    return 1125*np.log(1+f/700)


def mel2hz(m):
    return 700*(np.exp(m/1125) - 1)


def bin2hz(k, sample_rate, fft_length):
    return sample_rate * k / fft_length


def hz2bin(f, sample_rate, fft_length):
    return f * fft_length / sample_rate
