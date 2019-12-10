import numpy as np
import torch
from padertorch.base import Module
from torch import nn
from typing import Optional
import librosa


class MelTransform(Module):
    def __init__(
            self,
            sample_rate: int,
            fft_length: int,
            n_mels: int,
            fmin: Optional[int] = 50,
            fmax: Optional[int] = None,
            trainable: bool = False,
            log: bool = True,
            eps=1e-18,
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

        >>> mel_transform = MelTransform(16000, 512, 40)
        >>> spec = torch.zeros((100, 257))
        >>> logmelspec = mel_transform(spec)
        >>> logmelspec.shape
        torch.Size([100, 40])
        >>> rec = mel_transform.inverse(logmelspec)
        >>> rec.shape
        torch.Size([100, 257])
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log
        self.eps = eps

        fbanks = librosa.filters.mel(
            n_mels=self.n_mels,
            n_fft=self.fft_length,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=True,
            norm=None
        ).astype(np.float32)
        fbanks = fbanks / fbanks.sum(axis=-1, keepdims=True)
        self.fbanks = nn.Parameter(torch.from_numpy(fbanks.T), requires_grad=trainable)

    def forward(self, x):
        x = torch.mm(x, self.fbanks)
        if self.log:
            x = torch.log(x + self.eps)
        return x

    def inverse(self, x):
        """Invert the mel-filterbank transform."""
        ifbanks = torch.pinverse(self.fbanks.transpose(0, 1)).transpose(0, 1)
        if self.log:
            x = np.exp(x)
        return np.maximum(torch.mm(x, ifbanks), 0.)
