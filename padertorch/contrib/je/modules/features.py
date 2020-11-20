from functools import partial
from typing import Optional

import numpy as np
import torch
from padertorch.base import Module
from padertorch.contrib.je.modules.augment import (
    Scale, TimeWarping, GaussianBlur2d, Mask, Noise,
    LogUniformSampler, TruncExponentialSampler, LogTruncNormalSampler
)
from padertorch.modules.normalization import Normalization, InputNormalization
from torch import nn
from paderbox.transform.module_fbank import get_fbanks


class NormalizedLogMelExtractor(nn.Module):
    """
    >>> x = torch.ones((10,1,100,257,2))
    >>> NormalizedLogMelExtractor(40, 16000, 512, stft_scale_window=50)(x)[0].shape
    >>> NormalizedLogMelExtractor(40, 16000, 512, add_deltas=True, add_delta_deltas=True)(x)[0].shape
    """
    def __init__(
            self, n_mels, sample_rate, fft_length, fmin=50, fmax=None,
            add_deltas=False, add_delta_deltas=False,
            norm_statistics_axis='bt', norm_eps=1e-5, batch_norm=False,
            clamp=None,
            # augmentation
            frequency_warping_fn=None, time_warping_fn=None,
            blur_sigma=0, blur_kernel_size=5,
            n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=.2,
            n_mel_masks=0, max_masked_mel_steps=20, max_masked_mel_rate=.2,
            max_noise_scale=0.,
    ):
        super().__init__()
        self.mel_transform = MelTransform(
            n_mels=n_mels, sample_rate=sample_rate, fft_length=fft_length,
            fmin=fmin, fmax=fmax, log=True, warping_fn=frequency_warping_fn,
        )
        self.add_deltas = add_deltas
        self.add_delta_deltas = add_delta_deltas
        norm_cls = Normalization if batch_norm else InputNormalization
        self.norm = norm_cls(
            data_format='bcft',
            shape=(None, 1 + add_deltas + add_delta_deltas, n_mels, None),
            statistics_axis=norm_statistics_axis,
            shift=True,
            scale=True,
            eps=norm_eps,
            independent_axis=None,
            momentum=None,
        )
        self.clamp = clamp

        # augmentation
        if time_warping_fn is not None:
            self.time_warping = TimeWarping(warping_fn=time_warping_fn)
        else:
            self.time_warping = None

        if blur_sigma > 0:
            self.blur = GaussianBlur2d(
                kernel_size=blur_kernel_size,
                sigma_sampling_fn=TruncExponentialSampler(
                    shift=.1, scale=blur_sigma, truncation=blur_kernel_size
                )
            )
        else:
            self.blur = None

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_mel_masks > 0:
            self.mel_masking = Mask(
                axis=-2, n_masks=n_mel_masks,
                max_masked_steps=max_masked_mel_steps,
                max_masked_rate=max_masked_mel_rate,
            )
        else:
            self.mel_masking = None

        if max_noise_scale > 0.:
            self.noise = Noise(max_noise_scale)
        else:
            self.noise = None

    def forward(self, x, seq_len=None):
        with torch.no_grad():

            x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)

            if self.time_warping is not None:
                x, seq_len = self.time_warping(x, seq_len=seq_len)

            if self.blur is not None:
                x = self.blur(x)

            if self.add_deltas or self.add_delta_deltas:
                deltas = compute_deltas(x)
                if self.add_deltas:
                    x = torch.cat((x, deltas), dim=1)
                if self.add_delta_deltas:
                    delta_deltas = compute_deltas(deltas)
                    x = torch.cat((x, delta_deltas), dim=1)

            x = self.norm(x, sequence_lengths=seq_len)
            if self.clamp is not None:
                x = torch.clamp(x, -self.clamp, self.clamp)

            if self.time_masking is not None:
                x = self.time_masking(x, seq_len=seq_len)
            if self.mel_masking is not None:
                x = self.mel_masking(x)

            if self.noise is not None:
                # print(torch.std(x, dim=-1))
                x = self.noise(x)

        return x, seq_len

    def inverse(self, x):
        return self.mel_transform.inverse(
            self.norm.inverse(x).transpose(-2, -1)
        )


class MelTransform(Module):
    def __init__(
            self,
            n_mels: int,
            sample_rate: int,
            fft_length: int,
            fmin: Optional[float] = 50.,
            fmax: Optional[float] = None,
            log: bool = True,
            eps=1e-12,
            *,
            warping_fn=None,
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
                warping_fn=partial(self.warping_fn, n=x.shape[0])
            ).astype(np.float32)
            fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
            fbanks = torch.from_numpy(fbanks).transpose(-2, -1).to(x.device)
            while x.dim() > fbanks.dim():
                fbanks = fbanks[:, None]
        return nn.ReLU()(fbanks)

    def forward(self, x):
        x = x.matmul(self.get_fbanks(x))
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


def compute_deltas(specgram, win_length=5, mode="replicate"):
    # type: (Tensor, int, str) -> Tensor
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    !!!copy from torchaudio.functional!!!

    .. math::
        d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is (`win_length`-1)//2.

    Args:
        specgram (torch.Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int): The window length used for computing delta
        mode (str): Mode parameter passed to padding

    Returns:
        deltas (torch.Tensor): Tensor of audio of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = (
        torch
        .arange(-n, n + 1, 1, device=specgram.device, dtype=specgram.dtype)
        .repeat(specgram.shape[1], 1, 1)
    )

    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.reshape(shape)

    return output
