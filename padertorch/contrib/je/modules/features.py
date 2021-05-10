from typing import Optional

import numpy as np
import torch
from paderbox.transform.module_fbank import get_fbanks
from paderbox.utils.random_utils import TruncatedExponential
from padertorch.base import Module
from padertorch.contrib.je.modules.augment import (
    TimeWarping, GaussianBlur2d, Mixup, Mask, AdditiveNoise,
)
from padertorch.modules.normalization import Normalization, InputNormalization
from torch import nn
from scipy.signal import savgol_coeffs
from padertorch.ops.sequence.mask import compute_mask


class NormalizedLogMelExtractor(nn.Module):
    """
    >>> x = torch.randn((10,1,100,257,2))
    >>> NormalizedLogMelExtractor(16000, 512, 40)(x)[0].shape
    torch.Size([10, 1, 40, 100])
    >>> NormalizedLogMelExtractor(16000, 512, 40, add_deltas=True, add_delta_deltas=True)(x)[0].shape
    torch.Size([10, 3, 40, 100])
    """
    def __init__(
            self, sample_rate, stft_size, number_of_filters, num_channels=1,
            lowest_frequency=50, highest_frequency=None, htk_mel=True,
            add_deltas=False, add_delta_deltas=False,
            norm_statistics_axis='bt', norm_eps=1e-5, batch_norm=False,
            clamp=6,
            ipd_pairs=(),
            # augmentation
            frequency_warping_fn=None, time_warping_fn=None,
            blur_sigma=0, blur_kernel_size=5,
            mixup_prob=0., mixup_alpha=1.,
            n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=.2,
            n_frequency_masks=0, max_masked_frequency_bands=20, max_masked_frequency_rate=.2,
            max_noise_scale=0.,
    ):
        super().__init__()
        self.mel_transform = MelTransform(
            sample_rate=sample_rate,
            stft_size=stft_size,
            number_of_filters=number_of_filters,
            lowest_frequency=lowest_frequency,
            highest_frequency=highest_frequency,
            htk_mel=htk_mel,
            log=True,
            warping_fn=frequency_warping_fn,
        )
        if add_deltas:
            self.deltas_extractor = DeltaExtractor(order=1)
        else:
            self.deltas_extractor = None
        if add_delta_deltas:
            self.delta_deltas_extractor = DeltaExtractor(order=2)
        else:
            self.delta_deltas_extractor = None

        assert all([len(pair) == 2 for pair in ipd_pairs]), ipd_pairs
        assert all([c < num_channels for pair in ipd_pairs for c in pair]), ipd_pairs
        self.ipd_pairs = list(zip(*ipd_pairs))
        self.filter_max_indices = self.mel_transform.fbanks.argmax(0)
        norm_cls = Normalization if batch_norm else InputNormalization
        self.norm = norm_cls(
            data_format='bcft',
            shape=(
                None,
                (1 + add_deltas + add_delta_deltas) * num_channels,
                number_of_filters,
                None
            ),
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
                sigma_sampling_fn=TruncatedExponential(
                    loc=.1, scale=blur_sigma, truncation=blur_kernel_size
                )
            )
        else:
            self.blur = None

        assert 0 <= mixup_prob <= 1, mixup_prob
        if mixup_prob > 0:
            self.mixup = Mixup(p=mixup_prob, alpha=mixup_alpha)
        else:
            self.mixup = None

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_frequency_masks > 0:
            self.mel_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_bands,
                max_masked_rate=max_masked_frequency_rate,
            )
        else:
            self.mel_masking = None

        if max_noise_scale > 0.:
            self.noise = AdditiveNoise(max_noise_scale)
        else:
            self.noise = None

    def forward(self, x, seq_len=None, targets=None):
        with torch.no_grad():
            if self.ipd_pairs:
                x_re = x[..., self.filter_max_indices, 0]
                x_im = x[..., self.filter_max_indices, 1]
                channel_ref, channel_other = self.ipd_pairs
                ipds = torch.atan2(
                    x_im[:, channel_other] * x_re[:, channel_ref]
                    - x_re[:, channel_other] * x_im[:, channel_ref],
                    x_re[:, channel_other] * x_re[:, channel_ref]
                    + x_im[:, channel_other] * x_im[:, channel_ref]
                ).transpose(-2, -1)
            else:
                ipds = None
            x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)

            if self.time_warping is not None:
                x, seq_len = self.time_warping(x, seq_len=seq_len)

            if self.blur is not None:
                x = self.blur(x)

            if (self.deltas_extractor is not None) or (self.delta_deltas_extractor is not None):
                x_ = x
                if self.deltas_extractor is not None:
                    deltas = self.deltas_extractor(x_, seq_len=seq_len)
                    x = torch.cat((x, deltas), dim=1)
                if self.delta_deltas_extractor is not None:
                    delta_deltas = self.delta_deltas_extractor(x_, seq_len=seq_len)
                    x = torch.cat((x, delta_deltas), dim=1)

            x = self.norm(x, sequence_lengths=seq_len)

            if self.clamp is not None:
                x = torch.clamp(x, -self.clamp, self.clamp)

            if ipds is not None:
                x = torch.cat((x, torch.cos(ipds), torch.sin(ipds)), dim=1)

            if self.mixup is not None:
                x, seq_len, targets = self.mixup(x, seq_len, targets)

            if self.time_masking is not None:
                x = self.time_masking(x, seq_len=seq_len)
            if self.mel_masking is not None:
                x = self.mel_masking(x)

            if self.noise is not None:
                # print(torch.std(x, dim=-1))
                x = self.noise(x)
        if targets is None:
            return x, seq_len
        return x, seq_len, targets

    def inverse(self, x):
        return self.mel_transform.inverse(
            self.norm.inverse(x).transpose(-2, -1)
        ).sqrt().unsqueeze(-1)


class MelTransform(Module):
    def __init__(
            self,
            sample_rate: int,
            stft_size: int,
            number_of_filters: int,
            lowest_frequency: Optional[float] = 50.,
            highest_frequency: Optional[float] = None,
            htk_mel=True,
            log: bool = True,
            eps=1e-12,
            *,
            warping_fn=None,
            independent_axis=0,
    ):
        """
        Transforms linear spectrogram to (log) mel spectrogram.

        Args:
            sample_rate: sample rate of audio signal
            stft_size: fft_length used in stft
            number_of_filters: number of filters to be applied
            lowest_frequency: lowest frequency (onset of first filter)
            highest_frequency: highest frequency (offset of last filter)
            log: apply log to mel spectrogram
            eps:

        >>> sample_rate = 16000
        >>> highest_frequency = sample_rate/2
        >>> mel_transform = MelTransform(sample_rate, 512, 40)
        >>> spec = torch.rand((3, 1, 100, 257))
        >>> logmelspec = mel_transform(spec)
        >>> logmelspec.shape
        torch.Size([3, 1, 100, 40])
        >>> rec = mel_transform.inverse(logmelspec)
        >>> rec.shape
        torch.Size([3, 1, 100, 257])
        >>> from paderbox.transform.module_fbank import HzWarping
        >>> from paderbox.utils.random_utils import Uniform
        >>> warping_fn = HzWarping(\
                warp_factor_sampling_fn=Uniform(low=.9, high=1.1),\
                boundary_frequency_ratio_sampling_fn=Uniform(low=.6, high=.7),\
                highest_frequency=highest_frequency,\
            )
        >>> mel_transform = MelTransform(sample_rate, 512, 40, warping_fn=warping_fn)
        >>> mel_transform(spec).shape
        torch.Size([3, 1, 100, 40])
        >>> mel_transform = MelTransform(sample_rate, 512, 40, warping_fn=warping_fn, independent_axis=(0,1,2))
        >>> np.random.seed(0)
        >>> x = mel_transform(spec)
        >>> x.shape
        torch.Size([3, 1, 100, 40])
        >>> from paderbox.transform.module_fbank import MelTransform as MelTransformNumpy
        >>> mel_transform_np = MelTransformNumpy(sample_rate, 512, 40, warping_fn=warping_fn, independent_axis=(0,1,2))
        >>> np.random.seed(0)
        >>> x_ref = mel_transform_np(spec.numpy())
        >>> assert (x.numpy()-x_ref).max() < 1e-6
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.stft_size = stft_size
        self.number_of_filters = number_of_filters
        self.lowest_frequency = lowest_frequency
        self.highest_frequency = highest_frequency
        self.htk_mel = htk_mel
        self.log = log
        self.eps = eps
        self.warping_fn = warping_fn
        self.independent_axis = [independent_axis] if np.isscalar(independent_axis) else independent_axis

        fbanks = get_fbanks(
            sample_rate=self.sample_rate,
            stft_size=self.stft_size,
            number_of_filters=self.number_of_filters,
            lowest_frequency=self.lowest_frequency,
            highest_frequency=self.highest_frequency,
            htk_mel=htk_mel,
        ).astype(np.float32)
        fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
        self.fbanks = nn.Parameter(
            torch.from_numpy(fbanks.T), requires_grad=False
        )

    def forward(self, x, return_maxima=False):
        if not self.training or self.warping_fn is None:
            fbanks = self.fbanks
            x = x.matmul(fbanks)
        else:
            independent_axis = [ax if ax >= 0 else x.ndim+ax for ax in self.independent_axis]
            assert all([ax < x.ndim-1 for ax in independent_axis])
            size = [
                x.shape[i] if i in independent_axis else 1
                for i in range(x.ndim-1)
            ]
            fbanks = get_fbanks(
                sample_rate=self.sample_rate,
                stft_size=self.stft_size,
                number_of_filters=self.number_of_filters,
                lowest_frequency=self.lowest_frequency,
                highest_frequency=self.highest_frequency,
                htk_mel=self.htk_mel,
                warping_fn=self.warping_fn,
                size=size,
            ).astype(np.float32)
            fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
            fbanks = torch.from_numpy(fbanks).transpose(-2, -1).to(x.device)
            if fbanks.shape[-3] == 1:
                x = x.matmul(fbanks.squeeze(-3))
            else:
                x = x[..., None, :].matmul(fbanks).squeeze(-2)
        if self.log:
            x = torch.log(x + self.eps)
        if return_maxima:
            maxima = (fbanks.argmax(-2) + 1) * (fbanks.sum(-2) > 0) - 1
            return x, maxima
        return x

    def inverse(self, x):
        """Invert the mel-filterbank transform."""
        ifbanks = self.fbanks.T
        ifbanks = ifbanks / (ifbanks.sum(dim=-2, keepdim=True) + 1e-6)
        if self.log:
            x = torch.exp(x)
        x = x @ ifbanks
        return torch.max(x, torch.zeros_like(x))


class DeltaExtractor(nn.Module):
    """
    >>> f = DeltaExtractor(order=1, width=9)
    >>> f.kernel
    >>> x = torch.randn(4, 2, 40, 1000) - 40
    >>> deltas = f(x)
    >>> deltas.shape
    torch.Size([4, 2, 40, 1000])
    >>> deltas.max()
    >>> deltas[0, 0, :, :5]
    >>> f = DeltaExtractor(order=2, width=9)
    >>> f.kernel
    >>> delta_deltas = f(x)
    >>> delta_deltas.shape
    torch.Size([4, 2, 40, 1000])
    >>> delta_deltas[0, 0, :, :5]
    >>> delta_deltas.max()
    >>> from librosa import feature
    >>> librosa_deltas = feature.delta(x.numpy(), axis=-1, order=1)
    >>> librosa_delta_deltas = feature.delta(x.numpy(), axis=-1, order=2)
    >>> np.abs(deltas.numpy() - librosa_deltas)[..., 4:-4].max()
    >>> np.abs(delta_deltas.numpy() - librosa_delta_deltas)[..., 4:-4].max()
    """
    def __init__(self, width=5, order=1):
        super().__init__()
        self.width = width
        self.order = order
        kernel = savgol_coeffs(width, order, deriv=order, delta=1.0).astype(np.float32)
        self.kernel = nn.Parameter(
            torch.from_numpy((-1)**(order % 2)*kernel), requires_grad=False
        )

    def forward(self, x, seq_len=None):
        # pack batch
        shape = x.size()
        x = x.reshape(1, -1, shape[-1])

        assert self.width >= 3, self.width
        n = (self.width - 1) // 2

        kernel = self.kernel.repeat(x.shape[1], 1, 1)
        y = torch.nn.functional.conv1d(x, kernel, groups=x.shape[1])
        y = torch.nn.functional.pad(y, [n, n], mode="constant")

        # unpack batch
        y = y.reshape(shape)
        if seq_len is not None:
            y = y * compute_mask(y, np.array(seq_len) - n, batch_axis=0, sequence_axis=-1)

        return y
