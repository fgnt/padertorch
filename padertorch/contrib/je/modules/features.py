from typing import Optional

import numpy as np
import torch
import typing
from math import ceil
from torch.nn.functional import conv1d
from einops import rearrange

from paderbox.transform.module_fbank import get_fbanks
from paderbox.utils.random_utils import TruncatedExponential
from paderbox.transform.module_stft import (
    sample_index_to_stft_frame_index, _samples_to_stft_frames, _stft_frames_to_samples,
)
from padertorch.utils import to_list
from padertorch.base import Module
from padertorch.contrib.je.modules.augment import (
    MixBack, Scale, Superpose, GaussianBlur2d, Mixup, Mask, AdditiveNoise,
)
from padertorch.contrib.cb.transform import stft, istft
from padertorch.modules.normalization import Normalization, InputNormalization
from torch import nn
from scipy.signal import savgol_coeffs
from padertorch.ops.sequence.mask import compute_mask


class NormalizedLogMelExtractor(nn.Module):
    """
    >>> x = torch.randn((10,1,48000))
    >>> seq_lens = 10*[48000]
    >>> labels = np.array([[3, 1, 16000, 24000],[7, 4, 24000, 32000],[8, 2, 32000, 48000]])
    >>> NormalizedLogMelExtractor(16000, 160, 512, 40)(x)[0].shape
    torch.Size([10, 1, 40, 300])
    >>> NormalizedLogMelExtractor(16000, 160, 512, 40, add_deltas=True, add_delta_deltas=True)(x, seq_lens)[0].shape
    torch.Size([10, 3, 40, 300])
    >>> m, l, t = NormalizedLogMelExtractor(16000, 160, 512, 40)(x, seq_lens, labels=labels)
    >>> m.shape
    torch.Size([10, 1, 40, 300])
    >>> l
    array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
    >>> t
    array([[  3,   1, 100, 150],
           [  7,   4, 150, 200],
           [  8,   2, 200, 300]])
    >>> m, l, t = NormalizedLogMelExtractor(16000, 160, 512, 40, time_warping_anchor_sampling_fn = lambda: .5, time_warping_anchor_shift_sampling_fn = lambda: .0)(x, seq_lens, labels=labels)
    >>> m.shape
    torch.Size([10, 1, 40, 300])
    >>> l
    array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
    >>> t.shape
    array([[  3,   1, 100, 150],
           [  7,   4, 150, 200],
           [  8,   2, 200, 300]])
    >>> m, l, t = NormalizedLogMelExtractor(16000, 160, 512, 40, time_warping_anchor_sampling_fn = lambda: .5, time_warping_anchor_shift_sampling_fn = lambda: 1/6)(x, seq_lens, labels=labels)
    >>> m.shape
    torch.Size([10, 1, 40, 300])
    >>> l
    array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
    >>> t.shape
    array([[  3,   1, 133, 200],
           [  7,   4, 200, 233],
           [  8,   2, 233, 300]])
    >>> m, l, t = NormalizedLogMelExtractor(16000, 160, 512, 40, time_warping_anchor_sampling_fn = lambda: .5, time_warping_anchor_shift_sampling_fn = lambda: 1/6)(x, seq_lens, labels=(labels[:, :2], labels,))
    >>> m.shape
    torch.Size([10, 1, 40, 300])
    >>> l
    array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300])
    >>> t[0].shape, t[1].shape
    [array([[3, 1],
           [7, 4],
           [8, 2]]), array([[  3,   1, 133, 200],
           [  7,   4, 200, 233],
           [  8,   2, 233, 300]])]
    >>> NormalizedLogMelExtractor(16000, 160, 512, 40, pcen_channels=3)(x)[0].shape
    torch.Size([10, 1, 40, 300])
    """
    def __init__(
            self,
            sample_rate,
            stft_shift, stft_size,
            number_of_filters, *,
            window: [str, typing.Callable] = 'blackman',
            window_length: int = None,
            fading: typing.Optional[typing.Union[bool, str]] = 'half',
            pad: bool = True,
            symmetric_window: bool = False,
            lowest_frequency=50, highest_frequency=None, htk_mel=True,
            clamp=6,
            pcen_channels=0, trainable_pcen=True,
            pcen_lp_time_constant=.4, pcen_lp_filter_length=100,
            pcen_eps=1e-6, pcen_gain=.98, pcen_bias=2, pcen_power=.5,
            add_deltas=False, add_delta_deltas=False,
            num_waveform_feature_maps=0,
            norm_statistics_axis='bt', norm_eps=1e-5, batch_norm=False,
            num_channels=1, ipd_pairs=(),
            num_target_classes=None,
            # augmentation
            scale_sampling_fn=None,
            time_warping_anchor_sampling_fn=None,
            time_warping_anchor_shift_sampling_fn=None,
            superposition_prob=0.,
            frequency_warping_fn=None,
            max_mixback_scale=0., mixback_buffer_size=1,
            blur_sigma=0, blur_kernel_size=5,
            mixup_prob=0., mixup_alpha=2., mixup_beta=1., mixup_target_threshold=None,
            max_noise_scale=0.,
            n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=1.,
            min_masked_time_steps=0, min_masked_time_rate=0.,
            n_frequency_masks=0,
            max_masked_frequency_bands=20, max_masked_frequency_rate=1.,
            min_masked_frequency_bands=0, min_masked_frequency_rate=0.,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.stft = STFT(
            shift=stft_shift,
            size=stft_size,
            window_length=window_length,
            window=window,
            fading=fading,
            pad=pad,
            symmetric_window=symmetric_window,
            time_warping_anchor_sampling_fn=time_warping_anchor_sampling_fn,
            time_warping_anchor_shift_sampling_fn=time_warping_anchor_shift_sampling_fn,
        )
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
        if pcen_channels > 0:
            t_frames = np.array(pcen_lp_time_constant) * sample_rate / stft_shift
            print(t_frames)
            lp_filter_coeff = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
            print('lp_filter_coeff', lp_filter_coeff)
            self.pcen = PCEN(
                in_frequencies=number_of_filters,
                out_channels_per_in_channel=pcen_channels,
                lp_filter_coeff=lp_filter_coeff.tolist(),  # s
                lp_filter_length=pcen_lp_filter_length,
                gain=pcen_gain,  # alpha
                eps=pcen_eps,
                power=pcen_power,  # r
                bias=pcen_bias,  # delta
                trainable=trainable_pcen,
            )
        else:
            self.pcen = None
        if add_deltas:
            self.deltas_extractor = DeltaExtractor(order=1)
        else:
            self.deltas_extractor = None
        if add_delta_deltas:
            self.delta_deltas_extractor = DeltaExtractor(order=2)
        else:
            self.delta_deltas_extractor = None
        if num_waveform_feature_maps > 0:
            self.waveform_conv = WaveformConv(
                shift=stft_shift,
                window_length=window_length,
                out_channels=number_of_filters*num_waveform_feature_maps,
                fading=fading,
                pad=pad,
                time_warping_anchor_sampling_fn=time_warping_anchor_sampling_fn,
                time_warping_anchor_shift_sampling_fn=time_warping_anchor_shift_sampling_fn,
            )
        else:
            self.waveform_conv = None

        assert all([len(pair) == 2 for pair in ipd_pairs]), ipd_pairs
        assert all([c < num_channels for pair in ipd_pairs for c in pair]), ipd_pairs
        self.ipd_pairs = list(zip(*ipd_pairs))
        self.filter_max_indices = self.mel_transform.fbanks.argmax(0)
        norm_cls = Normalization if batch_norm else InputNormalization
        # ToDo: warn if not batch_norm with learnable parameters?
        self.norm = norm_cls(
            data_format='bcft',
            shape=(
                None,
                (1 + add_deltas + add_delta_deltas) * num_channels * (1 + pcen_channels) + num_waveform_feature_maps,
                number_of_filters,
                None
            ),
            statistics_axis=norm_statistics_axis,
            shift=True,
            scale=True,
            eps=norm_eps,
            independent_axis=None,
            momentum=0.95 if batch_norm else None,
        )
        self.clamp = clamp

        # augmentation
        if scale_sampling_fn is None:
            scale_fn = None
        else:
            scale_fn = Scale(scale_sampling_fn=scale_sampling_fn)

        assert 0. <= superposition_prob <= 1., superposition_prob
        if superposition_prob > 0.:
            self.superpose = Superpose(p=superposition_prob, scale_fn=scale_fn)
            self.scale = None
        else:
            self.superpose = None
            self.scale = scale_fn

        if blur_sigma > 0:
            self.blur = GaussianBlur2d(
                kernel_size=blur_kernel_size,
                sigma_sampling_fn=TruncatedExponential(
                    loc=.1, scale=blur_sigma, truncation=blur_kernel_size
                )
            )
        else:
            self.blur = None

        if max_mixback_scale > 0.:
            self.mixback = MixBack(max_mixback_scale, mixback_buffer_size)
        else:
            self.mixback = None

        assert 0 <= mixup_prob <= 1, mixup_prob
        if mixup_prob > 0:
            self.mixup = Mixup(p=mixup_prob, alpha=mixup_alpha, beta=mixup_beta, target_threshold=mixup_target_threshold)
        else:
            self.mixup = None

        if max_noise_scale > 0.:
            self.noise = AdditiveNoise(max_noise_scale)
        else:
            self.noise = None

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
                min_masked_steps=min_masked_time_steps,
                min_masked_rate=min_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_frequency_masks > 0:
            self.mel_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_bands,
                max_masked_rate=max_masked_frequency_rate,
                min_masked_steps=min_masked_frequency_bands,
                min_masked_rate=min_masked_frequency_rate,
            )
        else:
            self.mel_masking = None

        self.num_target_classes = num_target_classes

    def reset(self):
        self.norm.reset_parameters()
        if self.mixback is not None:
            self.mixback.reset()

    def freeze(self):
        self.norm.freeze(freeze_stats=True)

    def forward(self, x, seq_len=None, labels=None):
        with torch.no_grad():
            if self.scale is not None:
                x = self.scale(x)
            if self.superpose is not None:
                x, seq_len, labels = self.superpose(
                    x, seq_len=seq_len, labels=labels,
                )

        if self.waveform_conv is None:
            x_conv = seq_len_conv = time_warping_params = None
        else:
            x_conv, seq_len_conv, time_warping_params = self.waveform_conv(x, seq_len, return_time_warping_params=True)
            x_conv = rearrange(x_conv, 'b c t (n f) -> b (c n) f t', f=self.mel_transform.number_of_filters)

        with torch.no_grad():
            if labels is None:
                x, seq_len = self.stft(x, seq_len, time_warping_params=time_warping_params)
                targets = None
            elif isinstance(labels, np.ndarray):
                assert labels.ndim == 2, labels.shape
                if labels.shape[1] == 2:
                    x, seq_len = self.stft(x, seq_len, time_warping_params=time_warping_params)
                else:
                    assert labels.shape[1] == 4, labels.shape
                    labels = labels.copy()
                    x, seq_len, labels[:, 2:] = self.stft(x, seq_len, labels[:, 2:], time_warping_params=time_warping_params)
                targets = _labels_to_targets(labels, x.shape[0], x.shape[2], device=x.device, num_target_classes=self.num_target_classes)
            elif isinstance(labels, (list, tuple)):
                labels = list(labels)
                sample_indices = []
                split_indices = []
                for i in range(len(labels)):
                    assert isinstance(labels[i], np.ndarray)
                    assert labels[i].ndim == 2, labels[i].shape
                    if labels[i].shape[1] == 4:
                        sample_indices.append(labels[i][:, 2:])
                        split_indices.append(len(sample_indices[-1]))
                    else:
                        split_indices.append(0)
                        assert labels[i].shape[1] == 2, labels[i].shape
                if len(sample_indices) == 0:
                    x, seq_len = self.stft(x, seq_len, time_warping_params=time_warping_params)
                else:
                    sample_indices = np.concatenate(sample_indices)
                    split_indices = np.cumsum(split_indices)
                    x, seq_len, frame_indices = self.stft(x, seq_len, sample_indices, time_warping_params=time_warping_params)
                    for i, labels_i in enumerate(np.split(frame_indices, split_indices)):
                        if len(labels_i) > 0:
                            labels[i] = labels[i].copy()
                            labels[i][:, 2:] = labels_i
                targets = [_labels_to_targets(labels_i, x.shape[0], x.shape[2], device=x.device, num_target_classes=self.num_target_classes) for labels_i in labels]

            if self.ipd_pairs:
                x_re = x[..., self.filter_max_indices].real
                x_im = x[..., self.filter_max_indices].imag
                channel_ref, channel_other = self.ipd_pairs
                ipds = torch.atan2(
                    x_im[:, channel_other] * x_re[:, channel_ref]
                    - x_re[:, channel_other] * x_im[:, channel_ref],
                    x_re[:, channel_other] * x_re[:, channel_ref]
                    + x_im[:, channel_other] * x_im[:, channel_ref]
                ).transpose(-2, -1)
            else:
                ipds = None

            x = self.mel_transform((x.real**2 + x.imag**2)).transpose(-2, -1)

            if self.blur is not None:
                x = self.blur(x)

        if self.pcen is not None:
            x = torch.cat((x, self.pcen(torch.exp(x))), dim=1)

        if (self.deltas_extractor is not None) or (self.delta_deltas_extractor is not None):
            x_ = x
            if self.deltas_extractor is not None:
                deltas = self.deltas_extractor(x_, seq_len=seq_len)
                x = torch.cat((x, deltas), dim=1)
            if self.delta_deltas_extractor is not None:
                delta_deltas = self.delta_deltas_extractor(x_, seq_len=seq_len)
                x = torch.cat((x, delta_deltas), dim=1)

        if x_conv is not None:
            assert (len(seq_len_conv) == len(seq_len)), (len(seq_len_conv), len(seq_len))
            assert (seq_len_conv == seq_len).all(), (seq_len_conv, seq_len)
            x = torch.cat((x, x_conv), dim=1)

        x = self.norm(x, sequence_lengths=seq_len)

        if self.clamp is not None:
            x = torch.clamp(x, -self.clamp, self.clamp)

        if self.mixback is not None:
            x = self.mixback(x)

        if ipds is not None:
            x = torch.cat((x, torch.cos(ipds), torch.sin(ipds)), dim=1)

        if self.mixup is not None:
            x, seq_len, targets = self.mixup(
                x, seq_len=seq_len, targets=targets
            )

        if self.noise is not None:
            # print(torch.std(x, dim=-1))
            x = self.noise(x)

        if self.time_masking is not None:
            x, _ = self.time_masking(x, seq_len=seq_len)

        if self.mel_masking is not None:
            x, _ = self.mel_masking(x)
        if labels is None:
            return x, seq_len
        return x, seq_len, targets

    def inverse(self, x):
        return self.mel_transform.inverse(
            self.norm.inverse(x).transpose(-2, -1)
        ).sqrt().unsqueeze(-1)


def _labels_to_targets(labels, batch_size, time_steps, device, num_target_classes):
    assert labels.ndim == 2, labels.shape
    if num_target_classes is None:
        if labels.size == 0:
            num_target_classes = 0
        else:
            num_target_classes = int(np.max(labels[:, 1])+1)
    if labels.shape[1] == 2:
        # targets = torch.zeros(
        #     (batch_size, num_target_classes), dtype=torch.bool,
        #     device=device
        # )
        # targets[labels[:, 0], labels[:, 1]] = True
        targets = torch.sparse_coo_tensor(
            labels.T,
            np.ones(len(labels), dtype=np.float32),
            (batch_size, num_target_classes),
            device=device,
        )
    else:
        assert labels.shape[1] == 4, labels.shape
        labels[:, 2:] = np.minimum(labels[:, 2:], time_steps)

        # n_indices = (labels[:, 3]-labels[:, 2]).sum()
        # indices = np.zeros((3, n_indices))
        # time_range = np.arange(time_steps, dtype=int)
        # i = 0
        # for b, k, t_on, t_off in labels:
        #     if t_off >= len(time_range):
        #         time_range = np.arange(t_off, dtype=int)
        #     indices[0][i:i+t_off-t_on] = b
        #     indices[1][i:i+t_off-t_on] = k
        #     indices[2][i:i+t_off-t_on] = time_range[t_on:t_off]
        #     i += t_off - t_on
        # assert i == n_indices, (i, n_indices)
        if len(labels) == 0:
            indices = np.zeros((3,0), dtype=int)
        else:
            batch_indices = labels[:, 0, None]
            class_indices = labels[:, 1, None]
            event_onsets = labels[:, 2, None]
            event_offsets = labels[:, 3, None]
            max_len = np.max(event_offsets - event_onsets)
            time_indices = event_onsets + np.arange(max_len)
            batch_indices, class_indices, time_indices, event_offsets = np.broadcast_arrays(batch_indices, class_indices, time_indices, event_offsets)
            relevant_idx = time_indices < event_offsets
            indices = np.array([batch_indices[relevant_idx], class_indices[relevant_idx], time_indices[relevant_idx]])

        # targets = torch.zeros(
        #     (batch_size, num_target_classes, time_steps), dtype=torch.bool,
        #     device=device
        # )
        # targets[indices[0], indices[1], indices[2]] = True
        targets = torch.sparse_coo_tensor(
            indices,
            np.ones(len(indices[0]), dtype=np.float32),
            (batch_size, num_target_classes, time_steps),
            device=device,
        )
    return targets


class SlidingWindowTransform(Module):
    def __init__(
            self,
            shift: int,
            window_length: int,
            *,
            fading: typing.Optional[typing.Union[bool, str]] = 'full',
            pad: bool = True,
            # data augmentation
            time_warping_anchor_sampling_fn=None,
            time_warping_anchor_shift_sampling_fn=None,
    ):
        super().__init__()
        self.shift = shift
        self.window_length = window_length
        self.fading = fading
        self.pad = pad
        self.time_warping_anchor_sampling_fn = time_warping_anchor_sampling_fn
        self.time_warping_anchor_shift_sampling_fn = time_warping_anchor_shift_sampling_fn

    def _transform(self, x, shift, fading, pad):
        raise NotImplementedError

    def forward(
            self, x, sequence_lengths=None, sample_indices=None, *,
            time_warping_params=None, return_time_warping_params=False,
    ):
        """
        Performs stft

        Args:
            x: time signal

        Returns:

        """
        if self.training and (self.time_warping_anchor_sampling_fn is not None or self.time_warping_anchor_shift_sampling_fn is not None):
            assert callable(self.time_warping_anchor_sampling_fn), type(self.time_warping_anchor_sampling_fn)
            assert callable(self.time_warping_anchor_shift_sampling_fn), type(self.time_warping_anchor_shift_sampling_fn)
            expected_frames = _samples_to_stft_frames(
                x.shape[-1], self.window_length, self.shift, pad=self.pad, fading=self.fading,
            )
            if time_warping_params is None:
                time_warping_anchor = self.time_warping_anchor_sampling_fn()
                time_warping_anchor_shift = self.time_warping_anchor_shift_sampling_fn()
                time_warping_params = (time_warping_anchor, time_warping_anchor_shift)
            else:
                time_warping_anchor, time_warping_anchor_shift = time_warping_params
            warp_factor = (time_warping_anchor + time_warping_anchor_shift) / time_warping_anchor
            segment_shift = round(self.shift / warp_factor)
            x, sequence_lengths, sample_indices = self.pad_audio(x, sequence_lengths, sample_indices)
            num_samples = x.shape[-1]
            anchor_sample = (num_samples - self.window_length + self.shift) * time_warping_anchor + self.window_length - self.shift
            anchor_frame = round((anchor_sample - self.window_length + segment_shift) / segment_shift)
            anchor_sample = anchor_frame * segment_shift + self.window_length - segment_shift

            remaining_frames = expected_frames - anchor_frame
            segment_shifts = [
                segment_shift,
                ceil((num_samples - anchor_sample) / (remaining_frames + 1 - self.pad - 1e-6)),
            ]

            segment_onsets = [0, anchor_sample - self.window_length + segment_shifts[1]]
            segment_lengths = [anchor_sample, num_samples - segment_onsets[1]]

            y = []
            seq_lens = []
            frame_indices = []
            for i, (onset, seg_len, shift) in enumerate(zip(segment_onsets, segment_lengths, segment_shifts)):
                offset = onset + seg_len
                y.append(self._transform(
                    x[..., onset:offset],
                    shift=shift,
                    pad=(i == 1) and self.pad,
                    fading=None,
                ))
                if sequence_lengths is not None:
                    seq_lens.append(
                        np.minimum(
                            _samples_to_stft_frames(
                                np.minimum(np.maximum(np.array(sequence_lengths)-onset, 0), seg_len),
                                self.window_length, shift,
                                pad=self.pad, fading=None,
                            ),
                            y[-1].shape[-2]
                        )
                    )
                if sample_indices is not None:
                    frame_indices.append(
                        np.minimum(
                            sample_index_to_stft_frame_index(
                                np.minimum(np.maximum(np.array(sample_indices)-onset, 0), seg_len) + shift//2,  # ceil
                                self.window_length, shift, fading=None,
                            ),
                            y[-1].shape[-2]
                        )
                    )
            # print(expected_frames, y[0].shape[2] + y[1].shape[2], anchor_frame, y[0].shape[2], flush=True)
            y = torch.cat(y, dim=-2)
            if sequence_lengths is not None:
                sequence_lengths = np.sum(seq_lens, 0)
            if sample_indices is None:
                if return_time_warping_params:
                    return y, sequence_lengths, time_warping_params
                return y, sequence_lengths
            frame_indices = np.sum(frame_indices, 0)
        else:
            assert time_warping_params is None
            y = self._transform(
                x,
                shift=self.shift,
                fading=self.fading,
                pad=self.pad
            )  # (..., T, F)
            if sequence_lengths is not None:
                sequence_lengths = _samples_to_stft_frames(
                    np.array(sequence_lengths),
                    self.window_length, self.shift,
                    pad=self.pad, fading=self.fading,
                )
            if sample_indices is None:
                if return_time_warping_params:
                    return y, sequence_lengths, time_warping_params
                return y, sequence_lengths
            frame_indices = sample_index_to_stft_frame_index(
                np.array(sample_indices)+self.shift//2,   # ceil
                self.window_length, self.shift, fading=self.fading
            )
        if return_time_warping_params:
            return y, sequence_lengths, frame_indices, time_warping_params
        return y, sequence_lengths, frame_indices

    def pad_audio(self, audio, sequence_lengths, sample_indices):
        pad_width = 0
        if self.fading == "full":
            pad_width = self.window_length - self.shift
        elif self.fading == "half":
            pad_width += (self.window_length - self.shift)//2
        elif self.fading is not None:
            raise ValueError(f'Invalid fading {self.fading}.')
        if pad_width > 0:
            audio = torch.cat(
                (
                    torch.zeros((*audio.shape[:-1], pad_width), device=audio.device),
                    audio,
                    torch.zeros((*audio.shape[:-1], pad_width), device=audio.device)
                ),
                dim=-1
            )
        if sequence_lengths is not None:
            sequence_lengths = np.array(sequence_lengths) + 2*pad_width
        if sample_indices is not None:
            sample_indices = np.array(sample_indices) + pad_width
        return audio, sequence_lengths, sample_indices

    def inverse(self, x, num_samples=None):
        """
        Computes inverse stft

        Args:
            x: stft

        Returns:

        """
        #  x: (C, T, F)
        return istft(
            x,
            size=self.size,
            shift=self.shift,
            window_length=self.window_length,
            window=self.window,
            symmetric_window=self.symmetric_window,
            fading=self.fading,
            num_samples=num_samples,
        )


class STFT(SlidingWindowTransform):
    """
    Transforms audio data to STFT. Also allows to invert stft.

    >>> stft = STFT(160, 512, window_length=480, fading='half', pad=True)
    >>> audio_data = torch.zeros((3,8000))
    >>> x, seq_len, frame_indices = stft(audio_data, [8000, 4000, 2000], [[1000, 4000], [5000, 6000]])
    >>> x.shape
    torch.Size([3, 50, 257])
    >>> seq_len
    array([50, 25, 13])
    >>> frame_indices
    array([[ 6, 25],
           [31, 38]])
    >>> stft = STFT(160, 512, window_length=480, fading=None, pad=False)
    >>> audio_data = torch.zeros((3,4320))
    >>> x, seq_len, frame_indices = stft(audio_data, [4320, 3000, 2000], [[500, 2720], [2500, 3360], [3500, 4320]])
    >>> x.shape
    torch.Size([3, 25, 257])
    >>> seq_len
    array([25, 16, 10])
    >>> frame_indices
    array([[ 2, 16],
           [15, 20],
           [21, 26]])
    >>> stft = STFT(160, 512, window_length=480, fading='half', pad=True, time_warping_anchor_sampling_fn = lambda: .5, time_warping_anchor_shift_sampling_fn = lambda: .0)
    >>> audio_data = torch.zeros((3,8000))
    >>> x, seq_len, frame_indices = stft(audio_data, [8000, 4000, 2000], [[1000, 4000], [5000, 6000]])
    >>> x.shape
    torch.Size([3, 50, 257])
    >>> seq_len
    array([50, 25, 13])
    >>> frame_indices
    array([[ 6, 25],
           [31, 38]])
    >>> stft = STFT(160, 512, window_length=480, fading='half', pad=True, time_warping_anchor_sampling_fn = lambda: .4, time_warping_anchor_shift_sampling_fn = lambda: .2)
    >>> audio_data = torch.zeros((3,8000))
    >>> x, seq_len, frame_indices = stft(audio_data, [8000, 4000, 2000], [[1000, 4000], [5000, 6000]])
    >>> x.shape
    torch.Size([3, 50, 257])
    >>> seq_len
    array([50, 33, 19])
    >>> frame_indices
    array([[ 9, 32],
           [36, 41]])
    """

    def __init__(
            self,
            shift: int,
            size: int,
            *,
            window: [str, typing.Callable] = 'blackman',
            window_length: int = None,
            fading: typing.Optional[typing.Union[bool, str]] = 'full',
            pad: bool = True,
            symmetric_window: bool = False,
            # data augmentation
            time_warping_anchor_sampling_fn=None,
            time_warping_anchor_shift_sampling_fn=None,
    ):
        super().__init__(
            shift=shift,
            window_length=size if window_length is None else window_length,
            fading=fading,
            pad=pad,
            time_warping_anchor_sampling_fn=time_warping_anchor_sampling_fn,
            time_warping_anchor_shift_sampling_fn=time_warping_anchor_shift_sampling_fn,
        )
        self.size = size
        self.window = window
        self.symmetric_window = symmetric_window

    def _transform(self, x, shift, fading, pad):
        return stft(
            x,
            size=self.size,
            shift=shift,
            window_length=self.window_length,
            window=self.window,
            symmetric_window=self.symmetric_window,
            axis=-1,
            pad=pad,
            fading=fading,
        )

    def inverse(self, x, num_samples=None):
        """
        Computes inverse stft

        Args:
            x: stft

        Returns:

        """
        #  x: (C, T, F)
        return istft(
            x,
            size=self.size,
            shift=self.shift,
            window_length=self.window_length,
            window=self.window,
            symmetric_window=self.symmetric_window,
            fading=self.fading,
            num_samples=num_samples,
        )


class WaveformConv(SlidingWindowTransform):
    def __init__(
            self,
            shift: int,
            window_length: int,
            out_channels: int,
            *,
            fading: typing.Optional[typing.Union[bool, str]] = 'full',
            pad: bool = True,
            # data augmentation
            time_warping_anchor_sampling_fn=None,
            time_warping_anchor_shift_sampling_fn=None,
    ):
        super().__init__(
            shift=shift,
            window_length=window_length,
            fading=fading,
            pad=pad,
            time_warping_anchor_sampling_fn=time_warping_anchor_sampling_fn,
            time_warping_anchor_shift_sampling_fn=time_warping_anchor_shift_sampling_fn,
        )
        self.filters = nn.Parameter(
            torch.randn((out_channels, 1, window_length)),
            requires_grad=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.filters, gain=1.)

    def _transform(self, x, shift, fading, pad):
        pad_widths = [0, 0]
        if fading == "full":
            pad_widths[0] += self.window_length - self.shift
            pad_widths[1] += self.window_length - shift
        elif fading == "half":
            pad_widths[0] += (self.window_length - self.shift)//2
            pad_widths[1] += ceil((self.window_length - shift)/2)
        elif fading is not None:
            raise ValueError(f'Invalid fading {self.fading}.')
        if pad:
            pad_width = (shift - (x.shape[-1] + pad_widths[0] + pad_widths[1] + shift - self.window_length)) % shift
            if pad_width < shift:
                pad_widths[1] += pad_width
        if sum(pad_widths) > 0:
            x = torch.cat(
                (
                    torch.zeros((*x.shape[:-1], pad_widths[0]), device=x.device),
                    x,
                    torch.zeros((*x.shape[:-1], pad_widths[1]), device=x.device)
                ),
                dim=-1
            )
        b, c, t = x.shape
        x = rearrange(x, 'b c t -> (b c) t')[:, None]
        y = conv1d(x, self.filters, stride=shift)
        return rearrange(y, '(b c) n t -> b c t n', b=b, c=c)


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


class PCEN(Module):
    def __init__(
            self,
            in_frequencies,
            out_channels_per_in_channel=1,
            lp_filter_coeff=0.05,  # s
            lp_filter_length=100,
            gain=.98,  # alpha
            eps=1e-6,
            power=.5,  # r
            bias=2,  # delta
            trainable=False,
    ):
        """

        Args:
            in_frequencies:
            lp_filter_coeff:
            lp_filter_length:
            gain:
            eps:
            power:
            bias:
            trainable:

        >>> sample_rate = 16000
        >>> highest_frequency = sample_rate/2
        >>> mel_transform = MelTransform(sample_rate, 512, 40, log=False)
        >>> spec = torch.rand((3, 2, 100, 257))
        >>> melspec = mel_transform(spec)
        >>> melspec.shape
        >>> pcen = PCEN(40,2)
        >>> pcen_spec = pcen(melspec.transpose(-2,-1))
        >>> pcen_spec.shape

        """
        super().__init__()
        self.in_frequencies = in_frequencies
        self.out_channels_per_in_channel = out_channels_per_in_channel
        lp_filter_coeff = torch.tensor(
            to_list(lp_filter_coeff, out_channels_per_in_channel),
            dtype=torch.float32
        ).repeat(in_frequencies).unsqueeze(-1)
        self.lp_filter_coeff = nn.Parameter(
            lp_filter_coeff, requires_grad=trainable,
        )
        self.lp_filter = nn.Parameter(
            torch.arange(lp_filter_length).flip(0), requires_grad=False,
        )
        gain = torch.tensor(
            to_list(gain, out_channels_per_in_channel), dtype=torch.float32
        ).repeat(in_frequencies).unsqueeze(-1)
        self.gain = nn.Parameter(gain, requires_grad=trainable)
        self.eps = eps
        power = torch.tensor(
            to_list(power, out_channels_per_in_channel), dtype=torch.float32
        ).repeat(in_frequencies).unsqueeze(-1)
        self.power = nn.Parameter(power, requires_grad=trainable)
        bias = torch.tensor(
            to_list(bias, out_channels_per_in_channel), dtype=torch.float32
        ).repeat(in_frequencies).unsqueeze(-1)
        self.bias = nn.Parameter(bias, requires_grad=trainable)

    def forward(self, x):
        assert x.dim() == 4, x.shape
        b, c, f, t = x.shape
        x = rearrange(x, 'b c f t -> (b c) f t')
        lp_filter_coeff = torch.clip(self.lp_filter_coeff, min=0., max=1.)
        lp_filter = (1-lp_filter_coeff)**self.lp_filter
        lp_filter = lp_filter / torch.sum(lp_filter, dim=1, keepdim=True)
        m = conv1d(
            torch.cat(
                (torch.zeros((b*c, f, lp_filter.shape[-1]-1), device=x.device), x),
                dim=-1,
            ),
            lp_filter[:, None],
            groups=x.shape[1],
        )
        norm = torch.cat(
            (
                torch.cumsum(lp_filter.flip(-1)[..., :m.shape[-1]], dim=-1),
                torch.ones_like(m[0,:,lp_filter.shape[-1]:])
            ),
            dim=-1,
        )
        m = m / norm
        x = x.repeat_interleave(self.out_channels_per_in_channel, dim=1)
        gain = torch.clip(self.gain, min=0.)
        bias = torch.clip(self.bias, min=0.)
        power = torch.clip(self.power, min=0.)
        # print()
        # print(rearrange(lp_filter_coeff.detach()[...,0], '(f n) -> f n', f=f).mean(0))
        # print(rearrange(gain.detach()[...,0], '(f n) -> f n', f=f).mean(0))
        # print(rearrange(bias.detach()[...,0], '(f n) -> f n', f=f).mean(0))
        # print(rearrange(power.detach()[...,0], '(f n) -> f n', f=f).mean(0))
        y = (x / ((m + self.eps)**gain) + bias) ** power - bias ** power
        return rearrange(y, '(b c) (f n) t -> b (c n) f t', b=b, c=c, f=f, n=self.out_channels_per_in_channel)


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


class NormalizedImageExtractor(nn.Module):
    def __init__(
            self,
            image_shape,
            norm_statistics_axis='bft', norm_eps=1e-5, batch_norm=False,
            num_target_classes=None,
            # augmentation
            random_horizontal_flip=False, random_vertical_flip=False,
            max_mixback_scale=0., mixback_buffer_size=1,
            blur_sigma=0, blur_kernel_size=5,
            mixup_prob=0., mixup_alpha=2., mixup_beta=1., mixup_target_threshold=None,
            max_noise_scale=0.,
            n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=1.,
            min_masked_time_steps=0, min_masked_time_rate=0.,
            n_frequency_masks=0,
            max_masked_frequency_bands=20, max_masked_frequency_rate=1.,
            min_masked_frequency_bands=0, min_masked_frequency_rate=0.,
    ):
        super().__init__()

        self.image_shape = image_shape
        norm_cls = Normalization if batch_norm else InputNormalization
        self.norm = norm_cls(
            data_format='bcft',
            shape=(None, *image_shape),
            statistics_axis=norm_statistics_axis,
            shift=True,
            scale=True,
            eps=norm_eps,
            independent_axis=None,
            momentum=None,
        )

        # augmentation
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        if blur_sigma > 0:
            self.blur = GaussianBlur2d(
                kernel_size=blur_kernel_size,
                sigma_sampling_fn=TruncatedExponential(
                    loc=.1, scale=blur_sigma, truncation=blur_kernel_size
                )
            )
        else:
            self.blur = None

        if max_mixback_scale > 0.:
            self.mixback = MixBack(max_mixback_scale, mixback_buffer_size)
        else:
            self.mixback = None

        assert 0 <= mixup_prob <= 1, mixup_prob
        if mixup_prob > 0:
            self.mixup = Mixup(p=mixup_prob, alpha=mixup_alpha, beta=mixup_beta, target_threshold=mixup_target_threshold)
        else:
            self.mixup = None

        if max_noise_scale > 0.:
            self.noise = AdditiveNoise(max_noise_scale)
        else:
            self.noise = None

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
                min_masked_steps=min_masked_time_steps,
                min_masked_rate=min_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_frequency_masks > 0:
            self.mel_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_bands,
                max_masked_rate=max_masked_frequency_rate,
                min_masked_steps=min_masked_frequency_bands,
                min_masked_rate=min_masked_frequency_rate,
            )
        else:
            self.mel_masking = None

        self.num_target_classes = num_target_classes

    def reset(self):
        self.norm.reset_parameters()
        if self.mixback is not None:
            self.mixback.reset()

    def freeze(self):
        self.norm.freeze(freeze_stats=True)

    def forward(self, x, seq_len=None, labels=None):
        with torch.no_grad():
            if labels is None:
                targets = None
            elif isinstance(labels, np.ndarray):
                targets = _labels_to_targets(
                    labels, x.shape[0], x.shape[3], device=x.device,
                    num_target_classes=self.num_target_classes,
                )
            elif isinstance(labels, (list, tuple)):
                targets = [
                    _labels_to_targets(
                        labels_i, x.shape[0], x.shape[3], device=x.device,
                        num_target_classes=self.num_target_classes
                    ) for labels_i in labels
                ]

            x = x.flip(2)
            if self.training and self.random_vertical_flip and np.random.choice(2):
                x = x.flip(2)
            if self.training and self.random_horizontal_flip and np.random.choice(2):
                x = x.flip(3)

            if self.blur is not None:
                x = self.blur(x)

            x = self.norm(x, sequence_lengths=seq_len)

            if self.mixback is not None:
                x = self.mixback(x)

            if self.mixup is not None:
                x, seq_len, targets = self.mixup(
                    x, seq_len=seq_len, targets=targets
                )

            if self.noise is not None:
                # print(torch.std(x, dim=-1))
                x = self.noise(x)

            if self.time_masking is not None:
                x, _ = self.time_masking(x, seq_len=seq_len)

            if self.mel_masking is not None:
                x, _ = self.mel_masking(x)

        if labels is None:
            return x, seq_len
        return x, seq_len, targets

    def inverse(self, x):
        return self.norm.inverse(x).transpose(-2, -1)


class NormalizedImageLogMelsExtractor(nn.Module):
    def __init__(
            self,
            image_shape, number_of_filters,
            lowest_frequency=50, highest_frequency=None, htk_mel=True,
            norm_statistics_axis='bft', norm_eps=1e-5, batch_norm=False,
            num_target_classes=None,
            # augmentation
            random_horizontal_flip=False, random_vertical_flip=False,
            max_mixback_scale=0., mixback_buffer_size=1,
            blur_sigma=0, blur_kernel_size=5,
            mixup_prob=0., mixup_alpha=2., mixup_beta=1., mixup_target_threshold=None,
            max_noise_scale=0.,
            n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=1.,
            min_masked_time_steps=0, min_masked_time_rate=0.,
            n_frequency_masks=0,
            max_masked_frequency_bands=20, max_masked_frequency_rate=1.,
            min_masked_frequency_bands=0, min_masked_frequency_rate=0.,
    ):
        super().__init__()

        self.image_shape = image_shape
        self.number_of_filters = number_of_filters
        self.mel_transform = MelTransform(
            sample_rate=16000,
            stft_size=image_shape[1],
            number_of_filters=number_of_filters,
            lowest_frequency=lowest_frequency,
            highest_frequency=highest_frequency,
            htk_mel=htk_mel,
            log=True,
        )
        norm_cls = Normalization if batch_norm else InputNormalization
        self.norm = norm_cls(
            data_format='bcft',
            shape=(None, image_shape[0], self.number_of_filters, image_shape[2]),
            statistics_axis=norm_statistics_axis,
            shift=True,
            scale=True,
            eps=norm_eps,
            independent_axis=None,
            momentum=None,
        )

        # augmentation
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        if blur_sigma > 0:
            self.blur = GaussianBlur2d(
                kernel_size=blur_kernel_size,
                sigma_sampling_fn=TruncatedExponential(
                    loc=.1, scale=blur_sigma, truncation=blur_kernel_size
                )
            )
        else:
            self.blur = None

        if max_mixback_scale >= 0.:
            self.mixback = MixBack(max_mixback_scale, mixback_buffer_size)
        else:
            self.mixback = None

        assert 0 <= mixup_prob <= 1, mixup_prob
        if mixup_prob > 0:
            self.mixup = Mixup(p=mixup_prob, alpha=mixup_alpha, beta=mixup_beta, target_threshold=mixup_target_threshold)
        else:
            self.mixup = None

        if max_noise_scale > 0.:
            self.noise = AdditiveNoise(max_noise_scale)
        else:
            self.noise = None

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
                min_masked_steps=min_masked_time_steps,
                min_masked_rate=min_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_frequency_masks > 0:
            self.mel_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_bands,
                max_masked_rate=max_masked_frequency_rate,
                min_masked_steps=min_masked_frequency_bands,
                min_masked_rate=min_masked_frequency_rate,
            )
        else:
            self.mel_masking = None

        self.num_target_classes = num_target_classes

    def reset(self):
        self.norm.reset_parameters()
        if self.mixback is not None:
            self.mixback.reset()

    def freeze(self):
        self.norm.freeze(freeze_stats=True)

    def forward(self, x, seq_len=None, labels=None):
        with torch.no_grad():
            if labels is None:
                targets = None
            elif isinstance(labels, np.ndarray):
                targets = _labels_to_targets(
                    labels, x.shape[0], x.shape[3], device=x.device,
                    num_target_classes=self.num_target_classes,
                )
            elif isinstance(labels, (list, tuple)):
                targets = [
                    _labels_to_targets(
                        labels_i, x.shape[0], x.shape[3], device=x.device,
                        num_target_classes=self.num_target_classes
                    ) for labels_i in labels
                ]

            if self.training and self.random_vertical_flip and np.random.choice(2):
                x = x.flip(2)
            if self.training and self.random_horizontal_flip and np.random.choice(2):
                x = x.flip(3)
            x = torch.fft.rfft(x, dim=2).transpose(-2, -1)
            x = self.mel_transform((x.real**2 + x.imag**2)).transpose(-2, -1)

            if self.blur is not None:
                x = self.blur(x)

            x = self.norm(x, sequence_lengths=seq_len)

            if self.mixback is not None:
                x = self.mixback(x)

            if self.mixup is not None:
                x, seq_len, targets = self.mixup(
                    x, seq_len=seq_len, targets=targets
                )

            if self.noise is not None:
                # print(torch.std(x, dim=-1))
                x = self.noise(x)

            if self.time_masking is not None:
                x, _ = self.time_masking(x, seq_len=seq_len)

            if self.mel_masking is not None:
                x, _ = self.mel_masking(x)

        if labels is None:
            return x, seq_len
        return x, seq_len, targets

    def inverse(self, x):
        return self.norm.inverse(x).transpose(-2, -1)
