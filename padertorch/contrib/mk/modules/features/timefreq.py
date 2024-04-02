from functools import partial
import typing as tp
import warnings

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import librosa

from paderbox.transform.module_fbank import hz2mel, mel2hz

import padertorch as pt
from padertorch.modules.normalization import InputNormalization
from padertorch.ops import STFT as _STFT

from padertorch.contrib.je.modules.features import get_fbanks
from padertorch.contrib.je.modules.augment import Mask
from padertorch.contrib.mk.typing import TSeqLen, TSeqReturn

__all__ = [
    'Sequential',
    'Logarithm',
    'STFT',
    'MelTransform',
    'MFCC',
]


class Sequential(pt.Module):
    def forward(
        self, x: Tensor, sequence_lengths: TSeqLen = None
    ) -> TSeqReturn:
        return x, sequence_lengths


class Logarithm(pt.Module):
    """Take the logarithm of the input features.

    Args:
        log_base (str, int, float, bool, optional): Base of the logarithm.
            If False, disable the logarithm. If None, use natural logarithm.
            Defaults to 10.
        eps (float, optional): Small value to add to the input before taking
            the logarithm. Defaults to 1e-5.
    """
    def __init__(
        self,
        log_base: tp.Union[None, str, int, float, bool] = 10,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        if log_base is None or log_base == 'e':
            self.log_fn = torch.log
            self.power_fn = torch.exp
        elif log_base == 10.0:
            self.log_fn = torch.log10
            self.power_fn = partial(torch.pow, 10)
        elif log_base == 2.0:
            self.log_fn = torch.log2
            self.power_fn = partial(torch.pow, 2)
        elif log_base is False:
            self.log_fn = nn.Sequential()
            self.power_fn = nn.Sequential()
        else:
            raise ValueError(f'log_base {self.log_base} is not supported')

    def forward(self, x: Tensor) -> Tensor:
        x = self.log_fn(torch.maximum(
            torch.tensor(self.eps).to(x.device), x
        ))
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.power_fn(x)


class STFT(_STFT):
    """Extract STFT features from audio signals.

    Args:
        size (int): See paderbox.transform.module_stft.stft.
        shift (int): See paderbox.transform.module_stft.stft.
        window (str, callable): See paderbox.transform.module_stft.stft.
        window_length (int, optional): See paderbox.transform.module_stft.stft.
        fading (bool, str): See paderbox.transform.module_stft.stft.
        pad (bool): See paderbox.transform.module_stft.stft.
        symmetric_window (bool): See paderbox.transform.module_stft.stft.
        complex_representation (str): See padertorch.ops._stft.STFT.
        spectrogram (bool): If True, return the magnitude spectrogram. Defaults
            to False.
        power (float): If `spectrogram` is True, raise magnitude to `power`.
            If 2, computes the power spectrogram. Defaults to 1.
        scale_spec (bool): If True, scale the spectrogram by the reciprocal of
            STFT size. Set to True to make compatible to
            paderbox.transform.module_fbank.fbank. Defaults to False.
        log_base (str, int, float, bool, optional): See Logarithm. Defaults to
            False.
        sequence_last (bool): If True, move the sequence axis to the last
            position. Defaults to True.
        normalization (InputNormalization, optional): InputNormalization
            instance to perform z-normalization. Defaults to None.
    """
    def __init__(
        self,
        size: int = 1024,
        shift: int = 256,
        *,
        window: tp.Union[str, tp.Callable] = 'blackman',
        window_length: tp.Optional[int] = None,
        fading: tp.Optional[tp.Union[bool, str]] = 'full',
        pad: bool = True,
        symmetric_window: bool = False,
        complex_representation: str = 'complex',
        spectrogram: bool = False,
        power: float = 1.,
        scale_spec: bool = False,
        log_base: tp.Union[None, str, int, float, bool] = False,
        sequence_last: bool = True,
        normalization: tp.Union[InputNormalization, None] = None,
    ):
        if not spectrogram and log_base:
            raise ValueError(
                'log_base can only be used with spectrogram=True'
            )
        super().__init__(
            size=size,
            shift=shift,
            window=window,
            window_length=window_length,
            fading=fading,
            pad=pad,
            symmetric_window=symmetric_window,
            complex_representation=complex_representation,
        )
        # Keep references to window and symmetric_window
        self.window = window
        self.symmetric_window = symmetric_window
        self.spectrogram = spectrogram
        self.power = power
        self.scale_spec = scale_spec
        self.log = Logarithm(log_base=log_base)
        self.sequence_last = sequence_last
        self.normalization = normalization

    def __call__(
        self,
        inputs: Tensor,
        sequence_lengths: TSeqLen = None,
    ) -> TSeqReturn:
        """

        Args:
            inputs (Tensor): Batch of time-domain signals of shape
                (batch, time) or (batch, channels, time).
            sequence_lengths (list, optional): List of number of samples of
                time signals in `inputs`. Defaults to None.

        Returns:
            encoded (Tensor): Spectrogram of shape (batch, bins, time) if
                `spectrogram` is True else STFT of shape
                    - (batch, bins, time) if `complex_representation` is 'complex',
                    - (batch, bins, time, 2) if `complex_representation` is 'stacked', or
                    - (batch, channels, 2*bins, time) if `complex_representation` is 'concat'.
                If `sequence_last` is False, the time and bins axis are swapped.
            sequence_lengths (list, optional): List of number of frames of
                spectrograms in `encoded` if input `sequence_lengths` is not
                None.
        """
        encoded = super().__call__(inputs)
        if self.spectrogram:
            if self.complex_representation == 'complex':
                encoded = torch.abs(encoded)
            elif self.complex_representation == 'stacked':
                encoded = encoded.pow(2).sum(-1).sqrt()
            else:
                real, imag = torch.split(
                    encoded, encoded.shape[-1] // 2, dim=-1
                )
                encoded = (real.pow(2) + imag.pow(2)).sqrt()
            encoded = encoded ** self.power
            if self.scale_spec:
                encoded /= self.size
        encoded = self.log(encoded)
        if sequence_lengths is not None:
            sequence_lengths = self.samples_to_frames(
                np.asarray(sequence_lengths)
            )
        if self.sequence_last:
            if (
                self.complex_representation == 'stacked'
                and not self.spectrogram
            ):
                encoded = encoded.transpose(-2, -3)
            else:
                encoded = encoded.transpose(-2, -1)  # (..., bins, time)
        if self.normalization is not None:
            encoded = self.normalization(
                encoded, sequence_lengths=sequence_lengths
            )
        return encoded, sequence_lengths

    def inverse(
        self, x: torch.Tensor, sequence_lengths: TSeqLen = None
    ) -> torch.Tensor:
        if self.normalization is not None:
            x = self.normalization.inverse(x, sequence_lengths=sequence_lengths)
        x = self.log.inverse(x)
        # TODO: ISTFT and phase reconstruction
        return x

    def __repr__(self):
        return (
            f"STFT(size={self.size}, shift={self.shift},\n"
            f"window={self.window}, window_length={self.window_length},\n"
            f"fading={self.fading}, pad={self.pad},\n"
            f"symmetric_window={self.symmetric_window},\n"
            f"complex_representation={self.complex_representation},\n"
            f"spectrogram={self.spectrogram}, power={self.power},\n"
            f"scale_spec={self.scale_spec})"
        )


class MelTransform(pt.Module):
    """Extract mel spectrogram from audio signals.

    Args:
        sampling_rate (int):
        stft_size (int):
        stft (STFT, optional): STFT instance to compute the linear spectrogram.
            If None, inputs are expected to be the linear spectrogram.
        number_of_filters (int): Number of mel filters.
        lowest_frequency (int, float): Lowest frequency of the mel filterbank.
        highest_frequency (int, float): Highest frequency of the mel filterbank.
        htk_mel (bool): If True use HTK's hz to mel conversion definition else
            use Slaney's definition (cf. librosa.mel_frequencies doc).
        norm (str, int): If 'slaney', normalize the mel filterbank to be
            consistent with librosa's mel filterbank. If int, normalize the
            mel filterbank with the given norm. Defaults to 'slaney'.
        log_base (str, int, float, bool): Base of the logarithm. See Logarithm.
            Defaults to 10.
        use_state_dict (bool): If True, store mel filterbank in the state dict.
            Defaults to True.
        warping_fn (callable): Function to (randomly) remap fbank center
            frequencies.
        independent_axis (int, list): Axis for which independently warped
            filter banks are used.
        squeeze_channel_axis (bool): If True, squeeze the channel axis and
            always return a 3D tensor. Defaults to False.
        sequence_last (boo): If True, move the sequence axis to the last
            position. Defaults to True.
        normalization (InputNormalization, optional): InputNormalization
            instance to perform z-normalization. Defaults to None.
    """
    def __init__(
        self,
        sampling_rate: int,
        stft_size: int,
        stft: tp.Optional[STFT] = None,
        number_of_filters: int = 80,
        lowest_frequency: tp.Union[int, float] = 80,
        highest_frequency: tp.Union[int, float] = 7600,
        htk_mel: bool = False,
        norm: tp.Optional[tp.Union[str, int]] = 'slaney',
        log_base: tp.Union[None, str, int, float, bool] = 10,
        use_state_dict: bool = True,
        *,
        warping_fn=None,
        independent_axis: tp.Union[int, tp.Sequence[int]] = 0,
        squeeze_channel_axis: bool = False,
        sequence_last: bool = True,
        normalization: tp.Union[InputNormalization, None] = None,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.stft_size = stft_size

        self.stft = stft
        if self.stft is not None and not self.stft.spectrogram:
            raise ValueError(
                f'stft.spectrogram must be True but is {stft.spectrogram}'
            )

        self.number_of_filters = number_of_filters
        self.lowest_frequency = lowest_frequency
        self.highest_frequency = highest_frequency
        if self.highest_frequency is None:
            self.highest_frequency = self.sampling_rate // 2
        self.htk_mel = htk_mel
        self.norm = norm
        mel_basis = get_fbanks(
            sample_rate=sampling_rate,
            stft_size=stft_size,
            number_of_filters=number_of_filters,
            lowest_frequency=lowest_frequency,
            highest_frequency=highest_frequency,
            htk_mel=htk_mel,
        )
        if self.norm == 'slaney':
            # https://github.com/librosa/librosa/blob/main/librosa/filters.py#L243
            mel_basis = self._normalize(mel_basis)
        elif isinstance(self.norm, int):
            mel_basis = librosa.util.normalize(
                mel_basis, norm=self.norm, axis=-1
            )
        else:
            raise ValueError(f'Unknown norm: {self.norm}')

        mel_basis = torch.from_numpy(mel_basis.T).float()
        ifbanks = torch.linalg.pinv(mel_basis).float()
        if use_state_dict:
            self.mel_basis = torch.nn.Parameter(
                mel_basis, requires_grad=False
            )
            self.ifbanks = torch.nn.Parameter(
                ifbanks, requires_grad=False
            )
        else:
            self.mel_basis = mel_basis
            self.ifbanks = ifbanks

        self.log = Logarithm(log_base=log_base)

        self.warping_fn = warping_fn
        self.independent_axis = (
            [independent_axis] if np.isscalar(independent_axis)
            else independent_axis
        )

        self.squeeze_channel_axis = squeeze_channel_axis
        self.sequence_last = sequence_last
        self.normalization = normalization

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['stft'] = {
            'factory': STFT,
            'window': 'hann',
            'spectrogram': True,
            'size': config['stft_size'],
            'sequence_last': False,
        }

    def _normalize(self, mel_basis):
        """
        >>> from paderbox.transform.module_fbank import hz2mel, mel2hz
        >>> import librosa
        >>> f = mel2hz(
        ...     np.linspace(hz2mel(80, htk_mel=False), hz2mel(7600, htk_mel=False), 82),
        ...     htk_mel=False
        ... )
        >>> mel_f = librosa.filters.mel_frequencies(82, fmin=80, fmax=7600, htk=False)
        >>> np.allclose(f, mel_f)
        True
        """
        mel_f = mel2hz(
            np.linspace(
                hz2mel(self.lowest_frequency, htk_mel=self.htk_mel),
                hz2mel(self.highest_frequency, htk_mel=self.htk_mel),
                self.number_of_filters+2,
            ), htk_mel=self.htk_mel,
        )
        enorm = 2.0 / (
            mel_f[2: self.number_of_filters + 2]
            - mel_f[:self.number_of_filters]
        )
        mel_basis = mel_basis * enorm[:, None]
        return mel_basis.astype(np.float32)

    def forward(
        self,
        x: Tensor,
        sequence_lengths: TSeqLen = None,
    ) -> TSeqReturn:
        """Compute log-mel spectrogram from audio signals and linear spectrograms.
        
        Args:
            x (Tensor): Magnitude or power spectrogram of shape
                (batch, ..., time, bins) if `stft` is None else time signal
                of shape (batch, ..., time).
            sequence_lengths (list, optional): List of number of frames of  
                spectrograms or number of samples in `x`.
        Returns:
            x (Tensor): Mel spectrogram of shape
                (batch, ..., number_of_filters, time). If `sequence_last` is
                False, the time and number_of_filters axis are swapped.
            sequence_lengths (list, optional): List of number of frames of
                mel spectrograms in `x` if input `sequence_lengths` is not None.
        """
        x = x.float()

        if self.stft is not None:
            x, sequence_lengths = self.stft(x, sequence_lengths)
            if self.stft.sequence_last:
                x = x.transpose(-2, -1)

        if not self.training or self.warping_fn is None:
            x = torch.matmul(x, self.mel_basis.to(x.device))
        else:
            # Copied from padertorch.contrib.je.modules.features
            independent_axis = [
                ax if ax >= 0 else x.ndim + ax for ax in
                self.independent_axis
            ]
            assert all([ax < x.ndim - 1 for ax in independent_axis])
            size = [
                x.shape[i] if i in independent_axis else 1
                for i in range(x.ndim - 1)
            ]
            mel_basis = get_fbanks(
                sample_rate=self.sampling_rate,
                stft_size=self.stft_size,
                number_of_filters=self.number_of_filters,
                lowest_frequency=self.lowest_frequency,
                highest_frequency=self.highest_frequency,
                htk_mel=self.htk_mel,
                warping_fn=self.warping_fn,
                size=size,
            )
            mel_basis = torch.from_numpy(
                self._normalize(mel_basis)
            ).transpose(-2, -1).to(x.device)
            if mel_basis.shape[-3] == 1:
                x = torch.matmul(x, mel_basis.squeeze(-3))
            else:
                x = torch.matmul(x[..., None, :], mel_basis).squeeze(-2)

        x = self.log(x)

        if x.ndim == 4 and self.squeeze_channel_axis:
            x = x.squeeze(1)

        if self.sequence_last:
            x = x.transpose(-2, -1)  # (..., bins, time)

        if self.normalization is not None:
            x = self.normalization(x, sequence_lengths=sequence_lengths)

        return x, sequence_lengths

    def inverse(
        self, x: torch.Tensor, sequence_lengths: TSeqLen = None
    ) -> torch.Tensor:
        if self.normalization is not None:
            x = self.normalization.inverse(x, sequence_lengths=sequence_lengths)
        x = self.log.inverse(x)
        return x


class MFCC(pt.Module):
    def __init__(
        self,
        number_of_filters: int,
        transform: tp.Optional[tp.Union[MelTransform, STFT]] = None,
        axis: int = -1,
        channel_axis: int = 1,
        num_cep: tp.Union[None, int] = None,
        low_pass: bool = True,
        lifter_coeff: int = 0,
        normalization: tp.Union[InputNormalization, None] = None,
    ):
        """Extract mel-cepstral coefficients from audio.

        Args:
            number_of_filters: Number of filters in the filterbank.
            mel_transform: Optional `MelTransform` instance. If not None,
                expect time signal as input and compute the log (mel)
                spectrogram before extracting the cepstral coefficients.
            axis: Position of the frequency axis.
            channel_axis: Position of the channel axis. Can be set to None if
                the input has no channel axis.
            num_cep: Number of cepstral coefficients to keep. If None, all
                coefficients are kept.
            low_pass: If True and `num_cep` is not None, keep the lowest
                `num_cep` coefficients and discard the rest (default behavior).
                If False, keep the highest `number_of_filters-num_cep`
                coefficients (high-pass behavior).
            lifter_coeff: Liftering in the cepstral domain. See
                `paderbox.transform.module_mfcc`. If 0, no liftering is applied.
        """
        super().__init__()
        self.number_of_filters = number_of_filters
        self.transform = transform
        self.axis = axis
        self.channel_axis = channel_axis
        self.num_cep = num_cep
        self.low_pass = low_pass
        self.lifter_coeff = lifter_coeff
        self.normalization = normalization

    def _lifter(self, cepstra):
        # Adapted from paderbox.transform.module_mfcc
        if self.lifter_coeff > 0:
            axis = self.axis % cepstra.ndim
            n = torch.arange(self.num_cep)
            for _ in range(axis):
                n = n.unsqueeze(0)
            for _ in range(axis, cepstra.ndim-1):
                n = n.unsqueeze(-1)
            lift = (
                1 + (self.lifter_coeff / 2)
                * torch.sin(torch.pi * n / self.lifter_coeff).to(cepstra.device)
            )
            return lift * cepstra
        return cepstra

    def _pad(self, x, pad_width):
        pad_width = tuple(pad_width.flatten())
        if self.channel_axis is not None:
            channel_axis = self.channel_axis % x.ndim
            x_pad = []
            channels = torch.split(x, 1, dim=channel_axis)
            for channel in channels:
                x_pad.append(F.pad(
                    channel.squeeze(channel_axis), pad_width, mode='reflect'
                ))
            return torch.stack(x_pad, dim=channel_axis)
        else:
            return F.pad(x, pad_width, mode='reflect')

    def forward(
        self,
        x: torch.Tensor,
        sequence_lengths: tp.Optional[tp.List[int]] = None,
    ):
        """Compute mel-cepstral coefficients from log (mel) spectrogram
        
        Args:
            x: Log (mel) spectrogram of shape (..., number_of_filters, time) or
                (..., time, number_of_filters) if `mel_transform` is `None`
                else time signal of shape (batch, 1, time).
            sequence_lengths (list, optional): List of number of frames of  
                spectrograms or number of samples in `x`.
        """
        if self.transform is not None:
            x, sequence_lengths = self.transform(x, sequence_lengths)
        n_mels = x.shape[self.axis]
        num_cep = self.num_cep
        if num_cep is None:
            if self.low_pass:
                num_cep = n_mels
            else:
                num_cep = 0
        axis = self.axis % x.ndim
        pad_width = np.zeros((x.ndim - axis, 2), dtype='int32')
        pad_width[-1, 1] = n_mels - 2
        x_pad = self._pad(x, pad_width)
        x_mfcc = torch.fft.rfft(x_pad, axis=axis, norm='ortho').real
        if self.low_pass:
            indices = torch.arange(min(num_cep, n_mels)).to(x_mfcc.device)
        else:
            indices = torch.arange(num_cep, n_mels).to(x_mfcc.device)
        x_mfcc = torch.index_select(x_mfcc, self.axis, indices)
        x_mfcc = self._lifter(x_mfcc)
        if self.normalization is not None:
            x_mfcc = self.normalization(
                x_mfcc, sequence_lengths=sequence_lengths
            )
        return x_mfcc, sequence_lengths

    def inverse(self, x_mfcc: Tensor) -> Tensor:
        if self.normalization is not None:
            x_mfcc = self.normalization.inverse(x_mfcc)
        if self.lifter_coeff > 0:
            warnings.warn("Liftering is not inverted.", UserWarning)
        if self.num_cep is not None:
            shape = list(x_mfcc.shape)
            if self.low_pass:
                shape[self.axis] = self.number_of_filters - self.num_cep
                x_mfcc = torch.cat(
                    (x_mfcc, torch.zeros(shape).to(x_mfcc.device)),
                    dim=self.axis
                )
            else:
                shape[self.axis] = self.num_cep
                x_mfcc = torch.cat(
                    (torch.zeros(shape).to(x_mfcc.device), x_mfcc),
                    dim=self.axis
                )
        spect = torch.index_select(
            torch.fft.irfft(x_mfcc, axis=self.axis, norm='ortho'),
            self.axis, torch.arange(self.number_of_filters).to(x_mfcc.device)
        )
        return spect

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['transform'] = {'factory': MelTransform}
        if config['transform'] is not None:
            if (
                pt.configurable.import_class(config['transform']['factory'])
                == MelTransform
            ):
                config['number_of_filters'] = config['transform']\
                    ['number_of_filters']
            elif (
                pt.configurable.import_class(config['transform']['factory'])
                == STFT
            ):
                config['number_of_filters'] = config['transform']\
                    ['size'] // 2 + 1


class SpecAug(pt.Module):
    def __init__(
        self, feature_extractor,
        n_time_masks=0, max_masked_time_steps=70, max_masked_time_rate=1.,
        n_frequency_masks=0,
        max_masked_frequency_bands=20,
        max_masked_frequency_rate=1.,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor

        if n_time_masks > 0:
            self.time_masking = Mask(
                axis=-1, n_masks=n_time_masks,
                max_masked_steps=max_masked_time_steps,
                max_masked_rate=max_masked_time_rate,
            )
        else:
            self.time_masking = None

        if n_frequency_masks > 0:
            self.frequency_masking = Mask(
                axis=-2, n_masks=n_frequency_masks,
                max_masked_steps=max_masked_frequency_bands,
                max_masked_rate=max_masked_frequency_rate,
            )
        else:
            self.frequency_masking = None

    def forward(self, x, sequence_lengths=None):
        x, sequence_lengths = self.feature_extractor(x, sequence_lengths)
        if self.time_masking is not None:
            x = self.time_masking(x)
        if self.frequency_masking is not None:
            x = self.frequency_masking(x)
        return x, sequence_lengths
