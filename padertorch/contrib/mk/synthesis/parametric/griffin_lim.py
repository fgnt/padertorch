import typing

import numpy as np
import torch
from paderbox.transform import STFT as pbSTFT
import padertorch as pt
from padertorch.ops import STFT as ptSTFT

from ..base import Synthesis


__all__ = [
    'fast_griffin_lim',
    'FGLA',
]


def reshape_complex(signal, complex_representation):
    if complex_representation in (None, 'complex'):
        return signal
    if complex_representation == 'stacked':
        signal = torch.stack(
            (signal.real, signal.imag), dim=-1
        )
    else:
        signal = torch.cat(
            (signal.real, signal.imag), dim=-1
        )
    return signal


def griffin_lim_step(
    a: typing.Union[np.ndarray, torch.Tensor],
    reconstruction_stft: typing.Union[np.ndarray, torch.Tensor],
    stft: typing.Union[pbSTFT, ptSTFT],
    backend=None,
):
    """
    Args:
        a:
        reconstruction_stft:
        stft:
        backend:

    Returns:

    """
    if backend is None:
        if isinstance(a, np.ndarray):
            backend = np
        else:
            backend = torch

    # From paderbox.transform.module_phase_reconstruction
    reconstruction_angle = backend.angle(reconstruction_stft)
    proposal_spec = a * backend.exp(1.0j * reconstruction_angle)  # P_A

    audio = stft.inverse(
        reshape_complex(
            proposal_spec, getattr(stft, 'complex_representation', None)
        )
    )  # P_C
    stft_signal = stft(audio)
    if isinstance(stft_signal, np.ndarray):
        return stft_signal, audio
    if stft.complex_representation != 'complex':
        if stft.complex_representation == 'stacked':
            stft_signal = stft_signal[..., 0] + 1j * stft_signal[..., 1]
        else:
            size = stft_signal.shape[-1]
            stft_signal = (
                stft_signal[..., :size//2] + 1j * stft_signal[..., size//2:]
            )
    return stft_signal, audio


def fast_griffin_lim(
    a: typing.Union[np.ndarray, torch.Tensor],
    stft: [pbSTFT, ptSTFT],
    alpha=0.95,
    iterations=100,
    atol: float = 0.1,
    verbose=False,
    x=None,
):
    """Griffin-Lim algorithm with momentum for phase retrieval [1].

    >>> f_0 = 200  # Hz
    >>> f_s = 16_000  # Hz
    >>> t = np.linspace(0, 1, num=f_s)
    >>> sine = np.sin(2*np.pi*f_0*t)
    >>> sine.shape
    (16000,)
    >>> stft = STFT(256, 1024, window_length=None, window='hann', pad=True, fading='half')
    >>> mag_spec = np.abs(stft(sine))
    >>> mag_spec.shape
    (63, 513)
    >>> sine_hat = fast_griffin_lim(mag_spec, stft)
    >>> sine_hat.shape
    (16128,)

    [1]: Peer, Tal, Simon Welker, and Timo Gerkmann. "Beyond Griffin-LIM:
        Improved Iterative Phase Retrieval for Speech." 2022 International
        Workshop on Acoustic Signal Enhancement (IWAENC). IEEE, 2022.

    Args:
        a: Magnitude spectrogram of shape (*, num_frames, stft.size//2+1)
        stft: paderbox.transform.module_stft.STFT instance
        alpha: Momentum for GLA acceleration, where 0 <= alpha <= 1
        iterations: Number of optimization iterations
        atol:
        verbose: If True, print the reconstruction error after each iteration step
        x: Optional complex STFT output from a different phase retrieval algorithm
    """
    if isinstance(a, np.ndarray):
        backend = np
    else:
        backend = torch

    if x is None:
        # Random phase initialization
        if backend is np:
            angle = np.random.uniform(
                low=-np.pi, high=np.pi, size=a.shape
            )
        else:
            angle = torch.rand(a.shape).to(a.device) * 2 * torch.pi - torch.pi
    else:
        assert x.dtype in (np.complex64, np.complex128, torch.complex64), x.dtype
        angle = backend.angle(x)

    with torch.no_grad():
        x = a * backend.exp(1.0j * angle)
        y = x
        for n in range(iterations):
            x_, _ = griffin_lim_step(a, y, stft)
            y = x_ + alpha * (x_ - x)
            x = x_
            reconstruction_magnitude = backend.abs(x)
            diff = (backend.sqrt(
                backend.mean((reconstruction_magnitude - a) ** 2)
            ))
            if verbose:
                print(
                    'Reconstruction iteration: {}/{} RMSE: {} '.format(
                        n, iterations, diff
                    )
                )
            if diff < atol:
                break
    angle = backend.angle(x)
    x = a * backend.exp(1.0j * angle)
    signal = stft.inverse(
        reshape_complex(x, getattr(stft, 'complex_representation', None))
    )
    return signal


class FGLA(Synthesis):
    """Phase reconstruction using the Griffin-Lim algorithm (FGLA).
    """
    def __init__(
        self,
        sampling_rate: int,
        stft: typing.Union[pbSTFT, ptSTFT],
        alpha: float = .95,
        iterations: int = 30,
        atol: float = 0.1,
    ):
        """
        Args:
            sampling_rate: Sampling rate of the synthesized signal
            stft: paderbox or padertorch STFT instance that was used to obtain
                the magnitude spectrogram
            alpha: See fast_griffin_lim
            iterations: See fast_griffin_lim
            atol: See fast_griffin_lim
        """
        self.sampling_rate = sampling_rate
        self.stft = stft
        self.alpha = alpha
        self.iterations = iterations
        self.atol = atol

    def __call__(
        self,
        mag_spec: typing.Union[np.ndarray, torch.Tensor],
        sequence_lengths: typing.Optional[typing.List[int]] = None,
        target_sampling_rate: typing.Optional[int] = None,
    ) -> typing.Union[torch.Tensor, np.ndarray]:
        """
        Args:
            mag_spec: Magnitude spectrogram of shape
                (*, num_frames, stft.size//2+1)
            sequence_lengths: Ignored
            target_sampling_rate: If not None, resample to
                `target_sampling_rate`

        Returns: np.ndarray or torch.Tensor
            The synthesized waveform
        """
        del sequence_lengths
        if isinstance(mag_spec, np.ndarray) and isinstance(self.stft, ptSTFT):
            mag_spec = pt.data.example_to_device(mag_spec)
        elif (
            isinstance(mag_spec, torch.Tensor)
            and isinstance(self.stft, pbSTFT)
        ):
            mag_spec = pt.utils.to_numpy(mag_spec, detach=True)

        signal = fast_griffin_lim(
            mag_spec, self.stft, self.alpha, self.iterations, self.atol
        )
        return self._resample(signal, target_sampling_rate)
