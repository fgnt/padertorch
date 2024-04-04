import typing
from functools import partial

import numpy as np
import torch
from paderbox.transform.module_resample import resample_sox
import padertorch as pt


class Synthesis(pt.Configurable):
    sampling_rate: int

    def __init__(
        self,
        postprocessing: typing.Optional[typing.Callable] = None,
    ):
        super().__init__()
        self.postprocessing = postprocessing

    def __call__(
        self,
        time_signal: typing.Union[
            np.ndarray, torch.Tensor, typing.List[np.ndarray],
            typing.List[torch.Tensor]
        ],
        target_sampling_rate: typing.Optional[int] = None,
    ) -> typing.Union[
        np.ndarray, torch.Tensor, typing.List[np.ndarray],
        typing.List[torch.Tensor]
    ]:
        if self.postprocessing is not None:
            if isinstance(time_signal, list) or time_signal.ndim == 2:
                time_signal = list(map(self.postprocessing, time_signal))
            else:
                time_signal = self.postprocessing(time_signal)
        return self.resample(time_signal, target_sampling_rate)

    def _resample(
        self,
        wav: typing.Union[np.ndarray, torch.Tensor],
        target_sampling_rate: typing.Optional[int] = None,
    ) -> typing.Union[np.ndarray, torch.Tensor]:
        to_torch = False
        if (
            target_sampling_rate is None
            or target_sampling_rate == self.sampling_rate
        ):
            return wav
        if isinstance(wav, torch.Tensor):
            to_torch = True
            wav = pt.utils.to_numpy(wav, detach=True)
        wav = resample_sox(
            wav,
            in_rate=self.sampling_rate,
            out_rate=target_sampling_rate
        )
        if to_torch:
            wav = torch.from_numpy(wav)
        return wav

    def resample(
        self,
        wav: typing.Union[
            np.ndarray, torch.Tensor, typing.List[np.ndarray],
            typing.List[torch.Tensor]
        ],
        target_sampling_rate: typing.Optional[int] = None,
    ) -> typing.Union[
        np.ndarray, torch.Tensor, typing.List[np.ndarray],
        typing.List[torch.Tensor]
    ]:
        if isinstance(wav, list) or wav.ndim == 2:
            wav = list(map(
                partial(
                    self._resample, target_sampling_rate=target_sampling_rate
                ), wav
            ))
            try:
                m = np if isinstance(wav[0], np.ndarray) else torch
                wav = m.stack(wav)
            except (ValueError, RuntimeError):
                pass
            return wav
        return self._resample(wav, target_sampling_rate=target_sampling_rate)
