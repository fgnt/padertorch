import os
from pathlib import Path
import typing as tp
import tempfile

from einops import rearrange
import numpy as np
import torch

import padertorch as pt

from .nvidia_bigvgan import bigvgan
from ..base import Synthesis


class Vocoder(Synthesis):
    """
    Neural vocoder wrapping any models from https://github.com/NVIDIA/BigVGAN

    Provides an easy-to-use interface to download vocoders and to perform
    waveform synthesis from log-mel spectrogram. Vocoder models are identified
    by a vocoder tag of the form "bigvgan[_<version>]_<sampling_rate>khz_<mel_bands>band[_<fmax>k][_<upsampling_ratio>x]",
    where parameters in brackets are optional. The __call__
    has to take a log-mel spectrogram (np.ndarray or torch.Tensor), a list of
    sequence lengths (in case of a batched input), the desired output sampling
    rate, and any optional *keyword arguments* to control the synthesis.

    Attributes:
        sampling_rate (int): Sampling rate of the training data the vocoder was
            trained on.
    """
    def __init__(
        self,
        base_dir: tp.Optional[tp.Union[str, Path]] = \
            os.environ.get('BIGVGAN_BASE_DIR', None),
        vocoder_tag: tp.Optional[str] = None,
        version: tp.Optional[str] = 'v2',
        sampling_rate: tp.Optional[int] = 44_000,
        mel_bands: tp.Optional[int] = 128,
        fmax: tp.Optional[int] = None,
        upsampling_ratio: tp.Optional[int] = 256,
        device: tp.Union[str, int] = 'cpu',
        batch_axis: int = 0,
        sequence_axis: int = -1,
        postprocessing: tp.Optional[tp.Callable] = None,
        use_cuda_kernel: bool = False,
     ):
        """
        Args:
            database: The database the vocoder was trained on (see
                https://github.com/kan-bayashi/ParallelWaveGAN). Used to
                infer the vocoder model. If `vocoder_tag` is not None, this
                argument will be ignored
            base_dir: Path to folder where vocoders will be downloaded to.
                Vocoders will be dumped as directories indexed by their vocoder
                tag. If a vocoder is already downloaded, it will be loaded from
                disk instead of being downloaded again
            vocoder_model: Type of vocoder architecture as specified in the
                vocoder tag, e.g., "parallel_wavegan" or "hifigan". See
                https://github.com/kan-bayashi/ParallelWaveGAN for available
                architectures. If `vocoder_tag` is not None, this
                argument will be ignored
            vocoder_tag: Vocoder tag of the form
                <database>_<vocoder_model>.<version_no.>. If not None, will
                first look under `base_dir` and then try to download it
                from https://github.com/kan-bayashi/ParallelWaveGAN
            device: Device (CPU, GPU) used for inference
            consider_mpi: If True, load the weights on the master and distribute
                to all workers
            batch_axis: Axis along which the batches are stacked. If the input
                to __call__ is 2-dimensional, `batch_axis` will be ignored
            sequence_axis: Axis that contains time information
            postprocessing: Optional postprocessing function that is applied to
                the synthesized waveform
            use_cuda_kernel (bool): Enables faster inference on GPU. Defaults
                to False.
        """
        super().__init__(postprocessing=postprocessing)
        self.base_dir = base_dir
        self.vocoder_tag = vocoder_tag
        self.device = device
        self.batch_axis = batch_axis
        self.sequence_axis = sequence_axis

        if self.vocoder_tag is None:
            self.vocoder_tag = 'bigvgan'
            if version is not None:
                self.vocoder_tag += '_' + version
            self.vocoder_tag += f'_{sampling_rate//1000}khz_{mel_bands}band'
            if fmax is not None:
                self.vocoder_tag += f'_{fmax//1000}k'
            if upsampling_ratio is not None:
                self.vocoder_tag += f'_{upsampling_ratio}x'
        if self.base_dir is None:
            self.base_dir = (
                Path(tempfile.gettempdir()) / 'bigvgan_models'
            )
        else:
            self.base_dir = Path(self.base_dir)

        self.vocoder_model = bigvgan.BigVGAN.from_pretrained(
            f'nvidia/{self.vocoder_tag}', cache_dir=self.base_dir,
            use_cuda_kernel=use_cuda_kernel
        )
        self.vocoder_model.remove_weight_norm()
        self.vocoder_model.to(self.device).eval()

        self.vocoder_params = self.vocoder_model.h
        self.sampling_rate = self.vocoder_params.sampling_rate

    def __call__(
        self,
        mel_spec: tp.Union[torch.Tensor, np.ndarray],
        sequence_lengths: tp.Optional[tp.List[int]] = None,
        target_sampling_rate: tp.Optional[int] = None,
    ) -> tp.Union[
        torch.Tensor, np.ndarray, tp.List[np.ndarray],
        tp.List[torch.Tensor]
    ]:
        """
        Synthesize waveform from log-mel spectrogram with a neural vocoder

        >>> vocoder = Vocoder(sampling_rate=22000, mel_bands=80)
        >>> mel_spec = torch.zeros((1, 80, 100))
        >>> wav_gen = vocoder(mel_spec)
        >>> wav_gen.shape
        torch.Size([1, 25600])

        Args:
            mel_spec: (Batched) mel-spectrograms where shape must match as
                specified by `self.batch_axis` and `self.sequence_axis`
            sequence_lengths:
            target_sampling_rate: By default, vocoder will produce waveforms
                with sampling rate seen during training. If not None, resample
                to `target_sampling_rate`

        Returns: torch.Tensor or np.ndarray
            Synthesized waveform
        """
        to_numpy = isinstance(mel_spec, np.ndarray)
        if to_numpy:
            mel_spec = torch.from_numpy(mel_spec).to(self.device)
        sequence_axis = self.sequence_axis % mel_spec.ndim
        batch_axis = self.batch_axis % mel_spec.ndim
        if mel_spec.ndim == 2:
            mel_spec = torch.moveaxis(mel_spec, sequence_axis, -1).unsqueeze(0)
            with torch.inference_mode():
                y = self.vocoder_model(mel_spec).view(-1)  # (T,)
                if to_numpy:
                    y = pt.utils.to_numpy(y, detach=True)
        elif mel_spec.ndim == 3:
            feature_axis = set(range(3)).difference(
                {batch_axis, self.sequence_axis}
            ).pop()
            shape = list(map(
                lambda t: t[1],
                sorted(
                    zip(
                        [batch_axis, sequence_axis, feature_axis],
                        ['b', 't', 'f']
                    ), key=lambda t: t[0]
                )
            ))
            mel_spec = rearrange(mel_spec, f"{' '.join(shape)} -> b f t")
            if sequence_lengths is None:
                sequence_lengths = [mel_spec.shape[-1]] * mel_spec.shape[0]
            with torch.inference_mode():
                y = []
                for _mel_spec, seq_len in zip(mel_spec, sequence_lengths):
                    y_ = (
                        self.vocoder_model(
                            _mel_spec.unsqueeze(0)[..., :seq_len],
                        ).view(-1)
                    )  # (T,)
                    if to_numpy:
                        y_ = pt.utils.to_numpy(y, detach=True)
                    y.append(y_)
                try:
                    if to_numpy:
                        y = np.stack(y)
                    else:
                        y = torch.stack(y)
                except RuntimeError:
                    pass
        else:
            raise TypeError(
                'Expected 2- or 3-dim. spectrogram but got '
                f'{mel_spec.ndim}-dim. input with shape {mel_spec.shape}'
            )
        return super().__call__(y, target_sampling_rate)
