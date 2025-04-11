from contextlib import nullcontext
from math import ceil
import os
from pathlib import Path
import typing as tp

import einops
import numpy as np
from paderbox.transform.module_stft import (
    STFT, stft_frame_index_to_sample_index
)
import padertorch as pt
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.contrib.mk.visualization import compute_receptive_field_1d
from padertorch.ops.sequence.mask import compute_mask
from padertorch.utils import to_numpy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import parametrize
import torchaudio
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


def tuple_to_int(sequence) -> list:
    return list(map(lambda t: t[0], sequence))


class Wav2Vec2(pt.Module):
    """Extract wav2vec 2.0 features from raw waveform.

    Args:
        model_name (str): Name of the pretrained wav2vec 2.0 model to load.
            Defaults to "WAV2VEC2_BASE". Needs to match with the backend (see
            below).
        layer (int, optional): Index of the layer to extract features from.
            If None, all hidden features are extracted and returned as list.
            Defaults to -1 (last layer).
        freeze (bool): If True, freeze the weights of the encoder
            (i.e., no finetuning of Transformer layers). Defaults to True.
        freeze_feature_extractor (bool): If True, freeze the weights of the
            feature extractor (i.e., no finetuning of convolutional layers).
            Defaults to True.
        detach (bool): If True, detach the features from the computation graph.
            Defaults to False.
        backend (str): Backend to use for loading the model. Can be either
            "torchaudio" or "hf" (i.e., huggingface). If "torchaudio", the
            model is loaded from torchaudio.pipelines. If "hf", the model is
            loaded from huggingface.co. Defaults to "torchaudio".
        pad (bool):
        fading (str, bool, optional):
    """
    def __init__(
        self,
        model_name: str = "WAV2VEC2_BASE",
        layer: tp.Optional[int] = -1,
        freeze: bool = True,
        freeze_feature_extractor: bool = True,
        detach: bool = False,
        backend: str = "torchaudio",
        device: str = 'cpu',
        pad: bool = False,
        fading: tp.Optional[tp.Union[str, bool]] = None,
    ):
        super().__init__()
        if not freeze and detach:
            raise ValueError(
                'detach=True only supported if freeze=True\n'
                f'Got: freeze={self.freeze}, detach={self.detach}'
            )

        self.layer = layer
        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.detach = detach
        self.pad = pad
        self.fading = fading
        self.backend = backend
        self.device = device

        self._init_model(model_name)

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.freeze_feature_extractor:
            if backend == "torchaudio":
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False
            elif backend == "hf":
                self.model.freeze_feature_encoder()

        self._get_conv_params()
        self.downsample_factor = np.prod(self.strides)
        self.window_size = compute_receptive_field_1d(
            tuple_to_int(self.kernel_sizes),
            dilations=tuple_to_int(self.dilations),
            strides=tuple_to_int(self.strides),
        )[0]

        size = 2**(np.ceil(np.log2(self.window_size)))
        self._stft = STFT(
            shift=self.downsample_factor,
            size=size,
            window_length=self.window_size,
            pad=self.pad, fading=self.fading,
        )

    def _init_model(self, model_name):
        if self.backend == "hf":
            if "wav2vec2" not in model_name.lower():
                raise ValueError(
                    "Wav2Vec2 only supports wav2vec 2.0 models.\n"
                    f"model_name: {model_name}"
                )
            self.model = Wav2Vec2Model.from_pretrained(
                model_name, cache_dir=self.cache_dir
            ).to(self.device)
        elif self.backend == "torchaudio":
            bundle = getattr(torchaudio.pipelines, model_name)
            self.model = bundle.get_model().to(self.device)
            self.sampling_rate = bundle.sample_rate
            if self.layer == -1:
                self.layer = len(self.model.encoder.transformer.layers)-1
        else:
            raise ValueError(f'Unknown backend: {self.backend}')

    def _get_conv_params(self):
        self.kernel_sizes = list(map(
            lambda layer: layer.conv.kernel_size,
            self.model.feature_extractor.conv_layers
        ))
        self.dilations = list(map(
            lambda layer: layer.conv.dilation,
            self.model.feature_extractor.conv_layers
        ))
        self.strides = list(map(
            lambda layer: layer.conv.stride,
            self.model.feature_extractor.conv_layers
        ))

    @property
    def cache_dir(self):
        return (
            Path(os.environ.get('STORAGE_ROOT', '~/.cache'))
            / 'huggingface' / 'hub'
        )

    @property
    def context(self):
        if self.detach:
            return torch.no_grad()
        return nullcontext()

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                # Deprecated
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # This module didn't have weight norm
                pass
            try:
                parametrize.remove_parametrizations(m, 'weight')
            except ValueError:  # This module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def maybe_eval(self):
        if self.freeze or self.detach:
            self.model.eval()
        elif self.freeze_feature_extractor:
            if self.backend == "torchaudio":
                self.model.feature_extractor.eval()
            else:
                self.model.freeze_feature_encoder()

    def save_pretrained(
        self,
        save_directory,
        *args,
        **kwargs,
    ):
        if self.backend == "hf":
            self.model.save_pretrained(save_directory, *args, **kwargs)
        else:
            torch.save(
                self.model.state_dict(), str(save_directory / "wav2vec2.pth")
            )

    def from_pretrained(
        self,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        if self.backend == "hf":
            self.model = Wav2Vec2Model.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            self.model.load_state_dict(
                torch.load(str(pretrained_model_name_or_path / "wav2vec2.pth"))
            )
        return self

    def add_padding(self, sequence_lengths, *, signal=None, return_numpy=False):
        shift = self.downsample_factor
        length = self.window_size
        if isinstance(sequence_lengths, np.ndarray):
            sequence_lengths = torch.from_numpy(sequence_lengths).long()
        elif isinstance(sequence_lengths, (list, tuple)):
            sequence_lengths = torch.tensor(sequence_lengths).long()
        if signal is not None:
            sequence_lengths = sequence_lengths.to(signal.device)
        zero_padding = max(sequence_lengths) - sequence_lengths
        to_pad = torch.zeros((len(sequence_lengths), 2)).long()
        if self.fading not in [False, None]:
            if self.fading == 'half':
                pad_width= (
                    (length - shift) // 2,
                    ceil((length - shift) / 2)
                )
            else:
                pad_width = length - shift
                pad_width = (pad_width, pad_width)
            to_pad += torch.tensor(pad_width).unsqueeze(0)
            sequence_lengths = sequence_lengths + sum(pad_width)
        pad_size = 0
        if self.pad:
            if any(ilength < length for ilength in sequence_lengths):
                pad_size = torch.maximum(
                    torch.tensor(0.), length - sequence_lengths
                )
                batch_mask = pad_size == 0
            else:
                batch_mask = torch.ones_like(sequence_lengths).bool()
            if (
                shift != 1
                and any(
                    (ilen + shift - length) % shift != 0
                    for ilen in sequence_lengths
                )
            ):
                pad_size +=\
                    batch_mask * (
                        shift - ((sequence_lengths+shift-length) % shift)
                    )
            sequence_lengths = sequence_lengths + pad_size
        if return_numpy:
            sequence_lengths = np.array(to_numpy(sequence_lengths))
        if signal is not None:
            pad_sizes = torch.maximum(
                torch.tensor(0.), pad_size - zero_padding
            ).long()
            to_pad[:, 1] += pad_sizes
            signal = list(map(F.pad, signal, to_pad.tolist()))
            signal = pt.pad_sequence(list(
                map(lambda t: t.moveaxis(-1, 0), signal)
            )).moveaxis(0, -1)
            return signal, sequence_lengths
        return sequence_lengths

    def compute_output_lengths(self, input_lengths: TSeqLen) -> TSeqLen:
        """Compute the number of time frames for each batch entry.

        Args:
            input_lengths (list): List with number of samples per batch entry.

        Returns:
            output_lengths (list): List with number of time frames per batch
                entry.
        """
        if input_lengths is None:
            return input_lengths
        return self._stft.samples_to_frames(input_lengths)

    def sample_index_to_frame_index(self, sample_index):
        if isinstance(sample_index, int):
            sample_index = np.array([sample_index])
        return self._stft.sample_index_to_frame_index(sample_index)

    def frame_index_to_sample_index(self, frame_index):
        return stft_frame_index_to_sample_index(
            frame_index,
            self._stft.window_length,
            self._stft.shift,
            pad=self._stft.pad,
            fading=self._stft.fading,
        )

    def to_samples(self, frames):
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        num_samples = self.frame_index_to_sample_index(frames.shape[0])
        y = np.zeros((num_samples, *frames.shape[1:]), dtype=np.float32)
        last_sample = 0
        for frame_index, value in enumerate(frames):
            cur_sample = stft_frame_index_to_sample_index(
                frame_index,
                self._stft.window_length,
                self._stft.shift,
                pad=self._stft.pad,
                fading=self._stft.fading,
                num_samples=num_samples,
            )
            y[last_sample:cur_sample, ...] = value[None]
            last_sample = cur_sample
        return y[:cur_sample]

    def to_frames(self, samples, num_frames):
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        y = np.zeros(num_frames, dtype=np.float32)
        last_frame_index = 0
        for sample_index, value in enumerate(samples):
            frame_index = self._stft.sample_index_to_frame_index(sample_index)
            y[last_frame_index:frame_index] = value
            last_frame_index = frame_index
        return y

    def extract_features_from_latents(
        self, latents: Tensor, sequence_lengths: TSeqLen
    ):
        self.maybe_eval()
        if self.freeze_feature_extractor:
            latents = latents.detach()
        if isinstance(sequence_lengths, np.ndarray):
            sequence_lengths = torch.from_numpy(sequence_lengths).long()\
                .to(latents.device)
        with self.context:
            if self.backend == "torchaudio":
                x = self.model.encoder.extract_features(
                    latents,
                    lengths=sequence_lengths,
                    num_layers=self.layer,
                )
                if isinstance(self.layer, int):
                    x = x[-1]
                elif self.layer is None:
                    return x
                else:
                    raise NotImplementedError(self.layer)
                return x
            hidden_states, latents = self.model.feature_projection(
                latents
            )
            encoder_outputs = self.model.encoder(
                hidden_states,
                output_hidden_states=True,
                return_dict=True,
            )
            x = encoder_outputs.hidden_states
            if isinstance(self.layer, int):
                x = x[self.layer]
            elif self.layer is None:
                return x
            else:
                raise NotImplementedError(self.layer)
            return x

    def forward(
        self,
        time_signal: Tensor,
        sequence_lengths: TSeqLen = None,
        return_latents: bool = False,
    ) -> tp.Tuple[tp.Union[Tensor, tp.List[Tensor]], TSeqLen]:
        """Extract wav2vec 2.0 features from raw waveform.

        Args:
            time_signal (Tensor): Time signal of shape (batch, 1, time) or
                (batch, time) sampled at 16 kHz.
            sequence_lengths (list, optional): List with number of samples per
                batch entry.
            return_latents (bool): If True, return the latents of the
                feature extractor instead of the features. Defaults to False.

        Returns:
            x (Tensor): Wav2Vec2 features of shape (batch, 1, time frames, D).
            sequence_lengths (list): List with number of time frames per
                batch entry.
        """

        if time_signal.ndim == 3:
            time_signal = einops.rearrange(time_signal, 'b c t -> (b c) t')

        if sequence_lengths is not None:
            if isinstance(sequence_lengths, (list, tuple, np.ndarray)):
                sequence_lengths = torch.tensor(sequence_lengths).long()\
                    .to(time_signal.device)

            time_signal, pad_sequence_lengths = self.add_padding(
                sequence_lengths, signal=time_signal
            )
        else:
            pad_sequence_lengths = None

        self.maybe_eval()
        with self.context:
            if self.backend == "torchaudio":
                self.model: torchaudio.models.Wav2Vec2Model
                if return_latents:
                    z, sequence_lengths = self.model.feature_extractor(
                        time_signal, pad_sequence_lengths
                    )
                    sequence_lengths = np.array(
                        to_numpy(sequence_lengths, detach=True)
                    )
                    try:
                        sequence_lengths = sequence_lengths.item()
                        if sequence_lengths is not None:
                            sequence_lengths = np.array([sequence_lengths])
                    except ValueError:
                        pass
                    return z, sequence_lengths

                x, sequence_lengths = self.model.extract_features(
                    time_signal, lengths=pad_sequence_lengths,
                    num_layers=self.layer,
                )
                sequence_lengths = np.array(
                    to_numpy(sequence_lengths, detach=True)
                )
                if isinstance(self.layer, int):
                    x = x[-1]
                    return x, sequence_lengths
                if self.layer is None:
                    return x, sequence_lengths
                raise NotImplementedError(self.layer)

            self.model: Wav2Vec2Model
            out_sequence_lengths = self.compute_output_lengths(sequence_lengths)
            z = self.model.feature_extractor(time_signal.float())\
                .transpose(1, 2)
            out_sequence_lengths = np.minimum(out_sequence_lengths, z.shape[1])
            if return_latents:
                return z, out_sequence_lengths

            sequence_mask = compute_mask(z[..., 0], out_sequence_lengths)
            outputs = self.model.forward(
                time_signal.float(),
                attention_mask=sequence_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            if isinstance(self.layer, int):
                x = outputs.hidden_states[self.layer]
                if self.detach:
                    x = x.detach()
            elif self.layer is None:
                x = outputs.hidden_states
                if self.detach:
                    x = [h.detach() for h in x]
                return x, out_sequence_lengths
            else:
                raise ValueError(f'Unknown layer: {self.layer}')

        return x, out_sequence_lengths
