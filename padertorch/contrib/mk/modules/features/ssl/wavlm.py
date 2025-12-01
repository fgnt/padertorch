import typing as tp

import numpy as np
from padertorch.contrib.mk.typing import TSeqLen
import torch
from torch import Tensor
import torchaudio
from transformers.models.wavlm.modeling_wavlm import WavLMModel

from .wav2vec2 import Wav2Vec2, SAMPLING_RATE

# See https://ieeexplore.ieee.org/abstract/document/9814838, Fig. 2
PR_BASE_LAYER = 11
PR_LARGE_LAYER = 22
SID_BASE_LAYER = 4
SID_LARGE_LAYER = 6


def tuple_to_int(sequence) -> list:
    return list(map(lambda t: t[0], sequence))


class WavLM(Wav2Vec2):
    """Extract WavLM features from raw waveform.

    Args:
        model_name (str): Name of the pretrained WavLM model to load.
            Defaults to "WAVLM_BASE". Needs to match with the backend (see
            below).
        layer (int): Index of the layer to extract features from.
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
        model_name: str = "WAVLM_BASE",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

    def _init_model(self, model_name):
        if "wavlm" not in model_name.lower():
            raise ValueError(
                "WavLMExtractor only supports WavLM models.\n"
                f"model_name: {model_name}"
            )
        if self.backend == "hf":
            self.model = WavLMModel.from_pretrained(
                model_name, cache_dir=self.cache_dir
            ).to(self.device)
            self.sampling_rate = SAMPLING_RATE
        elif self.backend == "torchaudio":
            bundle = getattr(torchaudio.pipelines, model_name)
            self.model = bundle.get_model().to(self.device)
            if hasattr(self.model, 'model'):
                # WavLM Large specific
                self.model = self.model.model
            self.sampling_rate = bundle.sample_rate
            if self.layer == -1:
                self.layer = self.num_layers
        else:
            raise ValueError(f'Unknown backend: {self.backend}')

    def _forward(self, time_signal: Tensor, sequence_lengths: TSeqLen):
        # WavLM does not support sequence lengths
        del sequence_lengths
        return super()._forward(time_signal, None)
        # x, _ = self.model.extract_features(
        #     time_signal, lengths=None,
        #     num_layers=self.layer,
        # )
        # return x

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
                num_layers = None if isinstance(self.layer, str) else self.layer
                x = self.model.encoder.extract_features(
                    latents,
                    lengths=None,  # WavLM does not support sequence lengths
                    num_layers=num_layers,
                )
                return self.extract_layer(x)

            # hf backend
            hidden_states = self.model.feature_projection(latents)
            hidden_states = self.model._mask_hidden_states(hidden_states)
            encoder_outputs = self.model.encoder(
                hidden_states,
                output_hidden_states=True,
                return_dict=True,
            )
            x = encoder_outputs.hidden_states
            return self.extract_layer(x)

    def forward(
        self,
        time_signal: Tensor,
        sequence_lengths: TSeqLen = None,
        return_latents: bool = False,
    ) -> tp.Tuple[tp.Union[Tensor, tp.List[Tensor]], TSeqLen]:
        """Extract WavLM features from raw waveform.

        Args:
            time_signal (Tensor): Time signal of shape (batch, 1, time) or
                (batch, time) sampled at 16 kHz.
            sequence_lengths (list, optional): List with number of samples per
                batch entry.
            return_latents (bool): If True, return the latents of the
                feature extractor instead of the features. Defaults to False.

        Returns:
            x (Tensor): WavLM features of shape (batch, D, time frames).
            sequence_lengths (list): List with number of time frames per
                batch entry.
        """

        return super().forward(time_signal, sequence_lengths, return_latents)
