from functools import partial
from typing import Union, Optional
import torch
from einops import rearrange

import padertorch as pt
from torch.nn.utils.rnn import pad_sequence

from .tas_coders import TasEncoder, TasDecoder
from padertorch.modules.dual_path_rnn import DPRNN, apply_examplewise
from padertorch.ops.losses.regression import si_sdr_loss, log_mse_loss, \
    log1p_mse_loss
from padertorch.ops.mappings import ACTIVATION_FN_MAP


class TasNet(pt.Model):
    def __init__(
            self,
            encoder: torch.Module,
            separator: torch.Module,
            decoder: torch.Module,
            mask: bool = True,
            output_nonlinearity: Optional[str] = 'sigmoid',
            num_speakers: int = 2,
            additional_out_size: int = 0,
            sample_rate: int = 8000,
    ):
        """
        Args:
            encoder:
            separator:
            decoder:
            mask: If `True`, use the output of the NN as a mask in tas domain
                for separation. Otherwise, use the output directly as an
                estimation for the separated signals.
            output_nonlinearity: Nonlinearity applied to the output (right
                before masking/decoding)
            num_speakers: The number of speakers/output streams
            additional_out_size: Size of the additional output. Has no effect if
                set to 0.
            sample_rate: Sample rate of the audio. Only used for correct
                reporting to TensorBoard, has no effect on the model
                architecture.
        """
        super().__init__()

        self.mask = mask
        self.additional_out_size = additional_out_size
        hidden_size = separator.feat_size
        self.input_proj = torch.nn.Conv1d(hidden_size, hidden_size, 1)
        self.output_prelu = torch.nn.PReLU()
        self.output_proj = torch.nn.Conv1d(
            hidden_size, hidden_size * num_speakers + additional_out_size, 1
        )

        self.encoder = encoder

        self.encoded_input_norm = torch.nn.LayerNorm(hidden_size)

        self.separator = separator

        self.decoder = decoder

        self.output_nonlinearity = ACTIVATION_FN_MAP[output_nonlinearity]()

        self.num_speakers = num_speakers
        self.sample_rate = sample_rate

    def forward(self, batch: dict) -> dict:
        """
        Separates the time-signal in `sequence` into `self.num_speakers`
        separated audio streams.

        Now supports sequence lengths, but runs quite slow when used with
        sequence lengths.
        """
        sequence = pad_sequence(batch['y'], batch_first=True)
        sequence_lengths = batch['num_samples']
        if not torch.is_tensor(sequence_lengths):
            sequence_lengths = torch.tensor(sequence_lengths)

        # Call encoder
        encoded_raw, encoded_sequence_lengths = self.encoder(
            sequence, sequence_lengths)

        # Apply layer norm to the encoded signal
        encoded = rearrange(encoded_raw, 'b n l -> b l n')
        encoded = apply_examplewise(
            self.encoded_input_norm, encoded, encoded_sequence_lengths)

        # Apply convolutional layer if set
        if self.input_proj:
            encoded = rearrange(encoded, 'b l n -> b n l')
            encoded = self.input_proj(encoded)
            encoded = rearrange(encoded, 'b n l -> b l n')

        # Call DPRNN. Needs shape BxLxN
        processed = self.separator(encoded, encoded_sequence_lengths)
        processed = rearrange(processed, 'b l n -> b n l')

        processed = self.output_proj(self.output_prelu(processed))

        # Split a part of the output for an additional output
        if self.additional_out_size > 0:
            processed, additional_out = (
                processed[..., self.additional_out_size:, :],
                processed[..., :self.additional_out_size, :]
            )

        # Shape KxBxNxL
        processed = torch.stack(
            torch.chunk(processed, self.num_speakers, dim=1))
        processed = self.output_nonlinearity(processed)

        # The estimation can be a little longer than the input signal.
        # Shorten the estimation to match the input signal
        processed = processed[..., :encoded_raw.shape[-1]]
        assert encoded_raw.shape == processed.shape[1:], (
            processed.shape, encoded_raw.shape)

        if self.mask:
            # Mask if set
            processed = encoded_raw.unsqueeze(0) * processed

        # Decode stream for each speaker
        decoded = rearrange(
            self.decoder(rearrange(processed, 'k b n l -> (k b) n l')),
            '(k b) t -> k b t',
            k=processed.shape[0], b=processed.shape[1])

        # The length can be slightly longer than the input length if it is not
        # a multiple of a segment length.
        decoded = decoded[..., :sequence.shape[-1]]

        # This is necessary if an offset-invariant loss fn (e.g.,
        # SI-SNR from the TasNet paper) but an offset-variant evaluation metric
        # (e.g., SI-SDR) is used.
        # TODO: Fix the loss fn and remove this
        decoded = decoded - torch.mean(decoded, dim=-1, keepdim=True)

        out = {
            'out': rearrange(decoded, 'k b t -> b k t'),
            'encoded': rearrange(encoded_raw, 'b n l -> b l n'),
            'encoded_out': rearrange(processed, 'k b n l -> b k l n'),
            'encoded_sequence_lengths': encoded_sequence_lengths,
        }

        if self.additional_out_size > 0:
            additional_out = additional_out[..., :processed.shape[-1]]
            out['additional_out'] = additional_out

        return out

    def loss(self, inputs: dict, outputs: dict) -> dict:
        s = inputs['s']
        sequence_lengths = inputs['num_samples']
        x = outputs['out']

        loss_functions = {
            'si-sdr': si_sdr_loss,
            'log-mse': log_mse_loss,
            'log1p-mse': log1p_mse_loss,
        }
        losses = {k: [] for k in loss_functions.keys()}

        for seq_len, estimated, target in zip(sequence_lengths, x, s):
            for k, loss_fn in loss_functions.items():
                losses[k].append(
                    pt.ops.losses.pit_loss(
                        estimated[..., :seq_len],
                        target[..., :seq_len],
                        axis=0, loss_fn=loss_fn,
                    )
                )

        return {k: torch.mean(torch.stack(v)) for k, v in losses.items()}

    def review(self, inputs: dict, outputs: dict) -> dict:
        # Report audios
        audios = {
            'observation': pt.summary.audio(
                signal=inputs['y'][0], sampling_rate=self.sample_rate
            ),
        }

        for i, e in enumerate(outputs['out'][0]):
            audios[f'estimate/{i}'] = pt.summary.audio(
                signal=e, sampling_rate=self.sample_rate
            )

        for i, y in enumerate(inputs['s'][0]):
            audios[f'target/{i}'] = pt.summary.audio(
                signal=y, sampling_rate=self.sample_rate
            )

        return pt.summary.review_dict(
            losses=self.loss(inputs, outputs),
            audios=audios,
        )

    def flatten_parameters(self) -> None:
        self.separator.flatten_parameters()
