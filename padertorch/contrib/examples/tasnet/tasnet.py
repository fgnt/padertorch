from functools import partial
from typing import Union, Optional
import torch
from einops import rearrange

import padertorch as pt
from torch.nn.utils.rnn import pad_sequence

from padertorch.modules.tas_coders import TasEncoder, TasDecoder
from padertorch.modules.dual_path_rnn import DPRNN, apply_examplewise
from padertorch.ops.losses.regression import si_sdr_loss, log_mse_loss
from padertorch.ops.mappings import ACTIVATION_FN_MAP


class TasNet(pt.Model):
    def __init__(
            self,
            hidden_size: int = 64,
            encoder_block_size: int = 16,
            rnn_size: int = 128,
            dprnn_window_length: Union[int, str] = 100,
            dprnn_hop_size: Union[str, int] = 50,
            mask: bool = True,
            dprnn_layers: int = 6,
            output_nonlinearity: Optional[str] = 'sigmoid',
            inter_chunk_type: str = 'blstm',
            intra_chunk_type: str = 'blstm',
            num_speakers: int = 2,
            additional_out_size: int = 0,
    ):
        """
        Args:
            hidden_size: Size between the layers (N)
            encoder_block_size:
            rnn_size: Size of the NNs in the inter- or intra-chunk RNNs. This
                corresponds to H or H//2.
            dprnn_window_length: Window length of the DPRNN segmentation (K).
                If set to 'auto', it is determined for each example
                independently based on the input length and the "rule of thumb"
                K \approx sqrt(2L)
            dprnn_hop_size: Hop size of the DPRNN segmentation (P). If set to
                'auto' it is set to 50% of the block length K (half overlap).
            mask: If `True`, use the output of the NN as a mask in tas domain
                for separation. Otherwise, use the output directly as an
                estimation for the separated signals.
            dprnn_layers: Number of stacked DPRNN blocks
            output_nonlinearity: Nonlinearity applied to the output (right
                before masking/decoding)
            inter_chunk_type: Type of the inter-chunk RNN
            intra_chunk_type: Type of the intra-chunk RNN
            num_speakers: The number of speakers/output streams
            additional_out_size: Size of the additional output. Has no effect if
                set to 0.
            input_norm: If `True` normalize the input
        """
        super().__init__()

        self.mask = mask

        assert additional_out_size == 0 or self.conv and self.split_in_conv, (
            'Additional out size can only be provided if convolutions before '
            'and after the DPRNN are enabled and if speaker signals are '
            'split in the convolutional output layer (split_in_conv).'
        )
        self.additional_out_size = additional_out_size

        self.input_proj = torch.nn.Conv1d(hidden_size, hidden_size, 1)
        self.output_prelu = torch.nn.PReLU()
        self.output_proj = torch.nn.Conv1d(
            hidden_size, hidden_size * num_speakers + additional_out_size, 1
        )

        self.encoder = TasEncoder(
            L=encoder_block_size,
            N=hidden_size,
        )

        self.encoded_input_norm = torch.nn.LayerNorm(hidden_size)

        self.dprnn = DPRNN(
            feat_size=hidden_size,
            rnn_size=rnn_size,
            window_length=dprnn_window_length,
            hop_size=dprnn_hop_size,
            inter_chunk_type=inter_chunk_type,
            intra_chunk_type=intra_chunk_type,
            num_blocks=dprnn_layers,
        )

        self.decoder = TasDecoder(
            L=encoder_block_size,
            N=hidden_size
        )

        self.output_nonlinearity = ACTIVATION_FN_MAP[output_nonlinearity]()

        self.num_speakers = num_speakers

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
        processed = self.dprnn(encoded, encoded_sequence_lengths)
        processed = rearrange(processed, 'b l n -> b n l')

        processed = self.output_proj(self.output_prelu(processed))

        # Split a part of the output for an additional output
        if self.additional_out_size > 0:
            processed, additional_out = (
                processed[..., self.additional_out_size:, :],
                processed[..., :self.additional_out_size, :][0]
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
            'si-sdr-grad-stop': partial(si_sdr_loss, grad_stop=True),
        }
        losses = {k: [] for k in loss_functions.keys()}

        for seq_len, estimated, target in zip(sequence_lengths, x, s):
            for k, loss_fn in loss_functions.items():
                losses[k].append(
                    pt.ops.loss.pit_loss(
                        estimated[:seq_len],
                        target[:seq_len],
                        axis=0, loss_fn=loss_fn,
                    )
                )

        return {k: torch.mean(torch.stack(v)) for k, v in losses.items()}

    def review(self, inputs, outputs):
        # Report audios
        audios = {
            'observation': pt.summary.audio(
                signal=inputs['y'][0], sampling_rate=8000
            ),
        }

        for i, e in enumerate(outputs['out'][0]):
            audios[f'estimate/{i}'] = pt.summary.audio(
                signal=e, sampling_rate=8000
            )

        for i, y in enumerate(inputs['s'][0]):
            audios[f'target/{i}'] = pt.summary.audio(
                signal=y, sampling_rate=8000
            )

        return pt.summary.review_dict(
            losses=self.loss(inputs, outputs),
            audios=audios,
        )

    def flatten_parameters(self) -> None:
        self.dprnn.flatten_parameters()
