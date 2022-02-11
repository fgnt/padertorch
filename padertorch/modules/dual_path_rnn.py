"""
This file contains an implementation of the Dual-Path RNN [1]

References:
    [1]: Luo, Yi, Zhuo Chen, and Takuya Yoshioka. “Dual-Path RNN: Efficient
        Long Sequence Modeling for Time-Domain Single-Channel Speech
        Separation.” ArXiv Preprint ArXiv:1910.06379, 2019.
        https://arxiv.org/pdf/1910.06379.pdf
"""
import math
import warnings
from typing import Optional, Tuple, List

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, \
    PackedSequence, pad_sequence

import paderbox as pb


def segment(
        signal: torch.Tensor, hop_size: int, window_size: int,
        sequence_lengths: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Zero-pads and segments the input sequence `signal` along the time dimension `L` (-2).

    Examples:
        >>> import torch
        >>> hop_size = 10
        >>> segmented, _ = segment(torch.randn(1, 50, 3), hop_size, 2 * hop_size)

        # Shape is BxNxKxS (batch x feat x win x frames)
        >>> segmented.shape
        torch.Size([1, 3, 20, 6])

        # The first block is zero-padded with hop_size
        >>> segmented[..., :hop_size, 0]
        tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

        # Last block as well
        >>> segmented[..., -hop_size:, -1]
        tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

        # Sequence lengths are computed
        >>> segmented, sequence_lengths = segment(torch.cat([torch.randn(1, 30, 3), torch.zeros(1, 10, 3)], dim=1),
        ...                                         hop_size, 2*hop_size, torch.tensor([30]))
        >>> sequence_lengths
        tensor([4])

        # All data outside of sequence_lengths is zero
        >>> segmented[0, ..., sequence_lengths[0]:].flatten()
        tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # And the last segment within seuqence_lengths contains data, but zero padded at the end
        # (Conversion to uint8 is to make the doctest compatible with all PyTorch versions)
        >>> (segmented[0, ..., sequence_lengths[0] - 1] == 0).type(torch.uint8)
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
               dtype=torch.uint8)

        # Test the corner-cases for computation of sequence lengths

        # One above exact match
        >>> segment(1 + torch.arange(5)[None, :, None], 2, 4, torch.tensor(5))
        (tensor([[[[0, 1, 3, 5],
                  [0, 2, 4, 0],
                  [1, 3, 5, 0],
                  [2, 4, 0, 0]]]]), tensor(4))

        # Exact match
        >>> segment(1 + torch.arange(5)[None, :, None], 2, 4, torch.tensor(4))
        (tensor([[[[0, 1, 3, 5],
                  [0, 2, 4, 0],
                  [1, 3, 5, 0],
                  [2, 4, 0, 0]]]]), tensor(3))
        >>> segment(1 + torch.arange(4)[None, :, None], 2, 4, torch.tensor(4))
        (tensor([[[[0, 1, 3],
                  [0, 2, 4],
                  [1, 3, 0],
                  [2, 4, 0]]]]), tensor(3))

        # One below exact match
        >>> segment(1 + torch.arange(5)[None, :, None], 2, 4, torch.tensor(3))
        (tensor([[[[0, 1, 3, 5],
                  [0, 2, 4, 0],
                  [1, 3, 5, 0],
                  [2, 4, 0, 0]]]]), tensor(3))
        >>> segment(1 + torch.arange(3)[None, :, None], 2, 4, torch.tensor(3))
        (tensor([[[[0, 1, 3],
                  [0, 2, 0],
                  [1, 3, 0],
                  [2, 0, 0]]]]), tensor(3))

        # Shift != size // 2
        >>> segmented, seq_len = segment(torch.arange(5)[None, :, None], 3, 4, torch.tensor(5))
        >>> segmented.shape
        torch.Size([1, 1, 4, 2])
        >>> seq_len
        tensor(2)
        >>> segmented, seq_len = segment(torch.arange(5)[None, :, None], 1, 4, torch.tensor(5))
        >>> segmented.shape
        torch.Size([1, 1, 4, 8])
        >>> seq_len
        tensor(8)

        >>> segmented, seq_len = segment(torch.ones(1, 7912, 64), 50, 100, torch.tensor([7912]))
        >>> segmented.shape
        torch.Size([1, 64, 100, 160])
        >>> seq_len
        tensor([160])


    Args:
        signal ([Bx]LxN): 2D input signal with optional batch dimension
        hop_size: Hop size P
        window_size: Window size K
        sequence_lengths: These are not used for segmentation, but if provided, the resulting sequence lengths along the
            segment (S) dimension are returned in addition to the segmented signal. Then, the sequence length is the
            number of blocks that contain any part of the signal, and these might be 0-padded.

    Returns:
        [Bx]NxKxS
        S is the number of frames, K is the window size, N is the feature size
    """
    # Add padding for the first and last blocks. Should be each hop_size so
    # that the first half of the first block and the last half of the last
    # block are filled with 0s for the case of 50% overlap.
    padding = window_size - hop_size
    signal = F.pad(signal, [0, 0, padding, padding])

    segmented = pb.array.segment_axis(
        signal, window_size, hop_size, axis=-2, end='pad')
    segmented = rearrange(segmented, '... s k n -> ... n k s')

    if sequence_lengths is not None:
        sequence_lengths = sequence_lengths + 2 * padding
        sequence_lengths = (sequence_lengths - padding)
        sequence_lengths = torch.div(sequence_lengths - 1, hop_size, rounding_mode='floor')  + 1
    return segmented, sequence_lengths


def overlap_add(
        signal: torch.Tensor, hop_size: int, unpad: bool = True
) -> torch.Tensor:
    """
    Examples:
        >>> import torch

        # Overlap-adding a segmented range should produce 2*range
        >>> a = torch.arange(50).unsqueeze(0).unsqueeze(-1)
        >>> a.shape
        torch.Size([1, 50, 1])

        >>> segmented, _ = segment(a, 10, 20)
        >>> added = overlap_add(segmented, 10, unpad=True)

        # The shape is BxLxN again
        >>> added.shape
        torch.Size([1, 50, 1])

        >>> added[0, :, 0]
        tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
                36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70,
                72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98])

        >>> overlap_add(segment(torch.arange(5)[None, :, None], 2, 4)[0], 2)
        tensor([[[0],
                 [2],
                 [4],
                 [6],
                 [8],
                 [0]]])

        >>> overlap_add( segment(torch.arange(5)[None, :, None], 3, 4)[0], 3)
        tensor([[[0],
                 [1],
                 [4],
                 [3],
                 [4]]])

    Args:
        signal:
        hop_size:

    Returns:

    """
    B, N, K, S = signal.shape
    assert K > hop_size

    out = signal.new_zeros(B, S*hop_size + K - hop_size, N)

    signal = rearrange(signal, 'b n k s -> b k n s')
    for i in range(S):
        out[:, i * hop_size:i * hop_size + K, :] += signal[..., :, :, i]

    if unpad:
        out = out[..., K - hop_size:- (K - hop_size), :]

    return out


def pack(x: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Packs `x` such that it combines its batch (0) and time (1) axis and removes
    any padded values in between. It can be reverted with `unpack`.

    .. note::

        This is different from `pack_padded_sequence` in that it does not
        interleave the time steps and does not return a `PackedSequence`.
    """
    assert len(sequence_lengths) == len(x)
    return torch.cat([x_[:l] for x_, l in zip(x, sequence_lengths)])


def unpack(x: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Examples:
        # Packing and unpacking a zero-padded tensor gives the same tensor as the input tensor
        >>> import torch
        >>> a = torch.randn(3, 100)
        >>> a[0, 50:] = 0
        >>> a[1, 70:] = 0
        >>> sequence_lengths = torch.tensor([50, 70, 100])
        >>> packed = pack(a, sequence_lengths)
        >>> unpacked = unpack(packed, sequence_lengths)
        >>> a.shape
        torch.Size([3, 100])
        >>> packed.shape
        torch.Size([220])
        >>> unpacked.shape
        torch.Size([3, 100])
        >>> a.shape == unpacked.shape
        True
        >>> bool(torch.all(unpacked == a))
        True
    """
    segments = []
    start = 0
    for l in sequence_lengths:
        segments.append(x[start:start + l])
        start += l
    return pad_sequence(segments, batch_first=True)


def apply_examplewise(fn, x: torch.Tensor, sequence_lengths, time_axis=1):
    """
    Applies a function to each element of x (along batch (0) dimension) and
    respects the sequence lengths along time axis. Assumes that fn does not
    change the dimensions of its input (e.g., norm).
    """
    if sequence_lengths is None:
        return fn(x)
    else:
        # Check inputs
        assert time_axis != 0, 'The first axis must be the batch axis!'
        assert len(sequence_lengths) == x.shape[0], (
            'Number of sequence lengths and batch size must match!'
        )

        time_axis = time_axis % x.dim()
        selector = [slice(None)] * (time_axis - 1)
        out = torch.zeros_like(x)
        for b, l in enumerate(sequence_lengths):
            s = (b, *selector, slice(l))

            # Keep the batch dimension while processing
            out[s] = fn(x[s][None, ...])[0]
        return out


class _ChunkRNN(torch.nn.Module):
    """
    Base for one "ChunkRNN" block. It consists of an RNN, a fully connected
    layer and a normalization layer.

    Examples:
        To perform iteration over the segment dimension s:
        >>> chunk_rnn = _ChunkRNN(10, 20, '(b k) s n')

        The output shape is exactly the same as the input shape:
        >>> a = torch.randn(2, 10, 5, 3)
        >>> out = chunk_rnn(a)
        >>> out.shape
        torch.Size([2, 10, 5, 3])
        >>> out.shape == a.shape
        True

        Sequence lengths are supported, but they can be omitted if all examples
        in the batch have the same length. With sequence lengths enabled, the
        padded part will be 0-padded after execution of this function, even if
        the input was not 0-padded.
        >>> a[1, :, :, 2] = 0
        >>> out = chunk_rnn(a, torch.tensor([3, 2], dtype=torch.int64))
        >>> out[1, :, :, 2]
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]], grad_fn=<SelectBackward0>)

        And for this case as well output and input shapes match
        >>> out.shape == a.shape
        True

        Chunk RNN along the time dimension k
        >>> out3 = _ChunkRNN(10, 20, '(b s) k n')(a, torch.tensor([3, 2], dtype=torch.int64))
        >>> out3.shape
        torch.Size([2, 10, 5, 3])
        >>> out3[1, :, :, 2]
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]], grad_fn=<SelectBackward0>)

        Input and output are the same with and without sequence length. By
        default, handling of sequence lengths is disabled when all examples in
        a batch have the same length to speed up computations. This can be
        disabled by passing `may_deactivate_seq=False`.
        >>> a = torch.randn(1, 10, 5, 3)
        >>> out_no_seq = chunk_rnn(a)
        >>> out_seq = chunk_rnn(a, torch.tensor([3]), may_deactivate_seq=False)
        >>> bool(torch.all(out_no_seq == out_seq))
        True

        >>> padded_a = torch.cat([a, torch.zeros(1, 10, 5, 2)], dim=-1)
        >>> out_seq = chunk_rnn(a, torch.tensor([3]), may_deactivate_seq=False)
        >>> bool(torch.all(out_no_seq == out_seq[..., :3]))
        True
        >>> bool(torch.all(out_seq[..., 3:] == 0))
        True

        Same check for the chunk rnn along time dimension
        >>> chunk_rnn = _ChunkRNN(10, 20, '(b s) k n')
        >>> out_no_seq = chunk_rnn(a)
        >>> out_seq = chunk_rnn(a, torch.tensor([3]), may_deactivate_seq=False)
        >>> bool(torch.all(out_no_seq == out_seq))
        True

        >>> padded_a = torch.cat([a, torch.zeros(1, 10, 5, 2)], dim=-1)
        >>> out_seq = chunk_rnn(a, torch.tensor([3]), may_deactivate_seq=False)
        >>> bool(torch.all(out_no_seq == out_seq[..., :3]))
        True
        >>> bool(torch.all(out_seq[..., 3:] == 0))
        True

    """

    def __init__(self, feat_size: int, rnn_size: int, lstm_reshape_to: str,
                 rnn_type='blstm'):
        """
        Args:
            feat_size: The features size (N)
            rnn_size: Number of units in the RNN (in each direction if
                bidirectional)
            lstm_reshape_to: A shape string as used by `einops.rearrange` to
                reshape prior to processing. This string should contain the
                following dimensions:
                  - b: batch size
                  - n: feature size
                  - k: segment length
                  - s: segment count
                and must result in a 3-dimensional tensor. An example to
                perform processing along the segment dimension ("inter-chunk")
                is '(b k) s n'.
            rnn_type: The type of the network used for processing. Can be one of
                'lstm', 'blstm', 'cnn', 'gru', 'bgru'.
        """
        super().__init__()

        if rnn_type in ('lstm', 'blstm'):
            self.rnn = torch.nn.LSTM(
                input_size=feat_size,
                hidden_size=rnn_size,
                bidirectional=rnn_type == 'blstm',
                batch_first=True,
            )
        elif rnn_type == 'cnn':
            # TODO: what kernel size?
            self.rnn = torch.nn.Sequential(
                Rearrange('b l n -> b n l'),
                torch.nn.Conv1d(feat_size, rnn_size, 3, padding=1),
                Rearrange('b n l -> b l n'),
            )
        elif rnn_type in ('gru', 'bgru'):
            self.rnn = torch.nn.GRU(
                input_size=feat_size,
                hidden_size=rnn_size,
                num_layers=1,
                batch_first=True,
                bidirectional=rnn_type == 'bgru'
            )
        else:
            raise ValueError(f'Unknown rnn_type for chunk RNN: {rnn_type}')

        self.fc = torch.nn.Linear(
            in_features=2 * rnn_size if rnn_type == 'blstm' else rnn_size,
            out_features=feat_size
        )

        self.norm = torch.nn.LayerNorm((feat_size,))
        self.lstm_reshape_to = lstm_reshape_to
        self.feat_size = feat_size

    def forward(self, sequence: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor] = None,
                may_deactivate_seq: bool = True) -> torch.Tensor:
        """

        Args:
            sequence (B, N, K, S): Chunked input sequence
            sequence_lengths (B): Sequence lengths along segment dimension (S)
            may_deactivate_seq: If set to `True`, the handling of sequence
                lengths is disabled when all examples in the batch have the
                same length

        """
        # The handling of sequence lengths can be disabled if all examples in a
        # batch have the same length and this length matches the size of the
        # time axis of the input sequence (i.e., the signal is not 0-padded)
        # This speeds up the computations
        if may_deactivate_seq and sequence_lengths is not None and (
                len(sequence_lengths) == 1 or all(
            sequence_lengths[1:] == sequence_lengths[:-1])
        ) and sequence_lengths[0] == sequence.shape[-1]:
            sequence_lengths = None

        B, N, K, S = sequence.shape

        # LSTM only support 3-dim input. Reshape according to given shape
        lstm_in = rearrange(sequence, f'b n k s -> {self.lstm_reshape_to}')

        # Call lstm
        if sequence_lengths is not None:
            # TODO: don't hardcode this
            if 's' in self.lstm_reshape_to[:4]:
                packed = pack(rearrange(sequence, 'b n k s -> b s k n'),
                              sequence_lengths)
            else:
                assert self.lstm_reshape_to[1] == 'b'
                packed_sequence_lengths = rearrange(
                    sequence_lengths.reshape(B, 1, 1, 1).expand(B, 1, K, 1),
                    f'b n k s -> {self.lstm_reshape_to}'
                ).squeeze()
                packed = pack_padded_sequence(lstm_in, packed_sequence_lengths,
                                              batch_first=True)
        else:
            packed_sequence_lengths = None
            packed = lstm_in

        out = self.rnn(packed)
        if isinstance(out, tuple):
            out = out[0]

        if sequence_lengths is not None and 's' not in self.lstm_reshape_to[:4]:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=S)

        # FC projection layer
        out = self.fc(out)

        # Apply norm and rearrange back to BxNxKxS
        if sequence_lengths is not None and 's' in self.lstm_reshape_to[:4]:
            out = self.norm(out)
            out = rearrange(
                unpack(out, sequence_lengths),
                'b s k n -> b n k s'
            )
        else:
            out = apply_examplewise(self.norm, out, packed_sequence_lengths)
            out = rearrange(out, f'{self.lstm_reshape_to} -> b n k s', b=B, s=S,
                            n=self.feat_size, k=K)

        # Residual connection
        out = out + sequence

        return out

    def flatten_parameters(self) -> None:
        """
        Calls `flatten_parameters` on `self.rnn` if it is a RNN. Does nothing
        in case of CNN.
        """
        if hasattr(self.rnn, 'flatten_parameters'):
            self.rnn.flatten_parameters()


class DPRNNBlock(torch.nn.Module):
    """
    One DPRNN Block consisting of an intra-chunk and an inter-chunk RNN
    """

    def __init__(self, feat_size: int, rnn_size: int,
                 inter_chunk_type: str = 'blstm',
                 intra_chunk_type: str = 'blstm'):
        super().__init__()

        # Chunk RNN along chunk dimension (K)
        self.intra_chunk_rnn = _ChunkRNN(
            feat_size=feat_size,
            rnn_size=rnn_size,
            lstm_reshape_to='(b s) k n',
            rnn_type=intra_chunk_type,
        )

        # Chunk RNN along time dimension (S)
        self.inter_chunk_rnn = _ChunkRNN(
            feat_size=feat_size,
            rnn_size=rnn_size,
            lstm_reshape_to='(b k) s n',
            rnn_type=inter_chunk_type,
        )

    def forward(
            self,
            sequence: torch.Tensor,
            sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        sequence = self.intra_chunk_rnn(sequence, sequence_lengths)
        sequence = self.inter_chunk_rnn(sequence, sequence_lengths)
        return sequence

    def flatten_parameters(self) -> None:
        self.intra_chunk_rnn.flatten_parameters()
        self.inter_chunk_rnn.flatten_parameters()


class DPRNN(torch.nn.Module):
    """
    This is the Dual-Path RNN implementation, not the source separator.
    """

    def __init__(
            self,
            input_size: int,
            rnn_size: int,
            window_length: int,
            hop_size: int,
            num_blocks: int,
            inter_chunk_type: 'str' = 'blstm',
            intra_chunk_type='blstm',
    ):
        """

        Args:
            input_size: The feature size (N)
            rnn_size: The units of the RNNs in each direction
            window_length: Length of window for segmentation (in frames)
            hop_size: Hop size for segmentation (in frames)
            num_blocks: Number of DPRNN blocks in this DPRNN
            inter_chunk_type: NN type for the inter-chunk RNN
            intra_chunk_type: NN type for the inter-chunk RNN
        """
        super().__init__()
        self.window_size = window_length
        self.hop_size = hop_size

        # Naming is taken from torch.nn.LSTM. In the DPRNN, all sizes are
        # always equal
        self.input_size = self.hidden_size = input_size

        self.dprnn_blocks = torch.nn.Sequential(*[
            DPRNNBlock(
                feat_size=input_size,
                rnn_size=rnn_size,
                inter_chunk_type=inter_chunk_type,
                intra_chunk_type=intra_chunk_type,
            ) for _ in range(num_blocks)
        ])

    def calculate_window_and_hop_size(
            self, sequence: torch.Tensor,
            sequence_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[int, int]:
        """
        Determine parameters for segmentation. If set to 'auto', use the
        heuristics from [1] Sec. 2.2 K \approx \sqrt{2L}.
        """

        if self.window_size == 'auto' or self.hop_size == 'auto':
            assert self.window_size == self.hop_size == 'auto', (
                'Set both window_size and hop_size or none of them!'
            )
            assert sequence_lengths is None or len(sequence_lengths) == 1, (
                'Variable length window and hop size (window_size = hop_size = '
                '"auto") are not supported (impossible) with non-unique '
                'sequence lengths in one batch! Either supply examples without '
                'sequence length or reduce the batch size to 1.'
            )
            window_size = int(math.sqrt(2 * sequence.shape[-2]))
            hop_size = window_size // 2
        else:
            window_size = self.window_size
            hop_size = self.hop_size

        return window_size, hop_size

    def forward(
            self,
            sequence: torch.Tensor,
            sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Args:
            sequence (B, L, N):
            sequence_lengths:

        Returns:

        """
        if isinstance(sequence, PackedSequence):
            warnings.warn(
                'DPRNN does not support packed sequences. Unpacking it again!')
            sequence, sequence_lengths = pad_packed_sequence(
                sequence, batch_first=True
            )

        # Make sure that the sequence lengths are a Tensor
        if not torch.is_tensor(sequence_lengths) and sequence_lengths is not None:
            sequence_lengths = torch.tensor(sequence_lengths)

        # Segment
        window_size, hop_size = self.calculate_window_and_hop_size(
            sequence, sequence_lengths)

        segmented, sequence_lengths = segment(
            sequence, hop_size=hop_size, window_size=window_size,
            sequence_lengths=sequence_lengths)

        # Flatten parameters for the case of multi-gpu (no idea why this is
        # required or what impact it has on the performance, but this stops the
        # "RNN module weights are not part of a single contiguous chunk of
        # memory" warnings.)
        self.flatten_parameters()

        # Call DPRNN blocks. It is not possible to use torch.nn.Sequential here
        # because each iteration needs the sequence lengths if provided
        h = segmented
        for block in self.dprnn_blocks:
            h = block(h, sequence_lengths)

        # Overlap add
        out = overlap_add(h, hop_size=hop_size, unpad=True)

        return out

    def flatten_parameters(self) -> None:
        """
        Calls `flatten_parameters` on all contained RNN modules
        """
        for block in self.dprnn_blocks:
            block.flatten_parameters()
