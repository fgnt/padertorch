from typing import Tuple, Union

import torch
from torch.nn import functional as F


class TasEncoder(torch.nn.Module):
    """
    This class encapsulates just the TasNet encoder (1-D Conv + ReLU). It can
    be used as a module for feature extraction in other modules/models.
    """

    def __init__(self, L: int = 20, N: int = 256, stride: int = None,
                 bias: bool = False):
        """
        Args:
            L: The block size in samples (length of the filters).
            N: The feature size of the output (number of filters in the
                autoencoder).
            stride: The stride of the filters. Defaults to `L//2`
            bias: If `True`, a bias is added to the 1D convolution. This can
                improve the performance in some cases, but note that when set
                to `True`, the encoder is not scale-invariant anymore!
        """
        super().__init__()

        if stride is None:
            stride = L // 2

        self.L = L
        self.N = N
        self.stride = stride

        self.encoder_1d = torch.nn.Conv1d(1, N, L, stride=self.stride,
                                          padding=0, bias=bias)

    def forward(
            self,
            x: torch.Tensor,
            sequence_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """

        Args:
            x (B, T) or (T, ): Input signal as time series
            sequence_lengths: Optional sequence lengths of the input
                sequence(s). If this is not `None`, the resulting sequence
                lengths in the encoded domain will be computed and returned.

        Returns:
            (B, N, T_enc), sequence_lengths
            Encoded output, where T_enc is the time length in the encoded
            domain. `sequence_lengths` are the sequence lengths in the encoded
            domain or `None` if the passed `sequence_lengths` were `None`.
        """
        assert x.dim() in [1, 2], (
            f'The {self.__class__.__name__} ony supports 1D and 2D input, but '
            f'got {x.shape}.'
        )

        # Add batch dimension if not provided
        if x.ndimension() == 1:
            x = x.unsqueeze(0)

        # Add padding to the end to fill a whole window
        l = x.shape[-1]
        sq_offset = -1

        if l % (self.L // 2) > 0:
            padding = self.L // 2 - (l % (self.L // 2))
            x = F.pad(x, (0, padding))
            sq_offset = 0

        # Compute new sequence_lengths
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths // (self.L // 2) + sq_offset

        # Add channel dimension. Results in (B, 1, T)
        x = torch.unsqueeze(x, dim=1)

        # Call 1D encoder layer. Results in (B, N, T_enc)
        w = F.relu(self.encoder_1d(x))

        return w, sequence_lengths


class TasDecoder(torch.nn.Module):
    """
    Encapsulates the decoder of the TasNet in a separate module
    """

    def __init__(self, L: int = 20, N: int = 256, stride: int = None,
                 bias=False):
        """
        The arguments should match the used encoder.

        Args:
            L: The block size in samples (length of the filters).
            N: The feature size of the output (number of filters in the
                autoencoder).
            stride: The stride of the kernels. Defaults to `L//2` if not given.
            bias: Whether to use a bias in the 1D convolution or not. Note
                that the model will become scale-variant if `bias` is set to
                `True`!
        """
        super().__init__()

        if stride is None:
            stride = L // 2

        self.L = L
        self.N = N
        self.stride = stride

        self.decoder_1d = torch.nn.ConvTranspose1d(
            N, 1, kernel_size=L, stride=self.stride, bias=bias
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w (B, N, T_enc): The hidden representation to decode.

        Returns:
            (B, T)
            The reconstructed time signal
        """
        # Convolution from (B, N, T_enc) to (B, 1, T) and remove channel
        return self.decoder_1d(w)[:, 0, :]
