from typing import Tuple, Union

import torch
from torch.nn import functional as F
from padertorch.ops import STFT
from einops import rearrange


class TasEncoder(torch.nn.Module):
    """
    This class encapsulates just the TasNet encoder (1-D Conv + ReLU). It can
    be used as a module for feature extraction in other modules/models.
    """

    def __init__(self, window_length: int = 20, feature_size: int = 256,
                 stride: int = None, bias: bool = False):
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
            stride = window_length // 2

        self.window_length = window_length
        self.feature_size = feature_size
        self.stride = stride

        self.encoder_1d = torch.nn.Conv1d(
            1, feature_size, window_length,
            stride=self.stride, padding=0, bias=bias
        )

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

        if l % (self.window_length // 2) > 0:
            padding = self.window_length // 2 - (l % (self.window_length // 2))
            x = F.pad(x, (0, padding))
            sq_offset = 0

        # Compute new sequence_lengths
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths // (
                    self.window_length // 2) + sq_offset

        # Add channel dimension. Results in (B, 1, T)
        x = torch.unsqueeze(x, dim=1)

        # Call 1D encoder layer. Results in (B, N, T_enc)
        w = F.relu(self.encoder_1d(x))

        return w, sequence_lengths


class TasDecoder(torch.nn.Module):
    """
    Encapsulates the decoder of the TasNet in a separate module
    """

    def __init__(self, window_length: int = 20, feature_size: int = 256,
                 stride: int = None, bias=False):
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
            stride = window_length // 2

        self.window_length = window_length
        self.feature_size = feature_size
        self.stride = stride

        self.decoder_1d = torch.nn.ConvTranspose1d(
            feature_size, 1, kernel_size=window_length,
            stride=self.stride, bias=bias
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


class StftEncoder(torch.nn.Module):
    """
    >>> mixture = torch.rand((2, 6, 203))
    >>> stft_encoder = StftEncoder(feature_size=258)
    >>> encoded, num_frames = stft_encoder(mixture, [203, 150])
    >>> encoded.shape
    torch.Size([2, 6, 258, 20])
    >>> num_frames
    tensor([20, 14])
    >>> from paderbox.transform import stft
    >>> import numpy as np
    >>> stft_out = stft(\
            mixture.numpy(), 256, 10, window_length=20, \
            fading=None)
    >>> stft_encoded = np.concatenate(\
            [np.real(stft_out), np.imag(stft_out)], axis=-1\
        ).transpose(0,1,3,2)
    >>> np.testing.assert_allclose(stft_encoded, encoded, atol=1e-5)
    """
    def __init__(self, window_length: int = 20, feature_size: int = 256,
                 stride: int = None):
        super().__init__()
        self.window_length = window_length
        self.feature_size = feature_size
        self.stride = stride

        if stride is None:
            stride = window_length // 2

        # feature_size - 2 because the stft adds two uninformative
        # values for an even size.
        # Note that the STFT does not support uneven sizes at the moment.
        self.stft = STFT(
            size=feature_size - 2, shift=stride, window_length=window_length,
            fading=False, complex_representation='concat'
        )

    def forward(self, inputs, sequence_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
            inputs: shape: [..., T], T is #samples
            sequence_lengths: list or tensor of #samples
        Returns:
            [..., N, frames] the stft encoded signal
        """

        encoded = self.stft(inputs)
        encoded = rearrange(encoded, '... frames fbins -> ... fbins frames')
        if sequence_lengths is not None:
            num_frames = torch.tensor([self.stft.samples_to_frames(samples)
                                       for samples in sequence_lengths])
            return encoded, num_frames
        else:
            return encoded


class IstftDecoder(torch.nn.Module):
    """
    >>> stft_signal = torch.rand((2, 4, 258, 10))
    >>> decoder = IstftDecoder(feature_size=258)
    >>> decoded = decoder(stft_signal)
    >>> decoded.shape
    torch.Size([2, 4, 110])
    >>> from paderbox.transform import istft
    >>> import numpy as np
    >>> signal_np = stft_signal.numpy().transpose(0, 1, 3, 2)
    >>> complex_signal = signal_np[..., :129] + 1j* signal_np[..., 129:]
    >>> stft_decoded = istft(\
            complex_signal, 256, 10, window_length=20, fading=False)
    >>> np.testing.assert_allclose(stft_decoded, decoded, atol=1e-5)
    """
    def __init__(self, window_length: int = 20, feature_size: int = 256,
                 stride: int = None):
        super().__init__()
        # Hyper-parameter
        self.window_length = window_length
        self.feature_size = feature_size
        self.stride = stride

        if stride is None:
            stride = window_length // 2

        # feature_size - 2 because the stft adds two uninformative
        # values for an even size.
        # Note that the STFT does not support uneven sizes at the moment.
        self.stft = STFT(
            size=feature_size - 2, window_length=window_length,
            shift=stride, fading=False, complex_representation='concat'
        )

    def forward(self, stft_signal) -> torch.Tensor:
        """
        Args:
            stft_signal: shape: [B, ..., N, Frames]
        Returns:
            [B, ..., T]
        """

        stft_signal = rearrange(
            stft_signal, '... fbins frames  -> ... frames fbins')

        return self.stft.inverse(stft_signal)
