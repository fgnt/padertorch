import numpy as np
import torch
import torch.nn.functional as F
from padertorch.contrib.je.modules.conv import Pad
from torch import nn


class Scale(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> x = Scale(log_truncnormal_sampling_fn)(x)
    """
    def __init__(self, scale_sampling_fn, **kwargs):
        super().__init__()
        self.scale_sampling_fn = scale_sampling_fn
        self.kwargs = kwargs

    def forward(self, x):
        if not self.training:
            return x
        scales = self.scale_sampling_fn(x.shape[0], **self.kwargs)
        scales = torch.from_numpy(scales[(...,) + (x.dim()-1)*(None,)]).float()
        return x * scales.to(x.device)


class Shift(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> Shift(truncnormal_sampling_fn, scale=0.5)(x)
    """
    def __init__(self, shift_sampling_fn, **kwargs):
        super().__init__()
        self.shift_sampling_fn = shift_sampling_fn
        self.kwargs = kwargs

    def forward(self, x):
        if not self.training:
            return x
        shifts = self.shift_sampling_fn(x.shape[0], **self.kwargs)
        shifts = torch.from_numpy(shifts[(...,) + (x.dim()-1)*(None,)]).float()
        return x + shifts.to(x.device)


class TimeWarping(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 1, 4, 5)), -1) - 1
    >>> resampling_factors = np.array([1,2,3])
    >>> warping_fn=lambda seq_len: (\
            np.arange(max(seq_len))/resampling_factors[:, None],\
            np.minimum(resampling_factors*np.array(seq_len), max(seq_len))\
        )
    >>> TimeWarping(warping_fn=warping_fn)(x, seq_len=[3,4,5])
    (tensor([[[[0.0000, 1.0000, 2.0000, 3.0000, 4.0000],
              [0.0000, 1.0000, 2.0000, 3.0000, 4.0000],
              [0.0000, 1.0000, 2.0000, 3.0000, 4.0000],
              [0.0000, 1.0000, 2.0000, 3.0000, 4.0000]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[0.0000, 0.5000, 1.0000, 1.5000, 2.0000],
              [0.0000, 0.5000, 1.0000, 1.5000, 2.0000],
              [0.0000, 0.5000, 1.0000, 1.5000, 2.0000],
              [0.0000, 0.5000, 1.0000, 1.5000, 2.0000]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[0.0000, 0.3333, 0.6667, 1.0000, 1.3333],
              [0.0000, 0.3333, 0.6667, 1.0000, 1.3333],
              [0.0000, 0.3333, 0.6667, 1.0000, 1.3333],
              [0.0000, 0.3333, 0.6667, 1.0000, 1.3333]]]]), array([3, 5, 5]))
    """
    def __init__(self, warping_fn, batch_axis=0, sequence_axis=-1):
        super().__init__()
        self.warping_fn = warping_fn
        self.batch_axis = batch_axis
        self.sequence_axis = sequence_axis

    def forward(self, *tensors, seq_len):
        """

        Args:
            tensors:
            seq_len:

        Returns:

        """
        if self.training:
            assert seq_len is not None
            time_indices, seq_len = self.warping_fn(seq_len)
            time_indices_ceil = np.ceil(time_indices).astype(np.int)
            time_indices_floor = np.floor(time_indices).astype(np.int)
            batch_indices = np.arange(len(seq_len)).astype(np.int)[:, None]
            tensors = list(tensors)
            for i, tensor in enumerate(tensors):
                batch_axis = self.batch_axis
                seq_axis = self.sequence_axis
                if batch_axis < 0:
                    batch_axis = tensor.dim() + batch_axis
                if seq_axis < 0:
                    seq_axis = tensor.dim() + seq_axis
                if batch_axis != 0:
                    tensor = tensor.transpose(0, batch_axis)
                    if seq_axis == 0:
                        seq_axis = batch_axis
                if seq_axis != 1:
                    tensor = tensor.transpose(1, seq_axis)
                ceil_weights = \
                    torch.Tensor(1 - time_indices_ceil + time_indices).to(tensor.device)
                floor_weights = (
                    torch.Tensor(1 - time_indices + time_indices_floor)
                    * torch.Tensor(time_indices_floor != time_indices_ceil)
                ).to(tensor.device)
                for _ in range(tensor.dim() - 2):
                    ceil_weights = ceil_weights.unsqueeze(-1)
                    floor_weights = floor_weights.unsqueeze(-1)
                tensor = (
                    tensor[batch_indices, time_indices_ceil] * ceil_weights
                    + tensor[batch_indices, time_indices_floor] * floor_weights
                )
                if seq_axis != 1:
                    tensor = tensor.transpose(1, seq_axis)
                if batch_axis != 0:
                    tensor = tensor.transpose(0, batch_axis)
                tensors[i] = tensor
        return (*tensors, seq_len)


class Mask(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> x = Mask(axis=-1, max_masked_rate=1., max_masked_steps=10)(x, seq_len=[1,2,3])
    """
    def __init__(self, axis, n_masks=1, max_masked_steps=None, max_masked_rate=1.):
        super().__init__()
        self.axis = axis
        self.n_masks = n_masks
        self.max_masked_values = max_masked_steps
        self.max_masked_rate = max_masked_rate

    def __call__(self, x, seq_len=None):
        if not self.training:
            return x
        mask = torch.ones_like(x)
        idx = torch.arange(x.shape[self.axis]).float()
        axis = self.axis
        if axis < 0:
            axis = x.dim() + axis
        idx = idx[(...,) + (x.dim() - axis - 1)*(None,)]
        idx = idx.expand(x.shape)
        if seq_len is None:
            seq_len = x.shape[axis] * torch.ones(x.shape[0])
        else:
            seq_len = torch.Tensor(seq_len)
        max_width = self.max_masked_rate/self.n_masks * seq_len
        if self.max_masked_values is not None:
            max_width = torch.min(self.max_masked_values*torch.ones_like(max_width)/self.n_masks, max_width)
        max_width = torch.floor(max_width)
        for i in range(self.n_masks):
            width = torch.floor(torch.rand(x.shape[0]) * (max_width + 1))
            max_onset = seq_len - width
            onset = torch.floor(torch.rand(x.shape[0]) * (max_onset + 1))
            width = width[(...,) + (x.dim()-1)*(None,)]
            onset = onset[(...,) + (x.dim()-1)*(None,)]
            offset = onset + width
            mask = mask * ((idx < onset) + (idx >= offset)).float().to(x.device)
        return x * mask


class AdditiveNoise(nn.Module):
    """
    >>> x = torch.zeros((3, 4, 5))
    >>> AdditiveNoise(1.)(x)
    """
    def __init__(self, max_scale):
        super().__init__()
        self.max_scale = max_scale

    def forward(self, x):
        if self.training:
            B = x.shape[0]
            scale = torch.rand(B).to(x.device) * self.max_scale
            x = x + scale[(...,) + (x.dim()-1)*(None,)] * torch.randn_like(x)
        return x


class GaussianBlur2d(nn.Module):
    r"""Copied (and slightly adapted) from
    https://github.com/kornia/kornia/blob/master/kornia/filters

    Creates an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.
    Arguments:
        kernel_size: the size of the kernel.
        sigma_sampling_fn: the standard deviation of the kernel.
        pad_mode (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
    Returns:
        Tensor: the blurred tensor.
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    Examples::
        >>> x = torch.zeros(3, 1, 5, 5)
        >>> x[:,:, 2, 2] = 1
        >>> blur = GaussianBlur2d(3, lambda n: [.5, 1., 2.])
        >>> output = blur(x)
        >>> output.shape
        torch.Size([3, 1, 5, 5])
        >>> output
        tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0113, 0.0838, 0.0113, 0.0000],
                  [0.0000, 0.0838, 0.6193, 0.0838, 0.0000],
                  [0.0000, 0.0113, 0.0838, 0.0113, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0751, 0.1238, 0.0751, 0.0000],
                  [0.0000, 0.1238, 0.2042, 0.1238, 0.0000],
                  [0.0000, 0.0751, 0.1238, 0.0751, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.1019, 0.1154, 0.1019, 0.0000],
                  [0.0000, 0.1154, 0.1308, 0.1154, 0.0000],
                  [0.0000, 0.1019, 0.1154, 0.1019, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
    """

    def __init__(
            self, kernel_size, sigma_sampling_fn, pad_mode: str = 'reflect'
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_sampling_fn = sigma_sampling_fn
        assert pad_mode in ["constant", "reflect", "replicate", "circular"]
        self.pad_mode = pad_mode

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(kernel_size=' + str(self.kernel_size) + ', ' +\
            'sigma_sampling_fn=' + repr(self.sigma_sampling_fn) + ', ' +\
            'border_type=' + self.pad_mode + ')'

    def forward(self, x):

        if not x.dim() == 4:
            raise ValueError(
                f"Invalid input shape, we expect BxCxHxW. Got: {x.shape}")
        if not self.training:
            return x

        # pad the input tensor
        x = Pad(mode=self.pad_mode, side='both')(x, size=self.kernel_size-1)
        b, c, hp, wp = x.shape
        # convolve the tensor with the kernel.
        sigma = torch.from_numpy(np.array(self.sigma_sampling_fn(b))).float()
        kernel = get_gaussian_kernel2d(self.kernel_size, sigma).unsqueeze(1).to(x.device)
        return F.conv2d(
            x.transpose(0, 1), kernel, groups=b, padding=0, stride=1
        ).transpose(0, 1)


def get_gaussian_kernel2d(kernel_size, sigma, force_even: bool = False):
    r"""Copied (and slightly adapted) from
    https://github.com/kornia/kornia/blob/master/kornia/filters

    Function that returns Gaussian filter matrix coefficients.
    Args:
        kernel_size: filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma: gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
    Examples::
        >>> get_gaussian_kernel2d(3, 1.5)
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d(3, [1.5, .000001])
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
    """
    kernel_1d: torch.Tensor = get_gaussian_kernel1d(
        kernel_size, sigma, force_even
    )
    kernel_2d: torch.Tensor = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(-2)
    return kernel_2d


def get_gaussian_kernel1d(
        kernel_size: int, sigma: float, force_even: bool = False
) -> torch.Tensor:
    r"""Copied (and slightly adapted) from
    https://github.com/kornia/kornia/blob/master/kornia/filters

    Function that returns Gaussian filter coefficients.
    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size})`
    Examples::
        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])
        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            f"Got {kernel_size}"
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def gaussian(window_size, sigma):
    """Copied (and slightly adapted) from
    https://github.com/kornia/kornia/blob/master/kornia/filters

    Args:
        window_size:
        sigma:

    Returns:

    """
    x = torch.arange(window_size).float() - window_size // 2
    if torch.is_tensor(sigma) and sigma.dim() > 0:
        sigma = sigma[..., None]
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2))).float()
    return gauss / gauss.sum(-1, keepdim=True)
