import numpy as np
import torch
from scipy.stats import truncnorm, truncexpon
from torch import nn
from torch.nn.functional import interpolate
from paderbox.transform.module_fbank import hz2mel, mel2hz
from einops import rearrange
from padertorch.utils import to_list
from typing import Tuple, List

import torch.nn.functional as F
from padertorch.contrib.je.modules.conv import Pad
from einops import rearrange


def hz_warping(f, n, alpha_sampling_fn, fhi_sampling_fn):
    """
    >>> from functools import partial
    >>> np.random.seed(0)
    >>> sample_rate = 16000
    >>> fmin = 0
    >>> fmax = sample_rate/2
    >>> n_mels = 10
    >>> alpha_max = 1.2
    >>> kwargs = dict(
    ...     alpha_sampling_fn=partial(
    ...         log_uniform_sampling_fn, scale=2*np.log(alpha_max)
    ...     ),
    ...     fhi_sampling_fn=partial(
    ...         uniform_sampling_fn, center=.7, scale=.2
    ...     ),
    ... )
    >>> f = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels+2))
    >>> f
    array([   0.        ,  180.21928115,  406.83711843,  691.7991039 ,
           1050.12629534, 1500.70701371, 2067.29249375, 2779.74887082,
           3675.63149949, 4802.16459006, 6218.73051459, 8000.        ])
    >>> hz_warping(f, (), **kwargs)
    array([   0.        ,  183.45581459,  414.14345066,  704.2230295 ,
           1068.98537001, 1527.65800595, 2104.41871721, 2829.67000102,
           3741.64166342, 4888.40600786, 6305.18977698, 8000.        ])
    >>> hz_warping(f, 2, **kwargs)
    array([[   0.        ,  187.10057293,  422.37133266,  718.21398838,
            1090.22314517, 1558.00833455, 2146.22768187, 2885.88769767,
            3815.97770824, 4985.52508038, 6350.49184281, 8000.        ],
           [   0.        ,  183.19308056,  413.5503401 ,  703.21448495,
            1067.45443547, 1525.47018891, 2101.40489926, 2825.61752316,
            3736.28311632, 4881.40513599, 6293.32469307, 8000.        ]])
    >>> hz_warping(f, [2, 3], **kwargs).shape
    (2, 3, 12)
    """
    fmin = f[0]
    f = f - fmin
    fmax = f[-1]
    alphas = np.array(alpha_sampling_fn(n))
    fhi = np.array(fhi_sampling_fn(n) * fmax)
    breakpoints = fhi * np.minimum(alphas, 1) / alphas

    if breakpoints.ndim == 0:
        breakpoints = np.array(breakpoints)
    breakpoints[(breakpoints > fmax) + ((alphas * breakpoints) > fmax)] = fmax
    bp_value = alphas * breakpoints

    f, breakpoints, bp_value = np.broadcast_arrays(
        f, breakpoints[..., None], bp_value[..., None]
    )
    f_warped = alphas[..., None] * f
    idx = f > breakpoints
    f_warped[idx] = (
        fmax
        + (
            (f[idx] - fmax)
            * (fmax - bp_value[idx]) / (fmax - breakpoints[idx])
        )
    )
    return fmin + f_warped


def mel_warping(f, n, alpha_sampling_fn, fhi_sampling_fn):
    f = hz2mel(f)
    f = hz_warping(f, n, alpha_sampling_fn, fhi_sampling_fn)
    return mel2hz(f)


class HzWarping:
    def __init__(self, alpha_sampling_fn, fhi_sampling_fn):
        self.alpha_sampling_fn = alpha_sampling_fn
        self.fhi_sampling_fn = fhi_sampling_fn

    def __call__(self, f, n):
        return hz_warping(f, n, self.alpha_sampling_fn, self.fhi_sampling_fn)


class MelWarping(HzWarping):
    def __call__(self, f, n):
        return mel_warping(f, n, self.alpha_sampling_fn, self.fhi_sampling_fn)


def truncexponential_sampling_fn(n, shift=0., scale=1., truncation=3.):
    return truncexpon(truncation/scale, shift, scale).rvs(n)


def uniform_sampling_fn(n, center=0., scale=1.):
    """
    Same as:
        np.random.uniform(
            low=center - scale / 2,
            high=center + scale / 2,
            size=n
        )
    """
    if np.isscalar(n):
        return center - scale / 2 + scale * np.random.rand(n)
    else:
        return center - scale / 2 + scale * np.random.rand(*n)


def log_uniform_sampling_fn(n, center=0., scale=1.):
    return np.exp(uniform_sampling_fn(n, center, scale))


def truncnormal_sampling_fn(n, center=0., scale=.5, truncation=3.):
    return (
        truncnorm(-truncation / scale, truncation / scale, center, scale).rvs(n)
    )


def log_truncnormal_sampling_fn(n, center=0., scale=.5, truncation=3.):
    return np.exp(truncnormal_sampling_fn(n, center, scale, truncation))


class TruncExponentialSampler:
    def __init__(self, shift=0., scale=1., truncation=3.):
        self.shift = shift
        self.scale = scale
        self.truncation = truncation

    def __call__(self, n):
        return truncexponential_sampling_fn(
            n, shift=self.shift, scale=self.scale, truncation=self.truncation
        )


class UniformSampler:
    def __init__(self, center=0., scale=1.):
        self.center = center
        self.scale = scale

    def __call__(self, n):
        return uniform_sampling_fn(n, center=self.center, scale=self.scale)


class LogUniformSampler(UniformSampler):
    def __call__(self, n):
        return log_uniform_sampling_fn(n, center=self.center, scale=self.scale)


class TruncNormalSampler:
    def __init__(self, center=0., scale=1., truncation=3.):
        self.center = center
        self.scale = scale
        self.truncation = truncation

    def __call__(self, n):
        return truncnormal_sampling_fn(
            n, center=self.center, scale=self.scale, truncation=self.truncation
        )


class LogTruncNormalSampler(TruncNormalSampler):
    def __call__(self, n):
        return log_truncnormal_sampling_fn(
            n, center=self.center, scale=self.scale, truncation=self.truncation
        )


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


class Mixup(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), 0)
    >>> y = torch.arange(3).float()
    >>> mixup = Mixup(p=1., interpolate=True)
    >>> mixup(x, seq_len=[3,4,5])
    """
    def __init__(self, p, weight_sampling_fn=lambda n: np.random.beta(1., 1., n), interpolate=False):
        super().__init__()
        self.p = p
        self.weight_sampling_fn = weight_sampling_fn
        self.interpolate = interpolate

    def forward(self, *tensors, seq_len=None):
        if self.training:
            B = tensors[0].shape[0]
            shuffle_idx = np.random.permutation(B)
            lambda2 = np.random.binomial(1, self.p, B)
            if seq_len is not None:
                seq_len = np.maximum(seq_len, lambda2*np.array(seq_len)[shuffle_idx])
            lambda2 = lambda2 * self.weight_sampling_fn(B)
            lambda2 = torch.from_numpy(lambda2).float().to(tensors[0].device)
            if self.interpolate:
                assert all(lambda2 >= 0.) and all(lambda2 <= 1.)
                lambda1 = 1. - lambda2
            else:
                lambda1 = torch.ones_like(lambda2)
            tensors = list(tensors)
            for i, tensor in enumerate(tensors):
                x1 = tensor
                x2 = tensor[shuffle_idx]
                lambda1_ = lambda1[(...,) + (x1.dim() - 1) * (None,)]
                lambda2_ = lambda2[(...,) + (x2.dim() - 1) * (None,)]
                tensors[i] = lambda1_ * x1 + lambda2_ * x2
        return (*tensors, seq_len)


class Crop(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), -1)
    >>> Crop(min_crop_rate=0.5)(x, seq_len=[3,4,5])
    """
    def __init__(self, max_cutoff_rate=.1):
        super().__init__()
        self.max_cutoff_rate = max_cutoff_rate

    def forward(self, *tensors, seq_len=None, seq_axes=-1):
        """

        Args:
            tensors: features (BxFxT)
            seq_len:

        Returns:

        """
        if self.training:
            seq_axes = to_list(seq_axes, len(tensors))
            T = tensors[0].shape[seq_axes[0]]
            max_cutoff = int(self.max_cutoff_rate * min(seq_len))
            cutoff_front = int(np.random.rand() * (max_cutoff + 1))
            cutoff_end = int(np.random.rand() * (max_cutoff + 1))
            seq_len = np.minimum(
                np.array(seq_len) - cutoff_front,
                T - (cutoff_front + cutoff_end)
            ).astype(np.int)
            tensors = list(tensors)
            for i, tensor in enumerate(tensors):
                tensors[i] = tensor.narrow(
                    seq_axes[i], cutoff_front, T - cutoff_end
                )
        return (*tensors, seq_len)


class Resample(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), -1)
    >>> Resample(rate_sampling_fn=LogUniformSampler(scale=.5))(x, seq_len=[3,4,5])
    """
    def __init__(self, rate_sampling_fn):
        super().__init__()
        self.rate_sampling_fn = rate_sampling_fn

    def forward(self, *tensors, seq_len=None, interpolation_mode='linear'):
        """

        Args:
            tensors: features (BxFxT)
            seq_len:
            interpolation_mode:

        Returns:

        """
        if self.training:
            rate = self.rate_sampling_fn(1)[0]
            seq_len = (rate * np.array(seq_len)).astype(np.int)
            tensors = list(tensors)
            for i, tensor in enumerate(tensors):
                if tensor.dim() == 4:
                    tensor = rearrange(tensor, 'b c f t -> b (c f) t')
                assert tensor.dim() == 3, tensor.shape
                tensor = interpolate(tensor, scale_factor=rate, mode=interpolation_mode)
                if tensors[i].dim() == 4:
                    b, c, f, _ = tensors[i].shape
                    t = tensor.shape[-1]
                    tensor = tensor.view((b, c, f, t))
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


class Noise(nn.Module):
    """
    >>> x = torch.zeros((3, 4, 5))
    >>> Noise(1.)(x)
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
        >>> blur = GaussianBlur2d(5, lambda n: [.5, 1., 2.])
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
