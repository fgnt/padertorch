import numpy as np
import torch
from scipy.stats import truncnorm, truncexpon
from torch import nn
from torch.distributions import Beta
from torch.nn.functional import interpolate
from padertorch.contrib.je.modules.features import hz2mel, mel2hz


def linear_warping(f, n, alpha_sampling_fn, fhi_sampling_fn):
    fmin = f[0]
    f = f - fmin
    fmax = f[-1]
    alphas = np.array(alpha_sampling_fn(n))
    fhi = np.array(fhi_sampling_fn(n) * fmax)
    breakpoints = fhi * np.minimum(alphas, 1) / alphas

    if breakpoints.ndim == 0:
        breakpoints = breakpoints[None]
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
    f = linear_warping(f, n, alpha_sampling_fn, fhi_sampling_fn)
    return mel2hz(f)


def truncexponential_sampling_fn(n, shift=0., scale=.5, truncation=3.):
    return truncexpon(truncation/scale, shift, scale).rvs(n)


def uniform_sampling_fn(n, center=0., scale=1.):
    return center - scale / 2 + scale * np.random.rand(n)


def log_uniform_sampling_fn(n, center=0., scale=1.):
    return np.exp(uniform_sampling_fn(n, center, scale))


def truncnormal_sampling_fn(n, center=0., scale=.5, truncation=2.):
    return (
        truncnorm(-truncation / scale, truncation / scale, center, scale).rvs(n)
    )


def log_truncnormal_sampling_fn(n, center=0., scale=.5, truncation=2.):
    return np.exp(truncnormal_sampling_fn(n, center, scale, truncation))


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
    >>> mixup = Mixup(interpolate=True, p=0.5, shift=True)
    >>> x, seq_len, mixup_params = mixup(x, seq_len=[3,4,5], sequence_axis=-1)
    >>> mixup_params
    >>> mixup(y, mixup_params=mixup_params, cutoff_value=1.)[0]
    """
    def __init__(self, interpolate=False, alpha=1., p=1.):
        super().__init__()
        self.interpolate = interpolate
        self.beta_dist = Beta(alpha, alpha)
        self.p = p

    def forward(self, *arrays):
        if self.training:
            arr0 = arrays[0]
            shuffle_idx = np.random.permutation(arr0.shape[0])
            lambda2 = torch.from_numpy(
                np.random.binomial(1, self.p, arr0.shape[0])
            ).float().to(arr0.device)
            if self.interpolate:
                w = self.beta_dist.sample((arr0.shape[0],)).to(arr0.device)
                lambda2 = lambda2 * w
                lambda1 = 1. - lambda2
            else:
                lambda1 = torch.ones(arr0.shape[0]).to(arr0.device)
            arrays = list(arrays)
            for i, arr in enumerate(arrays):
                x1 = arr
                x2 = arr[shuffle_idx]
                lambda1_ = lambda1[(...,) + (x1.dim() - 1) * (None,)]
                lambda2_ = lambda2[(...,) + (x2.dim() - 1) * (None,)]
                arrays[i] = lambda1_ * x1 + lambda2_ * x2
        return (*arrays,)


class Resample(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), -1)
    >>> Resample(alpha_sampling_fn=log_uniform_sampling_fn, scale=.5)(x, seq_len=[3,4,5])
    """
    def __init__(self, alpha_sampling_fn, **kwargs):
        super().__init__()
        self.alpha_sampling_fn = alpha_sampling_fn
        self.kwargs = kwargs

    def forward(self, x, seq_len=None, alpha=None, interpolation_mode='linear'):
        """

        Args:
            x:
            x: features (BxFxT)
            seq_len:
            alpha:
            interpolation_mode:

        Returns:

        """
        assert x.dim() == 3, x.shape
        if alpha is not None or self.training:
            alpha = self.alpha_sampling_fn(1, **self.kwargs)[0] if alpha is None else alpha
            x = interpolate(x, scale_factor=alpha, mode=interpolation_mode)
            seq_len = (alpha * np.array(seq_len)).astype(np.int)
        return x, seq_len, alpha


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
