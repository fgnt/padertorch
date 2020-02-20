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
    cutoff = fhi * np.minimum(alphas, 1) / alphas

    if cutoff.ndim == 0:
        cutoff = cutoff[None]
    cutoff[(cutoff > fmax) + ((alphas * cutoff) > fmax)] = fmax
    cutoff_value = alphas * cutoff

    f, cutoff, cutoff_value = np.broadcast_arrays(
        f, cutoff[..., None], cutoff_value[..., None]
    )
    f_warped = alphas[..., None] * f
    idx = f > cutoff
    f_warped[idx] = (
        fmax
        + (
            (f[idx] - fmax)
            * (fmax - cutoff_value[idx]) / (fmax - cutoff[idx])
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
    def __init__(
            self, interpolate=False, alpha=1., p=1., shift=False, max_seq_len=None,
    ):
        super().__init__()
        self.interpolate = interpolate
        self.beta_dist = Beta(alpha, alpha)
        self.p = p
        self.shift = shift
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None, sequence_axis=None, cutoff_value=None, mixup_params=None):
        if mixup_params is not None or (
                self.training and (np.random.rand() < self.p)
        ):
            if mixup_params is not None:
                shuffle_idx, shift, lambdas = mixup_params
            else:
                shuffle_idx = np.random.permutation(x.shape[0])
                if self.shift:
                    assert sequence_axis is not None
                    seq_len_ = np.array(
                        x.shape[0] * [x.shape[sequence_axis]]
                    ) if seq_len is None else seq_len
                    max_shift = np.min(seq_len_)
                    if self.max_seq_len is not None:
                        max_shift = min(
                            max_shift, self.max_seq_len - np.max(seq_len_)
                        )
                    shift = int(np.random.rand() * (max_shift + 1))
                else:
                    shift = 0
                if self.interpolate:
                    lambda2 = self.beta_dist.sample((x.shape[0],)).to(x.device)
                    lambda1 = 1. - lambda2
                else:
                    lambda1 = lambda2 = torch.ones((x.shape[0],)).to(x.device)
                lambdas = (lambda1, lambda2)
            if shuffle_idx is not None:  # may be None when mixup_params are given from a call where no mixup was performed
                x1 = x
                x2 = x[shuffle_idx]
                if shift > 0 and sequence_axis is not None:
                    pad_shape = [*x.shape]
                    pad_shape[sequence_axis] = shift
                    pad = torch.zeros(tuple(pad_shape)).to(x.device)
                    x1 = torch.cat((x1, pad), dim=sequence_axis)
                    x2 = torch.cat((pad, x2), dim=sequence_axis)
                (lambda1, lambda2) = lambdas
                lambda1_ = lambda1[(...,) + (x.dim() - 1) * (None,)]
                lambda2_ = lambda2[(...,) + (x.dim() - 1) * (None,)]
                x = lambda1_ * x1 + lambda2_ * x2
                if cutoff_value is not None:
                    x = torch.min(x, cutoff_value*torch.ones_like(x))

                if seq_len is not None:
                    seq_len = np.array(seq_len)
                    seq_len = np.maximum(seq_len, (seq_len[shuffle_idx] + shift))
            mixup_params = (shuffle_idx, shift, lambdas)
        else:
            mixup_params = 3*(None,)
        return x, seq_len, mixup_params


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
    def __init__(self, axis, n_masks=1, max_masked_steps=None, max_masked_rate=0.2):
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
        for i in range(self.n_masks):
            if seq_len is None:
                seq_len = x.shape[axis] * torch.ones(x.shape[0])
            else:
                seq_len = torch.Tensor(seq_len)
            max_width = torch.floor(self.max_masked_rate * seq_len)
            if self.max_masked_values is not None:
                max_width = torch.min(self.max_masked_values*torch.ones_like(max_width), max_width)
            width = torch.floor(torch.rand(x.shape[0]) * (max_width + 1))
            max_onset = seq_len - width
            onset = torch.floor(torch.rand(x.shape[0]) * (max_onset + 1))
            width = width[(...,) + (x.dim()-1)*(None,)]
            onset = onset[(...,) + (x.dim()-1)*(None,)]
            offset = onset + width

            mask = mask * ((idx < onset) + (idx >= offset)).float().to(x.device)
        return x * mask
