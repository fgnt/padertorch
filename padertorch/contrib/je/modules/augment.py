import numpy as np
import torch
from scipy.stats import truncnorm, truncexpon
from torch import nn
from torch.distributions import Beta
from torch.nn.functional import interpolate
from padertorch.contrib.je.modules.features import hz2mel, mel2hz
from einops import rearrange
from padertorch.utils import to_list


def linear_warping(f, n, alpha_sampling_fn, fhi_sampling_fn):
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
    >>> linear_warping(f, (), **kwargs)
    array([   0.        ,  183.45581459,  414.14345066,  704.2230295 ,
           1068.98537001, 1527.65800595, 2104.41871721, 2829.67000102,
           3741.64166342, 4888.40600786, 6305.18977698, 8000.        ])
    >>> linear_warping(f, 2, **kwargs)
    array([[   0.        ,  187.10057293,  422.37133266,  718.21398838,
            1090.22314517, 1558.00833455, 2146.22768187, 2885.88769767,
            3815.97770824, 4985.52508038, 6350.49184281, 8000.        ],
           [   0.        ,  183.19308056,  413.5503401 ,  703.21448495,
            1067.45443547, 1525.47018891, 2101.40489926, 2825.61752316,
            3736.28311632, 4881.40513599, 6293.32469307, 8000.        ]])
    >>> linear_warping(f, [2, 3], **kwargs).shape
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
    f = linear_warping(f, n, alpha_sampling_fn, fhi_sampling_fn)
    return mel2hz(f)


class LinearWarping:
    def __init__(self, alpha_sampling_fn, fhi_sampling_fn):
        self.alpha_sampling_fn = alpha_sampling_fn
        self.fhi_sampling_fn = fhi_sampling_fn

    def __call__(self, f, n):
        return linear_warping(f, n, self.alpha_sampling_fn, self.fhi_sampling_fn)


class MelWarping(LinearWarping):
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
    >>> mixup = Mixup(interpolate=True, p=0.5)
    >>> x, seq_len = mixup(x, seq_len=[3,4,5])
    """
    def __init__(self, interpolate=False, alpha=1., p=1.):
        super().__init__()
        self.interpolate = interpolate
        self.beta_dist = Beta(alpha, alpha)
        self.p = p

    def forward(self, *arrays, seq_len=None):
        if self.training:
            arr0 = arrays[0]
            shuffle_idx = np.random.permutation(arr0.shape[0])
            lambda2 = np.random.binomial(1, self.p, arr0.shape[0])
            if seq_len is not None:
                seq_len = np.maximum(seq_len, lambda2*np.array(seq_len)[shuffle_idx])
            lambda2 = torch.from_numpy(lambda2).float().to(arr0.device)
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
        return (*arrays, seq_len)


class ShiftedMixup(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), 0)
    >>> y = torch.arange(3).float()
    >>> mixup = ShiftedMixup(interpolate=True, p=0.5)
    >>> x, y, seq_len = mixup(x, y, seq_len=[3,4,5], seq_axis=[-1, None])
    >>> x, y, seq_len
    """
    def __init__(self, interpolate=False, alpha=1., p=1., max_shift=None):
        super().__init__()
        self.interpolate = interpolate
        self.beta_dist = Beta(alpha, alpha)
        self.p = p
        self.max_shift = max_shift

    def forward(self, *arrays, seq_len=None, seq_axis=-2):
        if self.training and np.random.rand() < self.p:
            seq_axis = to_list(seq_axis, len(arrays))
            arr0 = arrays[0]
            shuffle_idx = np.random.permutation(arr0.shape[0])
            if seq_len is None:
                seq_len = arr0.shape[0] * [arr0.shape[seq_axis[0]]]
            max_shift = min(seq_len)
            if self.max_shift is not None:
                max_shift = min(max_shift, self.max_shift)
            shift = int(np.random.randn() * max_shift)

            seq_len = np.maximum(seq_len, shift + np.array(seq_len)[shuffle_idx])
            if self.interpolate:
                lambda2 = self.beta_dist.sample((arr0.shape[0],)).to(arr0.device)
                lambda1 = 1. - lambda2
            else:
                lambda1 = lambda2 = torch.ones(arr0.shape[0]).to(arr0.device)
            arrays = list(arrays)
            for i, arr in enumerate(arrays):
                x1 = arr
                x2 = arr[shuffle_idx]
                if seq_axis[i] is not None and shift > 0:
                    pad_shape = list(arr.shape)
                    pad_shape[seq_axis[i]] = shift
                    pad = torch.zeros(tuple(pad_shape)).to(arr.device)
                    x1 = torch.cat((x1, pad), dim=seq_axis[i])
                    x2 = torch.cat((pad, x2), dim=seq_axis[i])
                lambda1_ = lambda1[(...,) + (x1.dim() - 1) * (None,)]
                lambda2_ = lambda2[(...,) + (x2.dim() - 1) * (None,)]
                arrays[i] = lambda1_ * x1 + lambda2_ * x2
        return (*arrays, seq_len)


class Resample(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), -1)
    >>> from functools import partial
    >>> Resample(rate_sampling_fn=LogUniformSampler(scale=.5))(x, seq_len=[3,4,5])
    """
    def __init__(self, rate_sampling_fn):
        super().__init__()
        self.rate_sampling_fn = rate_sampling_fn

    def forward(self, *arrays, seq_len=None, interpolation_mode='linear'):
        """

        Args:
            arrays: features (BxFxT)
            seq_len:
            interpolation_mode:

        Returns:

        """
        if self.training:
            rate = self.rate_sampling_fn(1)[0]
            seq_len = (rate * np.array(seq_len)).astype(np.int)
            arrays = list(arrays)
            for i, arr in enumerate(arrays):
                if arr.dim() == 4:
                    arr = rearrange(arr, 'b c f t -> b (c f) t')
                assert arr.dim() == 3, arr.shape
                arr = interpolate(arr, scale_factor=rate, mode=interpolation_mode)
                if arrays[i].dim() == 4:
                    b, c, f, _ = arrays[i].shape
                    _, _, t = arr.shape[-1]
                    arr = arr.view((b, c, f, t))
                arrays[i] = arr
        return (*arrays, seq_len)


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
