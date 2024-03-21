import numpy as np
from math import ceil
import torch
import torch.nn.functional as F
from padertorch.contrib.je.modules.conv import Pad
from torch import nn


class Scale(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> x = Scale(log_truncnormal_sampling_fn)(x)
    """
    def __init__(self, scale_sampling_fn):
        super().__init__()
        self.scale_sampling_fn = scale_sampling_fn

    def forward(self, x):
        if not self.training:
            return x
        scales = self.scale_sampling_fn(x.shape[0])
        scales = torch.from_numpy(scales[(...,) + (x.dim()-1)*(None,)]).float()
        return x * scales.to(x.device)


class Shift(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> Shift(truncnormal_sampling_fn, scale=0.5)(x)
    """
    def __init__(self, shift_sampling_fn):
        super().__init__()
        self.shift_sampling_fn = shift_sampling_fn

    def forward(self, x):
        if not self.training:
            return x
        shifts = self.shift_sampling_fn(x.shape[0])
        shifts = torch.from_numpy(shifts[(...,) + (x.dim()-1)*(None,)]).float()
        return x + shifts.to(x.device)


class TimeWarping(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 1, 4, 5)), -1) - 1
    >>> resampling_factors = np.array([1,2,.5])
    >>> warping_fn=lambda seq_len: (\
            np.minimum(np.arange(max(seq_len))/resampling_factors[:, None], max(seq_len)-1),\
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


class Superpose(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), 0)
    >>> y = np.array([[1,0,0],[1,1,1],[1,2,2]])
    >>> superpose = Superpose(p=1.)
    >>> superpose(x, seq_len=[3,4,5], targets=y)
    """
    def __init__(self, p, scale_fn=None):
        super().__init__()
        self.p = p
        self.scale_fn = scale_fn

    def forward(self, x, seq_len=None, labels=None):
        if self.training:
            B = x.shape[0]
            shuffle_idx = np.roll(np.arange(B, dtype=np.int), np.random.choice(B-1)+1)
            lambda_ = np.random.binomial(1, self.p, B)
            if seq_len is not None:
                seq_len = np.maximum(seq_len, (lambda_ > 0.) * np.array(seq_len)[shuffle_idx])
            lambda_ = torch.tensor(lambda_, dtype=torch.bool, device=x.device)
            lambda_x = lambda_[(...,) + (x.dim() - 1) * (None,)].float()
            x_shuffled = x[shuffle_idx]
            if self.scale_fn is not None:
                x = self.scale_fn(x)
                x_shuffled = self.scale_fn(x_shuffled)
            x = x + lambda_x * x_shuffled
            # ToDo: fix for sparse targets
            if isinstance(labels, (list, tuple)):
                raise NotImplementedError
                # targets = list(targets)
                # for i in range(len(targets)):
                #     lambda_t = lambda_[(...,) + (targets[i].dim() - 1) * (None,)]
                #     targets[i] = targets[i] | (lambda_t & targets[i][shuffle_idx])
            elif labels is not None:
                raise NotImplementedError
                # lambda_t = lambda_[(...,) + (targets.dim() - 1) * (None,)]
                # targets = targets | (lambda_t & targets[shuffle_idx])
        return x, seq_len, labels


class Mixup(nn.Module):
    """
    >>> x = torch.cumsum(torch.ones((3, 4, 5)), 0)
    >>> y = torch.eye(3).float()
    >>> y_sparse = torch.sparse_coo_tensor([[0,1,2],[0,1,2]],[1,1,1],(3,3)).float()
    >>> mixup = Mixup(p=1., target_threshold=0.3)
    >>> mixup.roll_targets(y,1)
    >>> mixup.roll_targets(y_sparse,1).to_dense()
    >>> mixup(x, seq_len=[3,4,5], targets=y_sparse)
    """
    def __init__(self, p, alpha=2., beta=1., target_threshold=None):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.target_threshold = target_threshold

    def forward(self, x, seq_len=None, targets=None):
        if self.training:
            B = x.shape[0]
            shift = 1+np.random.choice(B-1)
            shuffle_idx = np.roll(np.arange(B, dtype=np.int), shift)
            lambda_ = np.maximum(
                np.random.binomial(1, 1 - self.p, B),
                np.random.beta(self.alpha, self.beta, B),
            )
            if seq_len is not None:
                seq_len = np.maximum(seq_len, (lambda_ < 1.) * np.array(seq_len)[shuffle_idx])
            lambda_ = torch.from_numpy(lambda_).float().to(x.device)
            assert all(lambda_ >= 0.) and all(lambda_ <= 1.)
            lambda_x = lambda_[(...,) + (x.dim() - 1) * (None,)]
            x = lambda_x * x + (1. - lambda_x) * x[shuffle_idx]
            if isinstance(targets, (list, tuple)):
                targets = list(targets)
                for i in range(len(targets)):
                    targets[i] = targets[i].float()
                    rolled_targets = self.roll_targets(targets[i], shift)
                    lambda_t = lambda_[(...,) + (targets[i].dim() - 1) * (None,)]
                    targets[i] = lambda_t * targets[i] + (1. - lambda_t) * rolled_targets
                    if self.target_threshold is not None:
                        targets[i] = self.threshold_targets(targets[i])
            elif targets is not None:
                targets = targets.float()
                rolled_targets = self.roll_targets(targets, shift)
                lambda_t = lambda_[(...,) + (targets.dim() - 1) * (None,)]
                targets = lambda_t * targets + (1. - lambda_t) * rolled_targets
                if self.target_threshold is not None:
                    targets = self.threshold_targets(targets)
        return x, seq_len, targets

    @staticmethod
    def roll_targets(targets, shift):
        B = targets.shape[0]
        if targets.is_sparse:
            targets = targets.coalesce()
            targets_indices = targets.indices()
            targets_indices[0] = (targets_indices[0]+shift)%B
            return torch.sparse_coo_tensor(
                indices=targets_indices,
                values=targets.values(),
                size=targets.shape,
                device=targets.device
            )
        return torch.roll(targets, shift, 0)

    def threshold_targets(self, targets):
        if targets.is_sparse:
            targets = targets.coalesce()
            return torch.sparse_coo_tensor(
                indices=targets.indices()[..., targets.values() > self.target_threshold],
                values=torch.ones_like(targets.values()[targets.values() > self.target_threshold]),
                size=targets.shape,
                device=targets.device
            )
        return targets > self.target_threshold


class MixBack(nn.Module):
    """
    >>> mixback = MixBack(.5)
    >>> x1 = torch.ones((4,6,10))*torch.tensor([[1],[-1],[1],[-1],[1],[-1]])
    >>> mixback(x1)
    >>> x2 = torch.ones((3,6,9))*torch.tensor([[1],[-1],[1],[-1],[1],[-1]])
    >>> mixback(x2)
    """
    def __init__(self, max_mixback_scale, buffer_size=1, norm_axes=(-2,-1)):
        super().__init__()
        self.max_mixback_scale = max_mixback_scale
        self.buffer_size = buffer_size
        self.norm_axes = norm_axes
        self._buffer = []

    def reset(self):
        self._buffer = []

    def forward(self, x_input):
        if self.training:
            self._buffer = [x_input.detach().cpu()] + self._buffer
            if len(self._buffer) > self.buffer_size:
                b_in, *_, t_in = x_input.shape
                x_mixback = self._buffer[-1][:b_in, ..., :t_in].to(x_input.device)
                b_mix, *_, t_mix = x_mixback.shape
                if b_mix < b_in or t_mix < t_in:
                    reps = len(x_mixback.shape) * [1]
                    reps[0] = ceil(b_in/b_mix)
                    reps[-1] = ceil(t_in/t_mix)
                    x_mixback = x_mixback.repeat(*reps)[:b_in, ..., :t_in]
                x_mixback = (x_mixback - x_mixback.mean(self.norm_axes, keepdim=True)) / (x_mixback.std(self.norm_axes, keepdim=True)+1e-3)
                scale = self.max_mixback_scale * torch.rand((b_in,) + (x_input.dim()-1)*(1,), device=x_input.device)
                x_input = x_input + scale * x_mixback
                self._buffer = self._buffer[:self.buffer_size]
        return x_input


class Mask(nn.Module):
    """
    >>> x = torch.ones((3, 4, 5))
    >>> x, _ = Mask(axis=-1, max_masked_rate=1., max_masked_steps=10, n_masks=2)(x, seq_len=[1,2,3])
    >>> x.shape
    """
    def __init__(self, axis, n_masks=1, max_masked_steps=None, max_masked_rate=1., min_masked_steps=0, min_masked_rate=0.):
        super().__init__()
        self.axis = axis
        self.n_masks = n_masks
        self.max_masked_values = max_masked_steps
        self.max_masked_rate = max_masked_rate
        self.min_masked_values = min_masked_steps
        self.min_masked_rate = min_masked_rate

    def __call__(self, x, seq_len=None, rng=None):
        if not self.training:
            return x, torch.ones_like(x)
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
        max_width = self.max_masked_rate * seq_len
        if self.max_masked_values is not None:
            max_width = torch.min(self.max_masked_values*torch.ones_like(max_width), max_width)
        max_width = torch.floor(max_width)
        min_width = torch.min(self.min_masked_rate * seq_len, self.min_masked_values * torch.ones_like(max_width))
        min_width = torch.floor(min_width)
        width = min_width + torch.rand(x.shape[0], generator=rng) * (max_width - min_width + 1)
        width = torch.floor(width/self.n_masks)
        for i in range(self.n_masks):
            max_onset = seq_len - width
            onset = torch.floor(torch.rand(x.shape[0], generator=rng) * (max_onset + 1))
            onset = onset[(...,) + (x.dim()-1)*(None,)]
            offset = onset + width[(...,) + (x.dim()-1)*(None,)]
            mask = mask * ((idx < onset) + (idx >= offset)).float().to(x.device)
        return x * mask, mask


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
