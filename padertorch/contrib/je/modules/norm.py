import numpy as np
import torch
from padertorch.base import Module
from torch import nn


class Norm(Module):
    """
    >>> norm = Norm(data_format='bct', shape=(None, 10, None), statistics_axis='bt', momentum=0.5, straightness=1.)
    >>> x, seq_len = 2*torch.ones((3,10,4)), [1, 2, 3]
    >>> mask = norm.compute_mask(x, seq_len=seq_len)
    >>> mask
    tensor([[[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.]]])
    >>> norm.compute_stats(x, mask)
    (tensor([[[2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.]]]), tensor([[[4.],
             [4.],
             [4.],
             [4.],
             [4.],
             [4.],
             [4.],
             [4.],
             [4.],
             [4.]]]), tensor([[[6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.]]]))
    >>> norm.running_mean
    tensor([[[0.],
             [0.],
             [0.],
             [0.],
             [0.],
             [0.],
             [0.],
             [0.],
             [0.],
             [0.]]])
    >>> norm.running_power
    tensor([[[1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.]]])
    >>> x = norm(x, seq_len)
    >>> norm.running_mean
    tensor([[[1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.],
             [1.]]])
    >>> norm.running_power
    tensor([[[2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000],
             [2.5000]]])
    """
    def __init__(
            self,
            data_format='bcft',
            shape=None,
            *,
            statistics_axis='bft',
            independent_axis='c',
            batch_axis='b',
            sequence_axis='t',
            scale=True,
            eps: float = 1e-5,
            momentum=0.95,
            interpolation_factor=0.,
    ):
        super().__init__()
        self.data_format = data_format.lower()
        self.batch_axis = data_format.index(batch_axis.lower())
        self.sequence_axis = data_format.index(sequence_axis.lower())
        self.statistics_axis = tuple(
            [data_format.index(ax.lower()) for ax in statistics_axis]
        )
        self.scale = scale
        self.eps = eps
        self.track_running_stats = batch_axis in statistics_axis
        if self.track_running_stats:
            reduced_shape = [*shape]
            for ax in self.statistics_axis:
                reduced_shape[ax] = 1
            assert not any([d is None for d in reduced_shape])
            self.register_buffer(
                'num_tracked_values', torch.zeros(reduced_shape)
            )
            self.register_buffer('running_mean', torch.zeros(reduced_shape))
            if scale:
                self.register_buffer('running_power', torch.ones(reduced_shape))
            else:
                self.register_parameter('running_power', None)
        else:
            self.register_parameter('num_tracked_values', None)
            self.register_parameter('running_mean', None)
            self.register_parameter('running_power', None)
        self.momentum = momentum
        assert 0. <= interpolation_factor <= 1., interpolation_factor
        self.interpolation_factor = interpolation_factor

        if independent_axis is not None:
            self.learnable_scale = Scale(
                shape,
                data_format=data_format,
                independent_axis=independent_axis
            )
            self.learnable_shift = Shift(
                shape,
                data_format=data_format,
                independent_axis=independent_axis
            )
        else:
            self.learnable_scale = None
            self.learnable_shift = None

    @property
    def running_var(self):
        return self.running_power - self.running_mean ** 2

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.num_tracked_values.zero_()
            if self.scale is not None:
                self.running_power.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.learnable_scale is not None:
            nn.init.ones_(self.learnable_scale.scale)
        if self.learnable_shift is not None:
            nn.init.zeros_(self.learnable_shift.shift)

    def forward(self, x, seq_len=None):
        mask = self.compute_mask(x, seq_len)
        if self.training or not self.track_running_stats:
            mean, power, n_values = self.compute_stats(x, mask)
            if self.track_running_stats:
                self.num_tracked_values += n_values.data
                if self.momentum is None:
                    momentum = 1 - n_values / self.num_tracked_values
                else:
                    momentum = self.momentum
                self.running_mean *= momentum
                self.running_mean += (1 - momentum) * mean.data
                if self.scale:
                    self.running_power *= momentum
                    self.running_power += (1 - momentum) * power.data
                if self.interpolation_factor > 0.:
                    # perform straight through backpropagation
                    # https://arxiv.org/pdf/1611.01144.pdf
                    mean = mean + self.interpolation_factor * (self.running_mean.data - mean).detach()
                    power = power + self.interpolation_factor * (self.running_power.data - power).detach()
            x = x - mean
            if self.scale:
                n = torch.max(n_values, 2. * torch.ones_like(n_values))
                var = n / (n - 1.) * (power - mean ** 2)
                x = x / (torch.sqrt(var) + self.eps)
        else:
            x = x - self.running_mean.data
            if self.scale:
                n = torch.max(self.num_tracked_values, 2. * torch.ones_like(self.num_tracked_values))
                running_var = (
                    n / (n - 1.) * (self.running_power - self.running_mean**2)
                )
                x = x / (torch.sqrt(running_var).data + self.eps)

        if self.learnable_scale is not None:
            x = self.learnable_scale(x)
        if self.learnable_shift is not None:
            x = self.learnable_shift(x)
        return x * mask

    def compute_mask(self, x, seq_len=None):
        if seq_len is not None:
            assert self.sequence_axis is not None, self.sequence_axis
            seq_len = torch.Tensor(seq_len).long()
            for dim in range(self.batch_axis + 1, x.dim()):
                seq_len = seq_len.unsqueeze(-1)
            idx = torch.arange(x.shape[self.sequence_axis])
            for dim in range(self.sequence_axis + 1, x.dim()):
                idx = idx.unsqueeze(-1)
            mask = (idx < seq_len).float().to(x.device).expand(x.shape)
        else:
            mask = torch.ones_like(x)
        return mask

    def compute_stats(self, x, mask):
        n_values = mask.sum(dim=self.statistics_axis, keepdim=True)
        x = x * mask
        mean = x.sum(dim=self.statistics_axis, keepdim=True) / torch.max(n_values, torch.ones_like(n_values))
        if not self.scale:
            return mean, None, n_values
        power = (x ** 2).sum(dim=self.statistics_axis, keepdim=True) / torch.max(n_values, torch.ones_like(n_values))
        return mean, power, n_values


class Shift(nn.Module):
    def __init__(self, shape, data_format='bft', independent_axis='f'):
        super().__init__()
        reduced_shape = len(data_format) * [1]
        for ax in independent_axis:
            ax = data_format.index(ax.lower())
            assert shape[ax] is not None, shape[ax]
            reduced_shape[ax] = shape[ax]
        self.shift = nn.Parameter(
            torch.zeros(reduced_shape), requires_grad=True
        )

    def forward(self, x):
        return x - self.shift


class Scale(nn.Module):
    def __init__(self, shape, data_format='bft', independent_axis='f'):
        super().__init__()
        reduced_shape = len(data_format) * [1]
        for ax in independent_axis:
            ax = data_format.index(ax.lower())
            assert shape[ax] is not None, shape[ax]
            reduced_shape[ax] = shape[ax]
        self.scale = nn.Parameter(
            torch.ones(reduced_shape), requires_grad=True
        )

    def forward(self, x):
        return x * self.scale


class MulticlassNorm(Module):
    """
    >>> norm = MulticlassNorm(data_format='bct', n_classes=3, shape=(None, 10, None), statistics_axis='bt', momentum=0.5)
    >>> x, seq_len, class_idx = 2*torch.ones((3,10,4)), [1, 2, 3], [0, 1, 0]
    >>> norm(x, 0, seq_len).shape
    torch.Size([3, 10, 4])
    >>> norm(x, class_idx, seq_len).shape
    torch.Size([3, 10, 4])
    """
    def __init__(
            self, n_classes, data_format='bcft', shape=None, *,
            independent_axis='c', **kwargs
    ):
        super().__init__()
        self.n_classes = n_classes
        self.norms = nn.ModuleList([
            Norm(
                data_format,
                shape,
                independent_axis=None,
                **kwargs
            )
            for _ in range(n_classes)
        ])
        assert self.norms[0].batch_axis == 0, self.norms[0].batch_axis

        if independent_axis is not None:
            self.learnable_scale = Scale(
                shape,
                data_format=data_format,
                independent_axis=independent_axis
            )
            self.learnable_shift = Shift(
                shape,
                data_format=data_format,
                independent_axis=independent_axis
            )
        else:
            self.learnable_scale = None
            self.learnable_shift = None

    def forward(self, x, class_idx, seq_len=None):
        class_idx = np.array(class_idx)
        assert class_idx.ndim <= 1, class_idx.shape
        if class_idx.ndim == 0:
            class_idx = class_idx[None]
        classes_contained = sorted(set(class_idx.tolist()))
        if len(classes_contained) == 1:
            class_idx = classes_contained[0]
            x = self.norms[class_idx](x, seq_len=seq_len)
        else:
            idx = np.arange(x.shape[0]).astype(np.int)
            sort_idx = np.argsort(class_idx).flatten()
            reverse_idx = np.zeros_like(idx)
            reverse_idx[sort_idx] = idx
            x = x[sort_idx]
            if seq_len is not None:
                seq_len = np.array(seq_len)[sort_idx]
            class_idx = np.array(class_idx)[sort_idx]
            classes_contained = sorted(set(class_idx.tolist()))
            idx_arrays = [
                np.argwhere(class_idx == c).flatten()
                for c in classes_contained
            ]
            x = torch.cat(
                tuple([
                    self.norms[c](
                        x[idx_array],
                        seq_len=None if seq_len is None else seq_len[idx_array]
                    )
                    for c, idx_array in zip(classes_contained, idx_arrays)
                ]),
                dim=0
            )
            x = x[reverse_idx]

        if self.learnable_scale is not None:
            x = self.learnable_scale(x)
        if self.learnable_shift is not None:
            x = self.learnable_shift(x)
            mask = self.norms[0].compute_mask(x, seq_len)
            x = x * mask  # only necessary if shifted because already masked in individual norms
        return x
