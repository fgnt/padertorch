import torch
from padertorch.base import Module
from padertorch.ops.sequence.mask import compute_mask
from torch import nn
from torch.autograd import Function


class Norm(Module):
    """
    >>> norm = Norm(data_format='bct', shape=(None, 10, None), statistics_axis='bt', momentum=0.5)
    >>> x, seq_len = 2*torch.ones((3,10,4)), [1, 2, 3]
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
            shift=True,
            scale=True,
            eps: float = 1e-5,
            momentum=0.95,
    ):
        super().__init__()
        self.data_format = data_format.lower()
        self.batch_axis = None if batch_axis is None \
            else data_format.index(batch_axis.lower())
        self.sequence_axis = None if sequence_axis is None \
            else data_format.index(sequence_axis.lower())
        self.statistics_axis = tuple(
            [data_format.index(ax.lower()) for ax in statistics_axis]
        )
        self.shift = shift
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
            if shift:
                self.register_buffer('running_mean', torch.zeros(reduced_shape))
            else:
                self.register_parameter('running_mean', None)
            if scale:
                self.register_buffer('running_power', torch.ones(reduced_shape))
            else:
                self.register_parameter('running_power', None)
        else:
            self.register_parameter('num_tracked_values', None)
            self.register_parameter('running_mean', None)
            self.register_parameter('running_power', None)
        self.momentum = momentum

        if independent_axis is not None:
            reduced_shape = len(data_format) * [1]
            for ax in independent_axis:
                ax = data_format.index(ax.lower())
                assert shape[ax] is not None, shape[ax]
                reduced_shape[ax] = shape[ax]
            if scale:
                self.gamma = nn.Parameter(
                    torch.ones(reduced_shape), requires_grad=True
                )
            else:
                self.gamma = None
            if self.shift:
                self.beta = nn.Parameter(
                    torch.zeros(reduced_shape), requires_grad=True
                )
            else:
                self.beta = None
        else:
            self.gamma = None
            self.beta = None

    @property
    def runnning_var(self):
        n = torch.max(
            self.num_tracked_values,
            2. * torch.ones_like(self.num_tracked_values)
        )
        running_var = self.running_power
        if self.shift:
            running_var = n / (n-1) * (running_var - self.running_mean ** 2)
        running_var = running_var + self.eps
        assert (running_var >= 0).all(), running_var.min()
        return running_var

    def reset_running_stats(self):
        if self.track_running_stats:
            self.num_tracked_values.zero_()
            if self.shift:
                self.running_mean.zero_()
            if self.scale:
                self.running_power.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.gamma is not None:
            nn.init.ones_(self.gamma.scale)
        if self.beta is not None:
            nn.init.zeros_(self.beta.shift)

    def forward(self, x, sequence_lengths=None):
        if self.training or not self.track_running_stats:
            x, mean, power, n_values = normalize(
                x, gamma=self.gamma, beta=self.beta,
                statistics_axis=self.statistics_axis,
                batch_axis=self.batch_axis, sequence_axis=self.sequence_axis,
                sequence_lengths=sequence_lengths,
                shift=self.shift, scale=self.scale,
                eps=self.eps
            )
            if self.track_running_stats:
                self.num_tracked_values += n_values.detach()
                if self.momentum is None:
                    momentum = 1 - n_values / self.num_tracked_values.detach()
                else:
                    momentum = self.momentum
                if self.shift:
                    self.running_mean *= momentum
                    self.running_mean += (1 - momentum) * mean.detach()
                    power = power.detach() + mean.detach() ** 2
                if self.scale:
                    self.running_power *= momentum
                    self.running_power += (1 - momentum) * power.detach()
        else:
            if self.shift:
                x = x - self.running_mean.detach()
            if self.scale:
                x = x / torch.sqrt(self.runnning_var.detach())
            if self.gamma is not None:
                x = x * self.gamma
            if self.beta is not None:
                x = x + self.beta
            x = x * compute_mask(
                x, sequence_lengths, self.batch_axis, self.sequence_axis
            )
        return x

    def inverse(self, x):
        if not self.track_running_stats:
            raise NotImplementedError
        if self.beta is not None:
            x = x - self.beta
        if self.gamma is not None:
            x = x / self.gamma
        if self.scale:
            x = torch.sqrt(self.running_var) * x
        if self.shift:
            x = x + self.running_mean
        return x


class Normalize(Function):
    """
    Normalization function incl. backward computation.
    The implementation of the backward step saves memory compared to simply
    using autograd of the forward operations.
    """
    @staticmethod
    def forward(
            ctx, x, gamma, beta, statistics_axis, batch_axis, sequence_axis,
            sequence_lengths, shift, scale, eps
    ):
        ctx.statistics_axis = statistics_axis
        ctx.batch_axis = batch_axis
        ctx.sequence_axis = sequence_axis
        ctx.seq_len = sequence_lengths
        ctx.shift = shift
        ctx.scale = scale
        ctx.eps = eps

        # compute mask
        mask = compute_mask(x, sequence_lengths, batch_axis, sequence_axis)

        # compute statistics
        n_values = mask.sum(dim=statistics_axis, keepdim=True)
        x = x * mask
        mean = x.sum(dim=statistics_axis, keepdim=True) / torch.max(
            n_values, torch.ones_like(n_values)
        )
        power = (x ** 2).sum(dim=statistics_axis, keepdim=True) / torch.max(
            n_values, torch.ones_like(n_values)
        )
        y = x
        if shift:
            y = y - mean
            power = power - mean**2
        if scale:
            y = y / torch.sqrt(power + eps)
        ctx.save_for_backward(x, gamma, beta, mean, power)

        if gamma is not None:
            assert gamma.dim() == x.dim(), gamma.shape
            y = y * gamma
        if beta is not None:
            assert beta.dim() == x.dim(), beta.shape
            y = y + beta
        return y*mask, mean, power, n_values

    @staticmethod
    def backward(ctx, grad_y, grad_mean, grad_power, _):
        # equations from https://arxiv.org/abs/1502.03167
        if (grad_mean != 0).any() or (grad_power != 0).any():
            raise NotImplementedError
        x, gamma, beta, mean, power = ctx.saved_tensors
        # compute mask
        mask = compute_mask(x, ctx.seq_len, ctx.batch_axis, ctx.sequence_axis)
        n_values = mask.sum(dim=ctx.statistics_axis, keepdim=True)

        grad_y = grad_y * mask
        x_hat = x
        scale = torch.sqrt(power + ctx.eps)
        if ctx.shift:
            x_hat = x_hat - mean
        if ctx.scale:
            x_hat = x_hat / scale
        if beta is None:
            grad_beta = None
        else:
            reduce_axis = [i for i in range(beta.dim()) if beta.shape[i] == 1]
            grad_beta = grad_y.sum(reduce_axis, keepdim=True)
        if gamma is None:
            grad_gamma = None
            grad_x_hat = grad_y
        else:
            reduce_axis = [i for i in range(gamma.dim()) if gamma.shape[i] == 1]
            grad_gamma = (grad_y * x_hat).sum(reduce_axis, keepdim=True)
            grad_x_hat = grad_y * gamma
        if ctx.shift:
            x = (x - mean) * mask
            grad_mean_ = -grad_x_hat.sum(ctx.statistics_axis, keepdim=True)
        if ctx.scale:
            grad_power_ = (
                    (grad_x_hat * x).sum(ctx.statistics_axis, keepdim=True)
                    * (-1 / 2) * (power + ctx.eps) ** (-3 / 2)
            )
            if ctx.shift:
                grad_mean_ = (
                    grad_mean_ / scale
                    - 2 * grad_power_
                        * x.sum(ctx.statistics_axis, keepdim=True) / n_values
                )

        grad_x = grad_x_hat
        if ctx.scale:
            grad_x = grad_x / scale + grad_power_ * 2 * x / n_values
        if ctx.shift:
            grad_x = grad_x + grad_mean_ / n_values
        return grad_x * mask, grad_gamma, grad_beta, None, None, None, None, None, None, None


def normalize(
        x, gamma, beta, statistics_axis, batch_axis, sequence_axis,
        sequence_lengths, shift, scale, eps
):
    """
    >>> x, seq_len = 2*torch.ones((3,10,4)), [1, 2, 3]
    >>> x, m, p, n = normalize(x, None, None, [0, 2], 0, 2, seq_len, True, True, 1e-3)
    >>> m
    tensor([[[2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.],
             [2.]]])
    >>> p
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
    >>> n
    tensor([[[6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.],
             [6.]]])
    """
    return Normalize.apply(
        x, gamma, beta, statistics_axis, batch_axis, sequence_axis,
        sequence_lengths, shift, scale, eps
    )
