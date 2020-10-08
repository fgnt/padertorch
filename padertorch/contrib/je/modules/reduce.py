import numpy as np
import torch
from torch import nn
from padertorch.ops.sequence.mask import compute_mask


class Sum(nn.Module):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> x = Sum(axis=seq_axis)(x, seq_len=[4,5,6])
    """
    def __init__(self, axis=-1, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x.sum(self.axis, keepdim=self.keepdims)
        else:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x * mask).sum(dim=self.axis, keepdim=self.keepdims)
        return x


class Mean(Sum):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> x = Mean(axis=seq_axis)(x, seq_len=[4,5,6])
    >>> x.shape
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> x = Mean(axis=seq_axis, keepdims=True)(x, seq_len=[4,5,6])
    >>> x.shape
    """
    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x.mean(self.axis, keepdim=self.keepdims)
        else:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x * mask).sum(dim=self.axis, keepdim=self.keepdims) / (mask.sum(dim=self.axis, keepdim=self.keepdims) + 1e-6)
        return x


class Max(nn.Module):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> Max(axis=seq_axis)(x, seq_len=[4,5,6])
    """
    def __init__(self, axis=-1, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is not None:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x + torch.log(mask))
        x = x.max(self.axis, keepdim=self.keepdims)
        return x


class TakeLast(nn.Module):
    """
    >>> x = torch.Tensor([[[1,2,3]],[[4,5,6]]])
    >>> TakeLast()(x, [2, 3])
    tensor([[2.],
            [6.]])
    """
    def __init__(self, axis=-1, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__()

    def __call__(self, x, seq_len=None):
        axis = self.axis
        if axis < 0:
            axis = x.dim() + axis
        if axis != 1:
            assert axis > 1, axis
            x = x.unsqueeze(1).transpose(1, axis+1).squeeze(axis + 1)
        if seq_len is None:
            x = x[:, -1]
        else:
            x = x[torch.arange(x.shape[0]), np.array(seq_len) - 1]
        if self.keepdims:
            x = x.unsqueeze(self.axis)
        return x


class AutoPool(nn.Module):
    """

    >>> autopool = AutoPool(10)
    >>> autopool(torch.cumsum(torch.ones(4, 10, 17), dim=-1), seq_len=[17, 15, 12, 9])
    """
    def __init__(self, n_classes, alpha=1., trainable=False):
        super().__init__()
        self.trainable = trainable
        if trainable:
            self.alpha = nn.Parameter(alpha*torch.ones((n_classes, 1)))
        else:
            self.alpha = alpha

    def forward(self, x, seq_len=None):
        x_ = self.alpha*x
        if seq_len is not None:
            seq_len = torch.Tensor(seq_len).to(x.device)[:, None, None]
            mask = (torch.cumsum(torch.ones_like(x_), dim=-1) <= seq_len).float()
            x_ = x_ * mask + torch.log(mask)
        weights = nn.Softmax(dim=-1)(x_)
        return (weights*x).sum(dim=-1)
