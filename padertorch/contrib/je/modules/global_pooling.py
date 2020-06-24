import numpy as np
import torch
from torch import nn


def compute_mask(x, seq_len, batch_axis=0, seq_axis=1):
    """
    >>> x, seq_len = 2*torch.ones((3,1,10,4)), [1, 2, 3]
    >>> mask = compute_mask(x, seq_len=seq_len, batch_axis=0, seq_axis=-1)
    >>> mask[:,0]
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

    Args:
        x:
        seq_len:
        batch_axis:
        seq_axis:

    Returns:

    """
    if seq_len is None:
        return torch.ones_like(x)
    if batch_axis < 0:
        batch_axis = x.dim() + batch_axis
    if seq_axis < 0:
        seq_axis = x.dim() + seq_axis
    seq_len = torch.Tensor(seq_len).long().to(x.device)
    for dim in range(batch_axis + 1, x.dim()):
        seq_len = seq_len.unsqueeze(-1)
    idx = torch.arange(x.shape[seq_axis]).to(x.device)
    for dim in range(seq_axis + 1, x.dim()):
        idx = idx.unsqueeze(-1)
    mask = (idx < seq_len).float().expand(x.shape)
    return mask


class Sum(nn.Module):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> x = Sum(axis=seq_axis)(x, seq_len=[4,5,6])
    """
    def __init__(self, axis=-1):
        self.axis = axis
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x.sum(self.axis)
        else:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x * mask).sum(dim=self.axis)
        return x


class Mean(Sum):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> x = Mean(axis=seq_axis)(x, seq_len=[4,5,6])
    """
    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x.mean(self.axis)
        else:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x * mask).sum(dim=self.axis) / (mask.sum(self.axis) + 1e-6)
        return x


class Max(nn.Module):
    """
    >>> seq_axis = 1
    >>> x = torch.cumsum(torch.ones((3,7,4)), dim=seq_axis)
    >>> Max(axis=seq_axis)(x, seq_len=[4,5,6])
    """
    def __init__(self, axis=-1):
        self.axis = axis
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is not None:
            mask = compute_mask(x, seq_len, 0, self.axis)
            x = (x + torch.log(mask))
        x = x.max(self.axis)
        return x


class TakeLast(nn.Module):
    """
    >>> x = torch.Tensor([[[1,2,3]],[[4,5,6]]])
    >>> TakeLast()(x, [2, 3])
    tensor([[2.],
            [6.]])
    """
    def __init__(self, axis=-1):
        self.axis = axis
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
