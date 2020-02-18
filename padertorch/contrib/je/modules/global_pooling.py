import torch
from torch import nn


class TakeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            x = x[:, -1]
        else:
            x = x[torch.arange(x.shape[0]), seq_len - 1]
        seq_len = None
        return x, seq_len


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

