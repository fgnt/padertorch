import torch
import numpy as np

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP


def scaled_dot_product_attention(q, k, v, mask=False):
    """
    >>> q = torch.zeros((2, 3, 4))
    >>> k = torch.zeros((2, 6, 4))
    >>> v = torch.randn((2, 6, 8))
    >>> x = scaled_dot_product_attention(q, k, v)
    >>> x.shape
    torch.Size([2, 3, 8])
    >>> q = torch.zeros((2, 6, 4))
    >>> x = scaled_dot_product_attention(q, k, v, mask=True)
    >>> (x[0,0] == v[0,0]).all()
    tensor(1, dtype=torch.uint8)
    >>> (torch.abs(x[0,-1] - v[0].mean(0)) < 1e-6).all()
    tensor(1, dtype=torch.uint8)

    :param q:
    :param k:
    :param v:
    :param mask
    :return:
    """
    y = q@k.transpose(-2, -1)/np.sqrt(k.shape[-1])
    if mask is True:
        assert q.shape[1] == v.shape[1], (q.shape[1], v.shape[1])
        mask = torch.ones_like(y).cumsum(-1) <= torch.ones_like(y).cumsum(-2)
        y = y + torch.log(mask.float())
    return torch.softmax(y, dim=-1)@v


class MultiHeadAttention(Module):
    """
    https://arxiv.org/abs/1706.03762

    >>> q = torch.randn((2, 3, 4))
    >>> k = torch.randn((2, 6, 4))
    >>> v = torch.randn((2, 6, 4))
    >>> attn = MultiHeadAttention(4, 8)
    >>> attn(q, k, v).shape
    torch.Size([2, 3, 8])
    >>> attn = MultiHeadAttention(4, 8, 2)
    >>> attn(q, k, v).shape
    torch.Size([2, 3, 8])
    """
    def __init__(
            self, input_size, output_size, num_heads=1, mask=False
    ):
        assert output_size % num_heads == 0
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.mask = mask
        self.lin_queue = torch.nn.Linear(input_size, output_size)
        self.lin_key = torch.nn.Linear(input_size, output_size)
        self.lin_value = torch.nn.Linear(input_size, output_size)
        self.out = torch.nn.Linear(output_size, output_size)

    def forward(self, q, k, v):
        B, Tq, _ = q.shape
        B, Tk, _ = k.shape
        q = self.lin_queue(q).view(
            B, Tq, self.num_heads, self.output_size//self.num_heads
        ).transpose(1, 2)
        k = self.lin_key(k).view(
            B, Tk, self.num_heads, self.output_size//self.num_heads
        ).transpose(1, 2)
        v = self.lin_value(v).view(
            B, Tk, self.num_heads, self.output_size//self.num_heads
        ).transpose(1, 2)
        x = scaled_dot_product_attention(q, k, v).contiguous().view(
            B, Tq, self.output_size)
        return self.out(x)


class SelfAttention(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, input_size, hidden_size, num_heads=1, mask=True,
            activation='leaky_relu', residual=False, norm=None
    ):
        assert hidden_size
        super().__init__()
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.residual = residual
        self.multiheadattention = MultiHeadAttention(
            input_size, hidden_size, num_heads, mask
        )
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, hidden_size)

        if norm is None:
            self.hidden_norm = None
            self.out_norm = None
        elif norm == 'batch':
            self.hidden_norm = torch.nn.BatchNorm1d(hidden_size)
            self.out_norm = torch.nn.BatchNorm1d(hidden_size)
        else:
            raise ValueError(f'{norm} normalization not known.')

    def forward(self, x):
        h = self.multiheadattention(x, x, x)
        if self.residual and h.shape == x.shape:
            h = h + x
        if self.hidden_norm is not None:
            h = self.hidden_norm(h.transpose(1, -1)).transpose(1, -1)
        y = self.out(self.activation(self.hidden(h)))
        if self.residual and y.shape == h.shape:
            y = y + h
        if self.out_norm is not None:
            y = self.out_norm(y.transpose(1, -1)).transpose(1, -1)
        return y


def self_attention_stack(
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        num_heads: int = 1,
        mask: bool = True,
        activation: str = 'leaky_relu',
        residual: bool = False,
        norm: str = None
):
    """
    https://arxiv.org/abs/1706.03762

    Args:
        input_size:
        hidden_size:
        num_layers:
        num_heads:
        mask:
        activation:
        residual:
        norm:

    Returns:

    >>> x = torch.zeros((2, 3, 8))
    >>> attn = self_attention_stack(8, 6, 1)
    >>> attn(x).shape
    torch.Size([2, 3, 6])
    >>> attn = self_attention_stack(8, 6, 2)
    >>> attn(x).shape
    torch.Size([2, 3, 6])
    """
    layers = []
    for i in range(num_layers):
        layers.append(
            SelfAttention(
                input_size, hidden_size, num_heads, mask, activation, residual,
                norm
            )
        )
        input_size = hidden_size
    return torch.nn.Sequential(*layers)
