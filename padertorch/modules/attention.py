import torch
import numpy as np

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP


def scaled_dot_product_attention(q, k, v, mask=True):
    """
    >>> q = torch.zeros((2, 3, 4))
    >>> k = torch.zeros((2, 6, 4))
    >>> v = torch.zeros((2, 6, 8))
    >>> scaled_dot_product_attention(q, k, v).shape
    torch.Size([2, 3, 8])

    :param q:
    :param k:
    :param v:
    :param mask
    :return:
    """
    y = q@k.transpose(-2, -1)/np.sqrt(k.shape[-1])
    if mask is True:
        mask = torch.ones_like(y).cumsum(-1) <= torch.ones_like(y).cumsum(-2)
        y = y + torch.log(mask.float())
    return torch.softmax(y, dim=-1)@v


class MultiHeadAttention(Module):
    """
    https://arxiv.org/abs/1706.03762

    >>> attn = MultiHeadAttention(4, 8, 2, 2)
    >>> q = torch.zeros((2, 3, 4))
    >>> k = torch.zeros((2, 6, 4))
    >>> v = torch.zeros((2, 6, 4))
    >>> attn(q, k, v).shape
    torch.Size([2, 3, 8])
    """
    def __init__(
            self, input_size, output_size, key_size, value_size, num_heads=8,
            mask=True
    ):
        super().__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.mask = mask
        self.lin_queue = torch.nn.Linear(input_size, num_heads*key_size)
        self.lin_key = torch.nn.Linear(input_size, num_heads*key_size)
        self.lin_value = torch.nn.Linear(input_size, num_heads*value_size)
        self.out = torch.nn.Linear(num_heads*value_size, output_size)

    def forward(self, q, k, v):
        B, Tq, _ = q.shape
        B, Tk, _ = k.shape
        q = self.lin_queue(q).view(
            B, Tq, self.num_heads, self.key_size).transpose(1, 2)
        k = self.lin_key(k).view(
            B, Tk, self.num_heads, self.key_size).transpose(1, 2)
        v = self.lin_value(v).view(
            B, Tk, self.num_heads, self.value_size).transpose(1, 2)
        x = scaled_dot_product_attention(q, k, v).contiguous().view(
            B, Tq, self.num_heads * self.value_size)
        return self.out(x)


class SelfAttention(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, input_size, hidden_size, key_size, value_size,
            num_heads=8, mask=True, norm=None, activation='leaky_relu'
    ):
        assert hidden_size
        super().__init__()
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.multiheadattention = MultiHeadAttention(
            input_size, hidden_size, key_size, value_size, num_heads, mask
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
            raise ValueError(f'{norm} normalization  not known.')

    def forward(self, x):
        h = self.multiheadattention(x, x, x)
        if h.shape == x.shape:
            h = h + x
        if self.hidden_norm is not None:
            h = self.hidden_norm(h)
        y = self.out(self.activation(self.hidden(h)))
        if y.shape == h.shape:
            y = y + h
        if self.out_norm is not None:
            y = self.out_norm(y)
        return y


def self_attention_stack(
        input_size: int,
        hidden_size: int,
        key_size: int = 64,
        value_size: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        mask: bool = True,
        norm: str = None,
        activation: str = 'leaky_relu'
):
    """
    https://arxiv.org/abs/1706.03762

    >>> attn = self_attention_stack(8, 6, 2, 3)
    >>> x = torch.zeros((2, 3, 8))
    >>> attn(x).shape
    torch.Size([2, 3, 6])

    :param input_size:
    :param hidden_size:
    :param key_size:
    :param value_size:
    :param num_layers:
    :param num_heads:
    :param mask:
    :param norm:
    :param activation:
    :return:
    """
    layers = []
    for i in range(num_layers):
        layers.append(
            SelfAttention(
                input_size, hidden_size, key_size, value_size,
                num_heads, mask, norm, activation
            )
        )
        input_size = hidden_size
    return torch.nn.Sequential(*layers)
