import torch
from torch import nn
import numpy as np

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.contrib.je.modules.norm import Norm
from padertorch.contrib.je.modules.global_pooling import compute_mask


def scaled_dot_product_attention(q, k, v, seq_len=None, bidirectional=False):
    """
    >>> q = torch.zeros((2, 3, 4))
    >>> k = torch.zeros((2, 6, 4))
    >>> v = torch.randn((2, 6, 8))
    >>> x = scaled_dot_product_attention(q, k, v)
    >>> x.shape
    torch.Size([2, 3, 8])
    >>> q = torch.zeros((2, 6, 4))
    >>> x = scaled_dot_product_attention(q, k, v, causal=True)
    >>> (x[0,0] == v[0,0]).all()
    tensor(1, dtype=torch.uint8)
    >>> (torch.abs(x[0,-1] - v[0].mean(0)) < 1e-6).all()
    tensor(1, dtype=torch.uint8)
    >>> x = scaled_dot_product_attention(q, k, v, seq_len=[6,4])
    """
    y = q@k.transpose(-2, -1)/np.sqrt(k.shape[-1])
    if not bidirectional:
        mask = get_causal_mask(y)
        y = y + torch.log((mask > 0).float())
    elif seq_len is not None:
        mask = compute_mask(y, seq_len, seq_axis=-1)
        y = y + torch.log((mask > 0).float())
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
    def __init__(self, input_size, output_size, num_heads=1, bidirectional=False):
        assert output_size % num_heads == 0
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.lin_queue = torch.nn.Linear(input_size, output_size)
        self.lin_key = torch.nn.Linear(input_size, output_size)
        self.lin_value = torch.nn.Linear(input_size, output_size)
        self.out = torch.nn.Linear(output_size, output_size)

    def forward(self, q, k, v, seq_len=None):
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
        x = scaled_dot_product_attention(
            q, k, v, seq_len=seq_len, bidirectional=self.bidirectional
        )
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.output_size)
        return self.out(x)


class TransformerBlock(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, input_size, hidden_size, num_heads=1, bidirectional=False,
            cross_attention=False, norm='layer', norm_kwargs={},
            activation='relu',
    ):
        super().__init__()
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.multi_head_self_attention = MultiHeadAttention(
            input_size, hidden_size, num_heads, bidirectional=bidirectional
        )
        self.cross_attention = cross_attention
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, hidden_size)

        norm_kwargs = {
            "data_format": 'btc',
            "shape": (None, None, hidden_size),
            "statistics_axis": 'bt',
            **norm_kwargs
        }
        if norm is None:
            self.norm = None
        elif norm == 'batch':
            norm_kwargs['statistics_axis'] = 'bt'
        elif norm == 'layer':
            norm_kwargs['statistics_axis'] = 'tc'
            # ToDo: where is the difference between layer norm and instance norm?
        else:
            raise ValueError(f'{norm} normalization not known.')
        self.self_attention_norm = Norm(**norm_kwargs)
        self.output_norm = Norm(**norm_kwargs)

        if cross_attention:
            self.multi_head_cross_attention = MultiHeadAttention(
                hidden_size, hidden_size, num_heads, bidirectional=True
            )
            self.cross_attention_norm = Norm(**norm_kwargs)

    def forward(self, x, v=None, seq_len_x=None, seq_len_v=None, state=None):
        x_ = x if state is None else torch.cat([state, x], 1)
        h = self.multi_head_self_attention(x, x_, x_, seq_len=seq_len_x)
        if h.shape == x.shape:
            h = h + x
        h = self.self_attention_norm(h, seq_len=seq_len_x)
        if self.cross_attention:
            assert v is not None
            q = h
            h = self.multi_head_cross_attention(q, v, v, seq_len=seq_len_v)
            if h.shape == q.shape:
                h = h + q
            h = self.cross_attention_norm(h, seq_len=seq_len_x)
        y = self.out(self.activation(self.hidden(h)))
        y = y + h
        y = self.output_norm(y, seq_len=seq_len_x)
        return y, x_


class TransformerStack(Module):
    def __init__(
            self, input_size, hidden_size, num_layers, output_size=None,
            num_heads=1, bidirectional=False, cross_attention=False,
            activation='relu', norm='layer', norm_kwargs={}
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            input_size:
            hidden_size:
            output_size:
            num_heads:
            bidirectional:
            cross_attention:
            activation:
            norm:
            norm_kwargs:

        Returns:

        >>> x = torch.zeros((2, 3, 8))
        >>> attn = TransformerStack(8, 6, 2, 6, 1, bidirectional=True)
        >>> attn(x, seq_len_x=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerStack(8, 6, 2, 6, 2, bidirectional=True)
        >>> attn(x, seq_len_x=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn(x, state=[torch.zeros((2, 6, 8)), torch.zeros((2, 6, 6))])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerStack(8, 6, 2, 6, 2, bidirectional=False)
        >>> attn(x)[0].shape
        torch.Size([2, 3, 6])
        >>> attn(x, state=[torch.zeros((2, 6, 8)), torch.zeros((2, 6, 6))])[0].shape
        torch.Size([2, 3, 6])
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = hidden_size if output_size is None else output_size
        stack = list()
        for i in range(num_layers):
            stack.append(
                TransformerBlock(
                    input_size, hidden_size, num_heads,
                    bidirectional=bidirectional, cross_attention=cross_attention,
                    activation=activation, norm=norm, norm_kwargs=norm_kwargs
                )
            )
            input_size = hidden_size
        self.stack = torch.nn.ModuleList(stack)
        if output_size is not None:
            self.output_layer = nn.Linear(input_size, output_size)
        else:
            self.output_layer = None

    def forward(self, x, v=None, seq_len_x=None, seq_len_v=None, state=None):
        new_state = []
        for i, layer in enumerate(self.stack):
            x, x_ = layer(
                x, v=v, seq_len_x=seq_len_x, seq_len_v=seq_len_v,
                state=None if state is None else state[i],
            )
            new_state.append(x_)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, new_state


def get_causal_mask(x):
    return torch.tril(torch.ones_like(x), diagonal=(x.shape[-1] - x.shape[-2]))
