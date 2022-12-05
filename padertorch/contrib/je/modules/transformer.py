import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.modules.normalization import Normalization
from padertorch.ops.sequence.mask import compute_mask


def scaled_dot_product_attention(q, k, v, seq_len=None, bidirectional=False, mask=None):
    """
    >>> q = torch.zeros((2, 3, 4))
    >>> k = torch.zeros((2, 6, 4))
    >>> v = torch.randn((2, 6, 8))
    >>> x, _ = scaled_dot_product_attention(q, k, v, bidirectional=True)
    >>> x.shape
    torch.Size([2, 3, 8])
    >>> q = torch.zeros((2, 6, 4))
    >>> x, _ = scaled_dot_product_attention(q, k, v, bidirectional=False)
    >>> (x[0,0] == v[0,0]).all()
    tensor(True)
    >>> (torch.abs(x[0,-1] - v[0].mean(0)) < 1e-6).all()
    tensor(True)
    >>> x, _ = scaled_dot_product_attention(q, k, v, seq_len=[6,4], bidirectional=True)
    """
    y = q@k.transpose(-2, -1)/np.sqrt(k.shape[-1])
    if mask is not None:
        y = y + torch.log((mask > 0).float())
    if not bidirectional:
        mask = get_causal_mask(y)
        y = y + torch.log((mask > 0).float())
    elif seq_len is not None:
        mask = compute_mask(y, seq_len, sequence_axis=-1)
        y = y + torch.log((mask > 0).float())
    y = torch.softmax(y, dim=-1)
    return y@v, y


class MultiHeadAttention(Module):
    """
    https://arxiv.org/abs/1706.03762

    >>> q = torch.randn((2, 3, 4))
    >>> k = torch.randn((2, 6, 6))
    >>> v = torch.randn((2, 6, 8))
    >>> attn = MultiHeadAttention(4, 6, 8, 4, 4)
    >>> y, w = attn(q, k, v)
    >>> y.shape
    torch.Size([2, 3, 4])
    >>> attn = MultiHeadAttention(4, 6, 8, 4, 4, num_heads=2)
    >>> y, w = attn(q, k, v)
    >>> y.shape
    torch.Size([2, 3, 4])
    """
    def __init__(
            self, queue_size, key_size, value_size, d_model, output_size,
            num_heads=8, bidirectional=False
    ):
        super().__init__()
        self.queue_size = queue_size
        self.d_model = d_model
        self.output_size = output_size
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.lin_queue = torch.nn.Linear(queue_size, self.d_model)
        self.lin_key = torch.nn.Linear(key_size, self.d_model)
        self.lin_value = torch.nn.Linear(value_size, self.d_model)
        self.out = torch.nn.Linear(self.d_model, self.output_size)

    def forward(self, q, k, v, seq_len=None, mask=None):
        B, Tq, _ = q.shape
        B, Tk, _ = k.shape
        q = self.lin_queue(q).view(
            B, Tq, self.num_heads, self.d_model//self.num_heads
        ).transpose(1, 2)
        k = self.lin_key(k).view(
            B, Tk, self.num_heads, self.d_model//self.num_heads
        ).transpose(1, 2)
        v = self.lin_value(v).view(
            B, Tk, self.num_heads, self.d_model//self.num_heads
        ).transpose(1, 2)
        x, attention_weights = scaled_dot_product_attention(
            q, k, v, seq_len=seq_len, bidirectional=self.bidirectional, mask=mask
        )
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out(x), attention_weights


class TransformerLayer(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, d_model=512, d_ff=2048, num_heads=8,
            bidirectional=True, cross_attention=False,
            norm='layer', norm_kwargs={}, norm_first=True,
            activation_ff='relu', dropout=.0,
    ):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(
            d_model, d_model, d_model, d_model, d_model,
            num_heads=num_heads, bidirectional=bidirectional
        )
        self.cross_attention = cross_attention
        self.hidden = torch.nn.Linear(d_model, d_ff)
        self.out = torch.nn.Linear(d_ff, d_model)

        if norm is None:
            self.self_attention_norm = None
            self.output_norm = None
        else:
            norm_kwargs = {
                "data_format": 'btc',
                "shape": (None, None, d_model),
                'eps': 1e-2,
                **norm_kwargs
            }
            if norm == 'batch':
                norm_kwargs['statistics_axis'] = 'bt'
            elif norm == 'layer':
                norm_kwargs['statistics_axis'] = 'c'
            else:
                raise ValueError(f'{norm} normalization not known.')
            self.self_attention_norm = Normalization(**norm_kwargs)
            self.output_norm = Normalization(**norm_kwargs)

        if cross_attention:
            self.multi_head_cross_attention = MultiHeadAttention(
                d_model, d_model, d_model, d_model, d_model,
                num_heads=num_heads, bidirectional=True
            )
            if norm is None:
                self.cross_attention_norm = None
            else:
                self.cross_attention_norm = Normalization(**norm_kwargs)
        self.norm_first = norm_first
        self.activation_ff = ACTIVATION_FN_MAP[activation_ff]()
        self.dropout = dropout

    def forward(
            self, x, seq_len, m=None, seq_len_m=None, state=None,
    ):
        if state is not None:
            assert self.multi_head_self_attention.bidirectional is False
        s = x if state is None else torch.cat((state, x), 1)
        h, _ = self.multi_head_self_attention(x, s, s, seq_len=seq_len)
        if self.training and self.dropout > 0.:
            h = F.dropout(h, self.dropout)
        if self.self_attention_norm is not None and self.norm_first:
            h = self.self_attention_norm(h, sequence_lengths=seq_len)
        h = h + x
        if self.self_attention_norm is not None and not self.norm_first:
            h = self.self_attention_norm(h, sequence_lengths=seq_len)
        if self.cross_attention:
            assert m is not None
            q = h
            h, _ = self.multi_head_cross_attention(q, m, m, seq_len=seq_len_m)
            if self.training and self.dropout > 0.:
                h = F.dropout(h, self.dropout)
            if self.cross_attention_norm is not None and self.norm_first:
                h = self.cross_attention_norm(h, sequence_lengths=seq_len)
            h = h + q
            if self.cross_attention_norm is not None and not self.norm_first:
                h = self.cross_attention_norm(h, sequence_lengths=seq_len)
        y = self.out(self.activation_ff(self.hidden(h)))
        if self.training and self.dropout > 0.:
            y = F.dropout(y, self.dropout)
        if self.output_norm is not None and self.norm_first:
            y = self.output_norm(y, sequence_lengths=seq_len)
        y = y + h
        if self.output_norm is not None and not self.norm_first:
            y = self.output_norm(y, sequence_lengths=seq_len)
        return y, s


class TransformerLayerStack(Module):
    def __init__(
            self, input_size, hidden_size=512, d_ff=2048, num_heads=8,
            num_layers=6, bidirectional=False, cross_attention=False,
            norm='layer', norm_kwargs={}, norm_first=True,
            activation_ff='relu', dropout=0., positional_encoding=True,
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            input_size:
            hidden_size: d_model
            d_ff:
            num_heads:
            bidirectional:
            cross_attention:
            norm:
            norm_kwargs:
            norm_first:
            activation_ff:
            dropout:
            positional_encoding:

        Returns:

        >>> x = torch.zeros((2, 3, 8))
        >>> attn = TransformerLayerStack(8, 6, 20, 1, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerLayerStack(8, 6, 20, 2, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerLayerStack(8, 6, 20, 2, 2, bidirectional=False)
        >>> attn(x, seq_len=None)[0].shape
        torch.Size([2, 3, 6])
        >>> attn(x, seq_len=None, state=[torch.zeros((2, 5, 6)), torch.zeros((2, 5, 6))])[0].shape
        torch.Size([2, 3, 6])
        """
        super().__init__()
        self.positional_encoding = positional_encoding
        self.dropout = dropout
        self.num_layers = num_layers
        self.lin = torch.nn.Linear(input_size, hidden_size)
        transformer_layers = list()
        for i in range(num_layers):
            transformer_layers.append(
                TransformerLayer(
                    hidden_size, d_ff, num_heads, bidirectional=bidirectional,
                    cross_attention=cross_attention,
                    norm=norm, norm_kwargs=norm_kwargs, norm_first=norm_first,
                    activation_ff=activation_ff, dropout=dropout,
                )
            )
        self.transformer_layers = torch.nn.ModuleList(transformer_layers)

    def add_positional_encoding(self, x):
        b, t, d = x.shape
        assert d % 2 == 0, x.shape
        positions = torch.arange(t, device=x.device)[:, None]
        dimensions = torch.arange(d//2, device=x.device)
        cos_encodings = torch.cos(positions/(10000**(2*dimensions/d)))
        sin_encodings = torch.sin(positions/(10000**(2*dimensions/d)))
        pos_encodings = torch.stack((cos_encodings, sin_encodings), dim=-1)
        pos_encodings = rearrange(pos_encodings, 't d1 d2 -> t (d1 d2)')
        return x + pos_encodings

    def forward(self, x, seq_len, m=None, seq_len_m=None, state=None):
        h = self.lin(x)
        if self.positional_encoding:
            h = self.add_positional_encoding(h)
        if state is None:
            state = len(self.transformer_layers) * [None]
        for i, layer in enumerate(self.transformer_layers):
            h, state[i] = layer(
                h, seq_len=seq_len, m=m, seq_len_m=seq_len_m,
                state=state[i],
            )
        return h, state


def get_causal_mask(x):
    return torch.tril(torch.ones_like(x), diagonal=(x.shape[-1] - x.shape[-2]))
