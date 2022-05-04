import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.modules.normalization import Normalization
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.rnn import reverse_sequence
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d


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
            self, queue_size, key_size, value_size, hidden_size, output_size, num_heads=1, bidirectional=False
    ):
        super().__init__()
        self.queue_size = queue_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.lin_queue = torch.nn.Linear(queue_size, self.hidden_size)
        self.lin_key = torch.nn.Linear(key_size, self.hidden_size)
        self.lin_value = torch.nn.Linear(value_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, q, k, v, seq_len=None, mask=None):
        B, Tq, _ = q.shape
        B, Tk, _ = k.shape
        q = self.lin_queue(q).view(
            B, Tq, self.num_heads, self.hidden_size//self.num_heads
        ).transpose(1, 2)
        k = self.lin_key(k).view(
            B, Tk, self.num_heads, self.hidden_size//self.num_heads
        ).transpose(1, 2)
        v = self.lin_value(v).view(
            B, Tk, self.num_heads, self.hidden_size//self.num_heads
        ).transpose(1, 2)
        x, attention_weights = scaled_dot_product_attention(
            q, k, v, seq_len=seq_len, bidirectional=self.bidirectional, mask=mask
        )
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.hidden_size)
        return self.out(x), attention_weights


class TransformerBlock(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, input_size, hidden_size, output_size, num_heads=1,
            bidirectional=False, cross_attention=False,
            norm='layer', norm_kwargs={}, activation='relu', dropout=.0,
    ):
        super().__init__()
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.dropout = dropout
        self.multi_head_self_attention = MultiHeadAttention(
            input_size, input_size, input_size, hidden_size, hidden_size,
            num_heads=num_heads, bidirectional=bidirectional
        )
        self.cross_attention = cross_attention
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)

        if norm is None:
            self.self_attention_norm = None
            self.output_norm = None
        else:
            norm_kwargs = {
                "data_format": 'btc',
                "shape": (None, None, hidden_size),
                'eps': 1e-2,
                **norm_kwargs
            }
            if norm == 'batch':
                norm_kwargs['statistics_axis'] = 'bt'
            elif norm == 'layer':
                norm_kwargs['statistics_axis'] = 'c'
            else:
                raise ValueError(f'{norm} normalization not known.')
            norm_kwargs["shape"] = (None, None, hidden_size)
            self.self_attention_norm = Normalization(**norm_kwargs)
            norm_kwargs["shape"] = (None, None, output_size)
            self.output_norm = Normalization(**norm_kwargs)

        if cross_attention:
            self.multi_head_cross_attention = MultiHeadAttention(
                hidden_size*num_heads, hidden_size*num_heads, hidden_size*num_heads,
                hidden_size, hidden_size*num_heads,
                num_heads, bidirectional=True
            )
            if norm is None:
                self.cross_attention_norm = None
            else:
                norm_kwargs["shape"] = (None, None, hidden_size)
                self.cross_attention_norm = Normalization(**norm_kwargs)

    def forward(self, x, seq_len, v=None, seq_len_v=None, state=None):
        x_ = x if state is None else torch.cat([state, x], 1)
        h, _ = self.multi_head_self_attention(x, x_, x_, seq_len=seq_len)
        if self.training and self.dropout > 0.:
            h = F.dropout(h, self.dropout)
        if h.shape == x.shape:
            h = h + x
        if self.self_attention_norm is not None:
            h = self.self_attention_norm(h, sequence_lengths=seq_len)
        if self.cross_attention:
            assert v is not None
            q = h
            h, _ = self.multi_head_cross_attention(q, v, v, seq_len=seq_len_v)
            if self.training and self.dropout > 0.:
                h = F.dropout(h, self.dropout)
            if h.shape == q.shape:
                h = h + q
            if self.cross_attention_norm is not None:
                h = self.cross_attention_norm(h, sequence_lengths=seq_len)
        y = self.out(self.activation(self.hidden(h)))
        if self.training and self.dropout > 0.:
            y = F.dropout(y, self.dropout)
        if h.shape == y.shape:
            y = y + h
        if self.output_norm is not None:
            y = self.output_norm(y, sequence_lengths=seq_len)
        return y, x_


class TransformerStack(Module):
    def __init__(
            self, input_size, hidden_size, output_net=None, num_heads=1,
            num_layers=1, bidirectional=False, reverse=False, cross_attention=False,
            activation='relu', norm='layer', norm_kwargs={},
            positional_encoding=True, dropout=0.,
            return_state=False,
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
        >>> attn = TransformerStack(8, 6, 6, 1, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2]).shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerStack(8, 6, 6, 2, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2]).shape
        torch.Size([2, 3, 6])
        >>> attn(x, seq_len=None, state=[torch.zeros((2, 6, 8)), torch.zeros((2, 6, 6))]).shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerStack(8, 6, 6, 2, 2, bidirectional=False)
        >>> attn(x, seq_len=None).shape
        torch.Size([2, 3, 6])
        >>> attn(x, seq_len=None, state=[torch.zeros((2, 6, 8)), torch.zeros((2, 6, 6))]).shape
        torch.Size([2, 3, 6])
        """
        super().__init__()
        self.reverse = reverse
        self.positional_encoding = positional_encoding
        self.dropout = dropout
        self.return_state = return_state
        stack = list()
        for i in range(num_layers):
            stack.append(
                TransformerBlock(
                    input_size, hidden_size, hidden_size, num_heads,
                    bidirectional=bidirectional, cross_attention=cross_attention,
                    activation=activation, norm=norm, norm_kwargs=norm_kwargs,
                    dropout=dropout,
                )
            )
            input_size = hidden_size
        self.stack = torch.nn.ModuleList(stack)
        self.output_net = output_net

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

    def forward(self, x, seq_len, v=None, seq_len_v=None, state=None):
        x = rearrange(x, 'b f t -> b t f')
        if self.reverse:
            x = reverse_sequence(x, seq_len=seq_len)
        new_state = []
        if self.positional_encoding:
            x = self.add_positional_encoding(x)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, self.dropout)
        for i, layer in enumerate(self.stack):
            x, x_ = layer(
                x, seq_len=seq_len, v=v, seq_len_v=seq_len_v,
                state=None if state is None else state[i],
            )
            new_state.append(x_)
        if self.reverse:
            x = reverse_sequence(x, seq_len=seq_len)
        x = rearrange(x, 'b t f -> b f t')
        if self.output_net is not None:
            x, seq_len = self.output_net(x, seq_len)
        if self.return_state:
            return x, seq_len, new_state
        return x, seq_len

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config['output_net'] is not None:
            config['output_net']['factory'] = CNN1d
            if config['output_net']['factory'] in [CNN1d, CNNTranspose1d]:
                config['output_net']['in_channels'] = config['hidden_size']
            else:
                raise ValueError(f'output_net factory {config["output_net"]["factory"]} not allowed.')


def get_causal_mask(x):
    return torch.tril(torch.ones_like(x), diagonal=(x.shape[-1] - x.shape[-2]))
