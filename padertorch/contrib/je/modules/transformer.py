import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch import nn
import math

from paderbox.array.segment import segment_axis
from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.modules.normalization import Normalization
from padertorch.ops.sequence.mask import compute_mask
from padertorch.contrib.je.modules.conv import Conv2d


def scaled_dot_product_attention(q, k, v, seq_len=None, bidirectional=False, mask=None, dropout=0.):
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
    return F.dropout(y, p=dropout)@v, y


def get_causal_mask(x):
    return torch.tril(torch.ones_like(x), diagonal=(x.shape[-1] - x.shape[-2]))


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
            num_heads=8, bidirectional=True, dropout=.0,
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
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.lin_value.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.lin_queue.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)

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
            q, k, v, seq_len=seq_len, bidirectional=self.bidirectional,
            mask=mask, dropout=self.dropout * self.training
        )
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return self.out(x), attention_weights


class TransformerEncoderLayer(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, d_model=512, d_ff=2048, num_heads=8, bidirectional=True,
            self_attention_norm='layer', self_attention_norm_kwargs={},
            ff_norm='layer', ff_norm_kwargs={},
            ff_activation='gelu', layer_scale=True,
            dropout=.0, attention_dropout=.0,
    ):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(
            d_model, d_model, d_model, d_model, d_model,
            num_heads=num_heads, bidirectional=bidirectional,
            dropout=attention_dropout,
        )
        self.hidden = torch.nn.Linear(d_model, d_ff)
        self.out = torch.nn.Linear(d_ff, d_model)

        self.self_attention_norm = self._get_norm(self_attention_norm, self_attention_norm_kwargs, d_model)
        self.ff_norm = self._get_norm(ff_norm, ff_norm_kwargs, d_model)

        self.ff_activation = ACTIVATION_FN_MAP[ff_activation]()
        self.dropout = dropout

        self.self_attention_scale = nn.Parameter(torch.ones((d_model)), requires_grad=layer_scale)
        self.ff_scale = nn.Parameter(torch.ones((d_model)), requires_grad=layer_scale)

    def _get_norm(self, norm, norm_kwargs, d_model):
        if norm is None:
            return None
        else:
            norm_kwargs = {
                "data_format": 'btc',
                "shape": (None, None, d_model),
                'eps': 1e-3,
                **norm_kwargs
            }
            if norm == 'batch':
                norm_kwargs['statistics_axis'] = 'bt'
            elif norm == 'layer':
                norm_kwargs['statistics_axis'] = 'c'
            else:
                raise ValueError(f'{norm} normalization not known.')
            return Normalization(**norm_kwargs)

    def forward(self, x, seq_len, state=None):
        if state is not None:
            assert self.multi_head_self_attention.bidirectional is False
        x_skip = x
        if self.self_attention_norm is not None:
            x = self.self_attention_norm(x, sequence_lengths=seq_len)
        s = x if state is None else torch.cat((state, x), 1)
        h, _ = self.multi_head_self_attention(x, s, s, seq_len=seq_len)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.self_attention_scale * h
        h = h + x_skip
        h_skip = h
        if self.ff_norm is not None:
            h = self.ff_norm(h, sequence_lengths=seq_len)
        y = self.out(self.ff_activation(self.hidden(h)))
        y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.ff_scale * y
        y = y + h_skip
        return y, s


class TransformerEncoder(Module):
    def __init__(
            self, input_size, hidden_size=512, d_ff=2048, num_heads=8,
            num_layers=6, bidirectional=True, input_projection=True,
            self_attention_norm='layer', self_attention_norm_kwargs={},
            ff_norm='layer', ff_norm_kwargs={},
            ff_activation='relu', attention_dropout=.0, dropout=.0,
            positional_encoding=True, class_token=False,
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            input_size:
            hidden_size: d_model
            d_ff:
            num_heads:
            bidirectional:
            self_attention_norm:
            self_attention_norm_kwargs:
            ff_activation:
            attention_dropout:
            dropout:
            positional_encoding:

        Returns:

        >>> x = torch.zeros((2, 3, 8))
        >>> attn = TransformerEncoder(8, 6, 20, 1, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerEncoder(8, 6, 20, 2, 2, bidirectional=True)
        >>> attn(x, seq_len=[1, 2])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerEncoder(8, 6, 20, 2, 2, bidirectional=False)
        >>> attn(x, seq_len=None)[0].shape
        torch.Size([2, 3, 6])
        >>> attn(x, seq_len=None, state=[torch.zeros((2, 5, 6)), torch.zeros((2, 5, 6))])[0].shape
        torch.Size([2, 3, 6])
        >>> attn = TransformerEncoder(8, 6, 20, 2, 2, bidirectional=True, class_token=True)
        >>> attn(x, seq_len=[1, 2])[0].shape
        torch.Size([2, 7, 6])
        """
        super().__init__()
        self.positional_encoding = positional_encoding
        self.num_layers = num_layers
        if input_projection:
            self.input_projection = torch.nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = None
            assert input_size == hidden_size, (input_size, hidden_size)
        transformer_layers = list()
        for i in range(num_layers):
            transformer_layers.append(
                TransformerEncoderLayer(
                    hidden_size, d_ff, num_heads, bidirectional=bidirectional,
                    self_attention_norm=self_attention_norm,
                    self_attention_norm_kwargs=self_attention_norm_kwargs,
                    ff_norm=ff_norm, ff_norm_kwargs=ff_norm_kwargs,
                    ff_activation=ff_activation,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
            )
        self.transformer_layers = torch.nn.ModuleList(transformer_layers)
        if class_token:
            self.class_token = nn.Parameter(
                torch.randn(hidden_size), requires_grad=True
            )
        else:
            self.class_token = None

    def forward(self, x, seq_len, state=None):
        if self.input_projection is None:
            h = x
        else:
            h = self.input_projection(x)
        if self.positional_encoding:
            h = add_positional_encoding(h)
        if self.class_token is not None:
            h, seq_len = add_class_token(h, seq_len, self.class_token)
        if state is None:
            state = len(self.transformer_layers) * [None]
        for i, layer in enumerate(self.transformer_layers):
            h, state[i] = layer(
                h, seq_len=seq_len, state=state[i],
            )
        return h, state


def add_positional_encoding(x, sequence_axis=-2, channel_axis=-1):
    if sequence_axis < 0:
        sequence_axis = x.dim() + sequence_axis
    if channel_axis < 0:
        channel_axis = x.dim() + channel_axis
    t = x.shape[sequence_axis]
    d = x.shape[channel_axis]
    assert d % 2 == 0, x.shape
    positions = torch.arange(t, device=x.device)[:, None]
    dimensions = torch.arange(d//2, device=x.device)
    cos_encodings = torch.cos(positions/(10000**(2*dimensions/d)))
    sin_encodings = torch.sin(positions/(10000**(2*dimensions/d)))
    pos_encodings = torch.cat((cos_encodings, sin_encodings), dim=-1)
    if sequence_axis > channel_axis:
        pos_encodings = pos_encodings.T
        for axis in list(range(channel_axis+1, sequence_axis)) + list(range(sequence_axis+1, x.dim())):
            pos_encodings = pos_encodings.unsqueeze(axis-channel_axis)
    else:
        for axis in list(range(sequence_axis+1, channel_axis)) + list(range(channel_axis+1, x.dim())):
            pos_encodings = pos_encodings.unsqueeze(axis-sequence_axis)
    return x + pos_encodings


def add_class_token(h, seq_len, class_token):
    b, t, c = h.shape
    if seq_len is None:
        h = torch.cat((h, class_token.expand((b, 1, c))), dim=1)
    else:
        h = torch.stack([
            torch.cat((
                h[i, :seq_len[i]], class_token[None], h[i, seq_len[i]:]
            ))
            for i in range(b)
        ])
    seq_len = np.asarray(seq_len) + 1
    return h, seq_len


class AST(Module):
    def __init__(
            self, in_channels, out_channels=512, d_ff=2048, num_heads=8,
            patch_size=16, patch_stride=10, input_height=128,
            max_timesteps=101, num_layers=12,
            patch_emb_norm='batch', patch_emb_norm_kwargs={},
            self_attention_norm='layer', self_attention_norm_kwargs={},
            ff_norm='layer', ff_norm_kwargs={},
            activation_fn='relu', output_net=None,
            attention_dropout=.0, dropout=.0,
            frequency_patchout=.0, n_frequency_patchout_blocks=1,
            temporal_patchout=.0, n_temporal_patchout_blocks=1,
            class_token=True, output_tokens=True,
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            in_channels:
            out_channels: d_model
            d_ff:
            num_heads:
            self_attention_norm:
            self_attention_norm_kwargs:
            activation_fn:
            dropout:

        Returns:

        >>> x = torch.randn(4,3,43,45)
        >>> seq_len = np.full(4, 45)
        >>> ast = AST(3, 16, 32, input_height=10, frequency_patchout=0.2, temporal_patchout=0.2, class_token=False)
        >>> y, seq_len_y = ast(x, seq_len)
        >>> y.shape, seq_len_y
        >>> ast = AST(3, 16, 32, input_height=10, frequency_patchout=0.2, temporal_patchout=0.2, class_token=True)
        >>> y, seq_len_y = ast(x, seq_len)
        >>> y.shape, seq_len_y
        """
        super().__init__()
        self.num_layers = num_layers
        self.patch_embed = Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=patch_stride, activation_fn=None,
            norm=patch_emb_norm, norm_kwargs=patch_emb_norm_kwargs,
        )
        self.output_net = output_net
        transformer_layers = list()
        for i in range(num_layers):
            transformer_layers.append(
                TransformerEncoderLayer(
                    out_channels, d_ff, num_heads,
                    bidirectional=True,
                    self_attention_norm=self_attention_norm,
                    self_attention_norm_kwargs=self_attention_norm_kwargs,
                    ff_norm=ff_norm, ff_norm_kwargs=ff_norm_kwargs,
                    ff_activation=activation_fn,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
            )
        self.transformer_layers = torch.nn.ModuleList(transformer_layers)
        self.frequency_patchout = frequency_patchout
        assert 0. <= self.frequency_patchout < 1., self.frequency_patchout
        self.n_frequency_patchout_blocks = n_frequency_patchout_blocks
        self.temporal_patchout = temporal_patchout
        assert 0. <= self.temporal_patchout < 1., self.temporal_patchout
        self.n_temporal_patchout_blocks = n_temporal_patchout_blocks

        patches_per_timestep = self.patch_embed.get_output_shape((1, in_channels, input_height, 1))[-2]
        self.frequency_encoding = nn.Parameter(
            torch.randn((out_channels, patches_per_timestep, 1)),
            requires_grad=True,
        )
        # torch.nn.init.normal_(self.frequency_encoding, std=0.02)
        self.time_encoding = nn.Parameter(
            torch.randn((out_channels, 1, max_timesteps)), requires_grad=True
        )
        # torch.nn.init.normal_(self.time_encoding, std=0.02)
        if output_tokens:
            self.output_tokens = nn.Parameter(
                torch.randn((out_channels, 1, max_timesteps)), requires_grad=True
            )
            # torch.nn.init.normal_(self.output_tokens, std=0.02)
        else:
            self.output_tokens = None
        if class_token:
            self.class_token = nn.Parameter(
                torch.randn(out_channels), requires_grad=True
            )
            # torch.nn.init.normal_(self.class_token, std=0.02)
        else:
            self.class_token = None

    def forward(self, x, seq_len):
        # if self.training:
        #     print(x.shape)
        h, seq_len = self.patch_embed(x, seq_len)
        h = h + self.frequency_encoding + self.time_encoding[..., :h.shape[-1]]
        b, c, f, t = h.shape
        n_temp_patchout = int(self.temporal_patchout * t) if self.training else 0
        n_freq_patchout = int(self.frequency_patchout * f) if self.training else 0
        if n_temp_patchout > 0:
            n_cum_patchout = np.linspace(0, n_temp_patchout, self.n_temporal_patchout_blocks+1).astype(int)
            n_patchout_per_block = n_cum_patchout[1:] - n_cum_patchout[:-1]
            indices = []
            for i in range(b):
                indices.append(np.arange(t))
                for j, nj in enumerate(n_patchout_per_block):
                    if nj == 0:
                        continue
                    block_onset = np.random.choice(t - n_cum_patchout[j+1])
                    idx = np.concatenate((np.arange(block_onset), np.arange(block_onset + nj, len(indices[-1]))))
                    indices[-1] = indices[-1][idx]
                assert len(indices[-1]) == (t - n_temp_patchout)
            indices = np.stack(indices)
            h = h[np.arange(b)[:, None, None, None], np.arange(c)[:, None, None], np.arange(f)[:, None], indices[:, None, None]]
            seq_len = (indices <= seq_len[:, None]).sum(-1)
            t = t - n_temp_patchout
        if n_freq_patchout > 0:
            n_cum_patchout = np.linspace(0, n_freq_patchout, self.n_frequency_patchout_blocks+1).astype(int)
            n_patchout_per_block = n_cum_patchout[1:] - n_cum_patchout[:-1]
            indices = []
            for i in range(b):
                indices.append(np.arange(f))
                for j, nj in enumerate(n_patchout_per_block):
                    if nj == 0:
                        continue
                    block_onset = np.random.choice(f - n_cum_patchout[j+1])
                    idx = np.concatenate((np.arange(block_onset), np.arange(block_onset + nj, len(indices[-1]))))
                    indices[-1] = indices[-1][idx]
                assert len(indices[-1]) == (f - n_freq_patchout)
            indices = np.stack(indices)
            h = h[np.arange(b)[:, None, None], np.arange(c)[:, None], indices[:, None]]
            f = f - n_freq_patchout
        if self.output_tokens is not None:
            h = torch.cat((self.output_tokens[..., :t].expand((b, c, 1, t)), h), dim=2)
            f += 1
        h = rearrange(h, 'b c f t -> b (t f) c')
        if self.class_token is not None:
            h, _ = add_class_token(
                h, np.asarray(seq_len)*f, self.class_token,
            )
            seq_len = np.asarray(seq_len) + 1
        for i, layer in enumerate(self.transformer_layers):
            h, _ = layer(h, seq_len=seq_len*f)
        if self.class_token is not None:
            h = torch.cat((h, torch.zeros((b, f-1, c), device=h.device)), dim=1)
            t += 1
        h = rearrange(h, 'b (t f) c -> b c f t', t=t, f=f)[:, :, 0]
        if self.output_net is not None:
            h, seq_len = self.output_net(h, seq_len)
        return h, seq_len

    _total_stride = None
    def get_total_stride(self):
        if self._total_stride is None:
            total_stride = self.patch_embed.stride[-1] if isinstance(self.patch_embed.stride, (list, tuple)) else self.patch_embed.stride
            if self.output_net is not None:
                out_stride_down, out_stride_up = self.output_net.get_total_stride()
                total_stride = total_stride * out_stride_down
                assert (total_stride % out_stride_up) == 0, (total_stride, out_stride_up)
                total_stride = total_stride // out_stride_up
            self._total_stride = total_stride
        return self._total_stride, 1


class LocalAST(Module):
    def __init__(
            self, in_channels, out_channels=512, d_ff=2048, num_heads=8,
            patch_size=16, patch_stride=10, input_height=128,
            rf_size=1, stride=1, num_layers=12,
            patch_emb_norm='layer', patch_emb_norm_kwargs={},
            self_attention_norm='layer', self_attention_norm_kwargs={},
            ff_norm='layer', ff_norm_kwargs={},
            activation_fn='relu', output_net=None,
            attention_dropout=.0, dropout=.0,
            frequency_patchout=.0, n_frequency_patchout_blocks=1,
            temporal_patchout=.0, n_temporal_patchout_blocks=1,
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            in_channels:
            out_channels: d_model
            d_ff:
            num_heads:
            self_attention_norm:
            self_attention_norm_kwargs:
            activation_fn:
            dropout:

        Returns:

        >>> x = torch.randn(4,3,43,45)
        >>> seq_len = np.full(4, 45)
        >>> last = LocalAST(3, 16, 32, rf_size=3, input_height=10, frequency_patchout=0.2, temporal_patchout=0.)
        >>> y, seq_len_y = last(x, seq_len)
        >>> y.shape, seq_len_y
        """
        super().__init__()
        self.num_layers = num_layers
        self.patch_embed = Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=patch_stride, activation_fn=None,
            norm=patch_emb_norm, norm_kwargs=patch_emb_norm_kwargs,
        )
        self.output_net = output_net
        self.rf_size = rf_size
        self.stride = stride
        transformer_layers = list()
        for i in range(num_layers):
            transformer_layers.append(
                TransformerEncoderLayer(
                    out_channels, d_ff, num_heads,
                    bidirectional=True,
                    self_attention_norm=self_attention_norm,
                    self_attention_norm_kwargs=self_attention_norm_kwargs,
                    ff_norm=ff_norm, ff_norm_kwargs=ff_norm_kwargs,
                    ff_activation=activation_fn,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
            )
        self.transformer_layers = torch.nn.ModuleList(transformer_layers)
        self.frequency_patchout = frequency_patchout
        assert 0. <= self.frequency_patchout < 1., self.frequency_patchout
        self.n_frequency_patchout_blocks = n_frequency_patchout_blocks
        self.temporal_patchout = temporal_patchout
        assert 0. <= self.temporal_patchout < 1., self.temporal_patchout
        self.n_temporal_patchout_blocks = n_temporal_patchout_blocks

        self.output_token = nn.Parameter(
            torch.randn(out_channels), requires_grad=True
        )
        # torch.nn.init.normal_(self.output_token, std=0.02)
        patches_per_timestep = self.patch_embed.get_output_shape((1, in_channels, input_height, 1))[-2]
        self.frequency_encoding = nn.Parameter(
            torch.randn((out_channels, patches_per_timestep, 1)),
            requires_grad=True,
        )
        # torch.nn.init.normal_(self.frequency_encoding, std=0.02)
        self.time_encoding = nn.Parameter(
            torch.randn((out_channels, 1, 1, rf_size)), requires_grad=True
        )

    def forward(self, x, seq_len):
        seq_len_orig = seq_len
        h, seq_len = self.patch_embed(x, seq_len)
        h = h + self.frequency_encoding
        b, c, f, t = h.shape

        n_freq_patchout = int(self.frequency_patchout * f) if self.training else 0
        if n_freq_patchout > 0:
            n_cum_patchout = np.linspace(0, n_freq_patchout, self.n_frequency_patchout_blocks+1).astype(int)
            n_patchout_per_block = n_cum_patchout[1:] - n_cum_patchout[:-1]
            freq_pathout_indices = []
            for i in range(b):
                freq_pathout_indices.append(np.arange(f))
                for j, nj in enumerate(n_patchout_per_block):
                    if nj == 0:
                        continue
                    block_onset = np.random.choice(f - n_cum_patchout[j+1])
                    idx = np.concatenate((np.arange(block_onset), np.arange(block_onset + nj, len(freq_pathout_indices[-1]))))
                    freq_pathout_indices[-1] = freq_pathout_indices[-1][idx]
                assert len(freq_pathout_indices[-1]) == (f - n_freq_patchout)
            freq_pathout_indices = np.stack(freq_pathout_indices)
            h = h[np.arange(b)[:, None, None], np.arange(c)[:, None], freq_pathout_indices[:, None]]
            f = f - n_freq_patchout

        if self.rf_size > 1:
            h = torch.cat((
                torch.zeros_like(h[..., :(self.rf_size - 1) // 2]),
                h,
                torch.zeros_like(h[..., :math.ceil((self.rf_size - 1) / 2)])
            ), dim=-1)
            h = segment_axis(h, self.rf_size, self.stride, end='cut', axis=-1)
            h = h + self.time_encoding
            h = rearrange(h, 'b c f t s -> b c (s f) t')
            f *= self.rf_size
        elif self.stride > 1:
            h = h[..., (self.stride-1)//2::self.stride] + self.time_encoding[:, 0]
        seq_len = 1 + (seq_len - 1 - (self.stride-1)//2) // self.stride
        t = t_orig = h.shape[-1]

        n_temp_patchout = int(self.temporal_patchout * t) if self.training else 0
        if n_temp_patchout > 0:
            n_cum_patchout = np.linspace(0, n_temp_patchout, self.n_temporal_patchout_blocks+1).astype(int)
            n_patchout_per_block = n_cum_patchout[1:] - n_cum_patchout[:-1]
            temp_patchout_indices = []
            for i in range(b):
                temp_patchout_indices.append(np.arange(t))
                for j, nj in enumerate(n_patchout_per_block):
                    if nj == 0:
                        continue
                    block_onset = np.random.choice(t - n_cum_patchout[j+1])
                    idx = np.concatenate((np.arange(block_onset), np.arange(block_onset + nj, len(temp_patchout_indices[-1]))))
                    temp_patchout_indices[-1] = temp_patchout_indices[-1][idx]
                assert len(temp_patchout_indices[-1]) == (t - n_temp_patchout)
            temp_patchout_indices = np.stack(temp_patchout_indices)
            h = h[np.arange(b)[:, None, None, None], np.arange(c)[:, None, None], np.arange(f)[:, None], temp_patchout_indices[:, None, None]]
            # seq_len = (temp_patchout_indices <= seq_len[:, None]).sum(-1)
            t = t - n_temp_patchout
        else:
            temp_patchout_indices = None
        h = rearrange(h, 'b c f t -> (b t) f c')
        h = torch.cat((
            self.output_token.expand((b * t, 1, c)), h,
        ), dim=1)
        f += 1
        for i, layer in enumerate(self.transformer_layers):
            h, _ = layer(h, seq_len=np.array(b*t*[h.shape[1]]))
        h = rearrange(h[:, 0], '(b t) c -> b c t', b=b, t=t)
        if temp_patchout_indices is not None:
            h = torch.stack([
                torch.index_add(
                    torch.zeros((h.shape[1], t_orig), device=h.device),
                    dim=1, index=torch.tensor(temp_patchout_indices[i], device=h.device),
                    source=h[i],
                )
                for i in range(h.shape[0])
            ])
        if self.output_net is not None:
            h, seq_len = self.output_net(h, seq_len)
        if self.get_total_stride()[0] == 1:
            seq_len = seq_len_orig
            h = h[..., :max(seq_len_orig)]
        return h, seq_len

    _total_stride = None
    def get_total_stride(self):
        if self._total_stride is None:
            patch_stride = self.patch_embed.stride[-1] if isinstance(self.patch_embed.stride, (list, tuple)) else self.patch_embed.stride
            total_stride = patch_stride * self.stride
            if self.output_net is not None:
                out_stride_down, out_stride_up = self.output_net.get_total_stride()
                total_stride = total_stride * out_stride_down
                assert (total_stride % out_stride_up) == 0, (total_stride, out_stride_up)
                total_stride = total_stride // out_stride_up
            self._total_stride = total_stride
        return self._total_stride, 1
