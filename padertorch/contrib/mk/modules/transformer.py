from collections import namedtuple
import copy
import functools
import logging
import math
import typing as tp

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import padertorch as pt
from padertorch.ops.mappings import _CallableDispatcher
from padertorch.contrib.je.modules.conv_utils import Pad
from padertorch.contrib.mk.modules.activations import GELU
from padertorch.contrib.mk.typing import TActivationFn, TSeqLen
from padertorch.contrib.mk.modules.utils import normalize
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.ops.sequence.mask import compute_mask

LOG = logging.getLogger(__name__)

NormOutputs = namedtuple('NORM_OUTPUTS', ['x', 'layer_scale'], defaults=[1])


def interleave(x, y, dim):
    """

    Args:
        x:
        y:
        dim:

    Returns:

    >>> interleave(torch.Tensor([[1,2,3]]),torch.Tensor([[4,5,6]]), dim=1)
    """
    dim = dim % x.ndim
    assert x.ndim > dim >= 0, dim
    shape = [*x.shape]
    shape[dim] *= 2
    return torch.stack((x, y), dim=dim+1).view(shape)


def positional_embedding(d_model: int, max_len: int = 5000):
    position = torch.arange(max_len).unsqueeze(1)
    half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, half) * (-math.log(float(max_len)) / half)
    )
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


class Linear(pt.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        magnitude_preserving: bool = False,
        chunks: tp.Optional[int] = None,
        generator: tp.Optional[torch.Generator] = None,
    ):
        super().__init__()

        self.magnitude_preserving = magnitude_preserving
        self.chunks = chunks

        if magnitude_preserving:
            w = torch.randn(out_features, in_features, generator=generator)
        else:
            w = torch.nn.init.xavier_uniform_(
                torch.empty(out_features, in_features),
                generator=generator,
            )
        self.weight = nn.Parameter(w)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor):
        w = self.weight
        if self.magnitude_preserving:
            if self.training:
                with torch.no_grad():
                    self.weight.copy_(
                        normalize(w, chunks=self.chunks)
                    )
            in_features = w.shape[1]
            w = normalize(w, chunks=self.chunks) / np.sqrt(in_features)
            if self.bias is not None:
                bias = 0.5*self.bias
                x = 0.5*x
                scale = np.sqrt(2)
            else:
                bias = self.bias
                scale = 1.
        else:
            w = self.weight
            bias = self.bias
            scale = 1.
        x = F.linear(x, w, bias=bias)/scale
        return x


class PositionalEncoding(pt.Module):
    """_summary_

    Args:
        d_model (int): _description_
        dropout (float, optional): _description_. Defaults to 0.
        max_len (int, optional): _description_. Defaults to 5000.
    """

    def __init__(
        self, d_model: int, dropout: float = 0., max_len: int = 5000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = positional_embedding(d_model, max_len)
        if batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def reset_parameters(self):
        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` if
                batch_first is False else ``[batch_size, seq_len, embedding_dim]``
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalConvEmbedding(pt.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 128,
        groups: int = 16,
        causal: bool = False,
        dropout: float = 0.,
        norm: tp.Optional[TActivationFn] = 'layer',
        batch_first: bool = True,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
        )
        self.pad = Pad("front" if causal else "both")
        self.activation_fn = GELU()
        if norm is None:
            self.norm = NormOutputs
        else:
            self.norm = NORM_MAP[norm](d_model)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        if use_weight_norm:
            self.apply_weight_norm()

    # https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/models/hifigan.py
    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                torch.nn.utils.parametrizations.weight_norm(m)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # This module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def reset_parameters(self, seed=None):
        generator = torch.Generator(
            device=self.conv.weight.device
        )
        if seed is not None:
            generator.manual_seed(seed)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=1., generator=generator
        )
        torch.nn.init.zeros_(self.conv.bias)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` if
                batch_first is False else
                ``[batch_size, seq_len, embedding_dim]``
        """
        if self.batch_first:
            x = x.moveaxis(1, -1)
        else:
            x = x.moveaxis(0, -1)
        pe = self.activation_fn(self.conv(self.pad(x[:, :self.d_model], self.kernel_size-1)))
        x = self.dropout(self.norm((x + pe).moveaxis(-1, 1)).x)
        return x


class RoPE(pt.Module):
    """Rotary Positional Embedding (RoPE) [1]_, [2]_.

    References:
        .. [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary
            position embedding." Neurocomputing 568 (2024)
        .. [2] https://github.com/ZhuiyiTechnology/roformer
    """
    def __init__(
        self, d_model: int,
        max_len: int = 10_000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.batch_first = batch_first

        half = d_model // 2
        theta = max_len ** (-2*torch.arange(half)/d_model)
        self.register_buffer(
            'theta', torch.repeat_interleave(theta, 2)[None, None]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, [nheads,] seq_len, embedding_dim]``
        """
        x_quad = interleave(-x[..., 1::2], x[..., ::2], dim=-1)
        pos = torch.arange(
            1, x.shape[-2]+1, device=x.device, dtype=x.dtype
        ).unsqueeze(-1)[None]
        if x.ndim == 4:
            pos = pos.unsqueeze(1)
            theta = self.theta.unsqueeze(1)
        else:
            theta = self.theta
        cos_pos = torch.cos(pos * theta)
        sin_pos = torch.sin(pos * theta)
        x = x * cos_pos + x_quad * sin_pos
        return x


class ScaledDotProductAttention(pt.Module):
    def __init__(
        self, *args,
        enable_flash: tp.Optional[bool] = None,
        enable_mem_efficient: tp.Optional[bool] = None,
        magnitude_preserving: bool = False,
        **kwargs
    ):
        super().__init__()
        self.enable_flash = enable_flash
        self.enable_mem_efficient = enable_mem_efficient
        self.magnitude_preserving = magnitude_preserving
        if not torch.cuda.is_available():
            self.enable_flash = False
            self.enable_mem_efficient = False

    @staticmethod
    def _call_attention(q, k, v, attn_mask, is_causal):
        if is_causal:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def _check(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor,
        is_causal: bool,
    ):
        if self.enable_flash is None:
            params = torch.backends.cuda.SDPAParams(
                q, k, v, attn_mask, 0., is_causal, False,
            )
            self.enable_flash = torch.backends.cuda\
                .can_use_flash_attention(params)
        if self.enable_mem_efficient is None:
            params = torch.backends.cuda.SDPAParams(
                q, k, v, attn_mask, 0., is_causal, False,
            )
            self.enable_mem_efficient = torch.backends.cuda\
                .can_use_efficient_attention(params)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor,
        attn_mask: tp.Optional[Tensor] = None, is_causal: bool = False,
    ):
        if self.magnitude_preserving:
            q = normalize(q)
            k = normalize(k)
            v = normalize(v)
        try:
            # Requires torch>=2.3
            from torch.nn.attention import sdpa_kernel, SDPBackend

            self._check(q, k, v, attn_mask, is_causal)
            backends = []
            if self.enable_flash:
                backends.append(SDPBackend.FLASH_ATTENTION)
                try:
                    with (
                        sdpa_kernel(backends),
                        torch.autocast(device_type='cuda')
                    ):
                        return self._call_attention(
                            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                        ), None
                except RuntimeError as exc:
                    raise exc
            if self.enable_mem_efficient:
                backends.append(SDPBackend.EFFICIENT_ATTENTION)
            if not (self.enable_flash or self.enable_mem_efficient):
                backends.extend([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])
            with sdpa_kernel(backends):
                return self._call_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                ), None
        except ImportError:
            message = (
                "torch>=2.3 is required for efficient attention. "
                "Falling back to default implementation"
            )
            LOG.warning(message)
            LOG.addFilter(lambda record: record == message)
            return self._call_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            ), None


ATTENTION_FN_MAP = _CallableDispatcher(
    scaled_dot_product=ScaledDotProductAttention,
)


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        cond_dim: tp.Optional[int] = None,
        activation_fn: TActivationFn = None,
        layer_scale: bool = False,
        zero_init: bool = False,
        magnitude_preserving: bool = False,
    ):
        super().__init__(
            normalized_shape=normalized_shape, eps=eps,
            elementwise_affine=elementwise_affine, bias=bias,
            device=device, dtype=dtype,
        )
        self.magnitude_preserving = magnitude_preserving
        self.shift = bias
        self.layer_scale = layer_scale
        if cond_dim is not None:
            self.cond_layer = Linear(
                cond_dim, (1+bias+layer_scale)*normalized_shape,
                magnitude_preserving=magnitude_preserving,
                chunks=(1+bias+layer_scale),
            )
            if activation_fn is not None:
                activation_fn = ACTIVATION_FN_MAP[activation_fn]()
                self.cond_layer = nn.Sequential(
                    activation_fn, self.cond_layer
                )
            if magnitude_preserving:
                self.gain = nn.Parameter(torch.tensor(0.))
            if layer_scale and zero_init:
                for param in self.cond_layer.parameters():
                    param[-normalized_shape:].data.zero_()
        else:
            if magnitude_preserving:
                raise NotImplementedError()
            self.cond_layer = None

    def expand(self, tensor: Tensor, target_shape: tuple):
        while tensor.ndim < len(target_shape):
            tensor = tensor.unsqueeze(1)
        return tensor

    def forward(
        self,
        x: Tensor,
        cond: tp.Optional[Tensor] = None,
    ):
        if self.elementwise_affine and cond is not None:
            raise ValueError(
                'Expected condition to be None when elementwise_affine=True, '
                f'but got input of type {type(cond)}.'
            )

        x = super().forward(x)
        alpha = None
        if cond is not None:
            params = self.cond_layer(cond)
            if self.shift or self.layer_scale:
                params = torch.chunk(
                    params, 1+self.shift+self.layer_scale, dim=-1
                )
                if self.shift and self.layer_scale:
                    gamma, beta, alpha = params
                elif self.shift:
                    gamma, beta = params
                    alpha = None
                else:
                    gamma, alpha = params
                    beta = None
            else:
                gamma = params
                alpha = beta = None
            gamma = self.expand(gamma, x.shape)
            if self.magnitude_preserving:
                gamma = gamma*self.gain+1
                x = x*gamma
            else:
                x = x*gamma
            if beta is not None:
                beta = self.expand(beta, x.shape)
                if self.magnitude_preserving:
                    x = 0.5*x+0.5*beta/math.sqrt(2)
                else:
                    x = x + beta
            if alpha is not None:
                alpha = self.expand(alpha, x.shape)
        return NormOutputs(x, alpha)


class DynamicTanh(pt.Module):
    """Implementation of Dynamic Tanh (DyT) [1]_

    [1]_: Zhu, Jiachen et al., "Transformers without Normalization",
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition (CVPR), 2025.
    """
    def __init__(
        self,
        num_features: int,
        alpha_init_value: float = 0.5,
        elementwise_affine: bool = True,
        bias: bool = True,
        cond_dim: tp.Optional[int] = None,
        activation_fn: TActivationFn = None,
        layer_scale: bool = False,
        zero_init: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.shift = bias
        self.layer_scale = layer_scale
        if cond_dim is not None:
            self.cond_layer = Linear(
                cond_dim, (1+bias+layer_scale)*num_features,
                chunks=(1+bias+layer_scale),
            )
            if activation_fn is not None:
                activation_fn = ACTIVATION_FN_MAP[activation_fn]()
                self.cond_layer = nn.Sequential(
                    activation_fn, self.cond_layer
                )
            if layer_scale and zero_init:
                for param in self.cond_layer.parameters():
                    param[-num_features:].data.zero_()
        else:
            self.cond_layer = None

    def expand(self, tensor: Tensor, target_shape: tuple):
        while tensor.ndim < len(target_shape):
            tensor = tensor.unsqueeze(1)
        return tensor

    def forward(self, x: Tensor, cond: tp.Optional[Tensor] = None):
        x = torch.tanh(self.alpha * x)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        alpha = None
        if cond is not None:
            params = self.cond_layer(cond)
            if self.shift or self.layer_scale:
                params = torch.chunk(
                    params, 1+self.shift+self.layer_scale, dim=-1
                )
                if self.shift and self.layer_scale:
                    gamma, beta, alpha = params
                elif self.shift:
                    gamma, beta = params
                    alpha = None
                else:
                    gamma, alpha = params
                    beta = None
            else:
                gamma = params
                alpha = beta = None
            gamma = self.expand(gamma, x.shape)
            x = x*gamma
            if beta is not None:
                beta = self.expand(beta, x.shape)
                x = x + beta
            if alpha is not None:
                alpha = self.expand(alpha, x.shape)
        return NormOutputs(x, alpha)


NORM_MAP = _CallableDispatcher(
    batch1d=nn.BatchNorm1d,
    layer=LayerNorm,
    adaln=functools.partial(LayerNorm, elementwise_affine=False),
    rms=nn.RMSNorm,
    dyt=DynamicTanh,
)


class MultiheadAttention(pt.Module):
    """
    https://arxiv.org/abs/1706.03762

    >>> q = torch.randn((2, 3, 4))
    >>> k = torch.randn((2, 6, 6))
    >>> v = torch.randn((2, 6, 8))
    >>> attn = MultiheadAttention(4, 4, kdim=6, vdim=8)
    >>> y, w = attn(q, k, v)
    >>> y.shape
    torch.Size([2, 3, 4])
    >>> attn = MultiheadAttention(4, 2, kdim=6, vdim=8)
    >>> y, w = attn(q, k, v)
    >>> y.shape
    torch.Size([2, 3, 4])

    Args:
        embed_dim (int):
        num_heads (int):
        dropout (float, optional): Defaults to 0..
        bias (bool, optional): Defaults to True.
        add_bias_kv (bool, optional): Defaults to False.
        kdim (int, optional): Defaults to None.
        vdim (int, optional): Defaults to None.
        attention_fn (TActivationFn, optional): Defaults to 'scaled_dot'.
        batch_first (bool, optional): Defaults to True.
    """
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.,
        bias: bool = True, add_bias_kv: bool = False,
        kdim: tp.Optional[int] = None, vdim: tp.Optional[int] = None, attention_fn: TActivationFn = 'scaled_dot_product',
        batch_first: bool = True,
        magnitude_preserving: bool = False,
        l2_normalization: bool = False,
        flash_attention: bool = False,
        rope: bool = False,
        rms_norm: bool = False,
        linear_attention_bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert batch_first, "Only batch_first=True is supported"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_fn = ATTENTION_FN_MAP[attention_fn](
            embed_dim, num_heads,
            magnitude_preserving=magnitude_preserving,
            **kwargs
        )
        self.lin_query = Linear(
            embed_dim, embed_dim, bias=bias,
            magnitude_preserving=magnitude_preserving,
        )
        self.lin_key = Linear(
            kdim or embed_dim, embed_dim, bias=bias,
            magnitude_preserving=magnitude_preserving,
        )
        self.lin_value = Linear(
            vdim or embed_dim, embed_dim, bias=bias,
            magnitude_preserving=magnitude_preserving,
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=bias,
            magnitude_preserving=magnitude_preserving,
        )
        self.dropout = nn.Dropout(dropout)
        self.l2_normalization = l2_normalization
        self.flash_attention = flash_attention

        self.add_bias_kv = add_bias_kv
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        if rope:
            self.rope = RoPE(embed_dim//num_heads, batch_first=batch_first)
        else:
            self.rope = None
        if rms_norm:
            self.q_norm = NORM_MAP["rms"](embed_dim//num_heads)
            self.k_norm = NORM_MAP["rms"](embed_dim//num_heads)
        else:
            self.q_norm = self.k_norm = None
        self.linear_attention_bias = linear_attention_bias

    def reset_parameters(self, seed=None):
        generator = torch.Generator(
            device=self.out_proj.weight.device
        )
        if seed is not None:
            generator.manual_seed(seed)
        for param in self.parameters():
            if param.ndim == 1:
                # additive bias
                nn.init.zeros_(param)
            elif param.ndim == 2:
                # linear weights
                nn.init.xavier_uniform_(param, generator=generator)
            elif (
                param.ndim == 3 and param.shape[:2] == (1, 1)
                and param.shape[2] == self.embed_dim
            ):
                # bias_kv
                nn.init.xavier_normal_(param, generator=generator)
            elif (
                param.ndim in (3, 4)
                and param.shape[-3:] == (
                    self.num_heads, 1, self.embed_dim // self.num_heads
                )
            ):
                # relative positional bias
                nn.init.zeros_(param)
            else:
                raise ValueError(f"Unexpected parameter shape: {param.shape}")

    def project(
        self,
        tensor: Tensor,
        projection: pt.Module,
        input_signal: str,
        norm_fn: tp.Optional[pt.Module] = None,
    ):
        B, T, _ = tensor.shape
        if input_signal in ('key', 'value'):
            bias = self.bias_k if input_signal == 'key' else self.bias_v
            if bias is not None:
                T = T + 1
                tensor = torch.cat(
                    [tensor, bias.repeat(tensor.shape[0], 1, 1)], dim=1
                )
        tensor = projection(tensor).view(
            B, T, self.num_heads, self.embed_dim // self.num_heads
        ).transpose(1, 2)
        if input_signal in ('query', 'key'):
            if norm_fn is not None:
                tensor = norm_fn(tensor)
            if self.rope is not None:
                tensor = self.rope(tensor)
        return tensor

    def prepare_padding_mask(
        self,
        query: Tensor,
        key: Tensor,
        linear_attention_bias: bool,
        key_padding_mask: tp.Optional[Tensor] = None,
    ):
        if self.bias_k is not None and key_padding_mask is not None:
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)\
                .contiguous()  # (B, 1, 1, S)
            if self.flash_attention:
                key = key * key_padding_mask.transpose(-1, -2).exp()
                key_padding_mask = None
        if linear_attention_bias:
            if key_padding_mask is None:
                key_padding_mask = torch.zeros_like(
                    key[:, :1, :, :1].moveaxis(-2, -1)
                )
            key_indexer = torch.arange(key.shape[2], device=key.device)
            query_indexer = torch.arange(query.shape[2], device=key.device)
            linear_mask = -(key_indexer[None, :] - query_indexer[:, None]).abs()
            linear_mask = linear_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, l, s)
            slopes = (
                torch
                .linspace(8/self.num_heads, 8, self.num_heads)
                .unsqueeze(0)
                .to(linear_mask.device)
            )
            linear_mask = linear_mask * slopes[..., None, None]
            key_padding_mask = key_padding_mask + linear_mask
        return key, key_padding_mask

    def prepare_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: tp.Optional[Tensor] = None,
    ):
        q = self.project(
            query, self.lin_query, input_signal='query', norm_fn=self.q_norm
        )
        k = self.project(
            key, self.lin_key, input_signal='key', norm_fn=self.k_norm
        )
        v = self.project(value, self.lin_value, input_signal='value')
        key, key_padding_mask = self.prepare_padding_mask(
            q, k, self.linear_attention_bias, key_padding_mask
        )
        return q, k, v, key_padding_mask

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        key_padding_mask: tp.Optional[Tensor] = None,
        is_causal: bool = False,
    ):
        B, Tq, _ = query.shape
        q, k, v, key_padding_mask = self.prepare_attention(
            query, key, value, key_padding_mask
        )
        if self.l2_normalization:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        x, attn_weights = self.attention_fn(
            q, k, v,
            attn_mask=key_padding_mask,
            is_causal=is_causal,
        )
        x = x.transpose(1, 2).float().contiguous().view(B, Tq, self.embed_dim)
        return self.dropout(self.out_proj(x)), attn_weights


class TransformerNormBlock(pt.Module):
    def __init__(
        self,
        norm: TActivationFn,
        d_model: int,
        cond_dim: tp.Optional[int] = None,
    ):
        super().__init__()
        kwargs = {}
        if cond_dim is not None:
            kwargs['cond_dim'] = cond_dim
        self.norm = NORM_MAP[norm](d_model, **kwargs)

    def forward(self, inputs: Tensor, cond: tp.Optional[Tensor] = None):
        if cond is None:
            return self.norm(inputs)
        norm_outputs = self.norm(inputs, cond=cond)
        h = norm_outputs.x
        return h, norm_outputs.layer_scale


class EncoderLayer(pt.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.,
        activation: TActivationFn = 'relu',
        norm: tp.Optional[TActivationFn] = 'layer',
        output_norm: tp.Optional[TActivationFn] = None,
        magnitude_preserving: bool = False,
        bias: bool = True,
        pre_activation: bool = False,
        cond_dim: tp.Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.activation = activation
        self.magnitude_presering = magnitude_preserving

        activation_fn = ACTIVATION_FN_MAP[activation]
        try:
            # Instantiate activation function
            activation_fn = activation_fn()
        except TypeError:
            # Already instantiated
            pass
        self.mlp = self.build_mlp(
            d_model,
            dim_feedforward,
            activation_fn,
            dropout=dropout,
            bias=bias,
            magnitude_preserving=magnitude_preserving,
            pre_activation=pre_activation,
        )
        if norm is None:
            self.sa_norm = None
            self.mlp_norm = None
        else:
            self.sa_norm = TransformerNormBlock(norm, d_model, cond_dim)
            self.mlp_norm = TransformerNormBlock(norm, d_model, cond_dim)

        if output_norm is not None:
            self.output_norm = TransformerNormBlock(output_norm, d_model)
        else:
            self.output_norm = None

    def build_mlp(
        self,
        d_model: int,
        dim_feedforward: int,
        activation_fn: TActivationFn,
        dropout: float = 0.,
        bias: bool = True,
        magnitude_preserving: bool = False,
        pre_activation: bool = False
    ):
        if not pre_activation:
            mlp = nn.Sequential(
                Linear(
                    d_model, dim_feedforward,
                    magnitude_preserving=magnitude_preserving,
                    bias=bias,
                ),
                activation_fn,
                Linear(
                    dim_feedforward, d_model,
                    magnitude_preserving=magnitude_preserving,
                    bias=bias,
                ),
                nn.Dropout(dropout),
            )
        else:
            mlp = nn.Sequential(
                activation_fn,
                Linear(
                    d_model, dim_feedforward,
                    magnitude_preserving=magnitude_preserving,
                    bias=bias,
                ),
                activation_fn,
                Linear(
                    dim_feedforward, d_model,
                    magnitude_preserving=magnitude_preserving,
                    bias=bias,
                ),
                nn.Dropout(dropout),
            )
        return mlp

    def reset_parameters(self, seed=None):
        generator = torch.Generator(
            device=self.mlp[-2].weight.device
        )
        if seed is not None:
            generator.manual_seed(seed)
        if isinstance(self.activation, str):
            gain = torch.nn.init.calculate_gain(self.activation)
        else:
            gain = 1.0
        for param in self.mlp.parameters():
            if param.ndim > 1:
                if not self.magnitude_presering:
                    nn.init.xavier_uniform_(
                        param, gain=gain, generator=generator
                    )
                else:
                    nn.init.normal_(param, generator=generator)
            else:
                nn.init.zeros_(param)


class TransformerEncoderLayer(EncoderLayer):
    def __init__(
        self,
        d_model: int,
        attention_fn: tp.Callable,
        activation: TActivationFn,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.,
        norm: tp.Optional[TActivationFn] = 'layer',
        output_norm: tp.Optional[TActivationFn] = None,
        bidirectional: bool = True,
        magnitude_preserving: bool = False,
        bias: bool = True,
        pre_activation: bool = False,
        cond_dim: tp.Optional[int] = None,
        normalize_skip_connections: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__(
            d_model, dim_feedforward, dropout, activation, norm,
            output_norm=output_norm,
            bias=bias,
            magnitude_preserving=magnitude_preserving,
            pre_activation=pre_activation,
            cond_dim=cond_dim,
        )
        self.nhead = nhead
        self.attention_fn = attention_fn
        self.normalize_skip_connections = normalize_skip_connections
        self.pre_norm = pre_norm

        self.bidirectional = bidirectional

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["attention_fn"] = {
            'factory': MultiheadAttention,
            'attention_fn': 'scaled_dot_product',
            'embed_dim': config['d_model'],
            'num_heads': config['nhead'],
            'dropout': config['dropout'],
            'magnitude_preserving': config['magnitude_preserving'],
            'bias': config["bias"],
        }
        config['activation'] = {
            'factory': GELU,
            'magnitude_preserving': config['magnitude_preserving'],
        }

    def _call_attention(
        self, q, padding_mask, *, k=None, v=None, attention_fn=None
    ):
        if k is None:
            k = q
        if v is None:
            v = k
        if attention_fn is None:
            attention_fn = self.attention_fn
        h, _ = attention_fn(
            q, k, v,
            key_padding_mask=padding_mask if self.bidirectional else None,
            is_causal=not self.bidirectional,
        )
        return h

    def _normalize_skip_connection(
        self, inputs, outputs, layer_scale: tp.Optional[Tensor] = None
    ):
        if not self.normalize_skip_connections:
            if layer_scale is not None:
                outputs = outputs * F.softplus(layer_scale)
            return inputs + outputs
        # Rescale norm of inputs+outputs to be the same as input
        inputs_norm = inputs.norm(p=2, dim=-1, keepdim=True)
        outputs_norm = outputs.norm(p=2, dim=-1, keepdim=True)
        scale = torch.div(
            inputs_norm,
            torch.sqrt(
                inputs_norm**2 + outputs_norm**2
                + 2*(inputs*outputs).sum(dim=-1, keepdim=True)
            )
        )
        return scale*(inputs+outputs)

    def reset_parameters(self, seed=None):
        try:
            self.attention_fn.reset_parameters(seed=seed)
        except AttributeError:
            pass
        super().reset_parameters(seed=seed)

    def forward(
        self, x: Tensor,
        padding_mask: tp.Optional[Tensor] = None,
        latent_cond: tp.Optional[Tensor] = None,
    ):
        """
        Args:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            padding_mask: Tensor, shape ``[batch_size, seq_len]``
        """
        # Self-attention block
        if self.pre_norm and self.sa_norm is not None:
            h, layer_scale = self.sa_norm(x, cond=latent_cond)
        else:
            h = x
            layer_scale = 1
        h = self._call_attention(h, padding_mask)
        h = self._normalize_skip_connection(x, h, layer_scale)
        if not self.pre_norm and self.sa_norm is not None:
            h, _ = self.sa_norm(h, cond=latent_cond)

        # Feed-forward block
        if self.pre_norm and self.mlp_norm is not None:
            o, layer_scale = self.mlp_norm(h, cond=latent_cond)
        else:
            o = h
            layer_scale = 1
        o = self.mlp(o)
        o = self._normalize_skip_connection(h, o, layer_scale)
        if not self.pre_norm and self.mlp_norm is not None:
            o, _ = self.mlp_norm(o, cond=latent_cond)
        elif self.output_norm is not None:
            o = self.output_norm(o)
        return o


class TransformerDecoderLayer(TransformerEncoderLayer):
    def __init__(
        self,
        cross_attention_fn: tp.Callable,
        *arg,
        **kwargs,
    ):
        super().__init__(*arg, **kwargs)
        self.cross_attention_fn = cross_attention_fn

        d_model = kwargs["d_model"]
        norm = kwargs.get("norm", "layer")
        cond_dims = kwargs.get("cond_dims", None)
        if norm is None:
            self.ca_norm = nn.Identity()
        else:
            if cond_dims is None:
                self.ca_norm = NORM_MAP[norm](d_model)
            else:
                self.ca_norm = nn.ModuleList([
                    NORM_MAP[norm](d_model, cond_dim=cond_dim)
                    for cond_dim in cond_dims
                ])

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config["dropout"] = 0.
        config["magnitude_preserving"] = False
        config["bias"] = True
        TransformerEncoderLayer.finalize_dogmatic_config(config)
        config["cross_attention_fn"] = {
            'factory': MultiheadAttention,
            'attention_fn': 'scaled_dot_product',
            'embed_dim': config['d_model'],
            'num_heads': config['nhead'],
            'dropout': config['dropout'],
            'magnitude_preserving': config['magnitude_preserving'],
            'bias': config["bias"],
        }

    def reset_parameters(self, seed=None):
        try:
            self.cross_attention_fn.reset_parameters(seed=seed)
        except AttributeError:
            pass
        super().reset_parameters(seed=seed)

    def forward(
        self, x: Tensor, cross_condtioning: Tensor,
        padding_mask: tp.Optional[Tensor] = None,
        key_padding_mask: tp.Optional[Tensor] = None,
        latent_cond: tp.Optional[tp.Sequence[Tensor]] = None,
        cond_layers: tp.Optional[tp.Sequence[int]] = None,
    ):
        """
        Args:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            padding_mask: Tensor, shape ``[batch_size, seq_len]``
        """
        # Self-attention block
        if self.pre_norm and self.sa_norm is not None:
            h, layer_scale = self.sa_norm(x, cond=latent_cond)
        else:
            h = x
            layer_scale = 1
        h = self._call_attention(h, padding_mask)
        h = self._normalize_skip_connection(x, h, layer_scale)
        if not self.pre_norm and self.sa_norm is not None:
            h, _ = self.sa_norm(h, cond=latent_cond)

        # Cross-attention block
        if self.pre_norm and self.ca_norm is not None:
            h2, layer_scale = self.ca_norm(h, cond=latent_cond)
        else:
            h2 = h
            layer_scale = 1
        h2 = self._call_attention(
            h2, key_padding_mask, k=cross_condtioning, v=cross_condtioning,
            attention_fn=self.cross_attention_fn,
        )
        h2 = self._normalize_skip_connection(h, h2, layer_scale)
        if not self.pre_norm and self.ca_norm is not None:
            h, _ = self.ca_norm(h, cond=latent_cond)

        # Feed-forward block
        if self.pre_norm and self.mlp_norm is not None:
            o, layer_scale = self.mlp_norm(h2, cond=latent_cond)
        else:
            o = h2
            layer_scale = 1
        o = self.mlp(o)
        o = self._normalize_skip_connection(h2, o, layer_scale)
        if not self.pre_norm and self.mlp_norm is not None:
            o, _ = self.mlp_norm(o, cond=latent_cond)
        elif self.output_norm is not None:
            o = self.output_norm(o)

        return o


class TransformerEncoder(pt.Module):
    def __init__(
        self,
        input_size: int,
        encoder_layer: TransformerEncoderLayer,
        feature_extractor: tp.Optional[pt.Module] = None,
        num_layers: int = 6,
        input_norm: TActivationFn = None,
        output_norm: TActivationFn = None,
        positional_encoding: tp.Optional[pt.Module] = None,
        dropout: float = 0.,
        mlp_head: tp.Optional[nn.Module] = None,
        use_activation_checkpointing: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.mlp_head = mlp_head

        if feature_extractor is None and input_size != encoder_layer.d_model:
            self.feature_extractor = Linear(
                input_size, encoder_layer.d_model
            )
        else:
            self.feature_extractor = feature_extractor

        if positional_encoding is None:
            self.positional_encoding = nn.Identity()
        else:
            self.positional_encoding = positional_encoding

        if input_norm is None:
            self.input_norm = NormOutputs
        else:
            self.input_norm = NORM_MAP[input_norm](encoder_layer.d_model)

        if output_norm is None:
            self.output_norm = NormOutputs
        else:
            self.output_norm = NORM_MAP[output_norm](
                encoder_layer.d_model
            )

        self.dropout = nn.Dropout(dropout)
        self.use_activation_checkpointing = use_activation_checkpointing

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = copy.deepcopy(encoder_layer)
            layer.reset_parameters()
            self.layers.append(layer)
        if zero_init:
            self.zero_init_()

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['encoder_layer'] = {'factory': TransformerEncoderLayer}
        config['positional_encoding'] = {
            'd_model': config['encoder_layer']['d_model'],
        }

    def __getitem__(self, item):
        return self.layers[item]

    def zero_init_(self):
        if self.mlp_head is not None:
            # Initialize linear with zeros, see ViT paper
            for param in self.mlp_head.parameters():
                param.detach().zero_()
            return True
        return False

    def reset_parameters(self, seed=None):
        try:
            self.positional_encoding.reset_parameters(seed=seed)
        except AttributeError:
            pass
        for layer in self.layers:
            layer.reset_parameters(seed=seed)

    def _forward(
        self, x: Tensor, sequence_lengths: TSeqLen = None, *,
        latent_cond: tp.Optional[Tensor] = None,
    ):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        h = self.positional_encoding(x)
        h = self.input_norm(h).x
        h = self.dropout(h)
        padding_mask = compute_mask(
            h[..., 0], sequence_lengths, batch_axis=0, sequence_axis=1
            ).float().log()
        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                padding_mask=padding_mask,
                latent_cond=latent_cond,
            )
        o = self.output_norm(h).x

        if self.mlp_head is None:
            return o, sequence_lengths
        o = self.mlp_head(o)
        return o, sequence_lengths

    def forward(
        self, x: Tensor, sequence_lengths: TSeqLen = None, *,
        cond: tp.Optional[Tensor] = None,
    ):
        """
        Args:
            x: Tensor, shape ``[batch_size, seq_len, input_size]`` or
                ``[batch_size, input_size, seq_len]``
            sequence_lengths: Tensor, shape ``[batch_size]``
        """
        if self.use_activation_checkpointing:
            return activation_checkpoint(
                self._forward, x, sequence_lengths,
                latent_cond=cond,
                use_reentrant=False,
            )
        return self._forward(
            x, sequence_lengths, latent_cond=cond
        )


class TransformerDecoder(pt.Module):
    def __init__(
        self,
        input_size,
        decoder_layer: TransformerDecoderLayer,
        input_normalization: TActivationFn = None,
        num_layers: int = 6,
        input_norm: TActivationFn = None,
        output_norm: TActivationFn = None,
        positional_encoding: tp.Optional[pt.Module] = None,
        dropout: float = 0.,
        mlp_head: tp.Optional[nn.Module] = None,
        use_activation_checkpointing: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_normalization = input_normalization
        self.mlp_head = mlp_head

        self.linear = Linear(input_size, decoder_layer.d_model)

        if positional_encoding is None:
            self.positional_encoding = nn.Identity()
        else:
            self.positional_encoding = positional_encoding

        if input_norm is None:
            self.input_norm = NormOutputs
        else:
            self.input_norm = NORM_MAP[input_norm](decoder_layer.d_model)

        if output_norm is None:
            self.output_norm = NormOutputs
        else:
            self.output_norm = NORM_MAP[output_norm](
                decoder_layer.d_model
            )

        self.dropout = nn.Dropout(dropout)
        self.use_activation_checkpointing = use_activation_checkpointing

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = copy.deepcopy(decoder_layer)
            layer.reset_parameters()
            self.layers.append(layer)
        if zero_init:
            self.zero_init_()

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['decoder_layer'] = {'factory': TransformerDecoderLayer}
        config['positional_encoding'] = {
            'd_model': config['decoder_layer']['d_model'],
        }

    def zero_init_(self):
        if self.mlp_head is not None:
            # Initialize linear with zeros, see ViT paper
            for param in self.mlp_head.parameters():
                param.detach().zero_()
            return True
        return False

    def reset_parameters(self, seed=None):
        try:
            self.positional_encoding.reset_parameters(seed=seed)
        except AttributeError:
            pass
        for layer in self.layers:
            layer.reset_parameters(seed=seed)

    def _forward(
        self, x: Tensor, cross_conditioning: Tensor,
        sequence_lengths: TSeqLen = None,
        cross_sequence_lengths: TSeqLen = None,
        *,
        latent_cond: tp.Optional[tp.Sequence[Tensor]] = None,
        cond_layers: tp.Optional[tp.Sequence[int]] = None,
    ):
        if self.input_normalization is not None:
            x = self.input_normalization(x, sequence_lengths)
            cross_conditioning = self.input_normalization(
                cross_conditioning, cross_sequence_lengths
            )

        x = self.linear(x)
        cross_conditioning = self.linear(cross_conditioning)

        h = self.positional_encoding(x)
        h_cross = self.positional_encoding(cross_conditioning)
        h = self.dropout(self.input_norm(h).x)
        h_cross = self.dropout(self.input_norm(h_cross).x)
        padding_mask = compute_mask(
            h[..., 0], sequence_lengths, batch_axis=0, sequence_axis=-1
        ).float().log()
        key_padding_mask = compute_mask(
            h_cross[..., 0], cross_sequence_lengths,
            batch_axis=0, sequence_axis=-1
        ).float().log()
        for _, layer in enumerate(self.layers):
            h = layer(
                h, h_cross,
                padding_mask=padding_mask,
                key_padding_mask=key_padding_mask,
                latent_cond=latent_cond,
                cond_layers=cond_layers,
            )
        o = self.output_norm(h).x

        if self.mlp_head is None:
            return o, sequence_lengths
        o = self.mlp_head(o)
        return o, sequence_lengths

    def forward(
        self, x: Tensor, cross_conditioning: Tensor,
        sequence_lengths: TSeqLen = None,
        cross_sequence_lengths: TSeqLen = None,
        *,
        latent_cond: tp.Optional[tp.Sequence[Tensor]] = None,
    ):
        """
        Args:
            x: Tensor, shape ``[batch_size, seq_len, input_size]`` or
                ``[batch_size, input_size, seq_len]``
            sequence_lengths: Tensor, shape ``[batch_size]``
        """
        if self.use_activation_checkpointing:
            return activation_checkpoint(
                self._forward, x, cross_conditioning,
                sequence_lengths=sequence_lengths,
                cross_sequence_lengths=cross_sequence_lengths,
                latent_cond=latent_cond,
                use_reentrant=False,
            )
        return self._forward(
            x, cross_conditioning,
            sequence_lengths=sequence_lengths,
            cross_sequence_lengths=cross_sequence_lengths,
            latent_cond=latent_cond
        )
