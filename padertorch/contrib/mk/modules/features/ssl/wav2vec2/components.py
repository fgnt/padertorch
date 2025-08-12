import math
from typing import Optional, Tuple

import padertorch as pt
import torch
from torch import Tensor, nn
from torchaudio.models.wav2vec2.components import SelfAttention


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

        pe = positional_embedding(d_model, max_len)
        if batch_first:
            pe = pe.transpose(0, 1)
        cos_pos = torch.repeat_interleave(
            pe[..., 1::2], 2, dim=-1
        )
        sin_pos = torch.repeat_interleave(
            pe[..., ::2], 2, dim=-1
        )
        self.register_buffer('cos_pos', cos_pos)
        self.register_buffer('sin_pos', sin_pos)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` if
                batch_first is False else ``[batch_size, seq_len, embedding_dim]``
        """
        x_quad = interleave(-x[..., 1::2], x[..., ::2], dim=-1)
        if self.batch_first:
            cos_pos = self.cos_pos[:, :x.size(1)]
            sin_pos = self.sin_pos[:, :x.size(1)]
        else:
            cos_pos = self.cos_pos[:x.size(0)]
            sin_pos = self.sin_pos[:x.size(0)]
        if x.ndim == 4:
            # Apply per attention head
            cos_pos = cos_pos.unsqueeze(-2)
            sin_pos = sin_pos.unsqueeze(-2)
        x = x * cos_pos + x_quad * sin_pos
        return x


class ALiBi(SelfAttention):
    """
    ALiBi (Attention with Linear Biases) is a method for improving the performance of self-attention
    mechanisms in transformer models. It introduces a linear bias to the attention scores based on
    the distance between tokens, allowing the model to better capture long-range dependencies.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)

    def get_attention_mask(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
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
        if attention_mask is None:
            return linear_mask
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        return attention_mask + linear_mask

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                "The expected input shape is "
                f"(batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attention_mask = self.get_attention_mask(q, k, attention_mask)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        return output, None


class RoPEAttention(SelfAttention):
    """
    RoPE (Rotary Positional Embedding) is a method for improving the performance of self-attention
    mechanisms in transformer models. It introduces a rotary positional embedding to the attention
    scores, allowing the model to better capture long-range dependencies.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.rope = RoPE(embed_dim//num_heads, batch_first=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                "The expected input shape is "
                f"(batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        q = self.rope(q)
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.rope(k)
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        return output, None


class KerpleLogAttention(SelfAttention):
    # https://github.com/chijames/KERPLE/blob/25be3d712e96a3353b374f1f8e36b97d86f97ffa/megatron/mpu/layers.py#L192

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.eps = 1e-2

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return nn.Parameter(torch.ones(
                    self.num_heads,
                    dtype=torch.float32,
                )[:,None,None]*scale)
            if init_method == 'uniform':
                return nn.Parameter(torch.rand(
                    (self.num_heads,),
                    dtype=torch.float32,
                )[:,None,None]*scale)
            raise ValueError(f"Unknown init method {init_method}")

        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

    def get_attention_mask(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [b, np, sq, sk]
        # attn_mat = query @ key.transpose(-2, -1)
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]

        diff = torch.tril(
            torch.arange(seq_len_k, device=query.device).view(seq_len_k, 1)\
                .repeat(1, seq_len_k)
            + torch.arange(0, -seq_len_k, -1, device=query.device)
        )
        diff = diff.to(query.dtype)
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        if attention_mask is None:
            return bias
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        return attention_mask + bias

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                "The expected input shape is "
                f"(batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attention_mask = self.get_attention_mask(q, k, attention_mask)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        return output, None


class LongformerAttention(SelfAttention):
    """
    Longformer is a transformer model that uses a combination of local and
    global attention mechanisms to efficiently process long sequences of data.
    It introduces a sparse attention pattern that allows the model to focus on
    relevant parts of the input while reducing computational complexity. This
    makes it suitable for tasks such as document classification, question
    answering, and language modeling, where long-range dependencies are
    important.
    """

    def __init__(
        self,
        window_size: int,
        embed_dim: int,
        num_heads: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.window_size = window_size
        self.dilation = dilation
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 is not implemented. "
                "Please use the LongformerAttention class with dilation=1."
            )

    def get_attention_mask(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]
        attn_mask = torch.ones(
            (seq_len_q, seq_len_k), dtype=query.dtype, device=query.device
        )
        attn_mask = (
            attn_mask.tril(diagonal=self.window_size//2)
            - attn_mask.tril(diagonal=-math.ceil(self.window_size/2))
        ).log()
        if attention_mask is None:
            return attn_mask
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.float()
        return attention_mask + attn_mask

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                "The expected input shape is "
                f"(batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attention_mask = self.get_attention_mask(q, k, attention_mask)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        output = self.out_proj(attn_output)
        return output, None
