import torch
import numpy as np

from padertorch.base import Module
from padertorch.ops.mappings import ACTIVATION_FN_MAP


def scaled_dot_product_attention(q, k, v, mask=None):
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
    if mask is not None:
        assert torch.is_tensor(mask)
        # mask = torch.ones_like(y).cumsum(-1) <= torch.ones_like(y).cumsum(-2)
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
    def __init__(self, input_size, output_size, num_heads=1):
        assert output_size % num_heads == 0
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.lin_queue = torch.nn.Linear(input_size, output_size)
        self.lin_key = torch.nn.Linear(input_size, output_size)
        self.lin_value = torch.nn.Linear(input_size, output_size)
        self.out = torch.nn.Linear(output_size, output_size)

    def forward(self, q, k, v, mask=None):
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
        if mask is not None:
            mask = mask[:, None]
        x = scaled_dot_product_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(B, Tq, self.output_size)
        return self.out(x)


class Norm(Module):
    # ToDo: replace by general norm module
    def __init__(self, method, size):
        super().__init__()
        self.method = method

        if method is None:
            self.norm = None
        elif method == 'batch':
            self.norm = torch.nn.BatchNorm1d(size)
        elif method == 'layer':
            self.norm = torch.nn.LayerNorm(size)
        else:
            raise ValueError(f'{method} normalization not known.')

    def forward(self, y):
        if self.method == 'batch':
            y = self.norm(y.transpose(1, -1)).transpose(1, -1)
        elif self.method == 'layer':
            y = self.norm(y)
        return y


class TransformerBlock(Module):
    """
    https://arxiv.org/abs/1706.03762
    """
    def __init__(
            self, input_size, hidden_size, num_heads=1,
            activation='leaky_relu', residual=False, norm='layer'
    ):
        super().__init__()
        self.activation = ACTIVATION_FN_MAP[activation]()
        self.residual = residual
        self.multiheadattention = MultiHeadAttention(
            input_size, hidden_size, num_heads
        )
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, hidden_size)

        self.norm_hidden = Norm(norm, hidden_size)
        self.norm_output = Norm(norm, hidden_size)

    def forward(self, q, k, v, mask=None):
        h = self.multiheadattention(q, k, v, mask=mask)
        if self.residual and h.shape == q.shape:
            h = h + q
        h = self.norm_hidden(h)
        y = self.out(self.activation(self.hidden(h)))
        if self.residual and y.shape == h.shape:
            y = y + h
        y = self.norm_output(y)
        return y


class Transformer(Module):
    def __init__(
            self, input_size, hidden_size, num_layers, num_heads=1,
            activation='leaky_relu', residual=False, norm='layer'
    ):
        """
        https://arxiv.org/abs/1706.03762

        Args:
            input_size:
            hidden_size:
            num_layers:
            num_heads:
            activation:
            residual:
            norm:

        Returns:

        >>> x = torch.zeros((2, 3, 8))
        >>> attn = Transformer(8, 6, 1, 1)
        >>> attn(x).shape
        torch.Size([2, 3, 6])
        >>> attn = Transformer(8, 6, 1, 2)
        >>> attn(x).shape
        torch.Size([2, 3, 6])
        """
        super().__init__()
        stack = list()
        for _ in range(num_layers):
            stack.append(
                TransformerBlock(
                    input_size, hidden_size, num_heads, activation, residual,
                    norm
                )
            )
            input_size = hidden_size
        self.stack = torch.nn.ModuleList(stack)

    def forward(self, x, mask=None):
        for layer in self.stack:
            x = layer(x, x, x, mask=mask)
        return x
