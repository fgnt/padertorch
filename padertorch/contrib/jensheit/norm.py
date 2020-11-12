"""
This code is adapted version of https://github.com/funcwj/conv-tasnet
"""

import torch
import torch.nn as nn
from einops import rearrange


class TransposedLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    >>> norm = TransposedLayerNorm(256)
    >>> norm(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        """
        x: N x F x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        x = rearrange(x, 'n f t -> n t f')
        # LN
        x = super().forward(x)
        x = rearrange(x, 'n t f -> n f t')
        return x

class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization

    >>> norm = GlobalChannelLayerNorm(256)
    >>> norm(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x F x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x F
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN

    >>> norm = build_norm('cLN', 256)
    >>> norm(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])

    >>> norm = build_norm('gLN', 256)
    >>> norm(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])

    >>> norm = build_norm('BN', 256)
    >>> norm(torch.rand(5, 256, 343)).shape
    torch.Size([5, 256, 343])
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return TransposedLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)