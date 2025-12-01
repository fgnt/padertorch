import typing as tp

import numpy as np
import torch
from torch import Tensor


def normalize(w: Tensor, eps: float = 1e-4, chunks: tp.Optional[int] = None):
    """Weight normalization

    From: Karras et al., Analyzing and Improving the Training Dynamics of
        Diffusion Models, 2024, Algorithm 1

    Args:
        w (Tensor): _description_

    Returns:
        _type_: _description_
    """
    if chunks is None:
        norm = torch.norm(w, dim=-1, keepdim=True)
        alpha = np.sqrt(norm.numel() / w.numel())
        return w / torch.add(eps, norm, alpha=alpha)
    w_ = torch.chunk(w, chunks, dim=-1)
    w = torch.stack(w_, dim=-1)
    norm = torch.norm(w, dim=-2, keepdim=True)
    alpha = np.sqrt(norm.numel() / w.numel())
    w = w / torch.add(eps, norm, alpha=alpha)
    w = torch.chunk(w, chunks, dim=-1)
    return torch.cat(w, dim=-2).squeeze(-1)
