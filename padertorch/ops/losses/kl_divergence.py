from torch.distributions import Normal, MultivariateNormal
from torch.distributions import kl_divergence as kld


__all__ = [
    'gaussian_kl_divergence',
]


def _batch_diag(bmat):
    """
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.reshape(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


def gaussian_kl_divergence(q, p):
    """
    Args:
        q: Normal posterior distributions (B1, ..., BN, D)
        p: (Multivariate) Normal prior distributions (K1, ..., KN, D)

    Returns: kl between all posteriors in batch and all components
        (B1, ..., BN, K1, ..., KN)

    """
    assert isinstance(q, Normal), type(q)
    batch_shape = q.loc.shape[:-1]
    D = q.loc.shape[-1]
    component_shape = p.loc.shape[:-1]
    assert p.loc.shape[-1] == D, (p.loc.shape[-1], D)

    p_loc = p.loc.contiguous().view(-1, D)
    if isinstance(p, MultivariateNormal):
        p_scale_tril = p.scale_tril.contiguous().view(-1, D, D)
        q_loc = q.loc.contiguous().view(-1, D)
        q_scale = q.scale.contiguous().view(-1, D)

        term1 = (
            _batch_diag(p_scale_tril).log().sum(-1)[:, None]
            - q_scale.log().sum(-1)
        )
        L = p_scale_tril.inverse()
        term2 = (L.pow(2).sum(-2)[:, None, :] * q_scale.pow(2)).sum(-1)
        term3 = (
                (p_loc[:, None, :] - q_loc) @ L.transpose(1, 2)
        ).pow(2.0).sum(-1)
        kl = (term1 + 0.5 * (term2 + term3 - D)).transpose(0, 1)
    elif isinstance(p, Normal):
        p_scale = p.scale.contiguous().view(-1, D)
        q_loc = q.loc.contiguous().view(-1, 1, D)
        q_scale = q.scale.contiguous().view(-1, 1, D)

        kl = kld(
            Normal(loc=q_loc, scale=q_scale), Normal(loc=p_loc, scale=p_scale)
        ).sum(-1)
    else:
        raise ValueError

    return kl.view(*batch_shape, *component_shape)
