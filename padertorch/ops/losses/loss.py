from functools import partial
import torch
import torch.nn.functional
from torch.distributions import Normal, MultivariateNormal
from torch.distributions import kl_divergence as kld
from torch.nn.utils.rnn import PackedSequence
import itertools
import padertorch as pt


__all__ = [
    'softmax_cross_entropy',
    'deep_clustering_loss',
    'pit_loss',
    'pit_mse_loss',
    'kl_divergence',
]


IGNORE_INDEX = -1


def softmax_cross_entropy(x, t):
    """Allow inputs to be of type `PackedSequence`.

    In my understanding, all dimensions but the last should be treated as
    independent dimensions. Therefore, I argue for x.size() == (..., K) where
    t.size() == (...). Similarly, for sequences x.size() == (T, B, ..., K) and
    t.size() == (T, B, ...).

    Check the test case for typical usage.

    Params:
        x: `Tensor` or `PackedSequence` holding a multidimensional array whose
            elements indicate unnormalized log probabilities (logits).
        t: Same object type as `x`. Holds integers of ground truth labels.

    Returns:

    >>> x = torch.randn(100, 3)
    >>> t = torch.randint(0, 3, size=(100,), dtype=torch.long)
    >>> softmax_cross_entropy(x, t).size()
    torch.Size([])
    """
    if isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
        pass
    elif isinstance(x, PackedSequence) and isinstance(t, PackedSequence):
        # Data is already organized such that no padding is necessary.
        x, t = x.data, t.data
    else:
        raise ValueError(f'Incompatible types: {type(x)}, {type(t)}')

    assert x.size()[:-1] == t.size(), f'{x.size()}, {t.size()}'
    # remember torch.nn.CrossentropyLoss already includes softmax
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return loss_fn(pt.ops.move_axis(x, -1, 1), t)


def deep_clustering_loss(x, t):
    """Deep clustering loss as in Hershey 2016 paper.

    yields losses in the range 0.01 to 1 due to the normalization with N^2.

    Args:
        x: Shape (N, E), where it is assumed that each embedding vector
            is normalized to unit norm.
        t: Target mask with shape (N, K).

    Returns:

    """
    N = x.size()[0]
    return (
        torch.sum(pt.ops.einsum('ne,nE->eE', x, x) ** 2)
        - 2 * torch.sum(pt.ops.einsum('ne,nK->eK', x, t) ** 2)
        + torch.sum(pt.ops.einsum('nk,nK->kK', t, t) ** 2)
    ) / N ** 2


def pit_loss(
        estimate: torch.Tensor,
        target: torch.Tensor,
        axis: int,
        loss_fn=torch.nn.functional.mse_loss,
        return_permutation: bool = False
):
    """
    Permutation invariant loss function. Calls `loss_fn` on every possible
    permutation between `estimate`s and `target`s and returns the minimum
    loss among them. The tensors are permuted along `axis`.

    Does not support batch dimension. Does not support PackedSequence.

    Args:
        estimate: Padded sequence. The speaker axis is specified with `axis`,
            so the default shape is (T, K, F)
        target: Padded sequence with the same shape as `estimate` (defaults
            to (T, K, F))
        loss_fn: Loss function to apply on each permutation. It must accept two
            arguments (estimate and target) of the same shape that this function
            receives the arguments.
        axis: Speaker axis K. The permutation is applied along this axis. axis=-2
            and an input shape of (T, K, F) corresponds to the old default
            behaviour.
        return_permutation: If `True`, this function returns the permutation
            that minimizes the loss along with the minimal loss otherwise it
            only returns the loss.

    Examples:
        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, K, F)
        >>> pit_loss(estimate, target, 1)
        tensor(1.)

        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, F, dtype=torch.int64)
        >>> pit_loss(estimate, target, 1, loss_fn=torch.nn.functional.cross_entropy)
        tensor(0.6931)

        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(K, F, T), torch.zeros(K, F, T)
        >>> pit_loss(estimate, target, 0)
        tensor(1.)

        >>> T, K, F = 4, 2, 5
        >>> estimate = torch.stack([torch.ones(F, T), torch.zeros(F, T)])
        >>> target = estimate[(1, 0), :, :]
        >>> pit_loss(estimate, target, axis=0, return_permutation=True)
        (tensor(0.), (1, 0))

        >>> K = 5
        >>> estimate, target = torch.ones(K), torch.zeros(K)
        >>> pit_loss(estimate, target, axis=0)
        tensor(1.)

        >>> A, B, K, C, F = 4, 5, 3, 100, 128
        >>> estimate, target = torch.ones(A, B, K, C, F), torch.zeros(A, B, K, C, F)
        >>> pit_loss(estimate, target, axis=-3)
        tensor(1.)
    """
    axis = axis % estimate.ndimension()
    sources = estimate.size()[axis]
    assert sources < 30, f'Are you sure? sources={sources}'
    if loss_fn in [torch.nn.functional.cross_entropy]:
        estimate_shape = list(estimate.shape)
        del estimate_shape[1]
        assert estimate_shape == list(target.shape), (
            f'{estimate.shape} (N, K, ...) does not match {target.shape} (N, ...)'
        )
    else:
        assert estimate.size() == target.size(), (
            f'{estimate.size()} != {target.size()}'
        )
    candidates = []
    filler = (slice(None),) * axis
    permutations = list(itertools.permutations(range(sources)))
    for permutation in permutations:
        candidates.append(loss_fn(
            estimate[filler + (permutation, )],
            target
        ))

    min_loss, idx = torch.min(torch.stack(candidates), dim=0)

    if return_permutation:
        return min_loss, permutations[int(idx)]
    else:
        return min_loss


def _batch_diag(bmat):
    """
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.reshape(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


def kl_divergence(q, p):
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
