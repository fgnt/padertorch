import torch
import torch.nn.functional
import itertools
import padertorch as pt


__all__ = [
    'deep_clustering_loss',
    'pit_loss',
]


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
    sources = estimate.size()[axis]
    assert sources < 30, f'Are you sure? sources={sources}'
    if loss_fn in [torch.nn.functional.cross_entropy]:
        assert axis % estimate.ndimension() == 1, axis
        estimate_shape = list(estimate.shape)
        del estimate_shape[axis]
        assert estimate_shape == list(target.shape), (
            f'{estimate.shape} (N, K, ...) does not match {target.shape} (N, ...)'
        )
    else:
        assert estimate.size() == target.size(), (
            f'{estimate.size()} != {target.size()}'
        )

    candidates = []
    indexer = [slice(None),] * estimate.ndim
    permutations = list(itertools.permutations(range(sources)))
    for permutation in permutations:
        indexer[axis] = permutation
        candidates.append(loss_fn(
            estimate[tuple(indexer)],
            target
        ))
    min_loss, idx = torch.min(torch.stack(candidates), dim=0)

    if return_permutation:
        return min_loss, permutations[int(idx)]
    else:
        return min_loss


def pair_wise_loss(
        estimate: torch.Tensor,
        target: torch.Tensor,
        axis: int,
        loss_fn=torch.nn.functional.mse_loss,
):
    """
    The function pit_loss can be more efficient implemented, when the
    loss allows to calculate a pair wise loss. The pair wise losses are
    then written to a matrix (each estimated signal vs each target signal).
    On the matrix with the pair wise losses the function
    `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) can find the
    best permutation.

    The runtime of `scipy.optimize.linear_sum_assignment` does not matter,
    so the runtime complexity decreases from faculty complexity to quadratic
    with respect to the number of speakers.
    For 2 speakers this is slightly slower, but for large numbers of speakers
    (e.g. 7) thiis function is significant faster.

    Limitation:
        Not every loss function can be factorized in pair_wise losses.
        And sometimes it is difficult to implement the pair wise loss
        (See the special implementation in this function for cross_entropy).
        One good point is, that most used loss functions can be factorized.

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

    Examples:
        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, K, F)
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, 1))
        tensor(1.)

        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, F, dtype=torch.int64)
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, 1, loss_fn=torch.nn.functional.cross_entropy), reduction='sum')
        tensor(0.6931)
        >>> pit_loss(estimate, target, 1, loss_fn=torch.nn.functional.cross_entropy)
        tensor(0.6931)

        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(K, F, T), torch.zeros(K, F, T)
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, 0))
        tensor(1.)

        >>> T, K, F = 4, 2, 5
        >>> estimate = torch.stack([torch.ones(F, T), torch.zeros(F, T)])
        >>> target = estimate[(1, 0), :, :]
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, axis=0), return_permutation=True)
        (tensor(0.), array([1, 0]))

        >>> K = 5
        >>> estimate, target = torch.ones(K), torch.zeros(K)
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, axis=0))
        tensor(1.)

        >>> A, B, K, C, F = 4, 5, 3, 100, 128
        >>> estimate, target = torch.ones(A, B, K, C, F), torch.zeros(A, B, K, C, F)
        >>> pit_loss_from_pair_wise(pair_wise_loss(estimate, target, axis=-3))
        tensor(1.)
    """
    sources = estimate.size()[axis]
    assert sources < 30, f'Are you sure? sources={sources}'
    if loss_fn in [torch.nn.functional.cross_entropy]:
        import einops

        assert axis % estimate.ndimension() == 1, axis
        estimate_shape = list(estimate.shape)
        del estimate_shape[1]
        assert estimate_shape == list(target.shape), (
            f'{estimate.shape} (N, K, ...) does not match {target.shape} (N, ...)'
        )

        assert loss_fn == torch.nn.functional.cross_entropy, loss_fn
        assert axis == 1, axis

        # torch.einsum does not support reduction of ...
        return einops.reduce(torch.einsum(
            'nc...,n...k->n...ck',
            -torch.nn.LogSoftmax(dim=1)(estimate),
            torch.nn.functional.one_hot(target, num_classes=sources).to(estimate.dtype)
        ), 'n ... c k -> c k', reduction='mean')

    else:
        assert estimate.size() == target.size(), (
            f'{estimate.size()} != {target.size()}'
        )

        assert estimate.shape == target.shape, (estimate.shape, target.shape)

        indexer_e = [slice(None), ] * estimate.ndim
        indexer_t = [slice(None), ] * target.ndim
        pair_wise_loss_matrix = []
        for i in range(sources):
            indexer_e[axis] = i
            for j in range(0, sources):
                indexer_t[axis] = j
                pair_wise_loss_matrix.append(loss_fn(
                    estimate[tuple(indexer_e)],
                    target[tuple(indexer_t)],
                ))
        return torch.stack(pair_wise_loss_matrix, 0).reshape(sources, sources)


def pit_loss_from_pair_wise(
        pair_wise_loss_matrix,
        *,
        reduction='mean',
        algorithm: ['optimal', 'greedy'] = 'optimal',
        return_permutation=False,
):
    """
    Calculates the PIT loss given a pair_wise_loss matrix.
    
    Args:
        pair_wise_loss_matrix: shape: (K, K)
        reduction: 'mean' or 'sum'
        algorithm:
        return_permutation:

    Returns:
        
    >>> import numpy as np
    >>> score_matrix = np.array([[11., 10, 0],[4, 5, 10],[6, 0, 5]])
    >>> score_matrix
    array([[11., 10.,  0.],
           [ 4.,  5., 10.],
           [ 6.,  0.,  5.]])
    >>> pair_wise_loss_matrix = torch.tensor(-score_matrix)
    >>> pit_loss_from_pair_wise(pair_wise_loss_matrix, reduction='sum', algorithm='optimal')
    tensor(-26., dtype=torch.float64)
    >>> pit_loss_from_pair_wise(pair_wise_loss_matrix, reduction='sum', algorithm='greedy')
    tensor(-21., dtype=torch.float64)

    """
    import scipy.optimize
    from padertorch.utils import to_numpy

    assert len(pair_wise_loss_matrix.shape) == 2, pair_wise_loss_matrix.shape
    assert pair_wise_loss_matrix.shape[-2] == pair_wise_loss_matrix.shape[-1], pair_wise_loss_matrix.shape
    sources = pair_wise_loss_matrix.shape[-1]

    pair_wise_loss_np = to_numpy(pair_wise_loss_matrix)

    if algorithm == 'optimal':
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            pair_wise_loss_np)
    elif algorithm == 'greedy':
        from pb_bss.permutation_alignment import _mapping_from_score_matrix
        col_ind = _mapping_from_score_matrix(-pair_wise_loss_np,
                                             algorithm='greedy')
        row_ind = range(sources)
    else:
        raise ValueError(algorithm)

    if reduction == 'mean':
        min_loss = pair_wise_loss_matrix[row_ind, col_ind].mean()
    elif reduction == 'sum':
        min_loss = pair_wise_loss_matrix[row_ind, col_ind].sum()
    else:
        raise ValueError(reduction)

    if return_permutation:
        return min_loss, col_ind
    else:
        return min_loss
