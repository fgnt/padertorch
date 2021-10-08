import torch


def _sqnorm(x, dim=None, keepdim=False):
    if dim is None:
        assert not keepdim
        return torch.sum(x * x)
    else:
        return torch.sum(x * x, dim=dim, keepdim=keepdim)


def _mse(estimate, target, dim=None):
    error = estimate - target
    if dim is None:
        return torch.mean(error * error)
    else:
        return torch.mean(error * error, dim=dim)


def _get_scaling_factor(target, estimate):
    return torch.unsqueeze(torch.einsum(
        '...t,...t->...', estimate, target
    ), -1) / _sqnorm(target, dim=-1, keepdim=True)


def _reduce(array, reduction):
    if reduction is None or reduction == 'none':
        return array
    if reduction == 'sum':
        return torch.sum(array)
    elif reduction == 'mean':
        return torch.mean(array)
    else:
        raise ValueError(
            f'Unknown reduction: {reduction}. Choose from "sum", "mean".')


def _get_threshold(soft_sdr_max):
    """Computes the threshold tau for the thresholded SDR"""
    if soft_sdr_max is None:
        return
    assert 1 < soft_sdr_max < 50, f'Uncommon value for soft_sdr_max: {soft_sdr_max}'
    return 10 ** (-soft_sdr_max / 10)


def mse_loss(estimate: torch.Tensor, target: torch.Tensor,
             reduction: str = 'sum'):
    """
    Computes the mse loss.
    The `reduction` only affects the speaker dimension; the time dimension is always
    reduced by a mean operation as in [1].

    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions

    Returns:

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> mse_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(9.3333)
    >>> mse_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([1.0000, 8.3333])
    """
    return _reduce(_mse(estimate, target, dim=-1), reduction=reduction)


def log_mse_loss(estimate: torch.Tensor, target: torch.Tensor,
                 reduction: str = 'sum', soft_sdr_max: float = None):
    """
    Computes the log-mse loss between `x` and `y` as defined in [1], eq. 11.
    The `reduction` only affects the speaker dimension; the time dimension is always
    reduced by a mean operation as in [1].

    The log-mse loss is defined as [1]:

    .. math::

        L^{\\text{T-LMSE}} = 10 K^{-1} \\sum_{k} \\log_10 \sum_t |x(t) - y(t)|^2


    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions
        soft_sdr_max: Soft limit for the SDR loss value, see [2] and [3]

    Returns:
        The log-mse error between `estimate` and `target`

    References:
        [1] Jens Heitkaemper, Darius Jakobeit, Christoph Boeddeker,
            Lukas Drude, and Reinhold Haeb-Umbach. “Demystifying
            TasNet: A Dissecting Approach.” ArXiv:1911.08895
            [Cs, Eess], November 20, 2019.
            http://arxiv.org/abs/1911.08895.
        [2] Wisdom, Scott, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss,
            Kevin Wilson, and John R. Hershey. “Unsupervised Speech Separation
            Using Mixtures of Mixtures.” In Advances in Neural Information
            Processing Systems, 33:3846--3857. Curran Associates, Inc., 2020.
            https://openreview.net/forum?id=qMMzJGRPT2d.
        [3] Wisdom, Scott, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel,
            Nicolas Turpault, Eduardo Fonseca, Justin Salamon,
            Prem Seetharaman, and John R. Hershey. “What’s All the Fuss about
            Free Universal Sound Separation Data?” In IEEE International
            Conference on Acoustics, Speech and Signal Processing (ICASSP),
            186–90, 2021. https://doi.org/10.1109/ICASSP39728.2021.9414774.


    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> log_mse_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(0.9208)
    >>> log_mse_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([0.0000, 0.9208])
    >>> log_mse_loss(torch.tensor(target), torch.tensor(target), soft_sdr_max=20)
    tensor(-0.8216)
    """
    # Use the PyTorch implementation for MSE, should be the fastest
    loss = _mse(estimate, target, dim=-1)
    if soft_sdr_max:
        loss = loss + _get_threshold(soft_sdr_max) * _sqnorm(target, dim=-1)
    return _reduce(torch.log10(loss), reduction=reduction)


def sdr_loss(estimate: torch.Tensor, target: torch.Tensor,
             reduction: str = 'mean', soft_sdr_max: float = None):
    """
    The (scale dependent) SDR or SNR loss.

    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions
        soft_sdr_max: Soft limit for the SDR loss value as proposed in [1]

    Returns:

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-6.5167)
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([-9.8528, -3.1806])
    >>> sdr_loss(torch.tensor(target), torch.tensor(target), soft_sdr_max=20)
    tensor(-20.)

    References:
        [1] Wisdom, Scott, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss,
            Kevin Wilson, and John R. Hershey. “Unsupervised Speech Separation
            Using Mixtures of Mixtures.” In Advances in Neural Information
            Processing Systems, 33:3846--3857. Curran Associates, Inc., 2020.
            https://openreview.net/forum?id=qMMzJGRPT2d.

    """
    # Calculate the SNR. The square in the power computation is moved to the
    # front, thus the 20 in front of the log
    target_norm = _sqnorm(target, dim=-1)
    denominator = _sqnorm(estimate - target, dim=-1)

    if soft_sdr_max is not None:
        denominator = denominator + _get_threshold(soft_sdr_max) * target_norm

    sdr = 10 * torch.log10(target_norm / denominator)

    return -_reduce(sdr, reduction=reduction)


def si_sdr_loss(estimate, target, reduction='mean', offset_invariant=False,
                grad_stop=False, soft_sdr_max: float = None):
    """
    Scale Invariant SDR (SI-SDR) or Scale Invariant SNR (SI-SNR) loss as defined in [1], section 2.2.4.

    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions
        offset_invariant: If `True`, mean-normalize before loss calculation.
            This makes the loss shift- and scale-invariant.
        grad_stop: If `True`, the gradient is not propagated through the
            calculation of the scaling factor.
        soft_sdr_max: Soft limit for the SDR loss value as proposed in [2]

    References:
        [1] TASNET: TIME-DOMAIN AUDIO SEPARATION NETWORK FOR REAL-TIME,
            SINGLE-CHANNEL SPEECH SEPARATION
        [2] Wisdom, Scott, Efthymios Tzinis, Hakan Erdogan, Ron J. Weiss,
            Kevin Wilson, and John R. Hershey. “Unsupervised Speech Separation
            Using Mixtures of Mixtures.” In Advances in Neural Information
            Processing Systems, 33:3846--3857. Curran Associates, Inc., 2020.
            https://openreview.net/forum?id=qMMzJGRPT2d.

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> si_sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-10.7099)
    >>> si_sdr_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([-18.2391,  -3.1806])

    >>> import numpy as np
    >>> from pb_bss.evaluation import si_sdr
    >>> np.random.seed(0)
    >>> rng = np.random.RandomState(0)
    >>> reference = torch.tensor(rng.randn(100))
    >>> reference[:10]
    tensor([ 1.7641,  0.4002,  0.9787,  2.2409,  1.8676, -0.9773,  0.9501, -0.1514,
            -0.1032,  0.4106], dtype=torch.float64)

    >>> def calculate(estimate, target):
    ...     print('Torch loss:', si_sdr_loss(estimate, target))
    ...     print('Numpy metric:', si_sdr(estimate.numpy(), target.numpy()))

    Perfect estimation
    >>> si_sdr(reference.numpy(), reference.numpy())
    inf
    >>> sdr_loss(reference, reference)
    tensor(-inf, dtype=torch.float64)
    >>> si_sdr_loss(reference, reference) < -300  # Torch CPU is not hardware independent
    tensor(True)
    >>> si_sdr_loss(reference.to(torch.float32), reference.to(torch.float32)) < -130  # Torch CPU is not hardware independent
    tensor(True)
    >>> si_sdr(reference.numpy(), (reference * 2).numpy())
    inf
    >>> si_sdr_loss(reference, reference * 2) < -300  # Torch CPU is not hardware independent
    tensor(True)

    >>> calculate(reference, torch.flip(reference, (-1,)))
    Torch loss: tensor(25.1277, dtype=torch.float64)
    Numpy metric: -25.127672346460717
    >>> calculate(reference, reference + torch.flip(reference, (-1,)))
    Torch loss: tensor(-0.4811, dtype=torch.float64)
    Numpy metric: 0.481070445785553
    >>> calculate(reference, reference + 0.5)
    Torch loss: tensor(-6.3705, dtype=torch.float64)
    Numpy metric: 6.3704606032577304
    >>> calculate(reference, reference * 2 + 1)
    Torch loss: tensor(-6.3705, dtype=torch.float64)
    Numpy metric: 6.3704606032577304
    >>> calculate(
    ...    torch.tensor([1., 0], dtype=torch.float64),
    ...    torch.tensor([0., 0], dtype=torch.float64))  # never predict only zeros nan
    Torch loss: tensor(nan, dtype=torch.float64)
    Numpy metric: nan
    >>> calculate(
    ...     torch.tensor([reference.numpy(), reference.numpy()]),
    ...     torch.tensor([reference.numpy() * 2 + 1, reference.numpy() * 1 + 0.5])
    ... )
    Torch loss: tensor(-6.3705, dtype=torch.float64)
    Numpy metric: [6.3704606 6.3704606]

    >>> calculate(
    ...     torch.tensor([0., 0], dtype=torch.float64),
    ...     torch.tensor([0., 0], dtype=torch.float64))  # never predict only zeros
    Torch loss: tensor(nan, dtype=torch.float64)
    Numpy metric: nan
    >>> calculate(
    ...     torch.tensor([0., 0], dtype=torch.float64),
    ...     torch.tensor([1., 0], dtype=torch.float64))  # never predict only zeros
    Torch loss: tensor(nan, dtype=torch.float64)
    Numpy metric: nan
    >>> si_sdr_loss(torch.tensor(target), torch.tensor(target), soft_sdr_max=20)
    tensor(-20.)
    """
    assert estimate.shape == target.shape, (estimate.shape, target.shape)
    assert len(estimate.shape) >= 1, estimate.shape
    assert len(estimate.shape) == 1 or estimate.shape[-2] < 10, (
        f'Number of speakers should be small (<10, not {estimate.shape[-2]})!'
    )

    # Remove mean to ensure scale-invariance
    if offset_invariant:
        estimate = estimate - torch.mean(estimate, dim=(-1,), keepdim=True)
        target = target - torch.mean(target, dim=(-1,), keepdim=True)

    # Compute the scaling factor (alpha)
    scaling_factor = _get_scaling_factor(target, estimate)
    if grad_stop:
        scaling_factor = scaling_factor.detach()

    # Compute s_target ([1] eq. 13)
    s_target = scaling_factor * target

    # The SDR loss computes e_noise ([1] eq. 14) and the ratio, here the
    # SI-SDR ([1] eq. 15)
    return sdr_loss(
        estimate, s_target, reduction=reduction, soft_sdr_max=soft_sdr_max
    )


def log1p_mse_loss(estimate: torch.Tensor, target: torch.Tensor,
                   reduction: str = 'sum'):
    """
    Computes the log1p-mse loss between `x` and `y` as defined in [1], eq. 4.
    The `reduction` only affects the speaker dimension; the time dimension is
    always reduced by a mean operation as in [1]. It has the advantage of not
    going to negative infinity in case of perfect reconstruction while keeping
    the logarithmic nature.

    The log1p-mse loss is defined as [1]:

    .. math::

        L^{\\text{T-L1PMSE}} = \\log_10 (1 + \sum_t |x(t) - y(t)|^2)


    Args:
        estimate (... x T): The estimated signal
        target (... x T, same as estimate): The target signal
        reduction: 'mean', 'sum' or 'none'/None for batch dimensions

    Returns:
        The log1p-mse error between `estimate` and `target`


    References:
        [1] Thilo von Neumann, Christoph Boeddeker, Lukas Drude, Keisuke
            Kinoshita, Marc Delcroix, Tomohiro Nakatani, and Reinhold
            Haeb-Umbach. „Multi-talker ASR for an unknown number of sources:
            Joint training of source counting, separation and ASR“.
            http://arxiv.org/abs/2006.02786.

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> log1p_mse_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(1.2711)
    >>> log1p_mse_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([0.3010, 0.9700])
    """
    return _reduce(
        torch.log10(1 + _mse(estimate, target, dim=-1)),
        reduction=reduction
    )


def source_aggregated_sdr_loss(
        estimate: torch.Tensor,
        target: torch.Tensor,
        soft_sdr_max: float = None,
) -> torch.Tensor:
    """
    The source-aggregated SDR loss. There is no `reduction` argument because
    the reduction takes place in before the SDR is computed and is always
    required. Its value is the same for sum and mean.

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> source_aggregated_sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-4.6133)

    Is equal to `sdr_loss` if the SDRs of each pair of estimate and target are
    equal.
    >>> estimate = torch.tensor([[1., 2, 3], [4, 2, 6]])
    >>> target = torch.tensor([[2., 3, 4], [6, 4, 8]])
    >>> sdr_loss(estimate, target)
    tensor(-9.8528)
    >>> source_aggregated_sdr_loss(estimate, target)
    tensor(-9.8528)
    """
    # Calculate the source-aggregated SDR: Sum the squares of all targets and
    # all errors before computing the ratio.
    target_norm = _sqnorm(target)
    denominator = _sqnorm(estimate - target)
    if soft_sdr_max is not None:
        denominator = denominator + _get_threshold(soft_sdr_max) * target_norm
    sa_sdr = 10 * torch.log10(target_norm / denominator)

    return -sa_sdr

# TODO: remove _reduce
