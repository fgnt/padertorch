from functools import partial
import torch
from torch.nn import functional as F


def _get_scaling_factor(targets, estimates):
    return torch.unsqueeze(torch.einsum(
        '...t,...t->...', estimates, targets
    ), -1) / torch.norm(targets, dim=-1, keepdim=True) ** 2


def _reduce(array, reduction):
    if reduction is None or reduction == 'none':
        return array
    if reduction == 'sum':
        return torch.sum(array)
    elif reduction == 'mean':
        return torch.mean(array)
    else:
        raise ValueError(f'Unknown reduce: {reduction}. Choose from "sum", '
                         f'"mean".')


def log_mse_loss(estimate: torch.Tensor, target: torch.Tensor,
                 reduction: str = 'sum'):
    """
    Computes the log-mse loss between `x` and `y` as defined in [1], eq. 11.
    The `reduction` only affects the speaker dimension; the time dimension is always
    reduced by a mean operation as in [1].

    The log-mse loss is defined as [1]:

    .. math::

        L^{\\text{T-LMSE}} = 10 K^{-1} \\sum_{k} \\log_10 \sum_t |x(t) - y(t)|^2


    Args:
        estimate (K x T): The estimated signal
        target (K x T, same as estimate): The target signal
        reduce:

    Returns:
        The log-mse error between `estimate` and `target`

    References:
        [1] Jens Heitkaemper, Darius Jakobeit, Christoph Boeddeker,
            Lukas Drude, and Reinhold Haeb-Umbach. “Demystifying
            TasNet: A Dissecting Approach.” ArXiv:1911.08895
            [Cs, Eess], November 20, 2019.
            http://arxiv.org/abs/1911.08895.

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> log_mse_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(0.9208)
    >>> log_mse_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([0.0000, 0.9208])
    """
    # Use the PyTorch implementation for MSE, should be the fastest
    return _reduce(
        F.mse_loss(estimate, target, reduction='none').mean(dim=-1).log10(),
        reduction=reduction
    )


def sdr_loss(estimates: torch.Tensor, targets: torch.Tensor,
             reduction: str = 'mean', axis=-1):
    """
    The (scale dependent) SDR or SNR loss.

    Args:
        estimates (KxT):
        targets (KxT, same as estimates):

    Returns:

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-6.5167)
    >>> sdr_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([-9.8528, -3.1806])

    """
    # Calculate the SNR. The square in the power computation is moved to the
    # front, thus the 20 in front of the log
    snr = 20 * torch.log10(
        torch.norm(targets, dim=axis) / torch.norm(estimates - targets, dim=axis)
    )

    return -_reduce(snr, reduction=reduction)


def si_sdr_loss(estimates, targets, reduction='mean', offset_invariant=False,
                grad_stop=False):
    """
    Scale Invariant SDR (SI-SDR) or Scale Invariant SNR (SI-SNR) loss as defined in [1], section 2.2.4.

    Args:
        estimates: shape ...xT
        targets: shape ...xT, same as estimates
        reduce: If `True`, the mean is computed over all inputs. If `False`,
            the output has shape ...
        offset_invariant: If `True`, mean-normalize before loss calculation.
            This makes the loss shift- and scale-invariant.
        grad_stop: If `True`, the gradient is not propagated through the
            calculation of the scaling factor.

    References:
        [1] TASNET: TIME-DOMAIN AUDIO SEPARATION NETWORK FOR REAL-TIME,
            SINGLE-CHANNEL SPEECH SEPARATION

    >>> estimate = [[1., 2, 3], [4, 5, 6]]
    >>> target = [[2., 3, 4], [4, 0, 6]]
    >>> si_sdr_loss(torch.tensor(estimate), torch.tensor(target))
    tensor(-10.7099)
    >>> si_sdr_loss(torch.tensor(estimate), torch.tensor(target), reduction=None)
    tensor([-18.2391,  -3.1806])

    >>> import numpy as np
    >>> from pb_bss.evaluation import si_sdr
    >>> np.random.seed(0)
    >>> reference = torch.tensor(np.random.randn(100))

    >>> def calculate(estimates, targets):
    ...     print('Torch loss:', si_sdr_loss(estimates, targets))
    ...     print('Numpy metric:', si_sdr(estimates.numpy(), targets.numpy()))

    Perfect estimation
    >>> calculate(reference, reference)
    Torch loss: tensor(-313.2011, dtype=torch.float64)
    Numpy metric: inf
    >>> si_sdr_loss(reference.to(torch.float32), reference.to(torch.float32))
    tensor(-141.5990)
    >>> calculate(reference, reference * 2)
    Torch loss: tensor(-313.2011, dtype=torch.float64)
    Numpy metric: inf

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
    """
    assert estimates.shape == targets.shape, (estimates.shape, targets.shape)
    assert len(estimates.shape) >= 1, estimates.shape
    assert len(estimates.shape) == 1 or estimates.shape[-2] < 10, (
        f'Number of speakers should be small (<10, not {estimates.shape[-2]})!'
    )

    # Remove mean to ensure scale-invariance
    if offset_invariant:
        estimates = estimates - torch.mean(estimates, dim=(-1,), keepdim=True)
        targets = targets - torch.mean(targets, dim=(-1,), keepdim=True)

    # Compute the scaling factor (alpha)
    scaling_factor = _get_scaling_factor(targets, estimates)
    if grad_stop:
        scaling_factor = scaling_factor.detach()

    # Compute s_target ([1] eq. 13)
    s_target = scaling_factor * targets

    # The SNR loss computes e_noise ([1] eq. 14) and the ratio, here the
    # SI-SNR ([1] eq. 15)
    return sdr_loss(estimates, s_target, reduction=reduction)


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
        reduce:

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
    # Use the PyTorch implementation for MSE, should be the fastest
    return _reduce(
        torch.log10(
            1 + F.mse_loss(estimate, target, reduction='none').mean(dim=-1)),
        reduction=reduction
    )

# TODO: Add log1p_mse loss from interspeech paper
# TODO: remove _reduce


time_domain_loss_functions = {
    'log-mse': log_mse_loss,
    'si-sdr': si_sdr_loss,
    'si-sdr-grad-stop': partial(si_sdr_loss, grad_stop=True),
    'mse': F.mse_loss,
    'sdr': sdr_loss,
}

