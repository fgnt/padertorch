from functools import partial
import torch
from torch.nn import functional as F


def _get_scaling_factor(targets, estimates):
    return torch.unsqueeze(torch.einsum(
        '...t,...t->...', estimates, targets
    ), -1) / torch.norm(targets, dim=-1, keepdim=True) ** 2


def _reduce(array, reduce):
    if reduce is None or reduce == 'none':
        return array
    if reduce == 'sum':
        return torch.sum(array)
    elif reduce == 'mean':
        return torch.mean(array)
    else:
        raise ValueError(f'Unknown reduce: {reduce}. Choose from "sum", '
                         f'"mean".')


def log_mse_loss(estimate: torch.Tensor, target: torch.Tensor,
                 reduce: str = 'sum'):
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
    """
    # Use the PyTorch implementation for MSE, should be the fastest
    return _reduce(
        F.mse_loss(estimate, target, reduce='none').mean(dim=-1).log10(),
        reduce=reduce
    )


def sdr_loss(estimates: torch.Tensor, targets: torch.Tensor,
             reduce: str = 'mean'):
    """
    The (scale dependent) SDR or SNR loss.

    Args:
        estimates (KxT):
        targets (KxT, same as estimates):

    Returns:

    """
    # Calculate the SNR. The square in the power computation is moved to the
    # front, thus the 20 in front of the log
    snr = 20 * torch.log10(
        torch.norm(targets, dim=-1) / torch.norm(estimates - targets, dim=-1)
    )

    return -_reduce(snr, reduce=reduce)


def si_sdr_loss(estimates, targets, reduce='mean', offset_invariant=False,
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
    return sdr_loss(estimates, s_target, reduce=reduce)

# TODO: Add log1p_mse loss from interspeech paper
# TODO: remove _reduce


time_domain_loss_functions = {
    'log-mse': log_mse_loss,
    'si-sdr': si_sdr_loss,
    'si-sdr-grad-stop': partial(si_sdr_loss, grad_stop=True),
    'mse': F.mse_loss,
    'sdr': sdr_loss,
}

