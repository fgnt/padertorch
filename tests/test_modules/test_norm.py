import torch
from padertorch.ops.sequence.mask import compute_mask
from padertorch.modules.normalization import normalize
import paderbox.testing as tc


def normalize_ref(x, gamma, beta, statistics_axis, batch_axis, sequence_axis, seq_len, shift, scale, eps):
        # compute mask
        if seq_len is not None:
            mask = compute_mask(x, seq_len, batch_axis, sequence_axis)
        else:
            mask = torch.ones_like(x)

        # compute statistics
        n_values = mask.sum(dim=statistics_axis, keepdim=True)
        x = x * mask
        mean = x.sum(dim=statistics_axis, keepdim=True) / torch.max(n_values, torch.ones_like(n_values))
        power = (x ** 2).sum(dim=statistics_axis, keepdim=True) / torch.max(n_values, torch.ones_like(n_values))
        y = x
        if shift:
            y = y - mean
            power = power - mean**2
        if scale:
            y = y / torch.sqrt(power + eps)

        if gamma is not None:
            assert gamma.dim() == x.dim(), gamma.shape
            y = y * gamma
        if beta is not None:
            assert beta.dim() == x.dim(), beta.shape
            y = y + beta
        return y*mask, mean, power, n_values


def test_grads():
    x = torch.randn((2, 3, 5), requires_grad=True)
    gamma = 1+torch.randn((1, 3, 1))
    gamma.requires_grad = True
    beta = torch.randn((1, 3, 1), requires_grad=True)
    seq_len = [5, 3]
    x_ref = x.clone().detach()
    x_ref.requires_grad = True
    gamma_ref = gamma.clone().detach()
    gamma_ref.requires_grad = True
    beta_ref = beta.clone().detach()
    beta_ref.requires_grad = True

    for shift in [True, False]:
        for scale in [True, False]:
            if x.grad is not None:
                x.grad.zero_()
                x_ref.grad.zero_()
                gamma.grad.zero_()
                gamma_ref.grad.zero_()
                beta.grad.zero_()
                beta_ref.grad.zero_()
            y, *_ = normalize(x, gamma, beta, [0, 2], 0, 2, seq_len, shift, scale, 1e-3)
            (y[0, [0, 1]] - y[0, 2]).sum().backward()
            y_ref, *_ = normalize_ref(x_ref, gamma_ref, beta_ref, [0, 2], 0, 2, seq_len, shift, scale, 1e-3)
            (y_ref[0, [0, 1]] - y_ref[0, 2]).sum().backward()

            tc.assert_array_almost_equal(y.detach().numpy(), y_ref.detach().numpy())
            tc.assert_array_almost_equal(x.grad.numpy(), x_ref.grad.numpy())
            tc.assert_array_almost_equal(gamma.grad.numpy(), gamma_ref.grad.numpy())
            tc.assert_array_almost_equal(beta.grad.numpy(), beta_ref.grad.numpy())
