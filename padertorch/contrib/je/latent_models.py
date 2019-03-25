import numpy as np
import torch
import torch.distributions as D
from padertorch.base import Model
from padertorch.contrib.je.conv1d import Pool1d, Unpool1d
from padertorch.ops.losses.loss import kl_divergence
from torch import nn
from torch.distributions.utils import broadcast_all


class LatentModel(Model):
    def pool(self, z, log_rho, *args):
        if self.pool_size is not None and self.pool_size > 1:
            # TODO: padding
            pool = Pool1d(kernel_size=self.pool_size)
            log_rho, (pool_indices_, pad_size) = pool(log_rho[:, None])
            log_rho = log_rho[:, 0]
            pool_indices = pool_indices_[:, 0]

            batch_indices = torch.cumsum(
                torch.ones_like(pool_indices), 0
            ).long() - 1
            z = z[batch_indices, pool_indices].transpose(1, 2)
            args = [
                arg[batch_indices, pool_indices] for arg in args
            ]

            unpool = Unpool1d(kernel_size=self.pool_size)
            _, pool_indices_ = broadcast_all(z, pool_indices_)
            z = unpool(
                z, indices=pool_indices_, pad_size=pad_size
            ).transpose(1, 2)
        else:
            pool_indices = None
        return (z, log_rho, pool_indices, *args)


class StandardNormal(LatentModel):
    """
    >>> sn = StandardNormal(pool_size=4)
    >>> mean, logvar, logrho, indices = sn.process(torch.ones(3,4,5), torch.zeros(3,4,5), torch.randn(3,4,5))
    >>> mean.shape, logvar.shape, logrho.shape, indices.shape
    (torch.Size([3, 4, 5]), torch.Size([3, 4, 5]), torch.Size([3, 1]), torch.Size([3, 1]))
    """

    def __init__(self, feature_size, pool_size=None):
        super().__init__()
        self.feature_size = feature_size
        self.pool_size = pool_size

    def forward(self, inputs):
        mean, log_var = inputs
        log_rho = 0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=-1)
        return self.pool(self.sample(mean, log_var), log_rho)

    def sample(self, mean, log_var):
        if self.training:
            qz = D.Normal(loc=mean, scale=torch.exp(0.5 * log_var))
            z = qz.rsample()
        else:
            z = mean
        return z

    def review(self, inputs, outputs):
        mean, log_var = inputs
        _, logrho, pool_indices = outputs
        kld = -logrho
        return dict(
            losses=dict(
                kld=kld.mean()
            ),
            histograms=dict(
                mean_=mean.flatten(),
                log_var_=log_var.flatten(),
                kld_=kld.flatten()
            )
        )


class GMM(StandardNormal):
    def __init__(
            self, feature_size, num_classes, init_std=1.0, pool_size=None,
            covariance_type='full', temperature=1.
    ):
        super(Model, self).__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.covariance_type = covariance_type
        self.temperature = temperature

        locs_init = init_std * np.random.randn(num_classes, feature_size)

        probs_init = np.ones(num_classes) * 1 / num_classes
        log_weights_init = np.log(probs_init)

        requires_grad = 2 * [True]
        if covariance_type == 'full':
            scales_init = np.broadcast_to(
                np.diag(
                    np.ones(feature_size) * init_std
                ),
                (num_classes, feature_size, feature_size)
            )
        elif covariance_type in ['diag', 'fix']:
            scales_init = np.ones((num_classes,  feature_size)) * init_std
        else:
            raise ValueError

        if covariance_type == 'fix':
            requires_grad.append(False)
        else:
            requires_grad.append(True)
        # scales_init *= (1 / num_classes) ** (1 / feature_size)

        self.log_weights, self.locs, self.scales = [
            nn.Parameter(torch.Tensor(init), requires_grad=requires_grad_)
            for init, requires_grad_ in zip(
                [log_weights_init, locs_init, scales_init], requires_grad
            )
        ]

    @property
    def log_probs(self):
        log_probs = torch.log_softmax(self.log_weights, dim=-1)
        log_probs = torch.max(
            log_probs, -100 * torch.ones_like(log_probs)
        )
        return log_probs

    @property
    def probs(self):
        return torch.exp(self.log_probs)

    @property
    def gaussians(self):
        if self.covariance_type == 'full':
            mask = torch.tril(torch.ones_like(self.scales.data[0]))
            return D.MultivariateNormal(
                loc=self.locs,
                scale_tril=(
                    self.scales * mask
                    + 0.001 * torch.diag(torch.ones_like(self.locs[0]))
                )
            )
        else:
            return D.Normal(
                loc=self.locs,
                scale=self.scales * + 0.001
            )

    def forward(self, inputs):
        mean, logvar = inputs
        qz = D.Normal(loc=mean, scale=torch.exp(0.5 * logvar))
        kl = kl_divergence(qz, self.gaussians)

        log_rho = self.log_probs - kl
        log_gamma = torch.log_softmax(log_rho, dim=-1)

        if self.temperature > 0.:
            gamma = torch.softmax(log_rho / self.temperature, dim=-1).detach()
            log_rho_ = (gamma * log_rho).sum(-1)
        else:
            assert self.temperature == 0.
            log_rho_ = log_rho[:, torch.argmax(log_gamma, dim=-1)]

        z = self.sample(mean, logvar)
        z, log_rho_, pool_indices, log_gamma = self.pool(
            z, log_rho_, log_gamma
        )
        return z, log_rho_, pool_indices, log_gamma

    def review(self, inputs, outputs):
        mean, log_var = inputs
        # ToDo: how to encourage sparse gamma?
        _, log_rho, _, log_gamma = outputs
        kld = -log_rho
        log_gamma_ = torch.max(
            log_gamma, -100 * torch.ones_like(log_gamma)
        ).sum(-1)
        gamma_max_, classes_ = torch.max(torch.exp(log_gamma), -1)
        return dict(
            losses=dict(
                kld=kld.mean(),
                log_prob=self.log_probs.sum(),
                log_gamma=log_gamma_.mean()
            ),
            histograms=dict(
                kld_=kld.flatten(),
                log_probs_=self.log_probs.flatten(),
                gamma_=torch.exp(log_gamma).flatten(),
                gamma_max_=gamma_max_.flatten(),
                classes_=classes_.flatten(),
                mean_=mean.flatten(),
                log_var_=log_var.flatten(),
            )
        )


class FBGMM(StandardNormal):
    def __init__(
            self, feature_size, num_classes, init_std=1.0,
            alpha_0=0.1, dataset_size=1000, pool_size=None
    ):
        super(Model, self).__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        self.pool_size = pool_size

        self.kappa_0 = 1.
        self.nu_0 = self.feature_size
        self.alpha_0 = alpha_0
        self.scatter_prior = torch.Tensor(
            np.diag(
                np.ones(feature_size) * init_std
                # * (1 / num_classes) ** (1 / feature_size)
            )
        )

        count_init = np.ones(num_classes)
        count_init = count_init / count_init.sum(-1) * dataset_size
        self.counts = nn.Parameter(torch.Tensor(count_init))

        locs_init = init_std * np.random.randn(
            num_classes, feature_size
        )
        unnormalized_locs_init = locs_init * count_init[:, None]
        self.unnormalized_locs = nn.Parameter(
            torch.Tensor(unnormalized_locs_init)
        )

        scales_init = np.broadcast_to(
            np.diag(
                np.ones(feature_size) * init_std
                # * (1 / num_classes) ** (1 / feature_size)
            ),
            (num_classes, feature_size, feature_size)
        )
        unnormalized_scatter_init = (
                (scales_init @ scales_init.transpose((0, 2, 1))
                 + locs_init[:, :, None] * locs_init[:, None, :])
                * count_init[:, None, None]
        )
        self.unnormalized_scatter = \
            nn.Parameter(torch.Tensor(unnormalized_scatter_init))

    @property
    def probs(self):
        return (self.alpha_0 + self.counts) / (self.alpha_0 + self.counts).sum()

    @property
    def locs(self):
        # Note that Params are scaled by counts and need to be normalized
        # loc prior (here assumed to be zero) has more influence if kappa_0
        # large!
        return self.unnormalized_locs / (self.kappa_0 + self.counts[:, None])

    @property
    def covs(self):
        # Note that Params are scaled by counts and need to be normalized
        # scatter prior (here chosen to be identity) has more influence if nu_0
        # large! Also refer to Bishop eq. 10.62
        locs = self.unnormalized_locs / self.counts[:, None]
        scatter = (self.unnormalized_scatter -
                   locs[:, :, None] * self.unnormalized_locs[:, None, :])
        return ((self.nu_0 * self.scatter_prior.to(scatter.device) + scatter
                 + self.nu_0 * self.counts[:, None, None]
                 / (self.nu_0 + self.counts[:, None, None])
                 * locs[:, :, None] * locs[:, None, :])
                / (self.nu_0 + self.counts[:, None, None]))

    @property
    def gaussians(self):
        return D.MultivariateNormal(
            loc=self.locs.detach(),
            covariance_matrix=self.covs.detach()
        )

    def forward(self, inputs):
        mean, log_var = inputs
        qz = D.Normal(loc=mean, scale=torch.exp(0.5 * log_var))

        gaussians = self.gaussians

        # Patricks Arbeit Gl. 3.14
        # Gl. 2.21:
        term1 = torch.digamma(self.alpha_0 + self.counts.detach())  # + const.

        # Gl. 2.22:
        term2 = torch.digamma(
            (self.nu_0 + self.counts[:, None].detach()
             - torch.arange(self.feature_size).float().to(term1.device)) / 2
        ).sum(-1) - torch.log(self.nu_0 + self.counts.detach())
        # 0.5*ln|\nu*W| = 0.5*(ln\nu + ln|W|) is part of kl in term3

        # Gl. 3.15
        term3 = (
            kl_divergence(qz, gaussians)
            + 0.5 * self.feature_size / (self.kappa_0 + self.counts.detach())
        )

        log_rho = term1 + 0.5 * term2 - term3
        log_gamma = torch.log_softmax(log_rho, dim=-1).detach()
        gamma = torch.exp(log_gamma)
        log_rho_ = (gamma * log_rho).sum(-1)
        z = self.sample(mean, log_var)
        z, log_rho_, pool_indices, log_gamma = self.pool(
            z, log_rho_, log_gamma
        )
        return z, log_rho_, pool_indices, log_gamma

    def review(self, inputs, outputs):
        mean, log_var = inputs
        _, log_rho, _, log_gamma = outputs
        kld = -log_rho
        gamma = torch.exp(log_gamma)
        log_gamma_ = torch.max(
            log_gamma, -100 * torch.ones_like(log_gamma)
        ).sum(-1)

        gamma_ = gamma.contiguous().view(-1, self.num_classes)
        locs_ = mean.contiguous().view(-1, self.feature_size)
        scales_ = torch.exp(0.5 * log_var).contiguous().view(
            -1, self.feature_size)

        counts = gamma_.sum(0)
        unnormalized_locs = gamma_.transpose(0, 1) @ locs_
        unnormalized_vars = gamma_.transpose(0, 1) @ (scales_.pow(2))
        diagonal_unnormalized_covs = torch.cat(
            [torch.diag(unnormalized_vars[k])[None]
             for k in range(self.num_classes)]
        )
        unnormalized_scatter = (
                (gamma_.transpose(0, 1)[:, None, :]
                 * locs_.transpose(0, 1)) @ locs_
                + diagonal_unnormalized_covs
        )
        B = gamma_.shape[0]
        for param, estimate in zip(
                [self.counts, self.unnormalized_locs,
                 self.unnormalized_scatter],
                [counts, unnormalized_locs, unnormalized_scatter]
        ):
            param.grad = (param.data - estimate.data * (self.dataset_size / B))
        gamma_max_, classes_ = torch.max(torch.exp(log_gamma)   , -1)
        return dict(
            losses=dict(
                kld=kld.mean(),
                log_gamma=log_gamma_.mean()
            ),
            scalars=dict(
                num_classes=(self.probs > 1e-5).sum()
            ),
            histograms=dict(
                kld_=kld.flatten(),
                probs_=self.probs.flatten(),
                gamma_=gamma.flatten(),
                gamma_max_=gamma_max_.flatten(),
                classes_=classes_.flatten(),
                mean_=mean.flatten(),
                log_var_=log_var.flatten()
            )
        )


class KeyValueAttention(LatentModel):
    def __init__(
            self, feature_size, num_classes, pool_size=None
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.pool_size = pool_size

        self.keys = torch.nn.Linear(self.feature_size, num_classes)
        self.values = torch.nn.Linear(num_classes, self.feature_size)

    def forward(self, inputs):
        x, = inputs

        y = self.keys(x)
        log_gamma = torch.log_softmax(y/np.sqrt(self.feature_size), dim=-1)
        gamma = torch.exp(log_gamma)
        log_gamma = torch.max(
            log_gamma, -100 * torch.ones_like(log_gamma)
        )
        z = self.values(gamma)
        z, _, pool_indices, log_gamma = self.pool(
            z, -log_gamma.sum(dim=-1), log_gamma
        )
        return z, pool_indices, log_gamma

    def review(self, inputs, outputs):
        _, _, log_gamma = outputs
        log_gamma_ = log_gamma.sum(-1)
        max_gamma, classes_ = torch.max(torch.exp(log_gamma), -1)
        return dict(
            losses=dict(
                log_gamma=log_gamma_.mean()
            ),
            histograms=dict(
                log_gamma_=log_gamma_.flatten(),
                max_gamma_=max_gamma.flatten(),
                classes_=classes_.flatten()
            )
        )
