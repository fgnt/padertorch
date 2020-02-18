import numpy as np
import torch
import torch.distributions as D
from padertorch.base import Module
from padertorch.ops.losses.loss import kl_divergence
from torch import nn
from sklearn import metrics


class GMM(Module):
    """
    >>> gmm = FBGMM(10, 3)
    >>> log_gamma, log_rho = gmm(D.Normal(loc=torch.zeros((8, 10)), scale=torch.ones((8, 10))))
    """
    def __init__(
            self, feature_size, num_classes, covariance_type='full',
            loc_init_std=1.0, scale_init_std=1.0
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.covariance_type = covariance_type

        locs_init = loc_init_std * np.random.randn(num_classes, feature_size)

        class_probs_init = np.ones(num_classes) * 1 / num_classes
        log_class_weights_init = np.log(class_probs_init)

        if covariance_type == 'full':
            scales_init = np.broadcast_to(
                np.diag(
                    np.ones(feature_size) * scale_init_std
                ),
                (num_classes, feature_size, feature_size)
            )
        elif covariance_type in ['diag', 'fix']:
            scales_init = (
                    np.ones((num_classes,  feature_size)) * scale_init_std
            )
        else:
            raise ValueError

        requires_grad = 2 * [True]
        if covariance_type == 'fix':
            requires_grad.append(False)
        else:
            requires_grad.append(True)

        self.log_weights, self.locs, self.scales = [
            nn.Parameter(torch.Tensor(init), requires_grad=requires_grad_)
            for init, requires_grad_ in zip(
                [log_class_weights_init, locs_init, scales_init], requires_grad
            )
        ]

    @property
    def log_class_probs(self):
        log_probs = torch.log_softmax(self.log_weights, dim=-1)
        log_probs = torch.max(
            log_probs, -20 * torch.ones_like(log_probs)
        )
        return log_probs

    @property
    def class_probs(self):
        return torch.exp(self.log_class_probs)

    @property
    def gaussians(self):
        if self.covariance_type == 'full':
            mask = torch.tril(torch.ones_like(self.scales.data[0]))
            return D.MultivariateNormal(
                loc=self.locs,
                scale_tril=(
                    self.scales * mask
                    + 0.1 * torch.diag(torch.ones_like(self.locs[0]))
                )
            )
        else:
            return D.Normal(
                loc=self.locs,
                scale=self.scales * + 0.1
            )

    def forward(self, qz):
        log_rho = -kl_divergence(qz, self.gaussians)
        log_class_posterior = torch.log_softmax(
            self.log_class_probs + log_rho,
            dim=-1
        )
        return log_class_posterior, log_rho


class FBGMM(Module):
    """
    >>> fbgmm = FBGMM(10, 3)
    >>> log_gamma, log_rho = fbgmm(D.Normal(loc=torch.zeros((8, 10)), scale=torch.ones((8, 10))))
    """
    def __init__(
            self, feature_size, num_classes, init_std=1.0,
            alpha_0=0.1, virtual_dataset_size=1000, momentum=0.99
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.virtual_dataset_size = virtual_dataset_size
        self.momentum = momentum

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
        count_init = count_init / count_init.sum(-1) * virtual_dataset_size
        self.register_buffer('counts', torch.Tensor(count_init))

        locs_init = init_std * np.random.randn(num_classes, feature_size)
        unnormalized_locs_init = locs_init * count_init[:, None]
        self.register_buffer(
            'unnormalized_locs', torch.Tensor(unnormalized_locs_init)
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
        self.register_buffer(
            'unnormalized_scatter', torch.Tensor(unnormalized_scatter_init)
        )

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
        return (
            self.nu_0 * self.scatter_prior.to(scatter.device)
            + scatter
            + self.nu_0 * self.counts[:, None, None] / (self.nu_0 + self.counts[:, None, None])
            * locs[:, :, None] * locs[:, None, :]
        ) / (self.nu_0 + self.counts[:, None, None])

    @property
    def gaussians(self):
        return D.MultivariateNormal(
            loc=self.locs.detach(),
            covariance_matrix=self.covs.detach()
        )

    def forward(self, qz):
        gaussians = self.gaussians

        # Patricks Arbeit Gl. 3.14
        # Gl. 2.21:
        term1 = torch.digamma(self.alpha_0 + self.counts)  # + const.

        # Gl. 2.22:
        term2 = torch.digamma(
            (self.nu_0 + self.counts[:, None]
             - torch.arange(self.feature_size).float().to(term1.device)) / 2
        ).sum(-1) - torch.log(self.nu_0 + self.counts)
        # 0.5*ln|\nu*W| = 0.5*(ln\nu + ln|W|) is part of kl in term3

        # Gl. 3.15
        term3 = (
            kl_divergence(qz, gaussians)
            + 0.5 * self.feature_size / (self.kappa_0 + self.counts)
        )

        log_rho = term1 + 0.5 * term2 - term3
        log_class_posterior = torch.log_softmax(log_rho, dim=-1)
        if self.training:

            gamma = log_class_posterior.exp()
            gamma_ = gamma.contiguous().view(-1, self.num_classes)
            counts = gamma_.sum(0)

            locs_ = qz.loc.contiguous().view((-1, self.feature_size))
            unnormalized_locs = gamma_.transpose(0, 1) @ locs_
            scales_ = qz.scale.contiguous().view((-1, self.feature_size))
            unnormalized_vars = gamma_.transpose(0, 1) @ (scales_.pow(2))
            diagonal_unnormalized_covs = torch.cat(
                [torch.diag(unnormalized_vars[k])[None] for k in range(self.num_classes)]
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
                param *= self.momentum
                param += (1 - self.momentum) * estimate.data * (self.virtual_dataset_size / B)

        return log_class_posterior, log_rho


# ToDo:
class HGMM(Module):
    def __init__(
            self, feature_size, num_scenes, num_events,
            covariance_type='full', event_temperature=1., scene_temperature=1.,
            loc_init_std=1.0, scale_init_std=1.0, supervised=False
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_scenes = num_scenes
        self.num_events = num_events
        self.covariance_type = covariance_type
        self.event_temperature = event_temperature
        self.scene_temperature = scene_temperature
        self.supervised = supervised

        locs_init = loc_init_std * np.random.randn(
            num_scenes, num_events, feature_size
        )

        scene_probs_init = np.ones(num_scenes) * 1 / num_scenes
        event_probs_init = np.ones((num_scenes, num_events)) * 1 / num_events
        log_scene_weights_init = np.log(scene_probs_init)
        log_event_weights_init = np.log(event_probs_init)

        if covariance_type == 'full':
            scales_init = np.broadcast_to(
                np.diag(
                    np.ones(feature_size) * scale_init_std
                ),
                (num_scenes, num_events, feature_size, feature_size)
            )
        elif covariance_type in ['diag', 'fix']:
            scales_init = (
                    np.ones((num_scenes, num_events, feature_size))
                    * scale_init_std
            )
        else:
            raise ValueError

        requires_grad = 3 * [True]
        if covariance_type == 'fix':
            requires_grad.append(False)
        else:
            requires_grad.append(True)

        self.log_scene_weights, self.log_event_weights, self.locs, self.scales = [
            nn.Parameter(torch.Tensor(init), requires_grad=requires_grad_)
            for init, requires_grad_ in zip(
                [
                    log_scene_weights_init, log_event_weights_init,
                    locs_init, scales_init
                ],
                requires_grad
            )
        ]

    @property
    def num_classes(self):
        return self.num_scenes * self.num_events

    @property
    def log_scene_probs(self):
        log_probs = torch.log_softmax(self.log_scene_weights, dim=-1)
        log_probs = torch.max(
            log_probs, -20 * torch.ones_like(log_probs)
        )
        return log_probs

    @property
    def scene_probs(self):
        return torch.exp(self.log_scene_probs)

    @property
    def log_event_probs(self):
        log_probs = torch.log_softmax(self.log_event_weights, dim=-1)
        log_probs = torch.max(
            log_probs, -20 * torch.ones_like(log_probs)
        )
        return log_probs

    @property
    def event_probs(self):
        return torch.exp(self.log_event_probs)

    @property
    def log_class_probs(self):
        log_probs = (
                self.log_event_probs + self.log_scene_probs[:, None]
        ).view(-1)
        log_probs = torch.max(
            log_probs, -20 * torch.ones_like(log_probs)
        )
        return log_probs

    @property
    def class_probs(self):
        return torch.exp(self.log_class_probs)

    @property
    def gaussians(self):
        if self.covariance_type == 'full':
            mask = torch.tril(torch.ones_like(self.scales.data[0, 0]))
            return D.MultivariateNormal(
                loc=self.locs,
                scale_tril=(
                    self.scales * mask
                    + 0.1 * torch.diag(torch.ones_like(self.locs[0, 0]))
                )
            )
        else:
            return D.Normal(
                loc=self.locs,
                scale=self.scales * + 0.1
            )

    def forward(self, inputs):
        mean, log_var = inputs['params']
        scene_labels = inputs['labels'] if self.supervised else None
        event_labels = None

        qz = D.Normal(loc=mean, scale=torch.exp(0.5 * log_var))
        gaussians = self.gaussians

        kld = kl_divergence(qz, gaussians)
        B, T, S, E = kld.shape

        log_event_posterior = torch.log_softmax(
            (self.log_event_probs - kld)
            / max(self.event_temperature, 1e-2),
            dim=-1
        )
        event_posterior = log_event_posterior.exp().detach()  # (B, T, S, E)

        log_scene_posterior = torch.log_softmax(
            (
                torch.logsumexp(self.log_event_probs - kld, dim=-1).sum(1)  # (B, S)
                + self.log_scene_probs
            ) / max(self.scene_temperature, 1e-2),
            dim=-1
        )
        scene_posterior = log_scene_posterior.exp().detach()
        log_class_posterior = (
            log_event_posterior + log_scene_posterior[:, None, :, None]
        ).view((B, T, S*E))

        if event_labels is None and self.event_temperature < 1e-2:
            event_labels = torch.argmax(log_event_posterior, dim=-1)
        if event_labels is None:
            kld = (event_posterior * kld).sum(-1)  # (B, T, S)
            event_ce = -(event_posterior * self.log_event_probs).sum(-1)
        else:
            while event_labels.dim() < 3:
                event_labels = event_labels[..., None]
            event_labels = event_labels.expand(kld.shape[:-1])
            kld = kld.gather(-1, event_labels[..., None]).squeeze(-1)
            event_ce = -self.log_event_probs[event_labels]

        if scene_labels is None and self.scene_temperature < 1e-2:
            scene_labels = torch.argmax(log_scene_posterior, dim=-1)
        if scene_labels is None:
            kld = (scene_posterior[:, None] * kld).sum(-1)  # (B, T)
            event_ce = (scene_posterior[:, None] * event_ce).sum(-1)  # (B, T)
            scene_ce = -(scene_posterior * self.log_scene_probs).sum(-1)
        else:
            scene_labels = scene_labels[:, None].expand(kld.shape[:-1])
            kld = kld.gather(-1, scene_labels[..., None]).squeeze(-1)
            event_ce = event_ce.gather(-1, scene_labels[..., None]).squeeze(-1)
            scene_ce = -self.log_scene_probs[scene_labels]

        z = self.sample(mean, log_var)
        return (
            z, kld, event_ce, scene_ce/float(T),
            log_event_posterior, log_scene_posterior, log_class_posterior
        )

    def review(self, inputs, outputs):
        (
            _, kld, event_ce, scene_ce,
            log_event_posterior, log_scene_posterior, log_class_posterior
        ) = outputs
        max_scene_posterior, scenes = torch.max(
            torch.exp(log_scene_posterior), -1
        )
        max_class_posterior, classes = torch.max(
            torch.exp(log_class_posterior), -1
        )
        mean, log_var = inputs['params']
        labels = inputs['labels']
        return dict(
            losses=dict(
                kld=kld.mean(),
                event_ce=event_ce.mean(),
                scene_ce=scene_ce.mean(),
                log_scene_prob=self.log_scene_probs.sum(),
                log_event_prob=self.log_event_probs.sum()
            ),
            scalars=dict(
                scene_temperature=self.scene_temperature,
                event_temperature=self.event_temperature,
                scenes=scenes.flatten(),
                labels=None if labels is None else labels.flatten(),
            ),
            histograms=dict(
                kld_=kld.flatten(),
                log_scene_probs_=self.log_scene_probs.flatten(),
                log_event_probs_=self.log_event_probs.flatten(),
                max_class_posterior_=max_class_posterior.flatten(),
                classes_=classes.flatten(),
                max_scene_posterior_=max_scene_posterior.flatten(),
                scenes_=scenes.flatten(),
                mean_=mean.flatten(),
                log_var_=log_var.flatten(),
            )
        )

    def modify_summary(self, summary):
        predictions = summary['scalars'].pop('scenes', None)
        labels = summary['scalars'].pop('labels', None)
        if predictions is not None and labels is not None:
            predictions = np.array(predictions)
            labels = np.array(labels)
            if not self.supervised:
                _, labels = np.unique(labels, return_inverse=True)
                _, predictions = np.unique(predictions, return_inverse=True)
                summary['scalars']['v_measure'] = metrics.v_measure_score(
                    labels, predictions
                )
                contingency_matrix = metrics.cluster.contingency_matrix(
                    labels, predictions
                )
                mapping = np.argmax(contingency_matrix, axis=0)
                predictions = mapping[predictions]
            summary['scalars']['accuracy'] = np.mean(predictions == labels)
            summary['scalars']['fscore'] = metrics.f1_score(
                labels, predictions, average='macro'
            )
        return summary
