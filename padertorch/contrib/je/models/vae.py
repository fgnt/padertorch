import numpy as np
import torch
import torch.distributions as D
from padertorch.base import Model, Module
from padertorch.contrib.je.modules.conv import HybridCNN, HybridCNNTranspose, CNN2d, CNN1d, MultiScaleCNN1d
from padertorch.ops.losses.loss import kl_divergence
from torch import nn
from torchvision.utils import make_grid
from sklearn import metrics


class VAE(Model):
    """
    >>> config = VAE.get_config(dict(\
            feature_key='log_mel',\
            encoder=dict(\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, hidden_channels=32, out_channels=32,\
                    num_layers=3, kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    hidden_channels=32, num_layers=3, kernel_size=3\
                ),\
                return_pool_data=True\
            ),\
            decoder=dict(cnn_transpose_1d=dict(in_channels=16))\
        ))
    >>> config['encoder']['cnn_1d']['in_channels']
    2560
    >>> config['encoder']['cnn_1d']['out_channels']
    32
    >>> config['decoder']['cnn_transpose_1d']['in_channels']
    16
    >>> vae = VAE.from_config(config)
    >>> inputs = {'log_mel': torch.zeros((4, 1, 80, 100))}
    >>> outputs = vae(inputs)
    >>> outputs[0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][1].shape
    torch.Size([4, 16, 100])
    >>> review = vae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose,
            feature_key, target_key=None
    ):
        super().__init__()
        # allow joint optimization of encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.feature_key = feature_key
        self.target_key = feature_key if target_key is None \
            else self.target_key
        self.n_params = 2

    def encode(self, inputs):
        x = inputs[self.feature_key]
        if self.encoder.return_pool_data:
            h, pool_indices, shapes = self.encoder(x)
            assert (
                pool_indices[-1][-1] is None
                if isinstance(self.encoder, HybridCNN)
                else pool_indices[-1] is None
            ), 'No pooling in output layer allowed'
        else:
            h = self.encoder(x)
            pool_indices = shapes = None
        assert not h.shape[1] % self.n_params
        params = tuple(torch.split(h, h.shape[1] // self.n_params, dim=1))
        return params, pool_indices, shapes

    def reparameterize(self, params):
        mu, logvar = params[:2]
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, pool_indices=None, shapes=None):
        x_hat = self.decoder(
            z, pool_indices=pool_indices, out_shapes=shapes
        )
        return x_hat  # (B, C, F, T)

    def forward(self, inputs):
        params, pool_indices, shapes = self.encode(inputs)
        z = self.reparameterize(params)
        x_hat = self.decode(
            z, pool_indices=pool_indices, shapes=shapes
        )
        return x_hat, params

    def review(self, inputs, outputs):
        # visualization
        x = inputs[self.target_key]
        x_hat, params, *_ = outputs
        (mu, log_var) = params[:2]
        mse = (x - x_hat).pow(2).sum(dim=((1, 2) if x.dim() == 4 else 1))
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
        x_hat = x_hat.contiguous()
        mu = mu.contiguous()
        review = dict(
            losses=dict(
                mse=mse.mean(),
                kld=kld.mean()
            ),
            histograms=dict(
                mse_=mse,
                kld_=kld.flatten(),
                mu_=mu.flatten(),
                log_var_=log_var.flatten()
            ),
            images=dict(
                targets=x[:3],
                latents=mu[:3],
                reconstructions=x_hat[:3]
            )
        )
        return review

    def modify_summary(self, summary):
        summary = super().modify_summary(summary)
        for key, image in summary['images'].items():
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['encoder']['factory'] = HybridCNN
        if config['encoder']['factory'] == HybridCNN:
            config['encoder'].update({
                'cnn_2d': {'factory': CNN2d},
                'cnn_1d': {
                    'factory': CNN1d,
                    'out_channels': 2*config['decoder']['cnn_transpose_1d']['in_channels']
                }
            })
        if config['encoder']['factory'] in (CNN1d, MultiScaleCNN1d):
            config['encoder']['out_channels'] = 2 * config['decoder']['in_channels']
        config['decoder'].update(
            config['encoder']['factory'].get_transpose_config(config['encoder'])
        )


class GMM(Module):
    def __init__(
            self, feature_size, num_classes,
            covariance_type='full', class_temperature=1.,
            loc_init_std=1.0, scale_init_std=1.0, supervised=False
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.covariance_type = covariance_type
        self.class_temperature = class_temperature
        self.supervised = supervised

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

    def forward(self, mu, log_var):
        qz = D.Normal(
            loc=mu.permute((0, 2, 1)),
            scale=torch.exp(0.5 * log_var.permute((0, 2, 1)))
        )
        kld = kl_divergence(qz, self.gaussians)
        log_class_posterior = torch.log_softmax(
            (self.log_class_probs - kld) / max(self.class_temperature, 1e-2),
            dim=-1
        )
        return log_class_posterior, kld


class GMMVAE(VAE):
    """
    >>> config = GMMVAE.get_config(dict(\
            feature_key='log_mel',\
            encoder=dict(\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, hidden_channels=32, out_channels=32,\
                    num_layers=3, kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    hidden_channels=32, num_layers=3, kernel_size=3\
                ),\
                return_pool_data=True\
            ),\
            decoder=dict(cnn_transpose_1d=dict(in_channels=16)),\
            gmm=dict(num_classes=10)\
        ))
    >>> config['encoder']['cnn_1d']['in_channels']
    2560
    >>> config['encoder']['cnn_1d']['out_channels']
    32
    >>> config['decoder']['cnn_transpose_1d']['in_channels']
    16
    >>> gmmvae = GMMVAE.from_config(config)
    >>> inputs = {'log_mel': torch.zeros((4, 1, 80, 100))}
    >>> outputs = gmmvae(inputs)
    >>> outputs[0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][1].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][2].shape
    torch.Size([4, 100, 10])
    >>> outputs[1][3].shape
    torch.Size([4, 100, 10])
    >>> review = gmmvae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose, gmm: GMM,
            feature_key, target_key=None, label_key=None, supervised=False
    ):
        super().__init__(
            encoder=encoder, decoder=decoder,
            feature_key=feature_key, target_key=target_key
        )
        self.gmm = gmm
        self.label_key = label_key
        self.supervised = supervised

    def encode(self, inputs):
        params, pool_indices, shapes = super().encode(inputs)
        params = (*params, *self.gmm(*params))
        return params, pool_indices, shapes

    def review(self, inputs, outputs):
        review = super().review(inputs, outputs)
        class_labels = inputs[self.label_key] if self.supervised else None

        log_class_posterior, kld = outputs[1][2:]
        class_posterior = log_class_posterior.exp().detach()
        if class_labels is None and self.gmm.class_temperature < 1e-2:
            class_labels = torch.argmax(log_class_posterior, dim=-1)
        if class_labels is None:
            kld = (class_posterior * kld).sum(-1)
            class_ce = -(class_posterior * self.gmm.log_class_probs).sum(-1)
        else:
            while class_labels.dim() < 2:
                class_labels = class_labels[..., None]
            class_labels = class_labels.expand(kld.shape[:-1])
            kld = kld.gather(-1, class_labels[..., None]).squeeze(-1)
            class_ce = -self.gmm.log_class_probs[class_labels]

        max_class_posterior, classes = torch.max(
            torch.exp(log_class_posterior), -1
        )
        review['losses'].update(dict(
            kld=kld.mean(),
            class_ce=class_ce.mean(),
            log_class_prob=self.gmm.log_class_probs.sum()
        ))
        review['scalars'] = dict(
            class_temperature=self.gmm.class_temperature,
        )
        if self.label_key is not None:
            labels = inputs[self.label_key]
            review['scalars'].update(dict(
                class_temperature=self.gmm.class_temperature,
                classes=classes.flatten(),
                labels=labels.flatten()
            ))
        review['histograms'].update(dict(
            kld_=kld.flatten(),
            log_class_probs_=self.gmm.log_class_probs.flatten(),
            max_class_posterior_=max_class_posterior.flatten(),
            classes_=classes.flatten()
        ))
        return review

    def modify_summary(self, summary):
        predictions = summary['scalars'].pop('classes', None)
        labels = summary['scalars'].pop('labels', None)
        if predictions is not None and labels is not None:
            _, labels = np.unique(labels, return_inverse=True)
            _, predictions = np.unique(predictions, return_inverse=True)
            if len(labels) < len(predictions):
                nframes = len(predictions) / len(labels)
                assert int(nframes) == nframes
                nframes = int(nframes)
            else:
                nframes = 1
            contingency_matrix = metrics.cluster.contingency_matrix(
                np.repeat(labels, nframes), predictions
            )
            ncm = contingency_matrix / np.sum(contingency_matrix, axis=0)
            label_probs = ncm[:, predictions.reshape((-1, nframes))]
            predictions = np.argmax(np.max(label_probs, axis=-1), axis=0)
            summary['scalars']['accuracy'] = np.mean(predictions == labels)
            summary['scalars']['fscore'] = metrics.f1_score(
                labels, predictions, average='macro'
            )
        summary = super().modify_summary(summary)
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        super().finalize_dogmatic_config(config)
        config['gmm'] = {
            'factory': GMM,
            'feature_size': config['decoder']['cnn_transpose_1d']['in_channels']
        }


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


class FBGMM(Module):
    def __init__(
            self, feature_size, num_classes, init_std=1.0,
            alpha_0=0.1, dataset_size=1000, pool_size=None
    ):
        super().__init__()
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
        self.unnormalized_scatter = nn.Parameter(
            torch.Tensor(unnormalized_scatter_init)
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


class KeyValueAttention(Model):
    def __init__(self, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes

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
        return z, log_gamma

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
