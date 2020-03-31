import numpy as np
import torch
from einops import rearrange
from padertorch.base import Model
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.dist import GMM
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.global_pooling import Mean
from padertorch.contrib.je.modules.hybrid import HybridCNN, HybridCNNTranspose
from sklearn import metrics
from torch.distributions import Normal
from torchvision.utils import make_grid


class VAE(Model):
    """
    >>> config = VAE.get_config(dict(\
            encoder=dict(\
                factory=HybridCNN,\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
            ),\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
        ))
    >>> config['encoder']['cnn_1d']['in_channels']
    2560
    >>> config['encoder']['cnn_1d']['out_channels']
    [32, 32, 32]
    >>> config['decoder']['cnn_transpose_1d']['in_channels']
    16
    >>> vae = VAE.from_config(config)
    >>> inputs = {'stft': torch.zeros((4, 1, 100, 257, 2)), 'seq_len': None}
    >>> outputs = vae(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[0][1].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1][0][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][0][1].shape
    torch.Size([4, 16, 100])
    >>> review = vae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose, *,
            feature_key='stft', feature_extractor=None, feature_size=None,
    ):
        super().__init__()
        # allow joint optimization of encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.feature_key = feature_key
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        self.n_params = 2

    def encode(self, x, seq_len=None):
        if isinstance(self.encoder, CNN1d):
            x = rearrange(x, 'b c f t -> b (c f) t')
        if self.encoder.return_pool_indices:
            h, seq_len, pool_indices = self.encoder(x, seq_len=seq_len)
        else:
            h, seq_len = self.encoder(x, seq_len=seq_len)
            pool_indices = None
        assert not h.shape[1] % self.n_params
        params = tuple(torch.split(h, h.shape[1] // self.n_params, dim=1))
        return params, seq_len, pool_indices

    def reparameterize(self, params):
        mu, logvar = params[:2]
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, seq_len=None, shapes=None, lengths=None, pool_indices=None):
        x_hat, seq_len = self.decoder(
            z, seq_len=seq_len, shapes=shapes, seq_lens=lengths, pool_indices=pool_indices
        )
        if x_hat.dim() == 3:
            b, f, t = x_hat.shape
            x_hat = x_hat.view((b, -1, self.feature_size, t))
        return x_hat, seq_len  # (B, C, F, T)

    def forward(self, inputs):
        x_target = inputs[self.feature_key]
        seq_len = inputs['seq_len']
        if self.feature_extractor is not None:
            x_target = self.feature_extractor(x_target, seq_len)
        params, seq_len, pool_indices = self.encode(x_target, seq_len)
        x_shape = x_target.shape
        if isinstance(self.encoder, CNN1d):
            x_shape = (x_shape[0], x_shape[1]*x_shape[2], x_shape[3])
        shapes = self.encoder.get_shapes(x_shape)
        lengths = self.encoder.get_seq_lens(seq_len)
        z = self.reparameterize(params)
        x_hat, _ = self.decode(
            z, seq_len, shapes=shapes, lengths=lengths, pool_indices=pool_indices
        )
        return (x_target, x_hat), (params, seq_len)

    def review(self, inputs, outputs):
        # visualization
        (x_target, *x_hats), (params, seq_len), *_ = outputs
        (mu, log_var) = params[:2]
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
        kld = Mean(axis=-1)(kld, seq_len)

        mu = mu.contiguous()
        review = dict(
            losses=dict(
                kld=kld.mean(),
            ),
            histograms=dict(
                kld_=kld.flatten(),
                mu_=mu.flatten(),
                log_var_=log_var.flatten(),
            ),
            images=dict(
                targets=x_target[:3],
                latents=mu[:3],
            )
        )
        seq_len = inputs['seq_len']
        for i, x_hat in enumerate(x_hats):
            mse = (x_hat - x_target).pow(2).sum(dim=(1, 2))
            mse = Mean(axis=-1)(mse, seq_len)
            review['losses'][f'mse{i}'] = mse.mean()
            review['histograms'][f'mse{i}_'] = mse
            review['images'][f'x_hat_{i}_'] = x_hat.contiguous()[:3]
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
        config['encoder']['factory'] = CNN1d
        config['feature_extractor'] = {
            'factory': NormalizedLogMelExtractor,
        }
        if config['feature_extractor'] is not None:
            if config['feature_extractor']['factory'] == NormalizedLogMelExtractor:
                config['feature_size'] = config['feature_extractor']['n_mels']
        feature_size = config['feature_size']
        if config['encoder']['factory'] == HybridCNN:
            if feature_size is not None:
                config['encoder'].update({
                    'input_size': feature_size,
                })
            content_emb_dim = config['encoder']['cnn_1d']['out_channels'][-1] // 2
        elif config['encoder']['factory'] == CNN1d:
            if feature_size is not None:
                config['encoder']['in_channels'] = feature_size
            content_emb_dim = config['encoder']['out_channels'][-1] // 2
        else:
            raise ValueError(f'Factory {config["encoder"]["factory"]} not allowed.')

        config['decoder'] = config['encoder']['factory'].get_transpose_config(config['encoder'])
        if config['decoder']['factory'] == HybridCNNTranspose:
            config['decoder']['cnn_transpose_1d']['in_channels'] = content_emb_dim
        elif config['decoder']['factory'] == CNNTranspose1d:
            config['decoder']['in_channels'] = content_emb_dim
        else:
            raise ValueError(f'Factory {config["decoder"]["factory"]} not allowed.')


class GMMVAE(VAE):
    """
    >>> config = GMMVAE.get_config(dict(\
            encoder=dict(\
                factory=HybridCNN,\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
            ),\
            decoder=dict(cnn_transpose_1d=dict(in_channels=16)),\
            gmm=dict(num_classes=10),\
            feature_extractor=dict(\
                factory=NormalizedLogMelExtractor,\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
        ))
    >>> config['encoder']['cnn_1d']['in_channels']
    2560
    >>> config['encoder']['cnn_1d']['out_channels']
    [32, 32, 32]
    >>> config['decoder']['cnn_transpose_1d']['in_channels']
    16
    >>> gmmvae = GMMVAE.from_config(config)
    >>> inputs = {'stft': torch.zeros((4, 1, 100, 257, 2)), 'seq_len': None}
    >>> outputs = gmmvae(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[0][1].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1][0][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][0][1].shape
    torch.Size([4, 16, 100])
    >>> outputs[2][0].shape
    torch.Size([4, 100, 10])
    >>> outputs[2][1].shape
    torch.Size([4, 100, 10])
    >>> review = gmmvae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose, gmm: GMM, *,
            feature_key='stft', feature_extractor=None, feature_size=None,
            label_key=None, supervised=False,
    ):
        super().__init__(
            encoder=encoder, decoder=decoder,
            feature_key=feature_key, feature_extractor=feature_extractor,
            feature_size=feature_size,
        )
        self.gmm = gmm
        self.label_key = label_key
        self.supervised = supervised

    def forward(self, inputs):
        (x, x_hat), ((mu, log_var), seq_len) = super().forward(inputs)
        qz = Normal(
            loc=mu.permute((0, 2, 1)),
            scale=torch.exp(0.5 * log_var.permute((0, 2, 1)))
        )
        log_class_posterior, log_rho = self.gmm(qz)
        return (x, x_hat), ((mu, log_var), seq_len), (log_class_posterior, log_rho)

    def review(self, inputs, outputs):
        review = super().review(inputs, outputs)
        class_labels = inputs[self.label_key] if self.supervised else None

        log_class_posterior, log_rho = outputs[-1]
        class_posterior = log_class_posterior.exp().detach()
        if class_labels is None:
            kld = -(class_posterior * log_rho).sum(-1)
            class_ce = -(class_posterior * self.gmm.log_class_probs).sum(-1)
        else:
            while class_labels.dim() < 2:
                class_labels = class_labels[..., None]
            class_labels = class_labels.expand(log_rho.shape[:-1])
            kld = -log_rho.gather(-1, class_labels[..., None]).squeeze(-1)
            class_ce = -self.gmm.log_class_probs[class_labels]

        max_class_posterior, classes = torch.max(
            torch.exp(log_class_posterior), -1
        )
        review['losses'].update(dict(
            kld=kld.mean(),
            class_ce=class_ce.mean(),
            log_class_prob=self.gmm.log_class_probs.sum()
        ))
        review['scalars'] = {}
        if self.label_key is not None:
            labels = inputs[self.label_key]
            review['scalars'].update(dict(
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
