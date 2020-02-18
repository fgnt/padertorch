import numpy as np
import torch
from einops import rearrange
from padertorch.base import Model
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.dist import GMM
from padertorch.contrib.je.modules.features import MelTransform
from padertorch.contrib.je.modules.hybrid import HybridCNN, HybridCNNTranspose
from padertorch.contrib.je.modules.norm import Norm
from sklearn import metrics
from torch.distributions import Normal
from torchvision.utils import make_grid


class VAE(Model):
    """
    >>> config = VAE.get_config(dict(\
            encoder=dict(\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
                return_pool_data=True,\
            ),\
            decoder=dict(cnn_transpose_1d=dict(in_channels=16)),\
            sample_rate=16000,\
            fft_length=512,\
            n_mels=80,\
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
    >>> outputs[0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[2].shape
    torch.Size([4, 16, 100])
    >>> outputs[3][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[3][1].shape
    torch.Size([4, 16, 100])
    >>> review = vae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose, *,
            sample_rate, fft_length, n_mels, fmin=50, fmax=None,
    ):
        super().__init__()
        # allow joint optimization of encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.n_params = 2
        self.mel_transform = MelTransform(
            n_mels=n_mels, sample_rate=sample_rate, fft_length=fft_length,
            fmin=fmin, fmax=fmax,
        )
        self.in_norm = Norm(
            data_format='bcft',
            shape=(None, 1, n_mels, None),
            statistics_axis='bt',
            scale=True,
            independent_axis=None,
            momentum=None,
            interpolation_factor=1.
        )

    def feature_extraction(self, x, seq_len=None):
        x = self.mel_transform(torch.sum(x**2, dim=(-1,))).transpose(-2, -1)
        x = self.in_norm(x, seq_len=seq_len)
        if isinstance(self.encoder, CNN1d):
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x

    def inverse_feature_extraction(self, x):
        if x.dim() == 3:
            b, f, t = x.shape
            x = x.view((b, -1, self.mel_transform.n_mels, t))
        return self.mel_transform.inverse(
            (
                torch.sqrt(self.in_norm.running_var) * x
                + self.in_norm.running_mean
            ).transpose(-2, -1)
        )

    def encode(self, x, seq_len=None):
        if self.encoder.return_pool_data:
            h, seq_len, shapes, lengths, pool_indices = self.encoder(
                x, seq_len=seq_len
            )
        else:
            h, seq_len = self.encoder(x)
            shapes = lengths = pool_indices = None
        assert not h.shape[1] % self.n_params
        params = tuple(torch.split(h, h.shape[1] // self.n_params, dim=1))
        return params, seq_len, shapes, lengths, pool_indices

    def reparameterize(self, params):
        mu, logvar = params[:2]
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, seq_len=None, shapes=None, lengths=None, pool_indices=None):
        x_hat, seq_len = self.decoder(
            z, seq_len=seq_len, out_shapes=shapes, out_lengths=lengths,
            pool_indices=pool_indices
        )
        return x_hat, seq_len  # (B, C, F, T)

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = inputs['seq_len']
        x = self.feature_extraction(x, seq_len)
        params, seq_len, shapes, lengths, pool_indices = self.encode(x, seq_len)
        z = self.reparameterize(params)
        x_hat, seq_len = self.decode(
            z, seq_len, shapes=shapes, lengths=lengths, pool_indices=pool_indices
        )
        return x, x_hat, z, params

    def review(self, inputs, outputs):
        # visualization
        seq_len = inputs['seq_len']
        x, x_hat, z, params, *_ = outputs
        (mu, log_var) = params[:2]
        mse = (x - x_hat).pow(2).sum(dim=((1, 2) if x.dim() == 4 else 1))
        kld = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)

        if seq_len is not None:
            seq_len = torch.Tensor(seq_len)[:, None].to(mse.device)
            idx = torch.arange(mse.shape[-1]).to(mse.device)
            mask = (idx < seq_len.long()).float()
            mse = (mse * mask).sum(dim=-1) / (seq_len.squeeze(-1) + 1e-6)
            kld = (kld * mask).sum(dim=-1) / (seq_len.squeeze(-1) + 1e-6)

        x_hat = x_hat.contiguous()
        mu = mu.contiguous()
        review = dict(
            losses=dict(
                mse=mse.mean(),
                kld=kld.mean(),
            ),
            histograms=dict(
                mse_=mse,
                kld_=kld.flatten(),
                mu_=mu.flatten(),
                log_var_=log_var.flatten(),
            ),
            images=dict(
                targets=x[:3],
                latents=mu[:3],
                reconstructions=x_hat[:3],
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
        if config['encoder']['factory'] == CNN1d:
            config['encoder']['in_channels'] = config['n_mels']
            config['encoder']['out_channels'] = 2 * config['decoder']['in_channels']
        config['decoder'].update(
            config['encoder']['factory'].get_transpose_config(config['encoder'])
        )


class GMMVAE(VAE):
    """
    >>> config = GMMVAE.get_config(dict(\
            encoder=dict(\
                input_size=80,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
                return_pool_data=True,\
            ),\
            decoder=dict(cnn_transpose_1d=dict(in_channels=16)),\
            gmm=dict(num_classes=10),\
            sample_rate=16000,\
            fft_length=512,\
            n_mels=80,\
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
    >>> outputs[0].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[1].shape
    torch.Size([4, 1, 80, 100])
    >>> outputs[2].shape
    torch.Size([4, 16, 100])
    >>> outputs[3][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[3][1].shape
    torch.Size([4, 16, 100])
    >>> outputs[4][0].shape
    torch.Size([4, 100, 10])
    >>> outputs[4][1].shape
    torch.Size([4, 100, 10])
    >>> review = gmmvae.review(inputs, outputs)
    """
    def __init__(
            self, encoder: HybridCNN, decoder: HybridCNNTranspose, gmm: GMM, *,
            sample_rate, fft_length, n_mels, fmin=50, fmax=None,
            label_key=None, supervised=False
    ):
        super().__init__(
            encoder=encoder, decoder=decoder,
            sample_rate=sample_rate, fft_length=fft_length, n_mels=n_mels,
            fmin=fmin, fmax=fmax,
        )
        self.gmm = gmm
        self.label_key = label_key
        self.supervised = supervised

    def forward(self, inputs):
        x, x_hat, z, (mu, log_var) = super().forward(inputs)
        qz = Normal(
            loc=mu.permute((0, 2, 1)),
            scale=torch.exp(0.5 * log_var.permute((0, 2, 1)))
        )
        log_class_posterior, log_rho = self.gmm(qz)
        return x, x_hat, z, (mu, log_var), (log_class_posterior, log_rho)

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
