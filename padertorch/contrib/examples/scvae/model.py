import torch
from padertorch.contrib.je.models.vae import VAE
from padertorch.contrib.je.modules.conv import HybridCNN, CNN1d, CNN2d, MultiScaleCNN1d
from copy import deepcopy
import numpy as np


class SCVAE(VAE):
    """
    >>> config = SCVAE.get_config(dict(\
            feature_key='log_mel',\
            condition_key='speaker',\
            n_conditions=10,\
            condition_dim=16,\
            encoder=dict(\
                in_channels=80, hidden_channels=32, num_layers=3,\
                kernel_size=3, return_pool_data=True\
            ),\
            decoder=dict(in_channels=32)\
        ))
    >>> config['encoder']['out_channels']
    32
    >>> vae = SCVAE.from_config(config)
    >>> inputs = {'log_mel': torch.zeros((4, 80, 100)), 'speaker': torch.arange(4).long()}
    >>> outputs = vae(inputs)
    >>> outputs[0].shape
    torch.Size([4, 80, 100])
    >>> outputs[1][0].shape
    torch.Size([4, 16, 100])
    >>> outputs[1][1].shape
    torch.Size([4, 16, 100])
    >>> review = vae.review(inputs, outputs)
    """
    def __init__(
            self, encoder, decoder, clf, feature_key, condition_key,
            n_conditions, condition_dim, target_key=None, detach_clf=False
    ):
        super().__init__(
            encoder=encoder, decoder=decoder,
            feature_key=feature_key, target_key=target_key
        )
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = encoder.out_channels // 2
        self.condition_key = condition_key
        self.n_conditions = n_conditions
        self.condition_dim = condition_dim
        self.embed = torch.nn.Embedding(
            self.n_conditions, self.condition_dim
        )
        initrange = 1.0
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.clf = clf
        self.detach_clf = detach_clf

    def forward(self, inputs):
        params, pool_indices, shapes = self.encode(inputs)
        z1 = self.reparameterize(params)

        condition = inputs[self.condition_key].long()
        z2 = self.embed(condition)
        if z2.dim() < 3:
            z2 = z2.unsqueeze(-1)
        z2 = z2.expand((*z2.shape[:2], z1.shape[-1]))
        z = torch.cat((z1, z2), dim=1)
        x_hat = self.decode(
            z, pool_indices=pool_indices, shapes=shapes
        )
        mu = params[0]
        mu = mu.detach() if self.detach_clf else 2 * mu.detach() - mu  # flip grad
        logits = self.clf(mu)
        return x_hat, params, logits

    def review(self, inputs, outputs):
        review = super().review(inputs, outputs)
        condition = inputs[self.condition_key].long()
        logits = outputs[-1]
        if condition.dim() == 1:
            condition = condition.unsqueeze(-1)
        condition = condition.expand(
            (condition.shape[0], logits.shape[-1])
        )
        ce = torch.nn.CrossEntropyLoss(reduction='none')(logits, condition)
        predictions = torch.argmax(outputs[-1], dim=1)
        review['losses']['ce'] = ce.mean()
        review['scalars'] = dict(
            predictions=predictions,
            labels=condition
        )
        review['histograms']['ce_'] = ce.mean()
        return review

    def modify_summary(self, summary):
        if 'labels' in summary['scalars']:
            labels = summary['scalars'].pop('labels')
            predictions = summary['scalars'].pop('predictions')
            summary['scalars']['accuracy'] = (
                    np.array(predictions) == np.array(labels)
            ).mean()
        summary = super().modify_summary(summary)
        return summary

    def conversion(self, features, condition):
        out = self.forward({self.feature_key: features, self.condition_key: condition})
        return out[0]

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['encoder']['factory'] = MultiScaleCNN1d
        if config['encoder']['factory'] == HybridCNN:
            config['encoder'].update({
                'cnn_2d': {'factory': CNN2d},
                'cnn_1d': {
                    'factory': CNN1d,
                    'out_channels': 2*(config['decoder']['cnn_transpose_1d']['in_channels'] - config['condition_dim'])
                }
            })
        if config['encoder']['factory'] in (CNN1d, MultiScaleCNN1d):
            config['encoder']['out_channels'] = 2 * (config['decoder']['in_channels'] - config['condition_dim'])
            config['clf'] = {}
            config['clf'].update(config['encoder']['factory'].get_transpose_config(config['encoder']))
            config['clf']['in_channels'] = config['encoder']['out_channels'] // 2
            config['clf']['out_channels'] = config['n_conditions']
        config['decoder'].update(
            config['encoder']['factory'].get_transpose_config(config['encoder'])
        )
