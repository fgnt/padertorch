import torch
from padertorch.base import Model
from padertorch.contrib.je.conv import CNN
from torch import nn
from torchvision import utils as vutils


class AE(Model):
    def __init__(
            self, encoder: CNN, decoder: CNN, feature_key="spectrogram",
            encoder_condition=None, decoder_condition=None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_key = feature_key
        self.encoder_condition = encoder_condition
        self.decoder_condition = decoder_condition
        if encoder_condition is not None:
            key, num_conditions = encoder_condition
            self.embed_encoder_cond = torch.nn.Embedding(
                num_conditions, encoder.condition_size
            )
            initrange = 1.0
            self.embed_encoder_cond.weight.data.uniform_(-initrange, initrange)
        if decoder_condition is not None:
            key, num_conditions = decoder_condition
            self.embed_decoder_cond = torch.nn.Embedding(
                num_conditions, decoder.condition_size
            )
            initrange = 1.0
            self.embed_decoder_cond.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, h=None):
        z, pooling_data = self.encoder(x, h)
        z = torch.split(z, self.decoder.input_size, dim=1)
        if pooling_data[-1][0] is not None:
            pooling_data[-1] = (
                torch.split(
                    pooling_data[-1][0], self.decoder.input_size, dim=1
                )[0],
                pooling_data[-1][0]
            )
        return z, pooling_data

    def decode(self, z, h=None, pooling_data=None):
        return self.decoder(z, h, pooling_data)

    def forward(self, inputs, command=None):
        if command == "encode" or command is None:
            x = inputs[self.feature_key]
            h = None
            if self.encoder_condition is not None:
                h = self.embed_encoder_cond(inputs[self.encoder_condition[0]])
            z, pooling_data = self.encode(x, h)
            if command == "encode":
                return z, pooling_data
            z = z[0]
        elif command == "decode":
            z, pooling_data = inputs
        else:
            raise ValueError
        h = None
        if self.decoder_condition is not None:
            h = self.embed_decoder_cond(inputs[self.decoder_condition[0]])
        x_hat = self.decode(z, h, pooling_data=pooling_data[::-1])
        return x_hat, z

    def review(self, inputs, outputs):
        # visualization
        x = inputs[self.feature_key]
        mse = (x - outputs[0]).pow(2).sum(dim=1)
        features = vutils.make_grid(
            x[:9].flip(1).unsqueeze(1),
            normalize=True, scale_each=False, nrow=3)
        latents = vutils.make_grid(
            outputs[1][:9].flip(1).unsqueeze(1),
            normalize=True, scale_each=False, nrow=3)
        reconstructions = vutils.make_grid(
            outputs[0][:9].flip(1).unsqueeze(1),
            normalize=True, scale_each=False, nrow=3)
        return dict(
            losses=dict(
                mse=mse.mean(),
            ),
            histograms=dict(
                mse_=mse,
            ),
            images=dict(
                features=features,
                latents=latents,
                reconstructions=reconstructions,
            )
        )

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['encoder'] = {'factory': CNN}
        config['decoder'] = {'factory': CNN, 'transpose': True}
        config['decoder']['output_size'] = config['encoder']['input_size']
        config['decoder']['num_layers'] = config['encoder']['num_layers']
        config['decoder']['pooling'] = config['encoder']['pooling']
        for key in [
            'hidden_sizes', 'kernel_sizes', 'n_scales', 'dilations',
            'strides', 'pool_sizes', 'paddings'
        ]:
            if isinstance(config['encoder'][key], (list, tuple)):
                config['decoder'][key] = config['encoder'][key][::-1]
            else:
                config['decoder'][key] = config['encoder'][key]
