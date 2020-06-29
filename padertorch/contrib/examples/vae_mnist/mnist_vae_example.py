"""
A very simple training example for a Variational Autoencoder(VAE)
using the mnist database, sacred and the padertorch trainer

Use the following command:
    python mnist_vae_example.py with logfilepath=/path/to/dir/ epochs=500

Other options for sacred are available, refer to the config() function.
"""
import os

import padertorch as pt
import padertorch.train.optimizer as pt_opt

import sacred

import torch
import torch.nn as nn

from einops import rearrange

from padertorch.contrib.je.data.transforms import Collate
from padertorch.ops.losses.kl_divergence import gaussian_kl_divergence
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.testing.test_db import MnistDatabase

from torch.distributions import Normal

ex = sacred.Experiment()


def get_dataset(batch_size=10):
    """
    load MNIST train and test dataset

    Args:
        batch_size (int):

    Returns:
        train_set and test_set

    """
    db = MnistDatabase()
    train_set = db.get_dataset('train')
    test_set = db.get_dataset('test')

    training_data = prepare_dataset(train_set, batch_size=batch_size, training=True)
    test_data = prepare_dataset(test_set, batch_size=batch_size, training=False)

    return training_data, test_data


def prepare_dataset(dataset, batch_size=10, training=False):
    if training:
        dataset = dataset.shuffle(reshuffle=True)
    return dataset.prefetch(
        num_workers=8, buffer_size=10*batch_size
    ).batch(batch_size=batch_size).map(Collate())


class VAEModel(pt.Model):
    def __init__(self, activation_function, hidden_channels):
        super().__init__()
        self.net = VAE(activation_function, hidden_channels)

    def forward(self, inputs):
        out, mu, std = self.net(inputs['image'][:, None, :, :])

        return dict(
            prediction=out,
            mean=mu,
            std=std
        )

    def review(self, inputs, outputs):
        images = rearrange(inputs['image'][:, None, :, :], 'b c x y -> b (c x y)')
        prediction = rearrange(outputs['prediction'], 'b c x y -> b (c x y)')

        mse = nn.functional.mse_loss(prediction, images)  # reconstruction error

        q = Normal(loc=outputs['mean'], scale=outputs['std'])  # posterior distribution
        mean = torch.zeros(outputs['mean'].shape[-1])
        std = torch.ones(outputs['std'].shape[-1])
        p = Normal(loc=mean, scale=std)  # prior distribution zero mean and unit variance
        kld = gaussian_kl_divergence(q=q, p=p)  # regularisation error

        loss = mse + torch.sum(kld)

        return dict(
            loss=loss,
        )


class VAE(nn.Module):
    """ A simple Variational Autoencoder"""

    def __init__(self, activation_function, hidden_channels):
        super().__init__()
        kernel_size = 4
        stride = int(kernel_size/2)
        # encoder
        self.conv_encoder = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride),
            activation_function(),
            torch.nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                            stride=stride),
            activation_function(),
            torch.nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=kernel_size, stride=1),
            activation_function(),
        )
        self.linear_1 = torch.nn.Linear(in_features=2*2, out_features=1)
        self.linear_2 = torch.nn.Linear(in_features=2*2, out_features=1)

        # decoder
        self.transp_conv_decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size,
                                     stride=stride),
            activation_function(),
            torch.nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels,
                                     kernel_size=kernel_size, stride=stride),
            activation_function(),
            torch.nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=kernel_size,
                                     stride=stride),
        )
        self.linear_3 = torch.nn.Linear(in_features=1, out_features=2*2)

    def encode(self, x):
        x = self.conv_encoder(x)
        x = rearrange(x, 'b c x y -> b (c x y)')
        mu = self.linear_1(x)  # mean value
        std = 0.5 * self.linear_2(x)  # standard deviation
        eps = torch.randn_like(std)
        return mu + eps*std, mu, std

    def decode(self, x):
        x = self.linear_3(x)
        x = rearrange(x, 'b (x y) -> b x y', y=2, x=2)
        x = x[:, None, :, :]
        return self.transp_conv_decoder(x)

    def forward(self, input):
        x, mu, std = self.encode(input)
        y = self.decode(x)
        return y[:, :, :28, :28], mu, std


@ex.config
def config():
    epochs = 2          # Use resume=True to train for more epochs
    batch_size = 10
    resume = False      # PT: Continue from checkpoints

    assert os.environ.get('STORAGE_ROOT') is not None, 'STORAGE_ROOT is not defined in the environment variables'
    logfilepath = os.environ['STORAGE_ROOT'] + '/mnist'  # PT creates log files

    activation_function = ACTIVATION_FN_MAP['relu']  # activation function after convolutional layers
    hidden_channels = 64


@ex.automain
def main(activation_function, batch_size, epochs, hidden_channels, resume, logfilepath):
    """
    Example Training for a Variational Autoencoder uses SGD for optimation

    Args:
        epochs: number of epochs, use resume=True to train for more epochs
        batch_size: number of batches
        resume: PT: Continue from checkpoints
        logfilepath: PT creates log files
                    The structure of produced storage_dir is:
                    .
                    ├── checkpoints
                    │   ├── ckpt_7122.pth
                    │   ├── ckpt_14244.pth
                    │   ├── ckpt_best_loss.pth -> ckpt_7122.pth
                    │   ├── ckpt_latest.pth -> ckpt_14244.pth
                    │   └── ckpt_ranking.json
                    ├── events.out.tfevents.1548851867.ntsim5
    """
    # load dataset
    train_set, test_set = get_dataset(batch_size=batch_size)

    # load model
    model = VAEModel(activation_function, hidden_channels)

    trainer = pt.trainer.Trainer(
        model=model,
        storage_dir=logfilepath,
        optimizer=pt_opt.SGD(),
        stop_trigger=(epochs, 'epoch'),
        #checkpoint_trigger=(5000, 'iteration'),
        summary_trigger=(1000, 'iteration'),
    )

    trainer.register_validation_hook(validation_iterator=test_set)
    trainer.train(train_set, resume=resume)
