"""
A very simple training example for a Variational Autoencoder(VAE)
using the mnist database, sacred and the padertorch trainer

Use the following command:
    python mnist_vae_example.py with logfilepath=/path/to/dir/ epochs=500

Other options for sacred are available, refer to the config() function.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

import padertorch as pt
import padertorch.train.optimizer as pt_opt

import sacred

import torch
import torch.nn as nn

from einops import rearrange

from paderbox.utils.timer import timeStamped

from padertorch.data.utils import collate_fn
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.testing.test_db import MnistDatabase
from padertorch.utils import to_numpy

from pathlib import Path

ex = sacred.Experiment()


def get_dataset(batch_size):
    """load MNIST train and test dataset"""
    db = MnistDatabase()
    train_set = db.get_dataset('train')
    test_set = db.get_dataset('test')

    training_data = prepare_dataset(batch_size, train_set, training=True)
    test_data = prepare_dataset(batch_size, test_set, training=False)

    return training_data, test_data


def prepare_dataset(batch_size, dataset, training=False):
    """preprocesses the dataset, shuffle the training set"""
    if training:
        dataset = dataset.shuffle(reshuffle=True)

    def Collate(batch):
        batch = collate_fn(batch)
        batch['image'] = np.array([tensor for tensor in batch['image']])
        return batch

    return dataset.prefetch(
        num_workers=1, buffer_size=10*batch_size
    ).batch(batch_size=batch_size).map(Collate)


class VAEModel(pt.Model):
    def __init__(self, activation_function, hidden_channels, kernel_size):
        super().__init__()
        self.net = VAE(activation_function, hidden_channels, kernel_size)

    def forward(self, inputs):
        images = inputs['image'][:, None, :, :]
        out, mu, log_var = self.net(images)

        return dict(
            inputs=images,
            mean=mu,
            prediction=out,
            log_var=log_var
        )

    def review(self, inputs, outputs):
        images = rearrange(outputs['inputs'], 'b c x y -> b (c x y)')
        prediction = rearrange(outputs['prediction'], 'b c x y -> b (c x y)')

        mse = nn.functional.mse_loss(prediction, images)  # Reconstruction error

        # Kullback leibler divergence
        kld = -0.5 * (1 + outputs['log_var'] - outputs['mean'].pow(2) - outputs['log_var'].exp()).sum()

        loss = mse + torch.sum(kld)

        return dict(
            loss=loss,
            images=self.add_images(outputs),
        )

    def add_images(self, batch):
        """Creates a dictionary of input and reconstructed images. The results can be seen in tensorboard"""
        images = dict()
        if batch['inputs'].size()[0] > 3:
            num_images = 3
        else:
            num_images = 1

        for name in ['inputs', 'prediction']:
            for idx in range(num_images):
                image = batch[name][idx, :, :, :].squeeze()
                image = to_numpy(image, detach=True)

                image = np.clip(image * 255, 0, 255)
                image = image.astype(np.uint8)

                images[name + '_example_' + str(idx)] = image[None]
        return images

    def create_mnist_data(self, logfilepath):
        """Samples from a unit distribution and the decoder creat a new image."""
        num_images = 3  # Number of images that should be created
        model = self.net.double()
        checkpoint = torch.load(logfilepath.joinpath("checkpoints/ckpt_best_loss.pth"))

        try:
            model.load_state_dict(checkpoint['model'])
        except:
            states = dict()
            for keys, value in zip(checkpoint['model'].keys(), checkpoint['model'].values()):
                states[keys[4::]] = value
            model.load_state_dict(states)

        inputs = np.random.normal(size=(num_images, 1))
        prediction = model.decode(torch.tensor(inputs).cuda()).cpu()[:, :, :28, :28]

        Path.mkdir(logfilepath.joinpath('Created_Images'))
        for idx in range(prediction.shape[0]):
            plt.imshow(prediction[idx].detach().numpy().squeeze())
            plt.savefig(logfilepath.joinpath('Created_Images/example_' + str(idx) + '_.png'))


class VAE(nn.Module):
    """ A simple Variational Autoencoder"""
    def __init__(self, activation_function, hidden_channels, kernel_size):
        super().__init__()
        self.n_layer = 3
        self.padding = 1
        self.latent_dimension = 28  # Image size 28*28
        activation_function = ACTIVATION_FN_MAP[activation_function]
        stride = int(kernel_size/2)

        for idx in range(self.n_layer):
            self.latent_dimension = int(
                (self.latent_dimension + 2 * self.padding - (kernel_size - 1) - 1) / stride + 1)

        # Encoder
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=hidden_channels[0], kernel_size=kernel_size, stride=stride,
                            padding=self.padding),
            activation_function(),
            torch.nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=kernel_size,
                            stride=stride, padding=self.padding),
            activation_function(),
            torch.nn.Conv2d(in_channels=hidden_channels[1], out_channels=1, kernel_size=kernel_size, stride=stride,
                            padding=self.padding),
            activation_function(),
        )
        self.linear_1 = torch.nn.Linear(in_features=self.latent_dimension**2, out_features=1)
        self.linear_2 = torch.nn.Linear(in_features=self.latent_dimension**2, out_features=1)

        # Decoder
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=1, out_channels=hidden_channels[1], kernel_size=kernel_size,
                                     stride=stride),
            activation_function(),
            torch.nn.ConvTranspose2d(in_channels=hidden_channels[1], out_channels=hidden_channels[0],
                                     kernel_size=kernel_size, stride=stride),
            activation_function(),
            torch.nn.ConvTranspose2d(in_channels=hidden_channels[0], out_channels=1, kernel_size=kernel_size,
                                     stride=stride),
        )
        self.linear_3 = torch.nn.Linear(in_features=1, out_features=self.latent_dimension**2)

    def decode(self, x):
        x = self.linear_3(x)
        x = rearrange(x, 'b (x y) -> b x y', y=self.latent_dimension, x=self.latent_dimension)
        x = x[:, None, :, :]
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)
        x = rearrange(x, 'b c x y -> b (c x y)')
        mu = self.linear_1(x)  # Mean value
        log_var = self.linear_2(x)  # Variance
        return mu, log_var

    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        x = self.sampling(mu, log_var)
        y = self.decode(x)
        return y[:, :, :28, :28], mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)
        return mu + std * eps


@ex.config
def config():
    activation_function = 'relu'  # Activation function after convolutional layers
    batch_size = 10
    epochs = 1  # Use resume=True to train for more epochs
    hidden_channels = [32, 64]
    kernel_size = 2
    assert kernel_size < 6, 'maximum kernel_size is 5'
    logfilepath = Path(os.environ['STORAGE_ROOT'] + '/vae_mnist/' + timeStamped('')[1:])  # PT creates log files
    assert os.environ.get('STORAGE_ROOT') is not None, 'STORAGE_ROOT is not defined in the environment variables'
    resume = False  # PT: Continue from checkpoints



@ex.automain
def main(activation_function, batch_size, epochs, hidden_channels, kernel_size, logfilepath, resume):
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
                    │   ├── ckpt_7122.pth
                    │   ├── ckpt_14244.pth
                    │   ├── ckpt_best_loss.pth -> ckpt_7122.pth
                    │   ├── ckpt_latest.pth -> ckpt_14244.pth
                    │   └── ckpt_ranking.json
                    ├── events.out.tfevents.1548851867.ntsim5
                    ├── created_images
                        ├── example_image.png
    """
    # load dataset
    train_set, test_set = get_dataset(batch_size=batch_size)

    # load model
    model = VAEModel(activation_function, hidden_channels, kernel_size)

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

    model.create_mnist_data(logfilepath)
