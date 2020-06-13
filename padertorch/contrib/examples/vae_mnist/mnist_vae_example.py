"""
A very simple training example for a Variational Autoencoder(VAE)
using the mnist database, sacred and the padertorch trainer

Use the following command:
    python mnist_vae_example.py with logfilepath=/path/to/dir/ epochs=500

Other options for sacred are available, refer to the config() function.
"""
import os

import padertorch
import padertorch.train.optimizer as pt_opt

import sacred

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from einops import rearrange

ex = sacred.Experiment()


def get_dataset(batch_size=128):
    """
    load MNIST dataset and store it in ./data

    Args:
        batch_size (int):

    Returns:
        trainloader and testloader

    """
    tf = transforms.Compose([transforms.ToTensor()])

    # Get Datasets
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=tf)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


class PadertorchModel(padertorch.base.Model):
    MSE_loss = nn.MSELoss()

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        images = inputs[0]
        out, mu, std = self.net(images)

        return dict(
            prediction=out,
            mean=mu,
            std=std
        )

    def review(self, inputs, outputs):
        images = rearrange(inputs[0], 'b c x y -> b (c x y)')
        prediction = rearrange(outputs['prediction'], 'b c x y -> b (c x y)')

        MSE = self.MSE_loss(prediction, images)  # reconstruction error
        # Kullback-Leibler divergence
        KLD = -0.5 * torch.sum(1 + outputs['std'] - outputs['mean'].pow(2) - outputs['std'].exp())

        loss = MSE + KLD

        return dict(
            loss=loss,
        )


class VAE(nn.Module):
    """ A simple Variational Autoencoder"""

    def __init__(self):
        super(VAE, self).__init__()
        self.kernel_size = 4

        # encoder
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=self.kernel_size,
                                      stride=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size,
                                      stride=2)
        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size,
                                      stride=1)
        self.linear_1 = torch.nn.Linear(in_features=2*2, out_features=1)
        self.linear_2 = torch.nn.Linear(in_features=2*2, out_features=1)
        self.lrelu = torch.nn.LeakyReLU()

        # decoder
        self.linear_3 = torch.nn.Linear(in_features=1, out_features=2*2)

        self.transconv_1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=self.kernel_size,
                                                   stride=2)
        self.transconv_2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=self.kernel_size,
                                                   stride=2)
        self.transconv_3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=self.kernel_size,
                                                   stride=2)
        self.relu = torch.nn.ReLU()

    def encode(self, x):
        x = self.conv_1(x)
        x = self.lrelu(x)
        x = self.conv_2(x)
        x = self.lrelu(x)
        x = self.conv_3(x)
        x = self.lrelu(x)
        x = rearrange(x, 'b c x y -> b (c x y)')
        mu = self.linear_1(x)  # mean value
        std = 0.5 * self.linear_2(x)  # standard deviation
        eps = torch.randn_like(std)
        return mu + eps*std, mu, std

    def decode(self, x):
        x = self.linear_3(x)
        x = rearrange(x, 'b (x y) -> b x y', y=2, x=2)
        x = x[:, None, :, :]
        x = self.transconv_1(x)
        x = self.relu(x)
        x = self.transconv_2(x)
        x = self.relu(x)
        x = self.transconv_3(x)
        return x

    def forward(self, input):
        x, mu, std = self.encode(input)
        y = self.decode(x)
        return y[:, :, :28, :28], mu, std


@ex.config
def config():
    epochs = 2          # Use resume=True to train for more epochs
    batch_size = 128
    resume = False      # PT: Continue from checkpoints
    logfilepath = os.environ['STORAGE_ROOT'] + '/mnist'  # PT creates log files


@ex.automain
def main(epochs, batch_size, resume, logfilepath):
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
    trainloader, testloader = get_dataset(batch_size=batch_size)

    # Setup net and model
    net = VAE()

    # load model
    model = PadertorchModel(net)

    trainer = padertorch.trainer.Trainer(
        model=model,
        storage_dir=logfilepath,
        optimizer=pt_opt.SGD(),
        stop_trigger=(epochs, 'epoch'),
        #checkpoint_trigger=(5000, 'iteration'),
        summary_trigger=(1000, 'iteration'),
    )
    trainer.register_validation_hook(validation_iterator=testloader)
    try:
        trainer.train(trainloader, resume=resume)
    except Exception:
        print('#' * 1000)
        raise
