import tempfile
from pathlib import Path
import contextlib

import numpy as np
import torch

import padertorch as pt
import paderbox as pb


class Model(pt.Model):

    def __init__(self):
        super().__init__()
        # self.norm = torch.nn.BatchNorm1d(28)
        self.drop = torch.nn.Dropout(0.5)
        self.l = torch.nn.Linear(28 * 28, 10)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        # clean = self.norm(clean)
        image = torch.reshape(clean, [-1])
        image = self.drop(image)
        return self.l(image)

    def review(self, inputs, output):
        digits = inputs['digit']

        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'losses': {'loss': ce}}


def get_iterators():
    db = pb.database.mnist.MnistDatabase()
    return (
        db.get_iterator_by_names('train'),
        db.get_iterator_by_names('test'),
    )


def test_single_model():
    it_tr, it_dt = get_iterators()

    config = pt.Trainer.get_config(
        updates=pb.utils.nested.deflatten({
            'model.factory': Model,
            'storage_dir': None,  # will be overwritten
            'max_trigger': None,  # will be overwritten
        })
    )

    pt.train.run_time_tests.test_run_from_config(
        config, it_tr, it_dt,
        test_with_known_iterator_length=False,
    )
    pt.train.run_time_tests.test_run_from_config(
        config, it_tr, it_dt,
        test_with_known_iterator_length=True,
    )


@contextlib.contextmanager
def assert_dir_unchanged_after_context(tmp_dir):
    tmp_dir = Path(tmp_dir)
    files_before = tuple(tmp_dir.glob('*'))
    if len(files_before) != 0:
        # no event files
        raise Exception(files_before)

    yield

    files_after = tuple(tmp_dir.glob('*'))
    if files_after != files_before:
        raise Exception(files_after, files_before)


def test_single_model_2():
    it_tr, it_dt = get_iterators()
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, max_trigger=(2., 'epoch')
        )
        with assert_dir_unchanged_after_context(tmp_dir):
            t.test_run(it_tr, it_dt)


class VAE(pt.Model):

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(28 * 28, 32)
        self.dec = torch.nn.Linear(16, 28 * 28)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])

        h = self.enc(image)
        mu, logvar = torch.split(h, h.shape[-1]//2, dim=-1)
        qz = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))
        if self.training:
            z = qz.rsample()
        else:
            z = mu
        x_hat = self.dec(z)
        return x_hat, mu, logvar

    def review(self, inputs, outputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])

        mse = (image - outputs[0]).pow(2).sum()
        return {'losses': {'mse': mse}}


class StandardNormal(Model):
    def forward(self, inputs):
        return ()

    def review(self, inputs, outputs):
        mean, logvar = inputs
        kld = -0.5 * (
            1 + logvar - mean.pow(2) - torch.exp(logvar)
        ).sum()
        return dict(
            losses=dict(
                kld=kld
            )
        )


class DictTrainer(pt.trainer.Trainer):
    def _step(self, example):
        example = pt.data.batch_to_device(
            example, self.device != 'cpu', self.device
        )
        review = dict()
        vae_out = self.model['vae'](example)
        pb.utils.nested.nested_update(review, self.model['vae'].review(
            example, vae_out))
        latent_out = self.model['latent'](vae_out[1:])
        pb.utils.nested.nested_update(review, self.model['latent'].review(
            vae_out[1:], latent_out))
        return (vae_out, latent_out), review


def test_dict_of_models():
    it_tr, it_dt = get_iterators()

    models = {'vae': VAE(), 'latent': StandardNormal()}
    optimizers = {'vae': pt.optimizer.Adam(), 'latent': None}
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = DictTrainer(
            models, storage_dir=tmp_dir, optimizer=optimizers,
            loss_weights={'mse': 1., 'kld': 1.}
        )

        with assert_dir_unchanged_after_context(tmp_dir):
            trainer.test_run(it_tr, it_dt)
