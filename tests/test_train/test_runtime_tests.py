import tempfile
from pathlib import Path
import contextlib

import pytest
import numpy as np
import torch

import padertorch as pt
import paderbox as pb
from padertorch.testing.test_db import MnistDatabase


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
        if not isinstance(digits, int):
            digits = digits.cpu().numpy()

        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'losses': {'loss': ce}}


def get_datasets():
    db = MnistDatabase()
    return (
        db.get_dataset('train'),
        db.get_dataset('test'),
    )


def test_single_model():
    tr_dataset, dt_dataset = get_datasets()

    config = pt.Trainer.get_config(
        updates=pb.utils.nested.deflatten({
            'model.factory': Model,
            'storage_dir': None,  # will be overwritten
            'stop_trigger': (1, 'epoch'),  # will be overwritten
        })
    )

    pt.train.runtime_tests.test_run_from_config(
        config, tr_dataset, dt_dataset,
        test_with_known_iterator_length=False,
    )
    pt.train.runtime_tests.test_run_from_config(
        config, tr_dataset, dt_dataset,
        test_with_known_iterator_length=True,
    )


@contextlib.contextmanager
def assert_dir_unchanged_after_context(tmp_dir):
    tmp_dir = Path(tmp_dir)
    files_before = tuple(tmp_dir.glob('*'))
    if len(files_before) != 0:
        raise Exception(files_before)

    yield

    files_after = tuple(tmp_dir.glob('*'))
    if files_after != files_before:
        raise Exception(files_after, files_before)


def test_single_model_dir_unchanged():
    tr_dataset, dt_dataset = get_datasets()
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, stop_trigger=(2, 'epoch')
        )
        with assert_dir_unchanged_after_context(tmp_dir):
            t.test_run(tr_dataset, dt_dataset)


# def test_lr_scheduler():
#     tr_dataset, dt_dataset = get_iterators()
#
#     config = pt.Trainer.get_config(
#         updates=pb.utils.nested.deflatten({
#             'model.factory': Model,
#             'lr_scheduler.factory': pt.train.optimizer.StepLR,
#             'storage_dir': None,  # will be overwritten
#             'stop_trigger': None,  # will be overwritten
#         })
#     )
#
#     pt.train.runtime_tests.test_run_from_config(
#         config, tr_dataset, dt_dataset,
#         test_with_known_iterator_length=False,
#     )
#     pt.train.runtime_tests.test_run_from_config(
#         config, tr_dataset, dt_dataset,
#         test_with_known_iterator_length=True,
#     )


class ZeroGradModel(pt.Model):

    def __init__(self):
        super().__init__()
        # self.norm = torch.nn.BatchNorm1d(28)
        self.l = torch.nn.Linear(28 * 28, 10)
        self.drop = torch.nn.Dropout(1)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        # clean = self.norm(clean)
        image = torch.reshape(clean, [-1])
        # drop all values -> parameters receive a 0 as update
        return self.drop(self.l(image))

    def review(self, inputs, output):
        digits = inputs['digit']
        if not isinstance(digits, int):
            digits = digits.cpu().numpy()

        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'losses': {'loss': ce}}


def test_single_grad_check():
    tr_dataset, dt_dataset = get_datasets()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            Model(),
            optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, stop_trigger=(2, 'epoch')
        )
        t.test_run(tr_dataset, dt_dataset)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            ZeroGradModel(), optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, stop_trigger=(2, 'epoch')
        )
        # AssertionError: The loss of the model did not change between two validations.
        with pytest.raises(AssertionError):
            t.test_run(tr_dataset, dt_dataset)


def test_single_virtual_minibatch():
    tr_dataset, dt_dataset = get_datasets()
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, stop_trigger=(2, 'epoch'),
            virtual_minibatch_size=4
        )
        with assert_dir_unchanged_after_context(tmp_dir):
            t.test_run(tr_dataset, dt_dataset)


class AE(pt.Model):

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(28 * 28, 16)
        self.dec = torch.nn.Linear(16, 28 * 28)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])
        z = self.enc(image)
        x_hat = self.dec(z)
        return x_hat

    def review(self, inputs, outputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])

        mse = (image - outputs[0]).pow(2).sum()
        return {'loss': mse}


def test_multiple_optimizers():
    tr_dataset, dataset_dt = get_datasets()

    model = AE()
    optimizers = {
        'enc': pt.optimizer.Adam(),
        'dec': pt.optimizer.Adam()
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = pt.Trainer(
            model, storage_dir=tmp_dir, optimizer=optimizers
        )

        with assert_dir_unchanged_after_context(tmp_dir):
            trainer.test_run(tr_dataset, dataset_dt)
