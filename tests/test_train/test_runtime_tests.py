import tempfile
from pathlib import Path
import contextlib
import copy

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

    pt.train.runtime_tests.test_run_from_config(
        config, it_tr, it_dt,
        test_with_known_iterator_length=False,
    )
    pt.train.runtime_tests.test_run_from_config(
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


def test_single_model_dir_unchanged():
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


def test_single_model_state_dict_unchanged():
    it_tr, it_dt = get_iterators()
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, max_trigger=(2., 'epoch')
        )
        pre_state_dict = copy.deepcopy(t.state_dict())
        with assert_dir_unchanged_after_context(tmp_dir):
            t.test_run(it_tr, it_dt)
        post_state_dict = copy.deepcopy(t.state_dict())

        pre_state_dict = pb.utils.nested.flatten(pre_state_dict)
        post_state_dict = pb.utils.nested.flatten(post_state_dict)

        assert pre_state_dict.keys() == post_state_dict.keys()

        pre_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, pre_state_dict)
        post_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, post_state_dict)

        np.testing.assert_equal(pre_state_dict, post_state_dict)


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
    it_tr, it_dt = get_iterators()

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
            trainer.test_run(it_tr, it_dt)
