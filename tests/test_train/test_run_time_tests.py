import tempfile
from pathlib import Path

import numpy as np
import torch

import padertorch as pt
import paderbox as pb


class Model(pt.Model):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(28 * 28, 10)

    def forward(self, inputs):
        clean = inputs['image']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        image = torch.reshape(clean, [-1])

        return self.l(image)

    def review(self, inputs, output):
        digits = inputs['digit']

        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'losses': {'ce': ce}}


def get_iterators():
    db = pb.database.mnist.MnistDatabase()
    return (
        db.get_iterator_by_names('train'),
        db.get_iterator_by_names('test'),
    )


def test_1():
    it_tr, it_dt = get_iterators()

    config = pt.Trainer.get_config(
        updates=pb.utils.nested.deflatten({
            'model.cls': Model,
            'storage_dir': None,  # will be overwritten
            'max_trigger': (10, 'epoch')
        })
    )

    pt.train.run_time_tests.test_run_from_config(
        config, it_tr, it_dt
    )


def test_2():
    it_tr, it_dt = get_iterators()
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, max_trigger=(2., 'epoch')
        )

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            # no event files
            raise Exception(files_before)

        t.test_run(it_tr, it_dt)

        files_after = tuple(tmp_dir.glob('*'))
        if files_after != files_before:
            raise Exception(files_after, files_before)
