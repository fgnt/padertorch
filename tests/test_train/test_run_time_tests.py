import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

import padertorch as pt
import paderbox as pb


class Model(pt.Model):

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(28 * 28, 10)

    def forward(self, inputs):
        clean = inputs['clean']

        if isinstance(clean, np.ndarray):
            clean = torch.tensor(clean)

        img = torch.reshape(
            clean, [-1]
        )

        return self.l(img)

    def review(self, inputs, output):
        digits = inputs['digits']

        target = torch.tensor(
            np.array(inputs['digits']).astype(np.int64)
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {
            'losses': {'ce': ce}
        }


def get_iterator(mode):
    # ToDo: remove tf dependency
    data = tf.keras.datasets.mnist.load_data()

    if mode == tf.estimator.ModeKeys.TRAIN:
        images = (data[0][0] / 255.0).astype(np.float32)
        digits = data[0][1].astype(np.int32)
    elif mode == tf.estimator.ModeKeys.EVAL:
        images = (data[1][0] / 255.0).astype(np.float32)
        digits = data[1][1].astype(np.int32)
    else:
        raise ValueError(mode, (
        tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL))

    examples = {
        f'example_{idx}':
            {'clean': example[0], 'digits': example[1]}
        for idx, example in enumerate(zip(images, digits))
    }

    return pb.database.iterator.ExamplesIterator(examples, name='mnist')


def test_1():
    it_tr = get_iterator('train')
    it_dt = get_iterator('eval')

    config = pt.Trainer.get_config(
        updates=pb.utils.nested.deflatten({
            'model.cls': Model,
            'storage_dir': None,  # will be overwritten
            'max_epochs': 10,
        })
    )

    pt.train.run_time_tests.test_run_from_config(
        config, it_tr, it_dt
    )


def test_2():
    it_tr = get_iterator('train')
    it_dt = get_iterator('eval')
    model = Model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        t = pt.Trainer(
            model, optimizer=pt.optimizer.Adam(),
            storage_dir=tmp_dir, max_epochs=2
        )

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 1:
            # One event file
            raise Exception(files_before)

        t.test_run(it_tr, it_dt)

        files_after = tuple(tmp_dir.glob('*'))
        if files_after != files_before:
            raise Exception(files_after, files_before)
