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
        return {'losses': {'loss': ce}}


def get_iterators():
    db = pb.database.mnist.MnistDatabase()
    return (
        db.get_iterator_by_names('train'),
        db.get_iterator_by_names('test'),
    )


def test_single_model():
    it_tr, it_dt = get_iterators()
    it_tr = it_tr[:2]
    it_dt = it_dt[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        config = pt.Trainer.get_config(
            updates=pb.utils.nested.deflatten({
                'model.cls': Model,
                'storage_dir': str(tmp_dir),
                'max_trigger': (2, 'epoch'),
                'keep_all_checkpoints': True,
            })
        )

        t = pt.Trainer.from_config(config)

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            # no event file
            raise Exception(files_before)

        t.train(
            train_iterator=it_tr,
            validation_iterator=it_dt,
            hooks=None,
            metrics={'loss': 'min'},
            n_best_checkpoints=1,
        )

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 2, files_after
        for file in files_after:
            if 'tfevents' in file.name:
                file_bytes = file.read_bytes()
                # 1333
                assert len(file_bytes) > 1000, len(file_bytes)
                assert len(file_bytes) < 1500, len(file_bytes)
            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 4, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_4.pth', 'ckpt_state.json'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)

        config['kwargs']['max_trigger'] = (4, 'epoch')
        config['kwargs']['init_checkpoint'] = tmp_dir / 'checkpoints' / 'ckpt_4.pth'
        t = pt.Trainer.from_config(config)

        t.train(
            train_iterator=it_tr,
            validation_iterator=it_dt,
            hooks=None,
            metrics={'loss': 'min'},
            n_best_checkpoints=1,
        )

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 2, files_after
        for file in files_after:
            if 'tfevents' in file.name:
                file_bytes = file.read_bytes()
                # 913
                assert len(file_bytes) > 800, len(file_bytes)
                assert len(file_bytes) < 1000, len(file_bytes)
            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 6, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    *[f'ckpt_{i}.pth'for i in [1, 2, 4, 6, 8]],
                    'ckpt_state.json',
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)
