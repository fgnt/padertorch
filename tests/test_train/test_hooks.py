import sys
import types
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock
import numpy as np

from IPython.lib.pretty import pretty
import pytest
import tensorboardX
import torch

import padertorch as pt
import paderbox as pb

from padertorch.testing.windows import skip_on_windows


class ProgresbarHookTest(unittest.TestCase):
    num_epochs = 2
    num_iterations = 5
    iterator_length = 4

    def test_max_iteration(self):
        self.train_loop_iteration(self.num_iterations, self.iterator_length)

    def test_max_epoch(self):
        self.train_loop_epoch(self.num_epochs, self.iterator_length)

    def test_max_epoch_no_iteration_length(self):
        self.train_loop_epoch(self.num_epochs, None)

    def train_loop_iteration(self, length, max_it_len):
        progressbar_hook = pt.train.hooks.ProgressBarHook(
            stop_trigger=(length, 'iteration'), max_it_len=max_it_len,
            update_interval=1
        )
        trainer = types.SimpleNamespace()
        trainer.iteration = 0
        iteration = 0
        try:
            for epoch in range(self.num_epochs):  # infinite loop
                trainer.epoch = epoch
                progressbar_hook.pre_step(trainer)
                print(trainer.iteration)
                for idx in range(self.iterator_length):
                    trainer.iteration = idx
                    if not idx == 0:
                        progressbar_hook.pre_step(trainer)
                    if iteration >= length:
                        raise pt.train.hooks.StopTraining
                    assert idx < self.num_iterations, (idx, epoch)
                    progressbar_hook.post_step(trainer, None, None, {
                        'loss': idx + 1})
                    iteration += 1
                assert idx == self.iterator_length - 1
        except pt.train.hooks.StopTraining:
            pass
        num_epochs = length // self.iterator_length
        assert trainer.epoch == num_epochs
        assert trainer.iteration == length - num_epochs * self.iterator_length , (
            trainer.iteration, length - num_epochs * self.iterator_length)

    def train_loop_epoch(self, length, max_it_len):
        progressbar_hook = pt.train.hooks.ProgressBarHook(
            stop_trigger=(length, 'epoch'), max_it_len=max_it_len,
            update_interval=1
        )
        trainer = types.SimpleNamespace()
        trainer.iteration = 0
        for epoch in range(self.num_epochs):  # infinite loop
            trainer.epoch = epoch
            progressbar_hook.pre_step(trainer)
            if not epoch == 0:
                assert iteration + 1 == self.iterator_length, iteration
                assert progressbar_hook.pbar.total == \
                       self.num_epochs * self.iterator_length, \
                    (progressbar_hook.pbar.total)
            for iteration in range(self.iterator_length):
                if not epoch == 0:
                    progressbar_hook.pre_step(trainer)
                trainer.iteration = iteration
                assert iteration < self.iterator_length * self.num_epochs
                progressbar_hook.post_step(trainer, None, None, {
                    'loss': iteration + 1})
        progressbar_hook.close(trainer)
        assert trainer.iteration == self.iterator_length - 1


def test_summary_hook():
    skip_on_windows()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        hook = pt.train.hooks.SummaryHook(
            (1, 'iteration'),
        )

        # Missing key 'loss' is now allowed, because the optimizer reports now
        # the summary directly to the hooks, instead of appending the review.
        # with pytest.raises(KeyError, match=r"'loss'") as excinfo:
        #     hook.update_summary({})

        # hook.update_summary({
        #     'loss': torch.tensor(1)
        # })

        hook.update_summary({
            # 'loss': torch.tensor(1),
            'scalars': {
                'a': 2,
                'b': torch.tensor(3),
            }
        })

        hook.update_summary({
            # 'loss': torch.tensor(1),
            'texts': {
                'c': 'abc',
            }
        })

        # ToDo: histograms, audios, images, figures

        class DummyTrainer:
            iteration = 10

            class Timer:
                as_dict = {}
                def clear(self): pass
            train_timer = Timer()

            class Model(pt.Model):
                def forward(self, inputs): pass
                def review(self, inputs, outputs): pass
            model = Model()

            class PaderOptimizer:
                class PytorchOptimizer:
                    param_groups = [{'lr': 1}]
                optimizer = PytorchOptimizer()
            optimizer = PaderOptimizer()

            writer = tensorboardX.SummaryWriter(str(tmp_dir / 'experiment_dir'))

        trainer = DummyTrainer()
        hook.finalize_summary(trainer)
        hook.dump_summary(trainer)
        DummyTrainer.writer.close()

        event_file, = (tmp_dir / 'experiment_dir').glob('*tfevents*')
        events = list(pt.summary.tfevents.load_events_as_dict(event_file))
        for e in events:
            del e['wall_time']

        if 'file_version' in events[0]:
            # Do not care about the brain.Event file_version.
            # In tensorboradX 1.6: file_version is missing
            # In tensorboradX 2: file_version is 'brain.Event:2'
            del events[0]['file_version']

        expect = [
            {},
            {'step': 10, 'summary': {'value': [
                {'tag': 'training/a', 'simple_value': 2.0}
            ]}},
            {'step': 10, 'summary': {'value': [
                {'tag': 'training/b', 'simple_value': 3.0}
            ]}},
            {'step': 10, 'summary': {'value': [
                {'tag': 'training/c/text_summary',
                 'tensor': {
                     'dtype': 7,
                     'tensor_shape': {
                         'dim': [{'size': 1}]},
                     'string_val': [b'abc']
                 },
                 'metadata': {'plugin_data': {'plugin_name': 'text'}}}
            ]}}
        ]

        assert events == expect, pretty([events, expect])


def test_summary_hook_fail_duplicate_key():
    skip_on_windows()

    hook = pt.train.hooks.SummaryHook((1, 'iteration'))

    hook.update_summary({
        # 'loss': torch.tensor(1),
        'scalars': {
            'a': 2,
            'b': torch.tensor(3),
        },
        'histograms': {
            'a': [1, 2]
        }
    })

    class DummyTrainer:
        iteration = 1
        writer = MagicMock()

    with pytest.raises(AssertionError) as excinfo:
        hook.dump_summary(DummyTrainer())

    expect = """The tag 'training/a' is used multiple times.

Tensorboard has problems, when different events have the same tag.
e.g. you cannot report the `grad_norm` as scalar and histogram.
A common workaround is to use `grad_norm` for the scalar and append an `_` for the histogram (i.e. `grad_norm_`)."""

    unittest.TestCase().assertMultiLineEqual(expect, str(excinfo.value))


class DummyModel(pt.Model):
    def __init__(self, validation_losses, exp_dir, optimizer):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10)

        self.validation_losses = validation_losses
        self.n = -1
        self.exp_dir = Path(exp_dir) / 'checkpoints'
        self.ckpt_log = []
        self.optimizer = optimizer
        self.lr_log = []

    def forward(self, inputs):
        return inputs

    def review(self, inputs, outputs):
        if self.training:
            l = self.lin.weight.sum()
            return {
                'loss': l.sum()
            }
        else:
            self.ckpt_log.append(
                [ckpt.name for ckpt in sorted(self.exp_dir.glob('*.pth'))]
            )
            self.lr_log.append(self.optimizer.optimizer.param_groups[0]['lr'])
            self.n += 1
            l = torch.Tensor([self.validation_losses[self.n]])
            l.requires_grad = True
            return {
                'loss': l.sum()
            }


def test_summary_hook_create_snapshot_flag():
    skip_on_windows()
    class Model(DummyModel):

        def __init__(self, validation_losses, exp_dir, optimizer):
            super().__init__(validation_losses, exp_dir, optimizer)
            self.create_snapshot_log = []

        def review(self, example, output):
            self.create_snapshot_log.append(self.create_snapshot)
            return super().review(example, output)

    ds = [0., 1., 2.]
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = pt.optimizer.Adam()
        model = Model([1, 2, 3] * 10, tmp_dir, optimizer)
        trainer = pt.Trainer(
            model, tmp_dir, optimizer, stop_trigger=(10, 'epoch'),
            summary_trigger=(1, 'epoch')
        )
        trainer.train(ds)
    assert model.create_snapshot_log == [True, False, False] * 10


def test_validation_hook_create_snapshot_flag():
    skip_on_windows()
    class Model(DummyModel):

        def __init__(self, validation_losses, exp_dir, optimizer):
            super().__init__(validation_losses, exp_dir, optimizer)
            self.train_create_snapshot_log = []
            self.validation_create_snapshot_log = []

        def review(self, example, output):
            if self.training:
                self.train_create_snapshot_log.append(self.create_snapshot)
            else:
                self.validation_create_snapshot_log.append(self.create_snapshot)
            return super().review(example, output)

    ds_train = [0., 1., 2.]
    ds_valid = [0., 1.]
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = pt.optimizer.Adam()
        model = Model([1, 2, 3, 4, 5] * 10, tmp_dir, optimizer)
        trainer = pt.Trainer(
            model, tmp_dir, optimizer, stop_trigger=(10, 'epoch'),
            summary_trigger=(1, 'epoch')
        )
        trainer.register_validation_hook(ds_valid)
        trainer.train(ds_train)
    assert model.train_create_snapshot_log == [True, False, False] * 10
    # The validation is executed num_epochs + 1 times, so 11
    assert model.validation_create_snapshot_log == [True, False] * 11


def test_validation_hook_modify_summary_training_flag():
    skip_on_windows()

    class Model(DummyModel):
        def review(self, example, output):
            summary = super().review(example, output)
            summary.setdefault('scalars', {})['training'] = self.training
            return summary

        def modify_summary(self, summary):
            if len(summary['scalars']['training']) == 0:
                # The first validation triggers to write the training summary,
                # which is empty.
                assert self.training is True
            else:
                assert set(summary['scalars']['training']) == {self.training}
            # self.modify_summary_log.append(self.training)
            return super().modify_summary(summary)

    ds_train = [0., 1., 2.]
    ds_valid = [0., 1.]
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = pt.optimizer.Adam()
        model = Model([1, 2, 3, 4, 5] * 10, tmp_dir, optimizer)
        trainer = pt.Trainer(
            model, tmp_dir, optimizer, stop_trigger=(10, 'epoch'),
            summary_trigger=(1, 'epoch')
        )
        trainer.register_validation_hook(ds_valid)
        trainer.train(ds_train)


def test_backoff():
    skip_on_windows()

    ds = [0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = pt.optimizer.Adam()
        model = DummyModel([3, 2, 1, 0, 1, 1, 1, 1, 1, 1], tmp_dir, optimizer)
        trainer = pt.Trainer(
            model, tmp_dir, optimizer, stop_trigger=(10, 'epoch')
        )
        trainer.register_validation_hook(
            ds, max_checkpoints=None,
            n_back_off=1, back_off_patience=2, early_stopping_patience=2
        )
        trainer.train(ds)
    assert model.ckpt_log == [
        [],
        ['ckpt_0.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_4.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_4.pth', 'ckpt_5.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_4.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
        ['ckpt_0.pth', 'ckpt_1.pth', 'ckpt_2.pth', 'ckpt_3.pth', 'ckpt_4.pth', 'ckpt_5.pth', 'ckpt_best_loss.pth', 'ckpt_latest.pth'],
    ]
    assert model.lr_log == 7*[0.001]+3*[0.0001]


def test_loss_weight_annealing_hook():
    skip_on_windows()
    class DummyTrainer:
        epoch = 0
        iteration = 0
        loss_weights = {'loss': .1}

    loss_weight_annealing_hook = pt.train.hooks.LossWeightAnnealingHook(
        (1, 'iteration'), [(0, 0), (5, 1), (10, 0)], 'iteration', 'loss'
    )
    trainer = DummyTrainer()
    values = []
    for i in range(11):
        trainer.iteration = i
        loss_weight_annealing_hook.pre_step(trainer)
        values.append(trainer.loss_weights['loss'])
    expected_values = np.linspace(0, .1, 6).tolist() + np.linspace(.08, 0, 5).tolist()
    pb.testing.assert_almost_equal(values, expected_values)


def test_model_attribute_annealing_hook():
    skip_on_windows()

    class DummyTrainer:
        epoch = 0
        iteration = 0

        class DummyModel:
            attr = .1
        model = DummyModel()

    attr_annealing_hook = pt.train.hooks.ModelAttributeAnnealingHook(
        (1, 'iteration'), [(0, 0), (5, 1), (10, 0)], 'iteration', 'attr'
    )
    trainer = DummyTrainer()
    values = []
    for i in range(11):
        trainer.iteration = i
        attr_annealing_hook.pre_step(trainer)
        values.append(trainer.model.attr)
    expected_values = np.linspace(0, .1, 6).tolist() + np.linspace(.08, 0, 5).tolist()
    pb.testing.assert_almost_equal(values, expected_values)


def test_lr_annealing_hook():
    skip_on_windows()

    class DummyTrainer:
        epoch = 0
        iteration = 0

        class PaderOptimizer:
            class PytorchOptimizer:
                param_groups = [{'lr': .1}]
            optimizer = PytorchOptimizer()
        optimizer = PaderOptimizer()

    lr_annealing_hook = pt.train.hooks.LRAnnealingHook(
        (1, 'iteration'), [(0, 0), (5, 1), (10, 0)], 'iteration'
    )
    trainer = DummyTrainer()
    values = []
    for i in range(12):
        trainer.iteration = i
        lr_annealing_hook.pre_step(trainer)
        values.append(trainer.optimizer.optimizer.param_groups[0]['lr'])
    expected_values = np.linspace(0, .1, 6).tolist() + np.linspace(.08, 0, 5).tolist() + [0]
    pb.testing.assert_almost_equal(values, expected_values)


def test_LRSchedulerHook():
    skip_on_windows()
    class DummyLRScheduler:
        def __init__(self):
            self.calls_iteration = []
            self.calls_epoch = []
        def step(self):
            self.calls_iteration.append(trainer.iteration)
            self.calls_epoch.append(trainer.epoch)

    class DummyTrainer:
        epoch = 0
        iteration = 0

    lr_scheduler = DummyLRScheduler()
    hook = pt.train.hooks.LRSchedulerHook(lr_scheduler, (1, 'epoch'))
    trainer = DummyTrainer()
    hook.PYTORCH_ge_1_1 = True
    for i in range(12):
        trainer.iteration = i
        trainer.epoch = i // 3
        hook.pre_step(trainer)
    assert lr_scheduler.calls_iteration == [3, 6, 9]
    assert lr_scheduler.calls_epoch == [1, 2, 3]

    lr_scheduler = DummyLRScheduler()
    hook = pt.train.hooks.LRSchedulerHook(lr_scheduler, (1, 'epoch'))
    trainer = DummyTrainer()
    hook.PYTORCH_ge_1_1 = False
    for i in range(12):
        trainer.iteration = i
        trainer.epoch = i // 3
        hook.pre_step(trainer)
    assert lr_scheduler.calls_iteration == [0, 3, 6, 9]
    assert lr_scheduler.calls_epoch == [0, 1, 2, 3]

    lr_scheduler = DummyLRScheduler()
    hook = pt.train.hooks.LRSchedulerHook(lr_scheduler, (2, 'iteration'))
    trainer = DummyTrainer()
    hook.PYTORCH_ge_1_1 = True
    for i in range(12):
        trainer.iteration = i
        trainer.epoch = i // 3
        hook.pre_step(trainer)
    assert lr_scheduler.calls_iteration == [2, 4, 6, 8, 10]
    assert lr_scheduler.calls_epoch == [0, 1, 2, 2, 3]

    lr_scheduler = DummyLRScheduler()
    hook = pt.train.hooks.LRSchedulerHook(lr_scheduler, (2, 'iteration'))
    trainer = DummyTrainer()
    hook.PYTORCH_ge_1_1 = False
    for i in range(12):
        trainer.iteration = i
        trainer.epoch = i // 3
        hook.pre_step(trainer)
    assert lr_scheduler.calls_iteration == [0, 2, 4, 6, 8, 10]
    assert lr_scheduler.calls_epoch == [0, 0, 1, 2, 2, 3]
