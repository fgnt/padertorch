import unittest
import types
import tempfile
from pathlib import Path

import pytest
import tensorboardX
import torch

import padertorch as pt


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
                assert progressbar_hook.pbar.max_value == \
                       self.num_epochs * self.iterator_length, \
                    (progressbar_hook.pbar.max_value)
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        hook = pt.train.hooks.SummaryHook(
            (1, 'iteration'),
            writer=tensorboardX.SummaryWriter(str(tmp_dir / 'experiment_dir')),
        )
        with pytest.raises(KeyError, match=r"'loss'") as excinfo:
            hook.update_summary({})

        hook.update_summary({
            'loss': torch.tensor(1)
        })

        hook.update_summary({
            'loss': torch.tensor(1),
            'scalars': {
                'a': 2,
                'b': torch.tensor(3),
            }
        })

        hook.update_summary({
            'loss': torch.tensor(1),
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

        trainer = DummyTrainer()
        hook.finalize_summary(trainer)
        hook.dump_summary(trainer)

        # ToDo: read summary
