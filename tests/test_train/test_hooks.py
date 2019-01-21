import padertorch as pt
import unittest
import types


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
            max_trigger=(length, 'iteration'), max_it_len=max_it_len,
            update_intervall=1
        )
        iteration = 0
        epoch = 0
        trainer = types.SimpleNamespace()
        trainer.iteration = iteration
        trainer.epoch = epoch
        try:
            for epoch in range(self.num_epochs):  # infinite loop
                if not epoch == 0:
                    progressbar_hook.post_step(trainer, None, None,
                                               {'losses': {'loss': iteration}})
                for idx in range(self.iterator_length):
                    if iteration >= length:
                        raise pt.train.hooks.StopTraining
                    if not iteration == 1:
                        assert iteration == progressbar_hook.pbar.value, \
                            (iteration, progressbar_hook.pbar.value)
                    assert iteration < self.num_iterations, (iteration, epoch)
                    trainer.iteration = iteration
                    trainer.epoch = epoch
                    progressbar_hook.post_step(trainer, None, None, {
                        'losses': {'loss': iteration + 1}})
                    iteration += 1
                assert iteration == self.iterator_length
        except pt.train.hooks.StopTraining:
            pass
        finally:
            assert progressbar_hook.pbar.value == length
            assert progressbar_hook.pbar.prefix == \
                   f'epochs: {epoch}, loss: {length} ', (
                progressbar_hook.pbar.prefix, epoch, iteration)
            progressbar_hook.close(trainer)

    def train_loop_epoch(self, length, max_it_len):
        progressbar_hook = pt.train.hooks.ProgressBarHook(
            max_trigger=(length, 'epoch'), max_it_len=max_it_len,
            update_intervall=1
        )
        iteration = 0
        epoch = 0
        trainer = types.SimpleNamespace()
        for epoch in range(self.num_epochs):  # infinite loop
            trainer.epoch = epoch
            if not epoch == 0:
                assert iteration // epoch == self.iterator_length, iteration
                progressbar_hook.post_step(trainer, None, None,
                                           {'losses': {'loss': iteration}})
                assert progressbar_hook.pbar.max_value ==\
                       self.num_epochs * self.iterator_length, \
                    (progressbar_hook.pbar.max_value, length)
            for idx in range(self.iterator_length):
                trainer.iteration = iteration
                if not iteration == 1:
                    assert iteration == progressbar_hook.pbar.value, \
                        (iteration, progressbar_hook.pbar.value)
                assert iteration < self.iterator_length * self.num_epochs
                progressbar_hook.post_step(trainer, None, None, {
                    'losses': {'loss': iteration + 1}})
                iteration += 1
        assert progressbar_hook.pbar.value == iteration
        assert progressbar_hook.pbar.prefix == \
               f'epochs: {epoch}, loss: {length * self.iterator_length} ', \
            (progressbar_hook.pbar.prefix, epoch, iteration)
        progressbar_hook.close(trainer)
