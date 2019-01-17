import padertorch as pt
import unittest
import types

class ProgresbarHookTest(unittest.TestCase):
    num_epochs = 2
    num_iterations = 5
    iterator_length = 4


    def test_max_iteration(self):
        self.train_loop(self.num_iterations, 'iteration', self.iterator_length)

    def train_loop(self, length, unit, max_it_len):
        progressbar_hook = pt.train.hooks.ProgressBarHook(
            max_trigger=(length, unit), max_it_len=max_it_len,
            update_intervall=1
        )
        iteration = 0
        try:
            for epoch in range(self.num_epochs):  # infinite loop
                print(epoch, 'epoch')
                for idx in range(self.iterator_length):
                    print(self.num_epochs, length)
                    if unit == 'iteration' and iteration >= length:
                        print('end', epoch, iteration)
                        raise pt.train.hooks.StopTraining
                    if not iteration == 1:
                        assert iteration == progressbar_hook.pbar.value,\
                            (iteration, progressbar_hook.pbar.value)

                    print('length', progressbar_hook.trigger.unit, progressbar_hook.trigger.period)
                    print(iteration, epoch, '\n')
                    assert iteration < 5, (iteration, epoch)
                    trainer = types.SimpleNamespace()
                    trainer.iteration = iteration
                    trainer.epoch = epoch
                    progressbar_hook.post_step(trainer, None, None, {'losses': {'loss': iteration}})
                    iteration += 1
                assert iteration == self.iterator_length
                progressbar_hook.post_step(trainer, None, None,
                                           {'losses': {'loss': iteration}})
        except pt.train.hooks.StopTraining:
            pass
        finally:
            progressbar_hook.close(trainer)




