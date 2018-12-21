import os
import inspect
import itertools
import contextlib
from unittest import mock

import numpy as np

import paderbox as pb
import padertorch as pt


def test_run(trainer, train_iterator, validation_iterator):
    """
    Run a test on the trainer instance (i.e. model test).

    Tests:
     - forward (train and validate)
     - deterministic output in eval
     - simple review dict test

    """
    with contextlib.ExitStack() as exit_stack:
        save_checkpoint = exit_stack.enter_context(mock.patch.object(
            pt.Trainer,
            'save_checkpoint',
            new_callable=mock.MagicMock,
        ))
        add_summary = exit_stack.enter_context(mock.patch.object(
            pt.Trainer,
            'add_summary',
            new_callable=mock.MagicMock,
        ))
        optimizer_step = exit_stack.enter_context(mock.patch.object(
            trainer.optimizer.optimizer,
            'step',
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'summary_trigger',
            new=pt.train.trainer.IntervalTrigger(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'validation_trigger',
            new=pt.train.trainer.IntervalTrigger(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'checkpoint_trigger',
            new=pt.train.trainer.IntervalTrigger(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'max_iterations',
            new=pt.train.trainer.EndTrigger(2, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            os,
            'makedirs',
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'iteration',
            new=0,
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'epoch',
            new=0,
        ))

        class SpyMagicMock(mock.MagicMock):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.spyed_return_values = []

            def __call__(self, *args, **kw):
                ret = super().__call__(*args, **kw)
                self.spyed_return_values += [ret]
                return ret

        review_mock = exit_stack.enter_context(mock.patch.object(
            trainer.model,
            'review',
            wraps=trainer.model.review,
            new_callable=SpyMagicMock,
        ))

        trainer.train(
            list(itertools.islice(train_iterator, 2)),
            list(itertools.islice(validation_iterator, 2)),
        )

        assert optimizer_step.call_count == 4, optimizer_step.call_count
        assert save_checkpoint.call_count == 3, save_checkpoint.call_count
        assert add_summary.call_count == 7, add_summary.call_count
        assert review_mock.call_count == 8, review_mock.call_count

        save_checkpoint.assert_called()
        add_summary.assert_called()

        def review_mock_to_inputs_output_review(review_mock):
            sig = inspect.signature(review_mock._mock_wraps)
            for call, review in zip(review_mock.call_args_list,
                                    review_mock.spyed_return_values):
                args, kwargs = tuple(call)
                inputs, output = sig.bind(*args, **kwargs).arguments.values()
                yield dict(inputs=inputs, output=output, review=review)

        tr1, tr2, dt1, dt2, tr3, tr4, dt3, dt4 = \
            review_mock_to_inputs_output_review(
                review_mock
            )

        pb.utils.nested.nested_op(
            np.testing.assert_allclose, dt1['output'], dt3['output']
        )
        pb.utils.nested.nested_op(
            np.testing.assert_allclose, dt2['output'], dt4['output']
        )
        pb.utils.nested.nested_op(
            np.testing.assert_allclose, dt1['review'], dt3['review']
        )
        pb.utils.nested.nested_op(
            np.testing.assert_allclose, dt2['review'], dt4['review']
        )
        assert 'losses' in dt1['review'], dt1['review']
        assert len(set(dt1['review'].keys()) - set(trainer.summary.keys())), (
        dt1['review'].keys(), trainer.summary.keys())

        trainer.reset_summary()
