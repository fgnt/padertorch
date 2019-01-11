import contextlib
import copy
import inspect
import itertools
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import paderbox as pb
import tqdm

import padertorch as pt


def test_run(trainer, train_iterator, validation_iterator):
    """
    Run a test on the trainer instance (i.e. model test).

    Tests:
     - forward (train and validate)
     - deterministic output in eval
     - simple review dict test

    """
    print('Start test run')
    with contextlib.ExitStack() as exit_stack:
        save_checkpoint = exit_stack.enter_context(mock.patch.object(
            pt.Trainer,
            'save_checkpoint',
            new_callable=mock.MagicMock,
        ))
        dump_summary = exit_stack.enter_context(mock.patch.object(
            pt.train.hooks.SummaryHook,
            'writer',
            new_callable=mock.MagicMock,
        ))
        exit_stack.enter_context(mock.patch.object(
            tqdm,
            'tqdm',
            new_callable=mock.MagicMock,
        ))
        optimizer_step = exit_stack.enter_context(mock.patch.object(
            trainer.optimizer.optimizer,
            'step',
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'summary_trigger',
            new=(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'validate_trigger',
            new=(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'checkpoint_trigger',
            new=pt.train.trigger.IntervalTrigger(1, 'epoch'),
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'max_trigger',
            new=(2, 'epoch'),
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

        validate_mock = exit_stack.enter_context(mock.patch.object(
            trainer,
            'validate',
            wraps=trainer.validate,
            new_callable=SpyMagicMock,
        ))
        get_default_hooks_mock = exit_stack.enter_context(mock.patch.object(
            trainer,
            'get_default_hooks',
            wraps=trainer.get_default_hooks,
            new_callable=SpyMagicMock,
        ))

        trainer.train(
            list(itertools.islice(train_iterator, 2)),
            list(itertools.islice(validation_iterator, 2)),
        )

        assert optimizer_step.call_count == 4, optimizer_step.call_count
        assert save_checkpoint.call_count == 3, save_checkpoint.call_count
        assert dump_summary.add_scalar.call_count >= 8, dump_summary.add_scalar.call_count
        assert validate_mock.call_count == 2, validate_mock.call_count
        assert review_mock.call_count == 8, review_mock.call_count
        assert get_default_hooks_mock.call_count == 1, get_default_hooks_mock.call_count

        save_checkpoint.assert_called()

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

        def nested_test_assert_allclose(struct1, struct2):
            def assert_func(array1, array2):
                array1 = pt.utils.to_numpy(array1)
                array2 = pt.utils.to_numpy(array2)
                np.testing.assert_allclose(
                    array1, array2,
                    rtol=1e-5,
                    atol=1e-5,
                )

            pb.utils.nested.nested_op(
                assert_func,
                struct1, struct2,
                handle_dataclass=True,
            )

        nested_test_assert_allclose(dt1['output'], dt3['output'])
        nested_test_assert_allclose(dt2['output'], dt4['output'])
        nested_test_assert_allclose(dt1['review'], dt3['review'])
        nested_test_assert_allclose(dt2['review'], dt4['review'])

        assert 'losses' in dt1['review'], dt1['review']

        if 0 != len(set(dt1['review'].keys()) - set(
                pt.trainer.SummaryHook.empty_summary_dict().keys())):
            got = set(dt1['review'].keys())
            allowed = set(trainer.summary.keys())
            raise ValueError(
                f'Found keys: {got}\n'
                f'Allowed: {allowed}\n'
                f'Delta: {got - allowed}'
            )

        hooks, = get_default_hooks_mock.spyed_return_values
        for hook in hooks:
            summary = getattr(hook, 'summary', {})
            assert all([
                len(s) == 0 for s in summary.values()
            ]), (hook, summary)

    print('Successfully finished test run')


def test_run_from_config(
        trainer_config,
        train_iterator,
        validation_iterator,
):
    trainer_config = copy.deepcopy(trainer_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['kwargs']['storage_dir'] = tmp_dir

        tmp_dir = Path(tmp_dir)
        t = pt.Trainer.from_config(trainer_config)

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            # no event file
            raise Exception(files_before)

        t.test_run(train_iterator, validation_iterator)

        files_after = tuple(tmp_dir.glob('*'))
        if files_after != files_before:
            raise Exception(files_after, files_before)
