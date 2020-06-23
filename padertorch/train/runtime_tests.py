import contextlib
import copy
import inspect
import itertools
import os
import tempfile
from pathlib import Path
from unittest import mock
import logging

import numpy as np
import torch
import tensorboardX

import lazy_dataset

import padertorch as pt
import paderbox as pb

from padertorch.train.hooks import (
    SummaryHook, CheckpointHook, StopTrainingHook, BackOffValidationHook
)


LOG = logging.getLogger('runtime_test')


def nested_test_assert_allclose(struct1, struct2, rtol=1e-5, atol=1e-5):
    def assert_func(array1, array2):
        if array1 is None:
            assert array2 is None, 'Validation step has not been deterministic'
        elif isinstance(array1, str):
            np.testing.assert_string_equal(array1, array2)
        else:
            # This function should fail when it is called for training data
            # (-> detach=False) because training is often not deterministic.
            # e.g.: dropout.
            if isinstance(array1, str) and isinstance(array2, str):
                # e.g. review['texts']
                np.testing.assert_equal(array1, array2)
            # elif ..:
            #     # ToDo: Does review['figures'] work?
            else:
                array1 = pt.utils.to_numpy(array1, detach=False)
                array2 = pt.utils.to_numpy(array2, detach=False)
                try:
                    np.testing.assert_allclose(
                        array1, array2,
                        rtol=rtol,
                        atol=atol,
                        err_msg='Validation step has not been deterministic.\n'
                                'This might be caused by layers changing their\n'
                                'internal state such as BatchNorm'
                    )
                except TypeError as e:
                    def get_type(array):
                        if hasattr(array, 'dtype'):
                            return (type(array), array.dtype)
                        else:
                            return type(array)
                    raise TypeError(
                        str(e)
                        + '\n\n'
                        + f'type1: {get_type(array1)} type2: {get_type(array2)}'
                        + f'\n\narray1:\n{array1}\n\narray2:\n{array2}'
                    )

    pb.utils.nested.nested_op(
        assert_func,
        struct1, struct2,
        handle_dataclass=True,
    )


def test_run(
        trainer: 'pt.Trainer',
        train_iterator,
        validation_iterator,
        device=0 if torch.cuda.is_available() else 'cpu',
        test_with_known_iterator_length=False,
):
    """
    Run a test on the trainer instance (i.e. model test).

    Does not work with layers updating their state such as BatchNorm

    Tests:
     - forward (train and validate)
     - deterministic output in eval
     - simple review dict test

    """
    print('Start test run')

    @contextlib.contextmanager
    def backup_state_dict(trainer: pt.Trainer):
        state_dict = copy.deepcopy(trainer.state_dict())
        try:
            yield
        finally:
            # pass
            trainer.load_state_dict(state_dict)

    with contextlib.ExitStack() as exit_stack:
        storage_dir = Path(
            exit_stack.enter_context(tempfile.TemporaryDirectory())
        ).expanduser().resolve()
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'iteration',
            new=-1,
        ))
        exit_stack.enter_context(mock.patch.object(
            trainer,
            'epoch',
            new=-1,
        ))

        class SpyMagicMock(mock.MagicMock):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                self.spyed_return_values = []

            def __call__(self, *args, **kw):
                ret = super().__call__(*args, **kw)
                self.spyed_return_values += [ret]
                return ret

        # Spy trainer.step, optimizer.step, trainer.validate and
        # trainer.get_default_hooks to check lates the output values and/or
        # the number of calls.
        trainer_step_mock = exit_stack.enter_context(mock.patch.object(
            trainer,
            'step',
            wraps=trainer.step,
            new_callable=SpyMagicMock,
        ))
        optimizer_step = pb.utils.nested.nested_op(
            lambda x: (exit_stack.enter_context(mock.patch.object(
                x,
                'step',
                wraps=x.step,
                new_callable=SpyMagicMock,
            )) if x is not None else None), trainer.optimizer)

        validate_mock = exit_stack.enter_context(mock.patch.object(
            trainer,
            'validate',
            wraps=trainer.validate,
            new_callable=SpyMagicMock,
        ))

        class Iterable:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                yield from self.data

            def __len__(self):
                raise TypeError(
                    f'object of type {self.__class__.__name__} has no len()'
                )

        virtual_minibatch_size = trainer.virtual_minibatch_size

        sub_train_iterator = list(itertools.islice(train_iterator, 2 * virtual_minibatch_size))
        sub_validation_iterator = list(itertools.islice(validation_iterator, 2))

        if test_with_known_iterator_length:
            sub_train_iterator = lazy_dataset.from_list(sub_train_iterator)
            sub_validation_iterator = lazy_dataset.from_list(sub_validation_iterator)
        else:
            sub_train_iterator = Iterable(sub_train_iterator)
            sub_validation_iterator = Iterable(sub_validation_iterator)

        @contextlib.contextmanager
        def ensure_unchanged_parameter(trainer):
            parameters_before = {
                name: parameter.detach().cpu().numpy().copy()
                for name, parameter in trainer.model.named_parameters()
            }

            yield

            parameters_after = {
                name: parameter.detach().cpu().numpy().copy()
                for name, parameter in trainer.model.named_parameters()
            }

            assert parameters_before.keys() == parameters_after.keys(), (
                parameters_before.keys(), parameters_after.keys()
            )
            for k in parameters_before.keys():
                np.testing.assert_equal(
                    parameters_before[k],
                    parameters_after[k],
                )

        # ================ Train Call ===================
        with ensure_unchanged_parameter(trainer):
            hooks = [
                SummaryHook((1, 'epoch')),
                CheckpointHook((1, 'epoch')),
                BackOffValidationHook(
                    (1, 'epoch'), sub_validation_iterator,
                    max_checkpoints=None
                ),
                StopTrainingHook((1, 'epoch'))
            ]
            exit_stack.enter_context(mock.patch.object(
                trainer,
                'hooks',
                new=hooks,
            ))
            with backup_state_dict(trainer):

                exit_stack.enter_context(mock.patch.object(
                    trainer,
                    'storage_dir',
                    new=storage_dir,
                ))


                trainer.train(
                    sub_train_iterator,
                    device=device
                )
            with backup_state_dict(trainer):
                storage_dir_2 = Path(
                    exit_stack.enter_context(tempfile.TemporaryDirectory())
                ).expanduser().resolve()
                exit_stack.enter_context(mock.patch.object(
                    trainer,
                    'storage_dir',
                    new=storage_dir_2,
                ))

                trainer.train(
                    sub_train_iterator,
                    device=device,
                )

        def assert_step(x):
            if x is not None:
                assert x.call_count == 4, x.call_count

        pb.utils.nested.nested_op(assert_step, optimizer_step)

        # before and after training for two trainings -> 4
        assert validate_mock.call_count == 4, validate_mock.call_count

        assert trainer_step_mock.call_count == (4 * virtual_minibatch_size + 8), (trainer_step_mock.call_count, virtual_minibatch_size)

        def trainer_step_mock_to_inputs_output_review(review_mock):
            # sig = inspect.signature(review_mock._mock_wraps)
            for call, (loss, inputs, output, review) in zip(
                    review_mock.call_args_list, review_mock.spyed_return_values):
                # args, kwargs = tuple(call)
                # model, example, timer, device
                # _, inputs, timer, _ = sig.bind(*args, **kwargs).arguments.values()
                yield dict(inputs=inputs, output=output, review=review, loss=loss)


        # trainer_step_mock_to_inputs_output_review
        step_returns = \
            trainer_step_mock_to_inputs_output_review(
                trainer_step_mock
            )
        step_returns = list(step_returns)
        step_returns_1 = step_returns[:len(step_returns) // 2]
        step_returns_2 = step_returns[len(step_returns) // 2:]

        if virtual_minibatch_size == 1:
            dt1, dt2, tr1, tr2, dt3, dt4 = step_returns_1
            dt5, dt6, tr3, tr4, dt7, dt8 = step_returns_2
        else:
            dt1, dt2 = step_returns_1[:2]
            dt3, dt4 = step_returns_1[-2:]
            dt5, dt6 = step_returns_2[:2]
            dt7, dt8 = step_returns_2[-2:]

        nested_test_assert_allclose(dt1['output'], dt5['output'])
        nested_test_assert_allclose(dt2['output'], dt6['output'])
        nested_test_assert_allclose(dt1['review'], dt5['review'])
        nested_test_assert_allclose(dt2['review'], dt6['review'])

        # Can not test these, because dropout makes them unequal
        # nested_test_assert_allclose(dt3['output'], dt7['output'])
        # nested_test_assert_allclose(dt4['output'], dt8['output'])
        # nested_test_assert_allclose(dt3['review'], dt7['review'])
        # nested_test_assert_allclose(dt4['review'], dt8['review'])

        # Expect that the initial loss is equal for two runs
        nested_test_assert_allclose(dt1['loss'], dt5['loss'], rtol=1e-6, atol=1e-6)
        nested_test_assert_allclose(dt2['loss'], dt6['loss'], rtol=1e-6, atol=1e-6)
        try:
            with np.testing.assert_raises(AssertionError):
                # Expect that the loss changes after training.
                nested_test_assert_allclose(dt1['loss'], dt3['loss'], rtol=1e-6, atol=1e-6)
                nested_test_assert_allclose(dt2['loss'], dt4['loss'], rtol=1e-6, atol=1e-6)
                nested_test_assert_allclose(dt5['loss'], dt7['loss'], rtol=1e-6, atol=1e-6)
                nested_test_assert_allclose(dt6['loss'], dt8['loss'], rtol=1e-6, atol=1e-6)
        except AssertionError:
            raise AssertionError(
                'The loss of the model did not change between two validations.'
                '\n'
                'This is usually caused from a zero gradient or the loss is'
                'independent of the parameters'
            )

        allowed_summary_keys = (
            {'loss', 'losses'} | set(
                pt.trainer.SummaryHook.empty_summary_dict().keys()
            )
        )
        if 0 != len(set(dt1['review'].keys()) - set(allowed_summary_keys)):
            got = set(dt1['review'].keys())
            raise ValueError(
                f'Found keys: {got}\n'
                f'Allowed: {allowed_summary_keys}\n'
                f'Delta: {got - allowed_summary_keys}'
            )
        # end trainer_step_mock_to_inputs_output_review

        # Test that the summary is empty
        for hook in hooks:
            summary = getattr(hook, 'summary', {})
            assert all([
                len(s) == 0 for s in summary.values()
            ]), (hook, summary)

        files = list(storage_dir.glob('*'))
        assert len(files) == 2, files

        for file in files:
            if 'tfevents' in file.name:
                pass
            elif file.name == 'checkpoints':
                checkpoint_names = {f.name for f in file.glob('*')}
                expect = {
                    'ckpt_latest.pth',
                    'ckpt_best_loss.pth',
                    f'ckpt_0.pth',
                    # f'ckpt_{2*virtual_minibatch_size}.pth',
                    f'ckpt_2.pth',
                }
                if checkpoint_names != expect:
                    os.system(f'ls -lha {file}')
                    raise AssertionError((checkpoint_names, expect))

                ckpt_best = (file / 'ckpt_best_loss.pth').resolve().name
                ckpt_last = (file / 'ckpt_latest.pth').resolve().name

                # This check does not always work, because it is not guaranteed,
                # that the training improves the loss on the validation data
                # assert ckpt_best == 'ckpt_2.pth', ckpt_best

                expected_ckpt_last = f'ckpt_2.pth'
                assert ckpt_last == expected_ckpt_last, (ckpt_last, expected_ckpt_last)

                # ckpt_state = pb.io.load_json(file / 'ckpt_state.json')
                # assert ckpt_state == {
                #     'latest_checkpoint_path':
                #         '/tmp/tmp_h0sygfv/checkpoints/ckpt_4.pth',
                #     'metrics': {
                #         'loss': {
                #             'criterion': 'min',
                #             'key': 'loss',
                #             'paths': ['ckpt_2.pth'],
                #             'values': [2.5040305852890015],
                #         }
                #     }
                # }, ckpt_state

    print('Successfully finished test run')


def test_run_from_config(
        trainer_config,
        train_iterator,
        validation_iterator,
        test_with_known_iterator_length=False,
):
    trainer_config = copy.deepcopy(trainer_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_config['storage_dir'] = tmp_dir

        tmp_dir = Path(tmp_dir)

        t = pt.Trainer.from_config(trainer_config)

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            raise Exception(files_before)
        test_run(
            t,
            train_iterator,
            validation_iterator,
            test_with_known_iterator_length=test_with_known_iterator_length
        )

        files_after = tuple(tmp_dir.glob('*'))
        if files_after != files_before:
            raise Exception(files_after, files_before)
