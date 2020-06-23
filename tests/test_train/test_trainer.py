import os
import tempfile
from pathlib import Path
import inspect
import textwrap
import collections
import copy
import itertools

from IPython.lib.pretty import pprint, pretty
import pytest

import numpy as np
import torch

from padertorch.testing.test_db import MnistDatabase
from padertorch.summary.tfevents import load_events_as_dict

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
        if not isinstance(digits, int):
            digits = digits.cpu()
        target = torch.tensor(
            np.array(digits).astype(np.int64),
            device=output.device,
        )[None]
        ce = torch.nn.CrossEntropyLoss()(output[None, :], target)
        return {'loss': ce}


def get_dataset():
    db = MnistDatabase()
    return (
        db.get_dataset('train'),
        db.get_dataset('test'),
    )


class TriggerMock(pt.train.trigger.Trigger):
    """
    Wrap a Trigger and logs each call of the trigger to the log_list.

    >>> import types
    >>> dummy_trainer = types.SimpleNamespace()
    >>> dummy_trainer.iteration = 0
    >>> dummy_trainer.epoch = 0

    >>> log_list = []
    >>> trigger = pt.train.trigger.EndTrigger(10, 'iteration')
    >>> hook = pt.train.hooks.StopTrainingHook(trigger)
    >>> hook.trigger = TriggerMock(hook.trigger, log_list)
    >>> hook.pre_step(dummy_trainer)
    >>> for log in log_list: print(log)
    I:0, E: 0, False, StopTrainingHook.pre_step
    >>> dummy_trainer.iteration = 1
    >>> hook.pre_step(dummy_trainer)
    >>> for log in log_list: print(log)
    I:0, E: 0, False, StopTrainingHook.pre_step
    I:1, E: 0, False, StopTrainingHook.pre_step

    """
    def __init__(self, trigger, log_list):
        self.trigger = trigger
        self.log_list = log_list
        self.last = (-1, -1)

    def __deepcopy__(self, memo):
        # The hooks make alwaiys a deepcopy of the trigger, so one trigger can
        # be used from multiple hooks. Here we need to disable the copy for
        # log_list.
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        result.log_list = self.log_list
        result.trigger = copy.deepcopy(self.trigger, memo)
        return result

    def __call__(self, iteration, epoch):
        ret = self.trigger(iteration, epoch)

        frame = inspect.stack()[1].frame

        if 'self' in frame.f_locals:
            name = frame.f_locals['self'].__class__.__name__

            string = f'I:{iteration}, E: {epoch}, {ret}, {name}.{inspect.stack()[1].function}'
            self.log_list.append(string)
        else:
            callerframerecord = inspect.stack()[2]
            frame = callerframerecord[0]
            name = frame.f_locals['self'].__class__.__name__
            assert name == 'OrTrigger', name

        return ret

    def set_last(self, iteration, epoch):
        return self.trigger.set_last(iteration=iteration, epoch=epoch)


def test_single_model():
    tr_dataset, dt_dataset = get_dataset()
    tr_dataset = tr_dataset[:2]
    dt_dataset = dt_dataset[:2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        config = pt.Trainer.get_config(
            updates=pb.utils.nested.deflatten({
                'model.factory': Model,
                'storage_dir': str(tmp_dir),
                'stop_trigger': (2, 'epoch'),
                'summary_trigger': (3, 'iteration'),
                'checkpoint_trigger': (2, 'iteration')
            })
        )

        t = pt.Trainer.from_config(config)
        pre_state_dict = copy.deepcopy(t.state_dict())

        files_before = tuple(tmp_dir.glob('*'))
        if len(files_before) != 0:
            # no event file
            raise Exception(files_before)

        t.register_validation_hook(
            validation_iterator=dt_dataset, max_checkpoints=None
        )

        # Wrap each trigger in each hook with TriggerMock.
        log_list = []
        for hook in t.hooks:
            for k, v in list(hook.__dict__.items()):
                if isinstance(v, pt.train.trigger.Trigger):
                    hook.__dict__[k] = TriggerMock(v, log_list)
        t.train(train_dataset=tr_dataset, resume=False)

        hook_calls = ('\n'.join(log_list))

        # CheckpointedValidationHook trigger is called two times
        #   (once for checkpointing once for validation)_file_name

        hook_calls_ref = textwrap.dedent('''
        I:0, E: 0, True, SummaryHook.pre_step
        I:0, E: 0, True, BackOffValidationHook.pre_step
        I:0, E: 0, True, CheckpointHook.pre_step
        I:0, E: 0, False, StopTrainingHook.pre_step
        I:1, E: 0, False, SummaryHook.pre_step
        I:1, E: 0, False, BackOffValidationHook.pre_step
        I:1, E: 0, False, CheckpointHook.pre_step
        I:1, E: 0, False, StopTrainingHook.pre_step
        I:2, E: 1, False, SummaryHook.pre_step
        I:2, E: 1, True, BackOffValidationHook.pre_step
        I:2, E: 1, True, CheckpointHook.pre_step
        I:2, E: 1, False, StopTrainingHook.pre_step
        I:3, E: 1, True, SummaryHook.pre_step
        I:3, E: 1, False, BackOffValidationHook.pre_step
        I:3, E: 1, False, CheckpointHook.pre_step
        I:3, E: 1, False, StopTrainingHook.pre_step
        I:4, E: 2, False, SummaryHook.pre_step
        I:4, E: 2, True, BackOffValidationHook.pre_step
        I:4, E: 2, True, CheckpointHook.pre_step
        I:4, E: 2, True, StopTrainingHook.pre_step
        ''').strip()

        print('#' * 80)
        print(hook_calls)
        print('#' * 80)

        if hook_calls != hook_calls_ref:
            import difflib
            raise AssertionError(
                '\n' +
                ('\n'.join(difflib.ndiff(
                    hook_calls_ref.splitlines(),
                    hook_calls.splitlines(),
            ))))

        old_event_files = []

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 2, files_after
        for file in sorted(files_after):
            if 'tfevents' in file.name:
                old_event_files.append(file)
                events = list(load_events_as_dict(file))

                tags = []
                # time_rel_data_loading = []
                # time_rel_train_step = []
                time_per_iteration = []

                relative_timings = collections.defaultdict(list)
                relative_timing_keys = {
                    'training_timings/time_rel_data_loading',
                    'training_timings/time_rel_to_device',
                    'training_timings/time_rel_forward',
                    'training_timings/time_rel_review',
                    'training_timings/time_rel_backward',
                    'training_timings/time_rel_optimize',
                }
                for event in events:
                    if 'summary' in event.keys():
                        value, = event['summary']['value']
                        tags.append(value['tag'])
                        if value['tag'] in relative_timing_keys:
                            relative_timings[value['tag']].append(value['simple_value'])
                        elif value['tag'] == 'training_timings/time_per_iteration':
                            time_per_iteration.append(value['simple_value'])

                c = dict(collections.Counter(tags))
                # Training summary is written two times (at iteration 3 when
                #   summary_trigger triggers and when training stops and
                #   summary_hook is closed).
                # Validation summary is written when checkpoint_trigger
                #   triggers, hence 3 times.
                #   non_validation_time can only be measured between
                #   validations => 2 values (one fewer than validation_time)
                expect = {
                    'training/grad_norm': 2,
                    'training/grad_norm_': 2,
                    'training/loss': 2,
                    'training/lr/param_group_0': 2,
                    'training_timings/time_per_iteration': 2,
                    'training_timings/time_rel_to_device': 2,
                    'training_timings/time_rel_forward': 2,
                    'training_timings/time_rel_review': 2,
                    'training_timings/time_rel_backward': 2,
                    'training_timings/time_rel_optimize': 2,
                    'training_timings/time_rel_data_loading': 2,
                    # 'training_timings/time_rel_step': 2,
                    'validation/loss': 3,
                    'validation_timings/time_per_iteration': 3,
                    'validation_timings/time_rel_to_device': 3,
                    'validation_timings/time_rel_forward': 3,
                    'validation_timings/time_rel_review': 3,
                    'validation_timings/time_rel_data_loading': 3,
                    # 'validation_timings/time_rel_step': 3,
                    # non validation time can only be measured between
                    # validations:
                    #  => # of non_val_time - 1 == # of val_time
                    'validation_timings/non_validation_time': 2,
                    'validation_timings/validation_time': 3,
                }
                pprint(c)
                if c != expect:
                    import difflib

                    raise AssertionError(
                        '\n' + ('\n'.join(difflib.ndiff(
                            [f'{k!r}: {v!r}'for k, v in sorted(expect.items())],
                            [f'{k!r}: {v!r}'for k, v in sorted(c.items())],
                        )))
                    )
                assert len(events) == 46, (len(events), events)

                assert relative_timing_keys == set(relative_timings.keys()), (relative_timing_keys, relative_timings)

                for k, v in relative_timings.items():
                    assert len(v) > 0, (k, v, relative_timings)

                # The relative timings should sum up to one,
                # but this model is really cheap.
                # e.g. 0.00108 and 0.000604 per iteration.
                # This may cause the mismatch.
                # Allow a calculation error of 15%.
                # ToDo: Get this work with less than 1% error.
                relative_times = np.array(list(relative_timings.values())).sum(axis=0)
                if not np.all(relative_times > 0.85):
                    raise AssertionError(pretty((relative_times, time_per_iteration, dict(relative_timings))))
                if not np.all(relative_times <= 1):
                    raise AssertionError(pretty((relative_times, time_per_iteration, dict(relative_timings))))

            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 5, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    'ckpt_0.pth', 'ckpt_2.pth', 'ckpt_4.pth',
                    'ckpt_best_loss.pth', 'ckpt_latest.pth'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
                ckpt_ranking = torch.load(file / 'ckpt_latest.pth')['hooks']['BackOffValidationHook']['ckpt_ranking']
                assert ckpt_ranking[0][1] > 0, ckpt_ranking
                for i, ckpt in enumerate(ckpt_ranking):
                    ckpt_ranking[i] = (ckpt[0], -1)
                expect = [
                    (f'ckpt_{i}.pth', -1) for i in [0, 2, 4]
                ]
                assert ckpt_ranking == expect, (ckpt_ranking, expect)

                for symlink in [
                        file / 'ckpt_latest.pth',
                        file / 'ckpt_best_loss.pth',
                ]:
                    assert symlink.is_symlink(), symlink

                    target = os.readlink(str(symlink))
                    if '/' in target:
                        raise AssertionError(
                            f'The symlink {symlink} contains a "/".\n'
                            f'Expected that the symlink has a ralative target,\n'
                            f'but the target is: {target}'
                        )
            else:
                raise ValueError(file)

        post_state_dict = copy.deepcopy(t.state_dict())
        assert pre_state_dict.keys() == post_state_dict.keys()

        equal_amount = {
            key: (
                    pt.utils.to_numpy(parameter_pre)
                    == pt.utils.to_numpy(post_state_dict['model'][key])
            ).mean()
            for key, parameter_pre in pre_state_dict['model'].items()
        }

        # ToDo: why are so many weights unchanged? Maybe the zeros in the image?
        assert equal_amount == {'l.bias': 0.0, 'l.weight': 0.6900510204081632}

        import time
        # tfevents use unixtime as unique indicator. Sleep 2 seconds to ensure
        # new value
        time.sleep(2)

        config['stop_trigger'] = (4, 'epoch')
        t = pt.Trainer.from_config(config)
        t.register_validation_hook(
            validation_iterator=dt_dataset, max_checkpoints=None
        )
        log_list = []
        for hook in t.hooks:
            for k, v in list(hook.__dict__.items()):
                if isinstance(v, pt.train.trigger.Trigger):
                    hook.__dict__[k] = TriggerMock(v, log_list)
        t.train(train_dataset=tr_dataset, resume=True)

        hook_calls = ('\n'.join(log_list))

        hook_calls_ref = textwrap.dedent('''
        I:4, E: 2, False, SummaryHook.pre_step
        I:4, E: 2, False, BackOffValidationHook.pre_step
        I:4, E: 2, False, CheckpointHook.pre_step
        I:4, E: 2, False, StopTrainingHook.pre_step
        I:5, E: 2, False, SummaryHook.pre_step
        I:5, E: 2, False, BackOffValidationHook.pre_step
        I:5, E: 2, False, CheckpointHook.pre_step
        I:5, E: 2, False, StopTrainingHook.pre_step
        I:6, E: 3, True, SummaryHook.pre_step
        I:6, E: 3, True, BackOffValidationHook.pre_step
        I:6, E: 3, True, CheckpointHook.pre_step
        I:6, E: 3, False, StopTrainingHook.pre_step
        I:7, E: 3, False, SummaryHook.pre_step
        I:7, E: 3, False, BackOffValidationHook.pre_step
        I:7, E: 3, False, CheckpointHook.pre_step
        I:7, E: 3, False, StopTrainingHook.pre_step
        I:8, E: 4, False, SummaryHook.pre_step
        I:8, E: 4, True, BackOffValidationHook.pre_step
        I:8, E: 4, True, CheckpointHook.pre_step
        I:8, E: 4, True, StopTrainingHook.pre_step
        ''').strip()

        print('#' * 80)
        print(hook_calls)
        print('#' * 80)

        if hook_calls != hook_calls_ref:
            import difflib
            raise AssertionError(
                '\n' +
                ('\n'.join(difflib.ndiff(
                    hook_calls_ref.splitlines(),
                    hook_calls.splitlines(),
            ))))

        files_after = tuple(tmp_dir.glob('*'))
        assert len(files_after) == 3, files_after
        for file in sorted(files_after):
            if 'tfevents' in file.name:
                if file in old_event_files:
                    continue

                events = list(load_events_as_dict(file))

                tags = []
                for event in events:
                    if 'summary' in event.keys():
                        value, = event['summary']['value']
                        tags.append(value['tag'])

                c = dict(collections.Counter(tags))
                assert len(events) == 38, (len(events), events)
                expect = {
                    'training/grad_norm': 2,
                    'training/grad_norm_': 2,
                    'training/loss': 2,
                    'training/lr/param_group_0': 2,
                    'training_timings/time_per_iteration': 2,
                    'training_timings/time_rel_to_device': 2,
                    'training_timings/time_rel_forward': 2,
                    'training_timings/time_rel_review': 2,
                    'training_timings/time_rel_backward': 2,
                    'training_timings/time_rel_optimize': 2,
                    'training_timings/time_rel_data_loading': 2,
                    # 'training_timings/time_rel_step': 2,
                    'validation/loss': 2,
                    # 'validation/lr/param_group_0': 2,
                    'validation_timings/time_per_iteration': 2,
                    'validation_timings/time_rel_to_device': 2,
                    'validation_timings/time_rel_forward': 2,
                    'validation_timings/time_rel_review': 2,
                    'validation_timings/time_rel_data_loading': 2,
                    # 'validation_timings/time_rel_step': 2,
                    # non validation time can only be measured between
                    # validations:
                    #  => # of non_val_time - 1 == # of val_time
                    'validation_timings/non_validation_time': 1,
                    'validation_timings/validation_time': 2,
                }
                if c != expect:
                    import difflib

                    raise AssertionError(
                        '\n' + ('\n'.join(difflib.ndiff(
                            [f'{k!r}: {v!r}'for k, v in sorted(expect.items())],
                            [f'{k!r}: {v!r}'for k, v in sorted(c.items())],
                        )))
                    )
            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 7, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    *[f'ckpt_{i}.pth'for i in [0, 2, 4, 6, 8]],
                    'ckpt_best_loss.pth',
                    'ckpt_latest.pth'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)


def test_virtual_minibatch_few__examples():
    test_virtual_minibatch(3, 1)
    test_virtual_minibatch(4, 1)


def test_virtual_minibatch(
        no_of_examples=7,
        expected_iterations_per_epoch=2,
):
    """
    Test idea:

    Create an example with 7 entries and use a virtual_minibatch_size of 4.
    The trainer will then need two iterations (first 4 examples, second
    3 examples) for one epoch.

    Further it will also be tested, if the model changed.

    virtual_minibatch_size is choosen such, that the first train call only
    accumulates the gradients, but do not apply them.
    The second call to train (where stop_trigger is increased) runs once the
    optimizer step, so the parameters are changed.
    """

    it_tr, it_dt = get_dataset()
    it_tr = it_tr[:no_of_examples]
    it_dt = it_dt[:no_of_examples]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        config = pt.Trainer.get_config(
            updates=pb.utils.nested.deflatten({
                'model.factory': Model,
                'storage_dir': str(tmp_dir),
                'stop_trigger': (1, 'epoch'),
                'summary_trigger': (3, 'iteration'),
                'checkpoint_trigger': (2, 'iteration'),
                'virtual_minibatch_size': 4,  # 2 epochs
            })
        )

        t = pt.Trainer.from_config(config)
        t.register_validation_hook(
            validation_iterator=it_dt, max_checkpoints=None
        )
        pre_state_dict = copy.deepcopy(t.state_dict())

        t.train(train_dataset=it_tr, resume=False)
        post_state_dict = copy.deepcopy(t.state_dict())

        pre_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, pre_state_dict)

        post_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, post_state_dict)

        assert pre_state_dict['iteration'] == np.array(-1)
        del pre_state_dict['iteration']

        # 7 examples / 4 virtual_minibatch_size = 2 iterations
        assert post_state_dict['iteration'] == expected_iterations_per_epoch

        del post_state_dict['iteration']

        assert pre_state_dict['epoch'] == np.array(-1)
        del pre_state_dict['epoch']
        del pre_state_dict['hooks']

        assert post_state_dict['epoch'] == 1
        del post_state_dict['epoch']
        del post_state_dict['hooks']

        with pytest.raises(AssertionError):
            np.testing.assert_equal(pre_state_dict, post_state_dict)


def test_released_tensors():
    import gc
    gc.collect()

    tr_dataset, dt_dataset = get_dataset()
    tr_dataset = tr_dataset[:2]
    dt_dataset = dt_dataset[:2]

    class ReleaseTestHook(pt.train.hooks.Hook):
        def get_all_tensors(self):
            import gc
            tensors = []
            for obj in gc.get_objects():
                if isinstance(obj, torch.Tensor):
                    tensors.append(obj)
            return tensors

        def get_all_parameters(self, trainer):
            return list(trainer.model.parameters())

        def get_all_optimizer_tensors(self, trainer):
            def get_tensors(obj):
                if isinstance(obj, (dict, tuple, list)):
                    if isinstance(obj, dict):
                        obj = obj.values()
                    return list(
                        itertools.chain(*[get_tensors(o) for o in obj]))
                else:
                    if isinstance(obj, torch.Tensor):
                        return [obj]
                    else:
                        return []

            return get_tensors(trainer.optimizer.optimizer.state)

        @classmethod
        def show_referrers_type(cls, obj, depth, ignore=list()):
            # Debug function to get all references to an object and the
            # references to the references up to a depth of `depth`.
            import gc
            import textwrap
            import inspect
            l = []
            if depth > 0:
                referrers = gc.get_referrers(obj)
                for o in referrers:
                    if not any({o is i for i in ignore}):
                        for s in cls.show_referrers_type(
                                o, depth - 1,
                                ignore=ignore + [referrers, o, obj]
                        ):
                            l.append(textwrap.indent(s, '  '))

            if inspect.isframe(obj):
                frame_info = inspect.getframeinfo(obj, context=1)
                if frame_info.function == 'show_referrers_type':
                    pass
                else:
                    info = f' {frame_info.function}, {frame_info.filename}:{frame_info.lineno}'
                    l.append(str(type(obj)) + str(info))
            else:
                l.append(str(type(obj)) + str(obj)[:200].replace('\n', ' '))
            return l

        def pre_step(self, trainer: 'pt.Trainer'):
            all_tensors = self.get_all_tensors()
            parameters = self.get_all_parameters(trainer)
            optimizer_tensors = self.get_all_optimizer_tensors(trainer)

            for p in all_tensors:
                if 'grad_fn' in repr(p) or 'grad_fn' in str(p):
                    txt = "\n".join(self.show_referrers_type(p, 2))
                    raise AssertionError(
                        'Found a tensor that has a grad_fn\n\n' + txt
                    )

            summary = [
                t.shape
                for t in all_tensors
                if any([t is p for p in parameters])
            ]

            import textwrap
            assert len(all_tensors) == len(parameters) + len(optimizer_tensors), (
                f'pre_step\n'
                f'{summary}\n'
                f'all_tensors: {len(all_tensors)}\n'
                + textwrap.indent("\n".join(map(str, all_tensors)), " "*8) + f'\n'
                f'parameters: {len(parameters)}\n'
                + textwrap.indent("\n".join(map(str, parameters)), " "*8) + f'\n'
                f'optimizer_tensors: {len(optimizer_tensors)}\n'
                + textwrap.indent("\n".join(map(str, optimizer_tensors)), " "*8) + f'\n'
            )

        def post_step(
                self, trainer: 'pt.Trainer', example, model_output, review
        ):
            all_tensors = self.get_all_tensors()
            parameters = list(trainer.model.parameters())
            assert len(all_tensors) > len(parameters), ('post_step', all_tensors, parameters)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        t = pt.Trainer(
            Model(),
            optimizer=pt.optimizer.Adam(),
            storage_dir=str(tmp_dir),
            stop_trigger=(1, 'epoch'),
            summary_trigger=(1, 'epoch'),
            checkpoint_trigger=(1, 'epoch'),
        )
        t.register_validation_hook(
            validation_iterator=dt_dataset, max_checkpoints=None
        )
        t.register_hook(ReleaseTestHook())  # This hook will do the tests
        t.train(tr_dataset)
