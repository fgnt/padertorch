import tempfile
from pathlib import Path
import inspect
import copy
import textwrap
import collections
import copy

from IPython.lib.pretty import pprint
import pytest

import numpy as np
import torch

import tensorflow as tf
from google.protobuf.json_format import MessageToDict

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
        return {'loss': ce}


def get_dataset():
    db = pb.database.mnist.MnistDatabase()
    return (
        db.get_iterator_by_names('train'),
        db.get_iterator_by_names('test'),
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


def load_tfevents_as_dict(
        path
):
    """

    >> path = '/net/home/boeddeker/sacred/torch/am/32/events.out.tfevents.1545605113.ntsim1'
    >> load_tfevents_as_dict(path)[2]
    {'wall_time': 1545605119.7274427, 'step': 1, 'summary': {'value': [{'tag': 'training/grad_norm', 'simple_value': 0.21423661708831787}]}}

    """
    # MessageToDict(e, preserving_proto_field_name=True)
    #   Converts int to str
    return [
        MessageToDict(e)
        for e in tf.train.summary_iterator(str(path))
    ]


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
        t.train(train_iterator=tr_dataset, resume=False)

        hook_calls = ('\n'.join(log_list))

        # CheckpointedValidationHook trigger is called two times
        #   (once for checkpointing once for validation)_file_name

        hook_calls_ref = textwrap.dedent('''
        I:0, E: 0, True, SummaryHook.pre_step
        I:0, E: 0, True, CheckpointHook.pre_step
        I:0, E: 0, True, ValidationHook.pre_step
        I:0, E: 0, False, StopTrainingHook.pre_step
        I:1, E: 0, False, SummaryHook.pre_step
        I:1, E: 0, False, CheckpointHook.pre_step
        I:1, E: 0, False, ValidationHook.pre_step
        I:1, E: 0, False, StopTrainingHook.pre_step
        I:2, E: 1, False, SummaryHook.pre_step
        I:2, E: 1, True, CheckpointHook.pre_step
        I:2, E: 1, True, ValidationHook.pre_step
        I:2, E: 1, False, StopTrainingHook.pre_step
        I:3, E: 1, True, SummaryHook.pre_step
        I:3, E: 1, False, CheckpointHook.pre_step
        I:3, E: 1, False, ValidationHook.pre_step
        I:3, E: 1, False, StopTrainingHook.pre_step
        I:4, E: 2, False, SummaryHook.pre_step
        I:4, E: 2, True, CheckpointHook.pre_step
        I:4, E: 2, True, ValidationHook.pre_step
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
                events = load_tfevents_as_dict(file)

                tags = []
                time_rel_data_loading = []
                time_rel_train_step = []
                for event in events:
                    if 'summary' in event.keys():
                        value, = event['summary']['value']
                        tags.append(value['tag'])
                        if value['tag'] == 'training_timings/time_rel_data_loading':
                            time_rel_data_loading.append(value['simpleValue'])
                        elif value['tag'] == 'training_timings/time_rel_step':
                            time_rel_train_step.append(value['simpleValue'])

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
                    'training_timings/time_per_iteration': 2,
                    'training_timings/time_rel_to_device': 2,
                    'training_timings/time_rel_forward': 2,
                    'training_timings/time_rel_review': 2,
                    'training_timings/time_rel_backward': 2,
                    'training_timings/time_rel_data_loading': 2,
                    'training_timings/time_rel_step': 2,
                    'validation/loss': 3,
                    'validation_timings/time_per_iteration': 3,
                    'validation_timings/time_rel_to_device': 3,
                    'validation_timings/time_rel_forward': 3,
                    'validation_timings/time_rel_review': 3,
                    'validation_timings/time_rel_backward': 3,
                    'validation_timings/time_rel_data_loading': 3,
                    'validation_timings/time_rel_step': 3,
                    # non validation time can only be measured between
                    # validations:
                    #  => # of non_val_time - 1 == # of val_time
                    'validation_timings/non_validation_time': 2,
                    'validation_timings/validation_time': 3,
                }
                pprint(c)
                assert c == expect, c
                assert len(events) == 50, (len(events), events)

                assert len(time_rel_data_loading) > 0, (time_rel_data_loading, time_rel_train_step)
                assert len(time_rel_train_step) > 0, (time_rel_data_loading, time_rel_train_step)
                np.testing.assert_allclose(
                    np.add(time_rel_data_loading, time_rel_train_step),
                    1,
                    err_msg=f'{time_rel_data_loading}, {time_rel_train_step})'
                )

            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 6, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    'ckpt_0.pth', 'ckpt_2.pth', 'ckpt_4.pth',
                    'ckpt_ranking.json', 'ckpt_best_loss.pth',
                    'ckpt_latest.pth'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
                ckpt_ranking = pb.io.load_json(file / 'ckpt_ranking.json')
                assert ckpt_ranking[0][1] > 0, ckpt_ranking
                for ckpt in ckpt_ranking:
                    ckpt[1] = -1
                expect = [
                    [str(t.checkpoint_dir / f'ckpt_{i}.pth'), -1]
                    for i in [0, 2, 4]
                ]
                assert ckpt_ranking == expect, (ckpt_ranking, expect)

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
        t.train(train_iterator=tr_dataset, resume=True)

        hook_calls = ('\n'.join(log_list))

        hook_calls_ref = textwrap.dedent('''
        I:4, E: 2, False, SummaryHook.pre_step
        I:4, E: 2, False, CheckpointHook.pre_step
        I:4, E: 2, False, ValidationHook.pre_step
        I:4, E: 2, False, StopTrainingHook.pre_step
        I:5, E: 2, False, SummaryHook.pre_step
        I:5, E: 2, False, CheckpointHook.pre_step
        I:5, E: 2, False, ValidationHook.pre_step
        I:5, E: 2, False, StopTrainingHook.pre_step
        I:6, E: 3, True, SummaryHook.pre_step
        I:6, E: 3, True, CheckpointHook.pre_step
        I:6, E: 3, True, ValidationHook.pre_step
        I:6, E: 3, False, StopTrainingHook.pre_step
        I:7, E: 3, False, SummaryHook.pre_step
        I:7, E: 3, False, CheckpointHook.pre_step
        I:7, E: 3, False, ValidationHook.pre_step
        I:7, E: 3, False, StopTrainingHook.pre_step
        I:8, E: 4, False, SummaryHook.pre_step
        I:8, E: 4, True, CheckpointHook.pre_step
        I:8, E: 4, True, ValidationHook.pre_step
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

                events = load_tfevents_as_dict(file)

                tags = []
                for event in events:
                    if 'summary' in event.keys():
                        value, = event['summary']['value']
                        tags.append(value['tag'])

                c = dict(collections.Counter(tags))
                assert len(events) == 40, (len(events), events)
                expect = {
                    'training/grad_norm': 2,
                    'training/grad_norm_': 2,
                    'training/loss': 2,
                    'training_timings/time_per_iteration': 2,
                    'training_timings/time_rel_to_device': 2,
                    'training_timings/time_rel_forward': 2,
                    'training_timings/time_rel_review': 2,
                    'training_timings/time_rel_backward': 2,
                    'training_timings/time_rel_data_loading': 2,
                    'training_timings/time_rel_step': 2,
                    'validation/loss': 2,
                    'validation_timings/time_per_iteration': 2,
                    'validation_timings/time_rel_to_device': 2,
                    'validation_timings/time_rel_forward': 2,
                    'validation_timings/time_rel_review': 2,
                    'validation_timings/time_rel_backward': 2,
                    'validation_timings/time_rel_data_loading': 2,
                    'validation_timings/time_rel_step': 2,
                    # non validation time can only be measured between
                    # validations:
                    #  => # of non_val_time - 1 == # of val_time
                    'validation_timings/non_validation_time': 1,
                    'validation_timings/validation_time': 2,
                }
                assert c == expect, c
            elif file.name == 'checkpoints':
                checkpoints_files = tuple(file.glob('*'))
                assert len(checkpoints_files) == 8, checkpoints_files
                checkpoints_files_name = [
                    f.name
                    for f in checkpoints_files
                ]
                expect = {
                    *[f'ckpt_{i}.pth'for i in [0, 2, 4, 6, 8]],
                    'ckpt_ranking.json',
                    'ckpt_best_loss.pth',
                    'ckpt_latest.pth'
                }
                assert expect == set(checkpoints_files_name), (
                    expect, checkpoints_files_name
                )
            else:
                raise ValueError(file)


def test_virtual_minibatch():
    """
    Test idea:

    virtual_minibatch_size is choosen such, that the first train call only
    accumulates the gradients, but do not apply them.
    The second call to train (where stop_trigger is increased) runs once the
    optimizer step, so the parameters are changed.
    """

    it_tr, it_dt = get_dataset()
    it_tr = it_tr[:2]
    it_dt = it_dt[:2]

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

        t.train(train_iterator=it_tr, resume=False)
        intermediate_state_dict = copy.deepcopy(t.state_dict())

        # Increase train end from 1 epoch to 2 epochs.
        # The first time that virtual_minibatch_size triggers is at the end of
        # epoch 2.
        config['stop_trigger'] = (2, 'epoch')

        t = pt.Trainer.from_config(config)
        t.register_validation_hook(
            validation_iterator=it_dt, max_checkpoints=None
        )
        t.train(train_iterator=it_tr, resume=True)
        post_state_dict = copy.deepcopy(t.state_dict())

        pre_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, pre_state_dict)
        intermediate_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, intermediate_state_dict)
        post_state_dict = pb.utils.nested.nested_op(
            pt.utils.to_numpy, post_state_dict)

        assert pre_state_dict['iteration'] == np.array(None)
        del pre_state_dict['iteration']
        assert intermediate_state_dict['iteration'] == 2
        del intermediate_state_dict['iteration']
        assert post_state_dict['iteration'] == 4
        del post_state_dict['iteration']

        assert pre_state_dict['epoch'] == np.array(None)
        del pre_state_dict['epoch']
        assert intermediate_state_dict['epoch'] == 1
        del intermediate_state_dict['epoch']
        assert post_state_dict['epoch'] == 2
        del post_state_dict['epoch']

        np.testing.assert_equal(pre_state_dict, intermediate_state_dict)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(pre_state_dict, post_state_dict)
