""" This module contains various hooks which perform actions during training.
"""
from collections import defaultdict
from enum import IntEnum
import json
import os
import copy
from pathlib import Path

import numpy as np
import torch
from cached_property import cached_property
from tensorboardX import SummaryWriter

from padertorch.train.trigger import IntervalTrigger, EndTrigger, OrTrigger


__all__ = [
    'SummaryHook',
    'SimpleCheckpointHook',
    'ValidationHook',
    'CheckpointedValidationHook',
    'ProgressBarHook',
    'StopTrainingHook',
    'StopTraining',
    'CKPT_EXT',
]


CKPT_EXT = 'pth'


class Priority(IntEnum):
    """
    Summary 50
    Print 40 NotImplemented
    ProgressBar(TQDM) 30 NotImplemented
    Validation 25
    Checkpoint 20
    End 10

    End has to be the last one
    Summary before Validation, clears timer information
    Print and ProgressBar may access Summary
    """
    END = 10
    DEFAULT = 15
    VALIDATION = 20
    CHECKPOINT = 25
    PROGRESS = 30
    PRINT = 40
    SUMMARY = 50


class BaseHook:

    def __init__(self, trigger=None):
        """
        :param trigger: Length of step between occurences or Trigger.
            It consists of an integer and either 'epoch' or 'iteration'
        """
        # if trigger is not None:  # CB: use case?
        self.trigger = IntervalTrigger.new(trigger)

    @property
    def priority(self):
        return Priority.DEFAULT

    def pre_step(self, trainer: 'pt.Trainer'):
        """
        function is called before each iteration of the train iterator
        :param trainer:
        :return:
        """
        pass

    def post_step(self, trainer: 'pt.Trainer', example, model_output,
                  review):
        """
        function is called after each train step
        :param trainer:
        :param example:
        :param model_output:
        :param review:
        :return:
        """
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass

    def set_last(self, iteration, epoch):
        self.trigger.set_last(iteration, epoch)


class SummaryHook(BaseHook):
    def __init__(self, trigger,
                 summary_prefix='training'):
        super().__init__(trigger)
        self.reset_summary()
        self.summary_prefix = summary_prefix
        self.storage_dir = None

    @property
    def priority(self):
        return Priority.SUMMARY

    @cached_property
    def writer(self):
        return SummaryWriter(str(self.storage_dir),
                             filename_suffix=self.summary_prefix)

    @staticmethod
    def empty_summary_dict():
        return dict(
            losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict()
        )

    def reset_summary(self):
        # Todo: add figures
        self.summary = self.empty_summary_dict()

    def update_summary(self, review):
        for key, loss in review.get('losses', dict()).items():
            self.summary['losses'][key].append(loss.item())
        for key, scalar in review.get('scalars', dict()).items():
            self.summary['scalars'][key].append(
                scalar.item() if torch.is_tensor(scalar) else scalar)
        for key, histogram in review.get('histograms', dict()).items():
            self.summary['histograms'][key] = np.concatenate(
                [self.summary['histograms'].get(key, np.zeros(0)),
                 histogram.clone().cpu().data.numpy().flatten()]
            )[-10000:]  # do not hold more than 10K values in memory
        for key, audio in review.get('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in review.get('images', dict()).items():
            self.summary['images'][key] = image  # snapshot

    def dump_summary(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        timer = trainer.timer
        prefix = self.summary_prefix
        for key, loss in self.summary['losses'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(loss), iteration)
        for key, scalar in self.summary['scalars'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(scalar), iteration)
        for key, scalar in timer.as_dict.items():
            if key in ['time_per_data_loading', 'time_per_train_step']:
                if 'time_per_step' in timer.as_dict.keys():
                    time_per_step = timer.as_dict['time_per_step']
                    if len(time_per_step) != len(scalar):
                        print(
                            'Warning: padertorch.Trainer timing bug.'
                            f'len(time_per_step) == {len(time_per_step)} '
                            f'!= len(scalar) == {len(scalar)}'
                        )
                    scalar = (
                        scalar.sum() / time_per_step.sum()
                    )
                    if key == 'time_per_data_loading':
                        key = 'time_rel_data_loading'
                    elif key == 'time_per_train_step':
                        key = 'time_rel_train_step'
                else:
                    # Something went wrong, most likely an exception.
                    pass
            self.writer.add_scalar(
                f'{prefix}/{key}', scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            self.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), iteration
            )
        for key, audio in self.summary['audios'].items():
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                self.writer.add_audio(
                    f'{prefix}/{key}', audio[0],
                    iteration, sample_rate=audio[1]
                )
            else:
                self.writer.add_audio(
                    f'{prefix}/{key}', audio,
                    iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            self.writer.add_image(f'{prefix}/{key}', image, iteration)
        self.reset_summary()
        trainer.reset_timer()

    def pre_step(self, trainer: 'pt.Trainer'):
        if(self.trigger(iteration=trainer.iteration, epoch=trainer.epoch)
           or trainer.iteration == 1):
            self.dump_summary(trainer)

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        if self.storage_dir is None:
            self.storage_dir = trainer.storage_dir
        else:
            assert self.storage_dir == trainer.storage_dir
        self.update_summary(review)

    def close(self, trainer: 'pt.Trainer'):
        self.dump_summary(trainer)


class SimpleCheckpointHook(BaseHook):
    """ Can be used to keep all checkpoints, e.g. for continuous evaluation
            (keep_all = False) or to only store the most recent checkpoint
            (keep_all = True).
            Cannot be used together with a CheckpointedValidationHook
    """
    def __init__(self, trigger, keep_all=False):
        super().__init__(trigger)
        self.keep_all = keep_all
        self.last_checkpoint_path = None

    @property
    def priority(self):
        return Priority.CHECKPOINT

    def pre_step(self, trainer: 'pt.Trainer'):
        checkpoint_path = trainer.default_checkpoint_path()
        trainer.save_checkpoint(checkpoint_path)
        if not(self.keep_all) and self.last_checkpoint_path.exists():
            self.last_checkpoint_path.unlink()
        self.last_checkpoint_path = checkpoint_path


class ValidationHook(SummaryHook):
    def __init__(self, trigger, iterator):
        super().__init__(trigger, summary_prefix='validation')
        self.iterator = iterator

    @property
    def priority(self):
        return Priority.VALIDATION

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            assert all([len(value) == 0 for value in self.summary.values()])
            assert len(trainer.timer.timings) == 0, trainer.timer
            print('Starting Validation')
            for model_out, review in trainer.validate(self.iterator):
                self.update_summary(review)
            self.dump_summary(trainer)
            assert len(trainer.timer.timings) == 0, trainer.timer
            print('Finished Validation')

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass


class _Metric:
    """ Bookkeeping of metrics (comparison, best value, checkpoint path,
        symlink) needed for CheckpointedValidationHook.
    """
    def __init__(self, metric_key, criterion, checkpoint_dir):
        self._key = metric_key
        self._criterion = criterion
        self._checkpoint_dir = checkpoint_dir
        self._symlink_name = f'ckpt_best_{metric_key}.{CKPT_EXT}'

        assert criterion in ('min', 'max'), criterion
        self._value = float('inf') if criterion == 'min' else -float('inf')

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self._key!r}, {self.criterion!r}, '
            f'best={self.resolved_symlink_path}'
            f')'
        )

    @property
    def name(self):
        return self._key

    @property
    def paths(self):
        return ([self.resolved_symlink_path]
                if self.resolved_symlink_path.exists() else [])

    @property
    def values(self):
        return [self._value] if abs(self._value) != float('Inf') else []

    def is_better(self, value):
        """ Decides whether current metric value is better than best
            previous one. Has to work for cost and gain objectives
            => See init for details.
        """
        if self._criterion == 'min':
            return value <= self._value
        elif self._criterion == 'max':
            return value >= self._value
        else:
            raise AssertionError(f'Should not ne reachable: {self._criterion}')

    def update(self, value, checkpoint_path):
        """ Update to best metric value, corresponding checkpoint path
            and set symlink to checkpoint.
        """
        self._value = value
        # create relative symlink to best checkpoint for metric
        if self.resolved_symlink_path.exists:
            self.resolved_symlink_path.unlink()
        self.resolved_symlink_path.symlink_to(checkpoint_path.name)

    @property
    def resolved_symlink_path(self):
        return (self._checkpoint_dir / self._symlink_name).resolve()

    def to_json(self):
        """ Dump metric state information into dictionary. """
        return dict(key=self._key,
                    criterion=self._criterion,
                    values=self.values,
                    paths=[path.relative_to(self._checkpoint_dir)
                           for path in self.paths])

    def set_state(self, state_dict):
        """ Set state of the metrics object from a state dictionary.
            This is usually useful when the state_dict stems from a json file
            and was previously created by to_json.
        """
        assert self._key == state_dict['key'], (self._key, state_dict['key'])
        assert self._criterion == state_dict['criterion'], (
            self._criterion, state_dict['criterion'])
        assert len(state_dict['paths']) == len(state_dict['values']), \
            state_dict
        if len(state_dict['paths']) == 0:
            pass
        elif len(state_dict['paths']) == 1:
            assert [self._symlink_name] == state_dict['paths'], (
                self._symlink_name, state_dict['paths'])
            self._value = state_dict['values'][0]
        else:
            assert False, state_dict['paths']


class CheckpointedValidationHook(ValidationHook):
    """ Performs model validation and keeps checkpoints for model states that
        perform best on a given set of metrics.
        Cannot be used together with a ValidationHook
        or a SimpleCheckpointHook.

    Checkpointing and validation:
     - save checkpoint
     - validate and collect summary
     - update best checkpoints according to metrics
     - dump summary to tfevents file
     - cleanup stale checkpoints
     - save CheckpointedValidationHook state in `_json_filename`

    """
    _json_filename = 'ckpt_state.json'

    def __init__(self, trigger, iterator, checkpoint_dir: Path,
                 metrics=None, keep_all=False, init_from_json=False):
        super().__init__(trigger, iterator)

        # ToDo: remove the checkpoint_trigger, see pre_step
        self.checkpoint_trigger = IntervalTrigger.new(trigger)

        assert isinstance(metrics, dict) and metrics,  \
            'The metrics dict must not be empty!'
        self._checkpoint_dir = checkpoint_dir
        self.metrics = self._convert_metrics_to_internal_layout(metrics)
        self._keep_all = keep_all
        if init_from_json:
            json_path = checkpoint_dir / self._json_filename
            assert checkpoint_dir.exists(), checkpoint_dir
            with json_path.open('r') as json_fd:
                json_state = json.load(json_fd)
            self.latest_checkpoint = Path(json_state['latest_checkpoint_path'])
            assert set(self.metrics.keys()) == set(json_state['metrics'].keys()), (
                self.metrics, json_state)
            for metric_key, metric in self.metrics.items():
                metric.set_state(json_state['metrics'][metric_key])
        else:
            if checkpoint_dir.exists():
                assert checkpoint_dir.is_dir(), checkpoint_dir
                assert len(list(checkpoint_dir.iterdir())) == 0, checkpoint_dir
            else:
                checkpoint_dir.mkdir()
            self.latest_checkpoint = None

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.checkpoint_trigger.set_last(iteration, epoch)

    def pre_step(self, trainer: 'pt.Trainer'):
        # A trigger triggers only once. So it is important to use here a copy
        # of self.trigger and super can use the original trigger.
        if (
                self.checkpoint_trigger(iteration=trainer.iteration, epoch=trainer.epoch)
                or trainer.iteration == 1
        ):

            self._save_latest_checkpoint(trainer)
        super().pre_step(trainer)

    def dump_summary(self, trainer: 'pt.Trainer'):
        """ This class needs to overload the dump_summary - even if the naming
            is suboptimal - because the ValidationHook class produces the
            necessary metrics in its pre_step and immediately calls
            dump_summary. However, the implementation in SummaryHook clears
            the summary content.
        """
        self._update_validated_checkpoints()
        super().dump_summary(trainer)
        self._cleanup_stale_checkpoints()
        self.dump_json()

    def dump_json(self):
        """ Store the state information of the hok object to a json.
        """
        assert all(metric_key == metric.name
                   for metric_key, metric in self.metrics.items()), \
            'Some metric keys do not match their names!'
        json_path = self._checkpoint_dir / self._json_filename
        content = dict(
            latest_checkpoint_path=str(self.latest_checkpoint),
            metrics={metric_key: metric.to_json()
                     for metric_key, metric in self.metrics.items()})
        with json_path.open('w') as json_file:
            json.dump(content, json_file)

    def close(self, trainer: 'pt.Trainer'):
        self._save_latest_checkpoint(trainer)
        self.dump_json()

    @property
    def best_checkpoints(self):
        """ return a set of the checkpoints referenced as best by the metrics.
        """
        return {path
                for metric in self.metrics.values()
                for path in metric.paths}

    def _convert_metrics_to_internal_layout(self, metrics):
        return {metric_key: _Metric(metric_key, criterion,
                                    self._checkpoint_dir)
                for metric_key, criterion in metrics.items()}

    def _save_latest_checkpoint(self, trainer: 'pt.Trainer'):
        """ Unconditionally save a checkpoint for the current model.
            This is needed for resume of training.
        """
        checkpoint_path = trainer.default_checkpoint_path()

        trainer.save_checkpoint(checkpoint_path)
        self.latest_checkpoint = checkpoint_path

    def _update_validated_checkpoints(self):
        """ Save a checkpoint if the current model improves one or multiple
            validation metrics, dump the metric information to a json file
            and remove old checkpoints.
        """
        for metric_key, metric in self.metrics.items():
            summary_value = np.mean(self.summary['scalars'][metric_key])
            if metric.is_better(summary_value):
                metric.update(summary_value, self.latest_checkpoint)

    def _cleanup_stale_checkpoints(self):
        """ Remove all checkpoints that became stale (i.e. have no associated
            metric where they perform best anymore).
        """
        if self._keep_all:
            return
        used_checkpoints = self.best_checkpoints | {self.latest_checkpoint}
        stored_checkpoints = [
            path for path in self._checkpoint_dir.glob(f'ckpt_*.{CKPT_EXT}')
            if path.is_file() and not path.is_symlink()]
        for checkpoint in stored_checkpoints:
            if checkpoint not in used_checkpoints:
                checkpoint.unlink()


class ProgressBarHook(BaseHook):
    """ Adds a progress bar to the console output. """
    def __init__(self, max_trigger, max_it_len=None, update_intervall=1):
        from progressbar import bar, UnknownLength
        """
        :param max_trigger: has to be defined if max_trigger unit is session
            integer with the length of the iterator
        :param max_it_len (int): length of iterator, only used if max_trigger
            uses unit epoch
        :param update_interval (int): Number of iterations to skip printing the
            progress bar.
        :param bar_length (int): Length of the progress bar in characters.
        :param disable: bool use to disable the entire progressbar wrapper
        """
        super().__init__((update_intervall, 'iteration'))
        self.ep_trigger = IntervalTrigger(1, 'epoch')
        self.update_intervall = update_intervall
        if isinstance(max_trigger, EndTrigger):
            length, unit = max_trigger.period, max_trigger.unit
        elif isinstance(max_trigger, (tuple, list)):
            length, unit = max_trigger
        else:
            raise ValueError(f'max_trigger is expected to be either a trigger'
                             f'or a list or tuple, but is {type(max_trigger)},'
                             f'{max_trigger}')
        if unit == 'iteration':
            max_iteration = length
        elif unit == 'epoch':
            if max_it_len is not None:
                max_iteration = length * max_it_len
            else:
                max_iteration = UnknownLength
        else:
            raise ValueError(f'unit {unit} is unknown,'
                             f' choose iteration or epoch')
        # self.pbar = tqdm(desc=f'epochs: {0}', total=max_iteration,
        #                  file=sys.stdout, ncols=ncols,
        #                  dynamic_ncols=dynamic_ncols)
        self.pbar = bar.ProgressBar(
            prefix=f'epochs: {0}',
            max_value=max_iteration,
            redirect_stderr=True,
            redirect_stdout=True
        )

    @property
    def priority(self):
        return Priority.PROGRESS

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.ep_trigger.set_last(iteration, epoch)
        self.pbar.value = iteration

    def post_step(self, trainer: 'pt.Trainer', example,
                  model_output, review):
        from progressbar import UnknownLength
        iteration = trainer.iteration
        epoch = trainer.epoch
        if self.trigger(iteration, epoch):
            self.pbar.update(self.update_intervall)
            if len(review['losses']) == 1:
                self.pbar.prefix = f'epochs: {epoch}, loss: ' \
                                 f'{list(review["losses"].values())[0]}'
        if (self.ep_trigger(iteration, epoch)
                and self.pbar.max_value is UnknownLength):
            self.pbar.max_value = iteration * epoch

    def close(self, trainer: 'pt.Trainer'):
        self.pbar.finish()


class StopTrainingHook(BaseHook):
    """ Raises a StopTraining exception if triggered. """
    def __init__(self, trigger):
        super().__init__(EndTrigger.new(trigger))

    @property
    def priority(self):
        return Priority.END

    def pre_step(self, trainer):
        if self.trigger(trainer.iteration, trainer.epoch):
            print(f'Training ended after {trainer.epoch} epochs and'
                  f' {trainer.iteration} iterations')
            raise StopTraining


class StopTraining(Exception):
    """ Rationale: Raised as signal to stop the training
        (e.g. when predefined number of iterations are completed.)
    """
    pass
