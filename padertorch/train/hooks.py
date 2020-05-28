""" This module contains various hooks which perform actions during training.

Hooks replace a huge amount of conditions in the trainer code.
Having individual hook modules allows to enable and disable specific
functionality.

E.g., adding a learning rate schedule without adding further conditions to the
trainer.

"""
import types
from collections import defaultdict
from enum import IntEnum
from pathlib import Path

import numpy as np
import padertorch as pt
import torch
from distutils.version import LooseVersion
from natsort import natsorted
from padertorch.train.trigger import IntervalTrigger, EndTrigger
from tqdm import tqdm

tqdm.monitor_interval = 0


__all__ = [
    'SummaryHook',
    'CheckpointHook',
    'ValidationHook',
    'BackOffValidationHook',
    'ProgressBarHook',
    'StopTrainingHook',
    'StopTraining',
    'LossWeightAnnealingHook',
    'ModelAttributeAnnealingHook',
    'LRAnnealingHook',
]


class Priority(IntEnum):
    """
    Summary 50
    Print 40 NotImplemented
    ProgressBar(TQDM) 30 NotImplemented
    Validation 20
    Checkpoint 11
    End 10

    End has to be the last one
    Summary before Validation, clears timer information
    Print and ProgressBar may access Summary
    """
    END = 10
    CHECKPOINT = 11  # CheckpointHook has to be called after all other hooks (except StopTrainingHook) to save latest hook states
    DEFAULT = 15
    VALIDATION = 20
    PROGRESS = 30
    PRINT = 40
    SUMMARY = 50


class Hook:
    @property
    def priority(self):
        return Priority.DEFAULT

    @property
    def uid(self):
        """
        A unique ID of a hook.

        The default `uid` disallows to register a state full hook more than once
        in the trainer. Hooks that may registered more than once should
        overwrite this property (e.g. `ModelAttributeAnnealingHook`)
        """
        return type(self).__qualname__

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def pre_step(self, trainer: 'pt.Trainer'):
        """
        function is called before each iteration of the train iterator

        Args:
            trainer:

        Returns:

        """
        pass

    def post_step(self, trainer: 'pt.Trainer', example, model_output,
                  review):
        """
        function is called after each train step

        Args:
            trainer:
            example:
            model_output:
            review:

        Returns:
        """
        pass

    def close(self, trainer: 'pt.Trainer'):
        pass

    def set_last(self, iteration, epoch):
        pass


class TriggeredHook(Hook):

    def __init__(self, trigger=None):
        """

        Args:
            trigger: tuple or Trigger.
                When Tuple, the first entry is the trigger interval length and
                the second the unit (i.e. 'epoch' or 'iteration').
                Example: (1, 'epoch')
        """
        self.trigger = IntervalTrigger.new(trigger)

    def set_last(self, iteration, epoch):
        self.trigger.set_last(iteration, epoch)


class SummaryHook(TriggeredHook):
    """
    Responsible to write a summary in the tfevents file.
    The tfevents can be visualised in the tensorboard.

    The summary consists of the returned scalars, images, audios, etc of the
    training that are returned by the model review function.
    Note: It does not contain the learned model parameters, they are saved at
    the checkpoints.

    To save results of the validation refer to ValidationHook.
    """

    def __init__(
            self,
            trigger,
            summary_prefix='training',
    ):
        super().__init__(trigger)
        self.reset_summary()
        self.summary_prefix = summary_prefix

    @property
    def priority(self):
        return Priority.SUMMARY

    @staticmethod
    def empty_summary_dict():
        # MappingProxyType is similar to a frozen dict (does not exist)
        #   Ensures that no key is added.
        return types.MappingProxyType(dict(
            # losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict(),
            texts=dict(),
            figures=dict(),
            timings=dict(),
            buffers=dict(),
            snapshots=dict()
        ))

    def reset_summary(self):
        # Todo: add figures
        self.summary = self.empty_summary_dict()

    def update_summary(self, review):
        allowed_keys = {
            'loss',
            'losses',
            'scalars',
            'histograms',
            'audios',
            'images',
            'texts',
            'figures',
            'buffers',
            'snapshots'
        }
        redundant_keys = set(review.keys()) - allowed_keys
        assert len(redundant_keys) == 0, (redundant_keys, review.keys(), allowed_keys)

        popped_review = {**review}  # copy for "pop"

        # note item is the pytorch function to get the value of a tensor
        self.summary['scalars']['loss'].append(popped_review.pop('loss').item())
        for key, loss in popped_review.pop('losses', dict()).items():
            self.summary['scalars'][key].append(loss.item())
        for key, scalars in popped_review.pop('scalars', dict()).items():
            self.summary['scalars'][key].extend(self._to_list(scalars))
        for key, histogram in popped_review.pop('histograms', dict()).items():
            self.summary['histograms'][key].extend(self._to_list(histogram))
            # do not hold more than 1M values in memory
            self.summary['histograms'][key] = \
                self.summary['histograms'][key][-1000000:]
        for key, buffer in popped_review.pop('buffers', dict()).items():
            self.summary['buffers'][key].extend(self._detach(buffer))
        for key, snapshot in popped_review.pop('snapshots', dict()).items():
            self.summary['snapshots'][key] = snapshot  # snapshot
        for key, audio in popped_review.pop('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in popped_review.pop('images', dict()).items():
            self.summary['images'][key] = image  # snapshot
        for key, figure in popped_review.pop('figures', dict()).items():
            self.summary['figures'][key] = figure  # snapshot
        for key, text in popped_review.pop('texts', dict()).items():
            assert isinstance(text, str), text
            self.summary['texts'][key] = text  # snapshot

        assert len(popped_review) == 0, (popped_review, review)

    @staticmethod
    def _to_list(scalars):
        if torch.is_tensor(scalars):
            scalars = scalars.clone().cpu().data.numpy()
        if isinstance(scalars, np.ndarray):
            scalars = scalars.flatten().tolist()
        if not isinstance(scalars, (list, tuple)):
            assert np.isscalar(scalars)
            scalars = [scalars]
        return scalars

    @staticmethod
    def _detach(buffer):
        if torch.is_tensor(buffer):
            buffer = buffer.detach()
        return buffer

    def compute_timings(self, timer: 'pt.trainer.ContextTimerDict'):
        timer_dict = timer.as_dict
        # Special handling for time_per_data_loading and time_per_train_step
        #  Calculate
        #   - time_per_iteration: time of loading plus train step per iteration
        #   - time_rel_data_loading: time_for_loading / time_per_step
        #   - time_rel_step: time_train_step / time_per_step
        # Note: It is not guarantied that the size of time_per_data_loading and
        #       time_per_train_step is equal, because the Summary Hook is
        #       called between dataloading and train step. So the loading can
        #       be part of the previous summary, while the train step is in the
        #       next summary.
        time_per_data_loading = timer_dict.pop('time_per_data_loading', [0])
        time_per_step = timer_dict.pop('time_per_step', [0])
        time_per_to_device = timer_dict.pop('time_per_to_device', [0])
        time_per_forward = timer_dict.pop('time_per_forward', [0])
        time_per_review = timer_dict.pop('time_per_review', [0])
        time_per_backward = timer_dict.pop('time_per_backward', [0])

        summary_timings = {}
        time_per_iteration = (
                np.mean(time_per_data_loading) + np.mean(time_per_step)
        )
        if time_per_iteration > 0:
            summary_timings['time_per_iteration'] = time_per_iteration

            sum_time_per_step = np.sum(time_per_step)
            sum_time_per_data_loading = np.sum(time_per_data_loading)
            sum_time_per_train_step_to_device = np.sum(time_per_to_device)
            sum_time_per_train_step_forward = np.sum(time_per_forward)
            sum_time_per_train_step_review = np.sum(time_per_review)
            sum_time_per_backward = np.sum(time_per_backward)

            sum_time_per_iteration = (
                    sum_time_per_data_loading + sum_time_per_step
            )
            summary_timings['time_rel_data_loading'] = \
                sum_time_per_data_loading / sum_time_per_iteration
            summary_timings['time_rel_step'] = \
                sum_time_per_step / sum_time_per_iteration
            if sum_time_per_step > 0:
                summary_timings['time_rel_to_device'] = \
                    sum_time_per_train_step_to_device / sum_time_per_step
                summary_timings['time_rel_forward'] = \
                    sum_time_per_train_step_forward / sum_time_per_step
                summary_timings['time_rel_review'] = \
                    sum_time_per_train_step_review / sum_time_per_step
                summary_timings['time_rel_backward'] = \
                    sum_time_per_backward / sum_time_per_step
        summary_timings.update({
            key: timing.mean() for key, timing in timer_dict.items()
        })
        timer.clear()
        return summary_timings

    def maybe_add_lr_to_summary(self, trainer):
        if 'loss' not in self.summary['scalars']:
            return self.summary
        if isinstance(trainer.optimizer, dict):
            for key, optim in trainer.optimizer.items():
                for i, param_group in enumerate(optim.optimizer.param_groups):
                    self.summary['scalars'][f'lr/{key}/param_group_{i}'] = param_group['lr']
        else:
            for i, param_group in enumerate(trainer.optimizer.optimizer.param_groups):
                self.summary['scalars'][f'lr/param_group_{i}'] = param_group['lr']
        return self.summary

    def finalize_summary(self, trainer):
        assert len(self.summary['timings']) == 0, self.summary['timings']

        for key, timing in self.compute_timings(trainer.train_timer).items():
            self.summary['timings'][key] = timing
        self.maybe_add_lr_to_summary(trainer)
        self.summary = trainer.model.modify_summary(self.summary)
        # Assert the intermediate types were converted in he modify summary
        assert len(self.summary['buffers']) == 0, "intermediate format buffers has to be converted during modify_summary"
        assert len(self.summary['snapshots']) == 0, "intermediate format snapshots has to be converted during modify summary"

    def dump_summary(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        prefix = self.summary_prefix

        time_prefix = f'{prefix}_timings'

        tags = set()

        def check_tag(tag):
            if tag in tags:
                # ToDo: Find an issue that describes this problem.
                #       Once this is solved, we can remove this exception.
                raise AssertionError(
                    f'The tag {tag!r} is used multiple times.\n\n'
                    'Tensorboard has problems, when different events have the '
                    'same tag.\n'
                    'e.g. you cannot report the `grad_norm` as scalar and '
                    'histogram.\n'
                    'A common workaround is to use `grad_norm` for the scalar '
                    'and append an `_` for the histogram (i.e. `grad_norm_`).'
                )
            tags.add(tag)
            return tag

        for key, scalar in self.summary['scalars'].items():
            tag = check_tag(f'{prefix}/{key}')
            trainer.writer.add_scalar(tag, scalar, iteration)
        for key, scalar in self.summary['timings'].items():
            tag = check_tag(f'{time_prefix}/{key}')
            trainer.writer.add_scalar(tag, scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            tag = check_tag(f'{prefix}/{key}')
            trainer.writer.add_histogram(tag, np.array(histogram), iteration)
        for key, audio in self.summary['audios'].items():
            tag = check_tag(f'{prefix}/{key}')
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                trainer.writer.add_audio(
                    tag, audio[0], iteration, sample_rate=audio[1]
                )
            else:
                trainer.writer.add_audio(
                    tag, audio, iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            tag = check_tag(f'{prefix}/{key}')
            trainer.writer.add_image(tag, image, iteration)
        for key, text in self.summary['texts'].items():
            tag = check_tag(f'{prefix}/{key}')
            trainer.writer.add_text(tag, text, iteration)
        for key, figure in self.summary['figures'].items():
            tag = check_tag(f'{prefix}/{key}')
            trainer.writer.add_figure(tag, figure, iteration)

        self.reset_summary()

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            self.finalize_summary(trainer)
            self.dump_summary(trainer)

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        self.update_summary(review)

    def close(self, trainer: 'pt.Trainer'):
        self.finalize_summary(trainer)
        self.dump_summary(trainer)

    def set_last(self, iteration, epoch):
        self.reset_summary()  # The reset is done for back_off
        super().set_last(iteration, epoch)


class CheckpointHook(TriggeredHook):
    """ Periodically saves trainer state to a checkpoint
    """
    @property
    def priority(self):
        return Priority.CHECKPOINT

    def _save_latest_checkpoint(self, trainer: 'pt.Trainer'):
        """ Unconditionally save a checkpoint for the current model.
            This is needed for resume of training.
        """
        checkpoint_path: Path = trainer.default_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint()

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            self._save_latest_checkpoint(trainer)

    def close(self, trainer):
        self._save_latest_checkpoint(trainer)

    def set_last(self, iteration, epoch):
        if self.trigger.last[0] > iteration:
            super().set_last(-1, -1)
            # has to be triggered after back off
        else:
            super().set_last(iteration, epoch)


class ValidationHook(SummaryHook):
    """ Performs model validation and deletes stale checkpoints
    (checkpoints that are not among the max_checkpoints best checkpoints).

    ValidationHook tasks:
     - validate and collect summary
     - update best checkpoints according to metric
     - dump summary to tfevents file
     - remove stale checkpoints

    """

    def __init__(
            self, trigger, iterator, metric='loss', maximize=False,
            max_checkpoints=1, early_stopping_patience=None
    ):
        """

        Args:
            trigger: tuple or Trigger. Do note that trigger must be the same as
                (or a multiple of) the trigger used for checkpointing!!
            iterator: validation data iterator
            metric: summary key of the metric that is to be used to track best
                performance
            maximize: If True metric is to be maximized else minimized
            max_checkpoints: the maximal number of best checkpoints
                When max_checkpoints is None, keep all checkpoints.
            early_stopping_patience: the number of allowed degradations before
                stopping training. Should be larger than back_off_patience.
        """
        super().__init__(trigger, summary_prefix='validation')
        self.iterator = iterator
        self.metric = metric
        self.maximize = maximize
        self.max_checkpoints = max_checkpoints
        self.early_stopping_patience = early_stopping_patience
        self.ckpt_ranking = []
        self.n_degradations = 0
        self.last_validation = -1

    @property
    def priority(self):
        return Priority.VALIDATION

    @property
    def _best_ckpt_name(self):
        return f"ckpt_best_{self.metric}.pth"

    def state_dict(self):
        return {
            'ckpt_ranking': self.ckpt_ranking,
            'n_degradations': self.n_degradations,
        }

    def load_state_dict(self, state_dict):
        self.ckpt_ranking = state_dict['ckpt_ranking']
        self.n_degradations = state_dict['n_degradations']

    def finalize_summary(self, trainer):
        # Do not call `super().finalize_summary(trainer)`.
        # This function replaces `trainer.train_timer` with
        # `trainer.validate_timer` from the super function.
        assert len(self.summary['timings']) == 0, self.summary['timings']
        for key, timing in self.compute_timings(trainer.validate_timer).items():
            self.summary['timings'][key] = timing
        self.maybe_add_lr_to_summary(trainer)
        self.summary = trainer.model.modify_summary(self.summary)

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            self.run_validation(trainer)
            self.last_validation = trainer.iteration
        if (
            self.early_stopping_patience is not None
            and self.n_degradations > self.early_stopping_patience
        ):
            print(f'Early stopping after {trainer.epoch} epochs and'
                  f' {trainer.iteration} iterations')
            raise StopTraining

    def run_validation(self, trainer: 'pt.Trainer'):
        ckpt_dir = trainer.checkpoint_dir
        ckpt_path: Path = trainer.default_checkpoint_path()
        # note that ckpt_path does not exist at this moment but will be written
        # after validation such that the state of this hook, which will be
        # saved in the checkpoint, includes the latest validation result.
        # post_step asserts that checkpoint is written and sets symlink to the
        # current best checkpoint.
        assert all([len(value) == 0 for value in self.summary.values()]), self.summary
        assert len(trainer.validate_timer.timings) == 0, trainer.validate_timer
        print('Starting Validation')
        at_least_one_value = False
        for model_out, review in trainer.validate(self.iterator):
            at_least_one_value = True
            self.update_summary(review)
        if not at_least_one_value:
            raise Exception(
                f'Got an empty validation iterator: {self.iterator}'
            )
        self.finalize_summary(trainer)
        score = self.summary['scalars'][self.metric]
        self.dump_summary(trainer)
        assert len(trainer.validate_timer.timings) == 0, trainer.validate_timer
        print(f'Finished Validation. Mean {self.metric}: {score}')

        # Only save the relative checkpoint path, so the folder can be
        # moved.
        self.ckpt_ranking.append((ckpt_path.name, score))
        # Sort the ckpt_ranking according to the score. The first entry
        # will then be the best checkpoint. When two scores are identical
        # the older checkpoint wins.
        self.ckpt_ranking = natsorted(self.ckpt_ranking, key=lambda x: (
                -x[1] if self.maximize else x[1],  # score
                x[0],  # ckpt name
        ))
        if self.max_checkpoints is not None:
            for i in range(
                len(self.ckpt_ranking) - 1, self.max_checkpoints - 1, -1
            ):
                ckpt_name = self.ckpt_ranking[i][0]
                if ckpt_name == ckpt_path.name:
                    continue
                ckpt = ckpt_dir / ckpt_name
                if ckpt.exists():  # may not exist anymore after backoff
                    ckpt.unlink()
                self.ckpt_ranking.pop(i)
        if self.ckpt_ranking[0][0] != ckpt_path.name:
            self.n_degradations += 1
        else:
            self.n_degradations = 0

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        if trainer.iteration == self.last_validation:
            ckpt_dir = trainer.checkpoint_dir
            ckpt_path: Path = trainer.default_checkpoint_path()
            if not ckpt_path.exists():
                raise RuntimeError(
                    'Before each validation the CheckpointHook has to write '
                    f'a checkpoint.\n'
                    f'Could not find {ckpt_path}.\n'
                    f'Found only:\n'
                    f'{[str(file) for file in ckpt_dir.iterdir()]}'
                )
            best_ckpt_path = ckpt_dir / self._best_ckpt_name
            if best_ckpt_path.is_symlink():
                best_ckpt_path.unlink()
            best_ckpt_path.symlink_to(self.ckpt_ranking[0][0])

    def close(self, trainer: 'pt.Trainer'):
        ckpt_name = trainer.default_checkpoint_path().name
        if ckpt_name not in [ckpt[0] for ckpt in self.ckpt_ranking]:
            # add to ranking to make sure it is deleted after resume
            self.ckpt_ranking.append((ckpt_name, -np.inf if self.maximize else np.inf))


class BackOffValidationHook(ValidationHook):
    """ Performs model validation and deletes stale checkpoints
    (checkpoints that are not among the max_checkpoints best checkpoints).

    ValidationHook tasks:
     - validate and collect summary
     - update best checkpoints according to metric
     - dump summary to tfevents file
     - remove stale checkpoints

    """

    def __init__(
            self, trigger, iterator, metric='loss', maximize=False,
            max_checkpoints=1, early_stopping_patience=None, n_back_off=0,
            lr_update_factor=1 / 10, back_off_patience=None
    ):
        """

        Args:
            trigger: tuple or Trigger. Do note that trigger must be the same as
                (or a multiple of) the trigger used for checkpointing!!
            iterator: validation data iterator
            metric: summary key of the metric that is to be used to track best
                performance
            maximize: If True metric is to be maximized else minimized
            max_checkpoints: the maximal number of best checkpoints
                When max_checkpoints is None, keep all checkpoints.
            early_stopping_patience: the number of allowed degradations before
                stopping training. Should be larger than back_off_patience.
            n_back_off: number of times the best checkpoint is reloaded to
                continue training with an updated learning rate.
            lr_update_factor: the factor by which the lr is multiplied in case
                of back off. Should be smaller than 1.
            back_off_patience: the number of allowed degradations before
                backing off
        """
        super().__init__(
            trigger, iterator,
            metric=metric, maximize=maximize, max_checkpoints=max_checkpoints,
            early_stopping_patience=early_stopping_patience
        )

        self.remaining_back_offs = n_back_off
        self.lr_update_factor = lr_update_factor
        if n_back_off > 0:
            assert lr_update_factor < 1, lr_update_factor
            assert back_off_patience is not None
        self.back_off_patience = back_off_patience
        if early_stopping_patience is not None \
                and back_off_patience is not None:
            assert early_stopping_patience >= back_off_patience, (
                early_stopping_patience, back_off_patience
            )

    def state_dict(self):
        return {
            'remaining_back_offs': self.remaining_back_offs,
            **super().state_dict()
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        assert state_dict['remaining_back_offs'] <= self.remaining_back_offs, (state_dict['remaining_back_offs'], self.remaining_back_offs)
        self.remaining_back_offs = state_dict['remaining_back_offs']

    def run_validation(self, trainer: 'pt.Trainer'):
        super().run_validation(trainer)
        if (
            self.remaining_back_offs > 0
            and self.n_degradations > self.back_off_patience
        ):
            self._back_off(trainer)

    def _back_off(self, trainer: 'pt.Trainer'):
        best_ckpt = self.ckpt_ranking[0][0]
        print(f'Back off to {best_ckpt}.')

        ckpt_dir = trainer.checkpoint_dir
        latest_symlink_path = (ckpt_dir / f'ckpt_latest.pth').absolute()
        if latest_symlink_path.is_symlink():  # CB: Change to assert?
            latest_symlink_path.unlink()
        latest_symlink_path.symlink_to(best_ckpt)

        best_iter = int(best_ckpt[len('ckpt_'): -len('.pth')])
        for j in range(1, len(self.ckpt_ranking)):
            ckpt = self.ckpt_ranking[-j][0]
            if int(ckpt[len('ckpt_'): -len('.pth')]) > best_iter:
                ckpt_path = ckpt_dir / ckpt
                if ckpt_path.exists():  # latest checkpoint does not exist because it is written after validation
                    ckpt_path.unlink()
                    self.ckpt_ranking.pop(-j)

        remaining_back_offs = self.remaining_back_offs
        trainer.load_checkpoint()
        self.n_degradations = 0
        self.remaining_back_offs = remaining_back_offs - 1

        def update_lr(optim):
            for param_group in optim.optimizer.param_groups:
                param_group['lr'] *= self.lr_update_factor

        optimizer = trainer.optimizer
        if isinstance(optimizer, dict):
            [update_lr(optim) for optim in optimizer.values()]
        else:
            update_lr(optimizer)


class LRSchedulerHook(TriggeredHook):
    """
    A hook that applies a learning rate scheduler from `torch.optim.lr_scheduler`
    to the training.

    Examples:
        >>> trainer = pt.Trainer(...)   # doctest: +SKIP
        >>> trainer.register_hook(LRSchedulerHook(
        ...     torch.optim.lr_scheduler.StepLR(
        ...         trainer.optimizer.optimizer, step_size=2, gamma=0.98)
        ... ))  # doctest: +SKIP

    Note:

        This hook can only be used with learning rate schedulers that
        don't require metrics.

    """
    # It is very likely that this check is exclusive to this hook
    # See https://github.com/pytorch/pytorch/pull/7889 and
    # https://github.com/pytorch/pytorch/pull/20203
    PYTORCH_ge_1_1 = LooseVersion(torch.__version__) >= '1.1.0'

    def __init__(self, lr_scheduler, trigger=(1, 'epoch')):
        super().__init__(trigger)
        self.lr_scheduler = lr_scheduler

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            if trainer.epoch > 0 or not self.PYTORCH_ge_1_1:
                self.lr_scheduler.step()

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)

        # Call step instead of setting `last_epoch` directly because step
        # updates the LR of the optimizer. Note that this might print
        # a warning message in PyTorch 1.1+ if this is called before
        # the first optimizer step.
        self.lr_scheduler.step(epoch=epoch)


class ProgressBarHook(TriggeredHook):

    """ Adds a progress bar to the console output. """
    def __init__(self, stop_trigger, max_it_len=None, update_interval=100):
        """
        :param stop_trigger: has to be defined if max_trigger unit is session
            integer with the length of the iterator
        :param max_it_len (int): length of iterator, only used if max_trigger
            uses unit epoch
        :param update_interval (int): Number of iterations to skip printing the
            progress bar.
        :param bar_length (int): Length of the progress bar in characters.
        """
        super().__init__((update_interval, 'iteration'))
        if isinstance(stop_trigger, EndTrigger):
            length, unit = stop_trigger.period, stop_trigger.unit
        elif isinstance(stop_trigger, (tuple, list)):
            length, unit = stop_trigger
        else:
            raise ValueError(f'max_trigger is expected to be either a trigger '
                             f'or a list or tuple, but is {type(stop_trigger)},'
                             f'{stop_trigger}')
        if unit == 'iteration':
            max_iteration = length
        elif unit == 'epoch':
            if max_it_len is not None:
                max_iteration = length * max_it_len
            else:
                self.num_epochs = length
                max_iteration = None
        else:
            raise ValueError(f'unit {unit} is unknown,'
                             f' choose iteration or epoch')
        self.loss = None
        self.pbar = tqdm(initial=1, total=max_iteration)


    @property
    def priority(self):
        return Priority.PROGRESS

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.pbar.n = iteration

    def pre_step(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        epoch = trainer.epoch
        if epoch == 1 and self.pbar.total is None:
            if hasattr(self, 'num_epochs'):
                # sets the max length of the bar after the first epoch
                self.pbar.total = (iteration + 1) * self.num_epochs
        if self.trigger(iteration, epoch) and iteration > 1:
            self.pbar.update(iteration - self.pbar.n)

    # def post_step(self, trainer: 'pt.Trainer', example,
    #               model_output, review):
    #     self.loss = pt.utils.to_numpy(review["loss"])

    def close(self, trainer: 'pt.Trainer'):
        self.pbar.close()


class StopTrainingHook(TriggeredHook):
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


class AnnealingHook(TriggeredHook):
    def __init__(self, trigger, breakpoints, unit, name):
        """
        Base class for piece-wise linear annealing. The piece-wise linear
        function is parameterized by its breakpoints. It can also be used for
        arbitrary annealing functions when stating breakpoints with an interval
        similar to the trigger interval.
        The annealing function is interpreted relative to the initial value,
        i.e., a breakpoint (i, 1) corresponds to an absolute value equal to the
        initial value of the parameter to be annealed.
        Before the first breakpoint there is a linear connection between
        (0, 1), which corresponds to the initial value, and the first breakpoint.
        After the last breakpoint the function stays constant at the value of
        the last breakpoint.
        Note that you can still start with values differing from the initial
        value by adding a breakpoint (0, y0).

        Args:
            trigger:
            breakpoints: list of (x, y) coordinates of the piecewise linear
                function. x is either iteration or epoch (see unit argument).
                y values are interpreted relative to the initial value of the
                parameter to be annealed.
            unit: states the unit of the breakpoints: "iteration" or "epoch"
            name: name of the attribute. You may use "attr1.attr11" to
                anneal a sub attribute
        """
        super().__init__(trigger)
        self.breakpoints = sorted(breakpoints, key=lambda x: x[0])
        self.unit = unit
        self.name = name
        self.scale = None
        self.last_break = None

    @property
    def uid(self):
        return super().uid + f"({self.name})"

    def get_value(self, trainer):
        raise NotImplementedError

    def set_value(self, trainer, value):
        raise NotImplementedError

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            if self.last_break is None:
                self.scale = self.get_value(trainer)
                self.last_break = (0, 1)
            if self.unit == "iteration":
                x = trainer.iteration
            elif self.unit == "epoch":
                x = trainer.epoch
            else:
                raise ValueError(f'{self.unit} is not a valid unit.')
            while len(self.breakpoints) > 0 and self.breakpoints[0][0] <= x:
                self.last_break = self.breakpoints.pop(0)
            if len(self.breakpoints) > 0:
                slope = (
                    (self.breakpoints[0][1]-self.last_break[1])
                    / (self.breakpoints[0][0]-self.last_break[0])
                )  # a = (y1 - y0) / (x1 - x0)
                value = (
                    self.last_break[1]
                    + slope * (x - self.last_break[0])
                )  # y = y0 + a * (x - x0)
                value *= self.scale  # relative to absolute
            else:
                value = self.last_break[1] * self.scale
            self.set_value(trainer, value)


class LossWeightAnnealingHook(AnnealingHook):
    """
    Anneals a loss weight within the loss_weights dict of the trainer.
    """
    def get_value(self, trainer):
        return trainer.loss_weights[self.name]

    def set_value(self, trainer, value):
        trainer.loss_weights[self.name] = value


class ModelAttributeAnnealingHook(AnnealingHook):
    """
    Anneals an (sub)attribute of the trainers model.
    """

    def get_module(self, trainer: 'pt.Trainer'):
        module = trainer.model
        name = self.name.split('.')[:-1]
        for attr_name in name:
            module = getattr(module, attr_name)
        return module

    def get_value(self, trainer):
        module = self.get_module(trainer)
        attr_name = self.name.split('.')[-1]
        return getattr(module, attr_name)

    def set_value(self, trainer, value):
        module = self.get_module(trainer)
        attr_name = self.name.split('.')[-1]
        setattr(module, attr_name, value)


class LRAnnealingHook(AnnealingHook):
    """
    Anneals an optimizer learning rate.
    """
    def __init__(self, trigger, breakpoints, unit, name=None):
        """See docstring of AnnealingHook.

        Args:
            trigger:
            breakpoints:
            name: states the key of the target optimizer when optimizer is a dict
        """
        super().__init__(trigger, breakpoints, unit, name)

    @property
    def uid(self):
        if self.name is None:
            return super(AnnealingHook, self).uid  # uid from TriggeredHook
        else:
            return super().uid  # uid from AnnealingHook

    def get_optimizer(self, trainer):
        optimizer = trainer.optimizer
        if self.name is not None:
            assert isinstance(optimizer, dict)
            optimizer = optimizer[self.name]
        return optimizer.optimizer

    def get_value(self, trainer):
        return self.get_optimizer(trainer).param_groups[0]['lr']

    def set_value(self, trainer, value):
        optimizer = self.get_optimizer(trainer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = value
