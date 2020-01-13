""" This module contains various hooks which perform actions during training.

Hooks replace a huge amount of conditions in the trainer code.
Having individual hook modules allows to enable and disable specific
functionality.

E.g., adding a learning rate schedule without adding further conditions to the
trainer.

"""
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
import types

from distutils.version import LooseVersion
import numpy as np
import progressbar
import torch
from natsort import natsorted

import paderbox as pb
import padertorch as pt
from padertorch.train.trigger import IntervalTrigger, EndTrigger

__all__ = [
    'SummaryHook',
    'CheckpointHook',
    'ValidationHook',
    'BackOffValidationHook',
    'ProgressBarHook',
    'StopTrainingHook',
    'StopTraining',
]


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


class Hook:
    @property
    def priority(self):
        return Priority.DEFAULT

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
        }
        redundant_keys = set(review.keys()) - allowed_keys
        assert len(redundant_keys) == 0, (redundant_keys, review.keys(), allowed_keys)

        poped_review = {**review}  # copy for "pop"

        # note item is the pytorch function to get the value of a tensor
        self.summary['scalars']['loss'].append(poped_review.pop('loss').item())
        for key, loss in poped_review.pop('losses', dict()).items():
            self.summary['scalars'][key].append(loss.item())
        for key, scalars in poped_review.pop('scalars', dict()).items():
            self.summary['scalars'][key].extend(self._to_list(scalars))
        for key, histogram in poped_review.pop('histograms', dict()).items():
            self.summary['histograms'][key].extend(self._to_list(histogram))
            # do not hold more than 1M values in memory
            self.summary['histograms'][key] = \
                self.summary['histograms'][key][-1000000:]
        for key, audio in poped_review.pop('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in poped_review.pop('images', dict()).items():
            self.summary['images'][key] = image  # snapshot
        for key, figure in poped_review.pop('figures', dict()).items():
            self.summary['figures'][key] = figure  # snapshot
        for key, text in poped_review.pop('texts', dict()).items():
            assert isinstance(text, str), text
            self.summary['texts'][key] = text  # snapshot

        assert len(poped_review) == 0, (poped_review, review)

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

    def dump_summary(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        prefix = self.summary_prefix

        time_prefix = f'{prefix}_timings'

        for key, scalar in self.summary['scalars'].items():
            trainer.writer.add_scalar(
                f'{prefix}/{key}', scalar, iteration)
        for key, scalar in self.summary['timings'].items():
            trainer.writer.add_scalar(
                f'{time_prefix}/{key}', scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            trainer.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), iteration
            )
        for key, audio in self.summary['audios'].items():
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                trainer.writer.add_audio(
                    f'{prefix}/{key}', audio[0],
                    iteration, sample_rate=audio[1]
                )
            else:
                trainer.writer.add_audio(
                    f'{prefix}/{key}', audio,
                    iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            trainer.writer.add_image(f'{prefix}/{key}', image, iteration)
        for key, text in self.summary['texts'].items():
            trainer.writer.add_text(f'{prefix}/{key}', text, iteration)
        for key, figure in self.summary['figures'].items():
            trainer.writer.add_figure(f'{prefix}/{key}', figure, iteration)

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


class ValidationHook(SummaryHook):
    """ Performs model validation and deletes stale checkpoints
    (checkpoints that are not among the max_checkpoints best checkpoints).

    ValidationHook tasks:
     - validate and collect summary
     - update best checkpoints according to metric
     - dump summary to tfevents file
     - remove stale checkpoints
     - save checkpoint ranking in `_json_filename`

    """
    _json_filename = 'validation_state.json'

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
        self._json_file = None
        self.ckpt_ranking = []
        self.n_degradations = 0

    @property
    def priority(self):
        return Priority.VALIDATION

    @property
    def json_file(self):
        if self._json_file is None:
            raise RuntimeError(
                'The property json_file will be lazy setted in the pre_step'
                'function.\n'
                'The trainer knows, where to store the "validation_state" and '
                'this hooks sees the trainer first in the pre_step.'
            )
        return self._json_file

    @property
    def _best_ckpt_name(self):
        return f"ckpt_best_{self.metric}.pth"

    def save_validation_state(self):
        validation_state = {
            'ckpt_ranking': self.ckpt_ranking,
            'n_degradations': self.n_degradations,
        }
        pb.io.dump_json(validation_state, self.json_file)

    def load_validation_state(self):
        validation_state = pb.io.load_json(self.json_file)
        self.ckpt_ranking = validation_state['ckpt_ranking']
        self.n_degradations = validation_state['n_degradations']

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

    def run_validation(self, trainer: 'pt.Trainer'):
        ckpt_dir = trainer.checkpoint_dir
        if self._json_file is None:
            self._json_file = ckpt_dir / self._json_filename
            if self._json_file.exists():
                self.load_validation_state()
        ckpt_path: Path = trainer.default_checkpoint_path()
        if not ckpt_path.exists():
            raise RuntimeError(
                'Before each validation the CheckpointHook has to write '
                f'a checkpoint.\n'
                f'Could not find {ckpt_path}.\n'
                f'Found only:\n'
                f'{[str(file) for file in ckpt_dir.iterdir()]}'
            )
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
        best_ckpt_path = ckpt_dir / self._best_ckpt_name
        if best_ckpt_path.is_symlink():
            best_ckpt_path.unlink()
        best_ckpt_path.symlink_to(self.ckpt_ranking[0][0])
        if self.max_checkpoints is not None:
            for ckpt_name, _ in self.ckpt_ranking[self.max_checkpoints:]:
                ckpt = ckpt_dir / ckpt_name
                if ckpt.exists():
                    ckpt.unlink()
            self.ckpt_ranking = self.ckpt_ranking[:self.max_checkpoints]
        if self.ckpt_ranking[0][0] != ckpt_path.name:
            self.n_degradations += 1
        else:
            self.n_degradations = 0
        self.save_validation_state()

        if (
            self.early_stopping_patience is not None
            and self.n_degradations > self.early_stopping_patience
        ):
            print(f'Early stopping after {trainer.epoch} epochs and'
                  f' {trainer.iteration} iterations')
            raise StopTraining

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        pass


class BackOffValidationHook(ValidationHook):
    """ Performs model validation and deletes stale checkpoints
    (checkpoints that are not among the max_checkpoints best checkpoints).

    ValidationHook tasks:
     - validate and collect summary
     - update best checkpoints according to metric
     - dump summary to tfevents file
     - remove stale checkpoints
     - save checkpoint ranking in `_json_filename`

    """
    _json_filename = 'validation_state.json'

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

    def save_validation_state(self):
        validation_state = {
            'ckpt_ranking': self.ckpt_ranking,
            'n_degradations': self.n_degradations,
            'remaining_back_offs': self.remaining_back_offs,
        }
        pb.io.dump_json(validation_state, self.json_file)

    def load_validation_state(self):
        validation_state = pb.io.load_json(self.json_file)
        self.ckpt_ranking = validation_state['ckpt_ranking']
        assert validation_state['remaining_back_offs'] <= self.remaining_back_offs, validation_state['remaining_back_offs']
        self.remaining_back_offs = validation_state['remaining_back_offs']
        self.n_degradations = validation_state['n_degradations']

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
            ckpt = self.ckpt_ranking[-j]
            if int(ckpt[len('ckpt_'): -len('.pth')]) > best_iter:
                ckpt_path = ckpt_dir / ckpt
                assert ckpt_path.exists(), ckpt_path
                ckpt_path.unlink()
                self.ckpt_ranking.pop(-j)

        trainer.load_checkpoint()

        def update_lr(optim):
            for param_group in optim.optimizer.param_groups:
                param_group['lr'] *= self.lr_update_factor

        optimizer = trainer.optimizer
        if isinstance(optimizer, dict):
            [update_lr(optim) for optim in optimizer.values()]
        else:
            update_lr(optimizer)
        self.n_degradations = 0
        self.remaining_back_offs -= 1
        self.save_validation_state()

    def post_step(self, trainer: 'pt.Trainer', example, model_out, review):
        pass


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
    IS_PYTORCH_1_1 = LooseVersion(torch.__version__) >= '1.1.0'

    def __init__(self, lr_scheduler, trigger=(1, 'epoch')):
        super().__init__(trigger)
        self.lr_scheduler = lr_scheduler

    def pre_step(self, trainer: 'pt.Trainer'):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            if trainer.epoch > 0 or not self.IS_PYTORCH_1_1:
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
    def __init__(self, stop_trigger, max_it_len=None, update_interval=10):
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
                max_iteration = progressbar.UnknownLength
        else:
            raise ValueError(f'unit {unit} is unknown,'
                             f' choose iteration or epoch')

        self.loss = None
        self.pbar = progressbar.ProgressBar(
            min_value=1,
            max_value=max_iteration,
            redirect_stderr=True,
            redirect_stdout=True,
            max_error=False,
        )

    @property
    def priority(self):
        return Priority.PROGRESS

    def set_last(self, iteration, epoch):
        super().set_last(iteration, epoch)
        self.pbar.value = iteration

    def pre_step(self, trainer: 'pt.Trainer'):
        iteration = trainer.iteration
        epoch = trainer.epoch
        if epoch == 1 and self.pbar.max_value is progressbar.UnknownLength:
            if hasattr(self, 'num_epochs'):
                # sets the max length of the bar after the first epoch
                self.pbar.max_value = (iteration + 1) * self.num_epochs
        if self.trigger(iteration, epoch) and iteration > 1:
            self.pbar.update(iteration)

    # def post_step(self, trainer: 'pt.Trainer', example,
    #               model_output, review):
    #     self.loss = pt.utils.to_numpy(review["loss"])

    def close(self, trainer: 'pt.Trainer'):
        self.pbar.finish()


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


class LossWeightAnnealingHook(TriggeredHook):
    """
    Anneals a loss weight within the los_weights dict of the trainer.
    """
    def __init__(self, name, factor, trigger, max_value=None, min_value=None):
        """

        Args:
            name: key of the loss_weight
            factor: factor by which to anneal the loss weight.
                factor > 1. results in an increase while factor < 1. results
                in a decrease
            trigger:
            max_value: upper bound of the weight
            min_value: lower bound of the weight
                (hint: can also be used to activate a loss weight after a
                certain number of iterations/epochs)
        """
        super().__init__(trigger)
        self.name = name
        self.factor = factor
        self.max_value = max_value
        self.min_value = min_value

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            weight = self.factor * trainer.loss_weights[self.name]
            if self.max_value is not None:
                weight = min(weight, self.max_value)
            if self.min_value is not None:
                weight = max(weight, self.min_value)
            trainer.loss_weights[self.name] = weight


class ModelAttributeAnnealingHook(TriggeredHook):
    """
    Anneals an attribute of the trainers model.
    """
    def __init__(
            self, name, trigger, factor=None, slope=None, max_value=None, min_value=None
    ):
        """

        Args:
            name: name of the attribute. You can use "attr1.attr11" to
                anneal a sub attribute
            factor: factor by which to anneal the attribute.
                factor > 1. results in an increase while factor < 1. results
                in a decrease
            trigger:
            max_value: upper bound of the weight
            min_value: lower bound of the weight
                (hint: can also be used to activate a loss weight after a
                certain number of iterations/epochs)
        """
        super().__init__(trigger)
        self.name = name.split('.')
        assert (factor is None) ^ (slope is None), (factor, slope)
        self.factor = factor
        self.slope = slope
        self.max_value = max_value
        self.min_value = min_value
        self.onset_value = None

    def get_module(self, trainer):
        module = trainer.model
        for attr_name in self.name[:-1]:
            module = getattr(module, attr_name)
        return module

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            module = self.get_module(trainer)
            value = getattr(module, self.name[-1])
            if self.onset_value is None:
                self.onset_value = value
            if self.factor is not None:
                value *= self.factor
            if self.slope is not None:
                value = self.onset_value + self.slope * trainer.iteration
            if self.max_value is not None:
                value = min(value, self.max_value)
            if self.min_value is not None:
                value = max(value, self.min_value)
            setattr(module, self.name[-1], value)

    def close(self, trainer):
        if self.onset_value is not None:
            module = self.get_module(trainer)
            setattr(module, self.name[-1], self.onset_value)


