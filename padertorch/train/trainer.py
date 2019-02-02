"""
    This module contains the Trainer class which can be used to train
    configurable padertorch models.
"""
import contextlib
import itertools
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tensorboardX

from paderbox.utils.nested import nested_update
import padertorch as pt
from padertorch.configurable import Configurable
from padertorch.train.optimizer import Optimizer, Adam
from padertorch.train.run_time_tests import test_run
from padertorch.train.hooks import *
from padertorch.train.trigger import AnyTrigger

__all__ = [
    'Trainer',
]


class Trainer(Configurable):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['optimizer'] = {'factory': Adam}

    def __init__(
            self,
            model,
            storage_dir,
            optimizer,
            loss_weights=None,
            summary_trigger=(1, 'epoch'),
            checkpoint_trigger=(1, 'epoch'),
            keep_all_checkpoints=False,
            max_trigger=(1, 'epoch'),
            gpu=0 if torch.cuda.is_available() else None
    ):
        """

        :param model: a `padertorch.base.Model` object or dict of Models
        :param storage_dir: The structure of produced storage_dir is:
            .
            ├── checkpoints
            │   ├── ckpt_7122.pth
            │   ├── ckpt_14244.pth
            │   ├── ckpt_best_loss.pth -> ckpt_7122.pth
            │   ├── ckpt_latest.pth -> ckpt_14244.pth
            │   └── ckpt_state.json
            ├── events.out.tfevents.1548851867.ntsim5
        :param optimizer: a `padertorch.train.optimizer.Optimizer` object
                    or dict of Optimizers
        :param loss_weights: dict of weights for model with multiple losses
        :param summary_trigger: `pytorch.train.trigger.IntervalTrigger` object
                    or tuple describing the interval when summaries
                    are written to event files
        :param checkpoint_trigger: `padertorch.train.trigger.IntervalTrigger`
                    object or tuple describing the interval when checkpoints
                    are saved
        :param keep_all_checkpoints: flag if False only latest and best
                    checkpoints are kept otherwise all checkpoints are kept
        :param max_trigger: `padertorch.train.trigger.EndTrigger` object
                    or tuple describing the endpoint of the training
        :param gpu: defines the gpu which shall be used, if None cpu is used
        """
        if isinstance(optimizer, dict):
            # Special case see Janek's example
            # ToDo: Hint to example

            model = torch.nn.ModuleDict(model)
            assert set(model.keys()) == set(optimizer.keys()), \
                (model, optimizer)
            optimizer = optimizer.copy()
            for key, opti in list(optimizer.items()):
                if opti is None:
                    del optimizer[key]
                else:
                    m = model[key]
                    opti.set_parameters(m.parameters())
        else:
            optimizer.set_parameters(model.parameters())

        self.optimizer = optimizer
        self.model = model
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                'Expect that the model is a subclass from padertorch.Module.\n'
                f'Got: type: {type(model)}\n{model}'
            )

        self.use_cuda = gpu is not None
        self.gpu_device = None
        if self.use_cuda:
            self.gpu_device = int(gpu)
            self.model.cuda(self.gpu_device)
        else:
            self.gpu_device = None

        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.reset_timer()
        self.iteration = None
        self.epoch = None

        self.summary_trigger = summary_trigger
        self.checkpoint_trigger = checkpoint_trigger
        self.keep_all_checkpoints = keep_all_checkpoints
        self.max_trigger = max_trigger

        self.loss_weights = loss_weights

    def reset_timer(self):
        self.timer = ContextTimerDict()

    def test_run(self, train_iterator, validation_iterator):
        """
        Run a test on the trainer instance (i.e. model test).

        Tests:
         - forward (train and validate)
         - deterministic output in eval
         - simple review dict test

        """
        test_run(
            self,
            train_iterator,
            validation_iterator,
        )

    def train(
            self,
            train_iterator,
            validation_iterator=None,
            *,
            hooks=None,
            metrics={'loss': 'min'},
            n_best_checkpoints=1,
            resume=False,
    ):
        """
        A simplified training loop::

            for epoch in range(1, ...):
                for example in train_iterator:
                    model_out = self.model(example)
                    review = self.model.review(example, model_out)
                    review = maybe_add_loss_from_losses(review)
                    review.backward()
                    self.optimizer.step()
                    add_review_to_tensorboardX(review)

        The remaining code takes care about calling validation and save the
        result to tensorboard (if the a validation_iterator is given), save
        checkpoints, cleanup checkpoints that are stale (not best according
        to metrics and not last) and display a progessbar.
        The code is designed that many aspects can be customized.
        (e.g. test_run_time_tests.py DictTrainer for multi model trainer)

        Args:
            train_iterator:
                The train_iterator is python iterable (e.g. tuple, list, ...)
                that can consumed multiple times (i.e. not generator).

                Usually it will be paderbox.database.BaseIterator that is
                returned from a database in paderbox.database.

            validation_iterator:
                Optional and same type as train_iterator. This iterator is used
                for validation.
            hooks:
                Add additional hooks to the default hooks
                (`Trainer.get_default_hooks`)
            metrics:
                The metrics that are used for the deciding which checkpoint is
                kept. The key is of each entry must be a key 'loss' or a key in
                review['losses'] or review['scalars']. The value indicate if
                the metric has to be maximised ('max') or minimised ('min').
            n_best_checkpoints:
                The numer of checkpoints to keep for each metric. In the moment
                only one checkpoints is supported.
                Use `keep_all_checkpoints=True` from the `__init__` to keep all
                checkpoints.
            resume:
                Whether to resume a training or start a fresh one.

        """
        if resume:
            self.load_checkpoint()
        else:
            assert not self.checkpoint_dir.exists(),\
                f'A checkpoint directory already exists. If you want to' \
                f'restart the training set init_checkpoint to True.'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        # Change model to train mode (e.g. activate dropout)
        self.model.train()

        hooks = self.get_default_hooks(
            hooks,
            train_iterator=train_iterator,
            validation_iterator=validation_iterator,
            metrics=metrics,
            n_best_checkpoints=n_best_checkpoints,
        )

        if self.iteration is None and self.epoch is None:
            self.iteration = 0
            self.epoch = 0
        else:
            # For training continue set the correct last value (default -1)
            for hook in hooks:
                hook.set_last(self.iteration, self.epoch)

        # ================ MAIN TRAINING LOOP! ===================
        try:
            # Count epochs up to infinity if not any stop condition is met. A
            # typical stop condition is a firing `StopTrainingHook`.
            for self.epoch in itertools.count(start=self.epoch):
                epoch_start = True
                for hook in hooks:
                    hook.pre_step(self)

                # Count up iteration up to infinity if not any stop condition,
                # (e.g. `StopIteration` exception) is reached.
                for self.iteration, example in self.timer(
                    key='time_per_data_loading',
                    iterable=enumerate(
                        train_iterator,
                        start=self.iteration,
                    )
                ):
                    if epoch_start:
                        epoch_start = False
                    else:
                        for hook in hooks:
                            hook.pre_step(self)

                    with self.timer['time_per_train_step']:
                        example = pt.data.batch_to_device(
                            example, self.use_cuda, self.gpu_device
                        )
                        # Todo: backup OutOfMemory
                        model_output, review = self.train_step(example)

                    for hook in hooks:
                        hook.post_step(self, example, model_output, review)

                # Fix for next loop
                self.iteration += 1

        except StopTraining:
            pass
        finally:
            for hook in hooks:
                hook.close(self)

    _start_non_validation_time = None

    def validate(self, validation_iterator):
        """
        used by ValidationHook

        :param validation_iterator:
        :return:
        """
        train_end_time = self.timer.timestamp()

        if self._start_non_validation_time is not None:
            self.timer.timings['non_validation_time'].append(
                train_end_time - self._start_non_validation_time
            )

        with self.timer['validation_time'], torch.no_grad():
            # Change model to eval mode (e.g. deactivate dropout)
            self.model.eval()
            try:
                for i, example in enumerate(validation_iterator):
                    example = pt.data.batch_to_device(
                        example, self.use_cuda, self.gpu_device
                    )
                    yield self.validation_step(example)
            finally:
                self.model.train()
                self._start_non_validation_time = self.timer.timestamp()

    def train_step(self, example):
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.zero_grad()
        else:
            self.optimizer.zero_grad()

        model_out, review = self.step(example)
        self.backward(review)
        grad_summary = self.clip_grad()

        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.step()
        else:
            self.optimizer.step()

        nested_update(review, grad_summary)
        return model_out, review

    def validation_step(self, example):
        return self.step(example)

    def _step(self, example):
        msg = 'Overwrite the step function of the trainer, ' \
              'when you have multiple models.'
        assert isinstance(self.model, torch.nn.Module), (self.model, msg)
        assert isinstance(self.optimizer, Optimizer), (self.optimizer, msg)
        model_out = self.model(example)
        return model_out, self.model.review(example, model_out)

    def _maybe_add_loss_to_review(self, review):
        if 'losses' in review:
            assert 'loss' not in review, review
            losses = review['losses']

            loss = 0.
            loss_weights = self.loss_weights
            if loss_weights is None and len(losses) != 1:
                raise Exception(
                    'You can not have multiple losses without specifying '
                    f'loss_weights. losses: {losses}'
                )
            for key, value in losses.items():
                weight = loss_weights[key] if loss_weights is not None else 1.
                loss += weight * value

            review['loss'] = loss
        else:
            assert 'loss' in review, review

        return review

    def step(self, example):
        model_out, review = self._step(example)
        return model_out, self._maybe_add_loss_to_review(review)

    def backward(self, review, retain_graph=False):
        review['loss'].backward(retain_graph=retain_graph)

    def get_default_hooks(
            self,
            hooks,
            *,
            train_iterator,
            validation_iterator,
            metrics,
            n_best_checkpoints,
    ):
        if n_best_checkpoints != 1:
            raise NotImplementedError(
                f'The implementation for more than one checkpoint is not'
                f'finished.\n'
                f'Requested number of checkponts: {n_best_checkpoints}'
            )

        if hooks is None:
            hooks = []
        try:
            max_it_len = len(train_iterator)
        except TypeError:
            # TypeError: object of type '...' has no len()
            max_it_len = None
        hooks = pt.utils.to_list(hooks)

        writer = tensorboardX.SummaryWriter(str(self.storage_dir))

        if validation_iterator is None:
            print(
                'Since no validation_iterator is provided to `Trainer.train`, '
                'disable validation.'
            )
            raise NotImplementedError(
                'ToDo: Check SimpleCheckpointHook for errors'
            )
            hooks.append(SimpleCheckpointHook(
                self.checkpoint_trigger,
                keep_all=self.keep_all_checkpoints,
            ))

            summary_trigger = self.summary_trigger
        else:
            hooks.append(CheckpointedValidationHook(
                trigger=self.checkpoint_trigger,
                iterator=validation_iterator,
                checkpoint_dir=self.checkpoint_dir,
                metrics=metrics,
                keep_all=self.keep_all_checkpoints,
                init_from_json=self.checkpoint_dir.exists(),
                writer=writer,
            ))

            summary_trigger = AnyTrigger(
                self.summary_trigger,
                self.checkpoint_trigger,
            )

        hooks.append(SummaryHook(summary_trigger, writer=writer))
        hooks.append(ProgressBarHook(self.max_trigger, max_it_len))
        hooks.append(StopTrainingHook(self.max_trigger))
        hooks = sorted(hooks, key=lambda h: h.priority, reverse=True)
        return hooks

    def clip_grad(self):
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm

        summary = dict(scalars=dict(), histograms=dict())
        if isinstance(self.optimizer, dict):
            for key, opti in self.optimizer.items():
                grad_norm = opti.clip_grad()

                summary['scalars'][f'{key}_grad_norm'] = grad_norm
                # underscore was necessary to obtain unique keys to prevent
                # tensorboard error
                summary['histograms'][
                    f'{key}_grad_norm_'] = torch.Tensor([grad_norm])
        else:
            grad_norm = self.optimizer.clip_grad()
            summary['scalars'][f'grad_norm'] = grad_norm
            summary['histograms'][f'grad_norm_'] = \
                torch.Tensor([grad_norm])

        return summary

    @property
    def checkpoint_dir(self):
        return self.storage_dir / 'checkpoints'

    def default_checkpoint_path(self):
        return self.checkpoint_dir / f'ckpt_{self.iteration}.{CKPT_EXT}'

    def save_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        if self.use_cuda:
            self.cpu()

        if isinstance(self.optimizer, dict):
            optimizer_state_dict = {
                k: opti.state_dict()
                for k, opti in self.optimizer.items()
            }
        else:
            optimizer_state_dict = self.optimizer.state_dict()

        torch.save(
            dict(
                model=self.model.state_dict(),
                iteration=self.iteration,
                epoch=self.epoch,
                optimizer=optimizer_state_dict,
            ),
            str(checkpoint_path)
        )
        if self.use_cuda:
            self.cuda(self.gpu_device)
        print(f"{datetime.now()}: Saved model and optimizer state "
              f"at iteration {self.iteration} to {checkpoint_path}")

    def load_checkpoint(self):
        checkpoint_path = self.checkpoint_dir / 'ckpt_latest.pth'
        assert checkpoint_path.is_file(), checkpoint_path
        checkpoint_dict = torch.load(str(checkpoint_path), map_location='cpu')

        self.model.load_state_dict(checkpoint_dict['model'])
        if isinstance(self.optimizer, dict):
            assert set(self.optimizer.keys() == set(checkpoint_dict['optimizer'].keys())), \
                (self.optimizer, checkpoint_dict['model'])
            for key, otim in self.optimizer.items():
                otim.load_state_dict(
                    checkpoint_dict['optimizer'][key]
                )
        else:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

        iteration = checkpoint_dict['iteration']
        self.iteration = iteration
        self.epoch = checkpoint_dict['epoch']

        print(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")

    def cpu(self):
        self.model.cpu()
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.cpu()
        else:
            self.optimizer.cpu()

    def cuda(self, device):
        self.model.cuda(device)
        if isinstance(self.optimizer, dict):
            for opti in self.optimizer.values():
                opti.cuda(device)
        else:
            self.optimizer.cuda(device)


class ContextTimerDict:
    """
    To be able to keep the measurements, we need to create the object before.
    Then each measurement can be started with a context manager.

    >>> np.set_printoptions(precision=2)
    >>> timer = ContextTimerDict()
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test']:
    ...     time.sleep(0.1)
    >>> with timer['test_2']:
    ...     time.sleep(0.1)
    >>> for _ in timer('test_3', range(3)):
    ...     time.sleep(0.1)

    Ignore timing when an exception is raised
    >>> with contextlib.suppress(Exception), timer['test_2']:
    ...     raise Exception
    >>> timer
    ContextTimerDict: {'test': array([0.1, 0.1]), 'test_2': array([0.1]), 'test_3': array([1.96e-06, 4.80e-06, 3.87e-06])}
>>> timer.as_dict
    {'test': array([0.1, 0.1]), 'test_2': array([0.1]), 'test_3': array([1.96e-06, 4.80e-06, 3.87e-06])}
"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.timestamp = time.perf_counter  # time.process_time

    @contextlib.contextmanager
    def __getitem__(self, item):
        assert isinstance(item, str), item
        start = self.timestamp()
        yield
        end = self.timestamp()
        self.timings[item].append(end - start)

    @property
    def as_dict(self):
        return {k: np.array(time) for k, time in self.timings.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}: ' + repr(self.as_dict)

    def __str__(self):
        return str(self.as_dict)

    def __call__(self, key, iterable):
        iterator = iter(iterable)
        while True:
            with self[key]:
                example = next(iterator)
            yield example



# ToDO: write function for those to functions outside of trainer
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)