import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import itertools
import operator

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from pytorch_sanity.utils import to_list, nested_update


__all__ = [
    'Trainer',
]


class Trainer:
    def __init__(
            self,

            models,
            train_iterator,
            validation_iterator,

            optimizers,
            learning_rates,
            loss_weights,
            batch_size,
            summary_step,
            checkpoint_step,
            validation_step,
            storage_dir,
            gpu=0 if torch.cuda.is_available() else None,
            max_epochs=None,
            max_iterations=None,
            weight_decays=None,
            init_checkpoint=None,
            seed=0,
    ):
        # self.config = config
        self.models = to_list(models)
        self.optimizers = [
            optimizer.set_params(
                self.models[i].parameters(),
                lr=learning_rates[i],
                weight_decay=weight_decays[i]
            )
            if len(list(self.models[i].parameters())) else None
            for i, optimizer in enumerate(to_list(optimizers, len(self.models)))
        ]
        self.train_iterator = train_iterator
        self.validation_iterator = validation_iterator

        self.storage_dir = Path(storage_dir).expanduser().absolute()
        self.reset_summary()
        self.iteration = 0
        self.epoch = 0
        self.writer = SummaryWriter(self.storage_dir)
        if init_checkpoint is not None:
            self.load_checkpoint(
                Path(init_checkpoint).expanduser().absolute(),
            )
        self.use_cuda = gpu is not None
        self.gpu_device = int(gpu)
        self.seed = seed
        self.batch_size = batch_size
        # self.max_epochs = max_epochs
        assert operator.xor(
            max_iterations is None,
            max_epochs is None,
        ), (max_iterations, max_epochs)
        if max_iterations is not None:
            self.max_iterations = EndTrigger(max_iterations, unit='iteration')
        elif max_epochs is not None:
            self.max_iterations = EndTrigger(max_iterations, unit='epoch')
        else:
            raise Exception(max_epochs, max_iterations)

        self.summary_step = IntervallTrigger.new(summary_step)
        self.checkpoint_step = IntervallTrigger.new(checkpoint_step)
        self.validation_step = IntervallTrigger.new(validation_step)

        self.loss_weights = loss_weights

    def reset_summary(self):
        # Todo: add figures
        self.summary = dict(
            losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict()
        )

    def train(self):
        os.makedirs(str(self.storage_dir / 'checkpoints'), exist_ok=True)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Todo: batch outside of trainer
        if self.batch_size is not None:
            train_iterator = self.train_iterator.batch(
                self.batch_size, collate_fn=default_collate
            )
        else:
            train_iterator = self.train_iterator
        # train_iterator = train_iterator  # .tile(self.max_epochs)

        # Todo: unit(s) for steps?
        max_iterations = self.max_iterations
        if self.use_cuda:
            self.cuda(self.gpu_device)
        # Change model to train mode (e.g. activate dropout)
        [m.train() for m in self.models]
        # ================ MAIN TRAINNIG LOOP! ===================
        try:
            for self.epoch in itertools.count(self.epoch):  # infinite loop
                if self.max_iterations(
                        iteration=self.iteration, epoch=self.epoch
                ):
                    return
                for self.iteration, batch in enumerate(
                        train_iterator.map(self.batch_to_device),
                        start=self.iteration
                ):
                    if self.max_iterations(
                            iteration=self.iteration, epoch=self.epoch
                    ):
                        return

                    # Todo: backup OOM
                    self.train_step(batch)

                    if self.summary_step(
                            iteration=self.iteration, epoch=self.epoch
                    ) or self.iteration == 0:
                        self.add_summary('training')
                    if self.checkpoint_step(
                            iteration=self.iteration, epoch=self.epoch
                    ) or self.iteration == 0:
                        self.save_checkpoint()
                    if self.validation_step(
                            iteration=self.iteration, epoch=self.epoch
                    ):
                        # Todo: allow continuous evaluation
                        self.add_summary('training')
                        self.validate()
                        [m.train() for m in self.models]
                    self.iteration += 1
        finally:
            self.add_summary('training')
            self.save_checkpoint()

    def validate(self):
        print('Starting Validation')
        # Change model to eval mode (e.g. deactivate dropout)
        [m.eval() for m in self.models]
        validation_iterator = self.validation_iterator.batch(
            self.batch_size, collate_fn=default_collate)
        for i, batch in enumerate(validation_iterator.map(
                self.batch_to_device)):
            self.validation_step(batch)
        self.add_summary('validation')
        print('Finished Validation')

    def batch_to_device(self, batch):
        for key, value in batch.items():
            if torch.is_tensor(value):
                if self.use_cuda:
                    value = value.cuda(self.gpu_device)
                else:
                    value = value.cpu()
            batch[key] = value
        return batch

    def train_step(self, batch):
        assert len(self.model) == 1, (
            self.model, 'Overwrite the train_step and validation_step, when you have multiple models.'
        )
        [opti and opti.zero_grad() for opti in self.optimizers]
        review = dict()
        for model in self.models:
            model_out = model(batch)
            nested_update(review, model.review(batch, model_out))
        self.backward(review)
        self.clip_grad()
        [opti and opti.step() for opti in self.optimizers]
        self.update_summary(review)

    def validation_step(self, batch):
        assert len(self.model), (
            self.model, 'Overwrite the train_step and validation_step, when you have multiple models.'
        )
        model_out = self.models[0](batch)
        review = self.models[0].review(batch, model_out)
        self.update_summary(review)

    def backward(self, review, retain_graph=False):
        loss = 0.
        loss_weights = self.loss_weights
        if loss_weights is None and len(review['losses']) != 1:
            raise Exception(
                'You can not have multiple losses without specifying '
                f'loss_weights. losses: {review["losses"]}'
            )
        for key, value in review['losses'].items():
            weight = loss_weights[key] if loss_weights is not None else 1.
            loss += weight * value
        loss.backward(retain_graph=retain_graph)

    def clip_grad(self, prefix: str = None):
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm
        for i, model in enumerate(self.models):
            grad_norm = self.optimizers[i].clip_grad(
                model.parameters(), prefix
            )
            self.summary['scalars'][f'{prefix_}grad_norm_{i}'].append(grad_norm)
            self.summary['histograms'][f'{prefix_}grad_norm_{i}_'].append(grad_norm)

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

    def add_summary(self, prefix):
        for key, loss in self.summary['losses'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(loss), self.iteration)
        for key, scalar in self.summary['scalars'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(scalar), self.iteration)
        for key, histogram in self.summary['histograms'].items():
            self.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), self.iteration,
                # bins='doane'
            )
        for key, audio in self.summary['audios'].items():
            self.writer.add_audio(
                f'{prefix}/{key}', audio[1], self.iteration, sample_rate=audio[0]
            )
        for key, image in self.summary['images'].items():
            self.writer.add_image(f'{prefix}/{key}', image, self.iteration)
        self.reset_summary()

    def save_checkpoint(self):
        checkpoint_path = str(
            self.storage_dir / 'checkpoints' / f'ckpt_{self.iteration}')
        if self.use_cuda:
            self.cpu()
        torch.save(
            dict(
                models=[m.state_dict() for m in self.models],
                iteration=self.iteration,
                epoch=self.epoch,
                optimizers=[
                    opti and opti.state_dict() for opti in self.optimizers
                ]
            ),
            checkpoint_path
        )
        if self.use_cuda:
            self.cuda(self.gpu_device)
        print(f"{datetime.now()}: Saved model and optimizer state at iteration "
              f"{self.iteration} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path), checkpoint_path
        checkpoint_dict = torch.load(str(checkpoint_path), map_location='cpu')
        [m.load_state_dict(d) for m, d in zip(
            self.models, checkpoint_dict['models'])]
        iteration = checkpoint_dict['iteration']
        self.iteration = iteration + 1
        self.epoch = checkpoint_dict['epoch']
        [opti.load_state_dict(d) if opti is not None else None
         for opti, d in zip(self.optimizers, checkpoint_dict['optimizers'])]
        print(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")

    def cpu(self):
        self.models = [m.cpu() for m in self.models]
        for opti in self.optimizers:
            if opti is None:
                continue
            for state in opti.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()

    def cuda(self, device):
        self.models = [m.cuda(self.gpu_device) for m in self.models]
        for opti in self.optimizers:
            if opti is None:
                continue
            for state in opti.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.gpu_device)


class IntervallTrigger:
    """

    https://www.cntk.ai/pythondocs/cntk.logging.progress_print.html
    Is a geometric schedule interesting as opposite to arithmetic?
        geometric: [1, 2, 4, 8, 16, ...] times period
        arithmetic: [1, 2, 3, 4, 5, ...] times period

    """

    @classmethod
    def new(cls, intervall_trigger):
        if isinstance(intervall_trigger, IntervallTrigger):
            return cls(
                intervall_trigger.period,
                intervall_trigger.unit,
            )
        else:
            assert len(intervall_trigger) == 2, intervall_trigger
            return cls(
                **intervall_trigger
            )

    def __init__(self, period, unit):
        """

        Args:
            period:
            unit: 'epoch' or 'iteration' (i.e. number of minibatches)


        >>> trigger = IntervallTrigger(2, 'epoch')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 False
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 False
        6 2 True
        7 2 False
        8 2 False
        9 3 False
        >>> trigger = IntervallTrigger(2, 'iteration')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 True
        1 0 False
        2 0 True
        3 1 False
        4 1 True
        5 1 False
        6 2 True
        7 2 False
        8 2 True
        9 3 False
        """
        self.period = period
        assert unit == 'epoch' or unit == 'iteration', unit
        self.unit = unit
        self.last = 0

    def __call__(self, iteration, epoch):
        if self.unit == 'epoch':
            if self.last == epoch:
                return False
            else:
                self.last = epoch
                return (epoch % self.period) == 0
        elif self.unit == 'iteration':
            return (iteration % self.period) == 0
        else:
            raise ValueError(self.unit, 'Expect epoch or iteration')

class EndTrigger(IntervallTrigger):
    def __call__(self, iteration, epoch):
        """
        >>> trigger = EndTrigger(2, 'epoch')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 False
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 False
        6 2 True
        7 2 True
        8 2 True
        9 3 True
        >>> trigger = EndTrigger(5, 'iteration')
        >>> for i in range(10):
        ...     epoch = i // 3
        ...     print(i, epoch, trigger(i, epoch))
        0 0 False
        1 0 False
        2 0 False
        3 1 False
        4 1 False
        5 1 True
        6 2 True
        7 2 True
        8 2 True
        9 3 True
        """
        if self.unit == 'epoch':
            return epoch >= self.period
        elif self.unit == 'iteration':
            return iteration >= self.period
        else:
            raise ValueError(self.unit, 'Expect epoch or iteration')