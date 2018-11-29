import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from nt.database.iterator import BaseIterator

from pytorch_sanity.utils import to_list, nested_update


OPTIMIZER_MAP = dict(
    adam=torch.optim.Adam,
    sgd=torch.optim.SGD
)


class Trainer:
    def __init__(
            self, models, train_iterator, validation_iterator, config,
            storage_dir
    ):
        self.config = config
        self.models = to_list(models)
        optimizers = to_list(config['optimizers'], len(self.models))
        learning_rates = to_list(config['learning_rates'], len(self.models))
        weight_decays = to_list(config['weight_decays'], len(self.models))
        self.optimizers = [
            OPTIMIZER_MAP[optimizers[i]](
                m.parameters(),
                lr=learning_rates[i],
                weight_decay=weight_decays[i]
            )
            if len(list(m.parameters())) else None
            for i, m in enumerate(self.models)
        ]
        self.train_iterator = train_iterator
        self.validation_iterator = validation_iterator

        self.storage_dir = Path(storage_dir).expanduser().absolute()
        self.reset_summary()
        self.iteration = 0
        self.writer = SummaryWriter(self.storage_dir)
        if config.get('init_checkpoint', None) is not None:
            self.load_checkpoint(Path(config['init_checkpoint']).absolute())
        self.use_cuda = config["use_cuda"]

    def reset_summary(self):
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

        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

        train_iterator = BatchIterator(self.train_iterator, self.config['batch_size'])
        train_iterator = RepeatIterator(train_iterator, self.config['max_epochs'])
        max_iterations = self.config['max_iterations']
        if self.use_cuda:
            self.cuda()
        [m.train() for m in self.models]
        # ================ MAIN TRAINNIG LOOP! ===================
        try:
            for batch in train_iterator.map(self.batch_to_device)():
                if max_iterations is not None and self.iteration >= max_iterations:
                    break
                self.train_step(batch)

                if self.iteration % self.config['summary_step'] == 0:
                    self.add_summary('training')
                if self.iteration % self.config['checkpoint_step'] == 0:
                    self.save_checkpoint()
                if self.iteration % self.config['validation_step'] == 0 \
                        and self.iteration > 0:
                    self.add_summary('training')
                    self.validate()
                    [m.train() for m in self.models]
                self.iteration += 1
        finally:
            self.add_summary('training')
            self.save_checkpoint()

    def validate(self):
        print('Starting Validation')
        [m.eval() for m in self.models]
        validation_iterator = BatchIterator(
            self.validation_iterator, self.config['batch_size'])
        for i, batch in enumerate(validation_iterator.map(
                self.batch_to_device)()):
            self.validation_step(batch)
        self.add_summary('validation')
        print('Finished Validation')

    def batch_to_device(self, batch):
        for key, value in batch.items():
            if torch.is_tensor(value):
                if self.use_cuda:
                    value = value.cuda()
                else:
                    value = value.cpu()
            batch[key] = value
        return batch

    def train_step(self, batch):
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
        model_out = self.models[0](batch)
        review = self.models[0].review(batch, model_out)
        self.update_summary(review)

    def backward(self, review, retain_graph=False):
        loss = 0.
        loss_weights = self.config.get('loss_weights', None)
        for key, value in review['losses'].items():
            weight = loss_weights[key] if loss_weights is not None else 1.
            loss += weight * value
        loss.backward(retain_graph=retain_graph)

    def clip_grad(self, prefix=''):
        prefix_ = f'{prefix}_' if prefix else ''
        grad_clips = to_list(self.config[f'{prefix_}gradient_clips'], len(self.models))
        for i, model in enumerate(self.models):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clips[i]
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
                f'{prefix}/{key}', audio, self.iteration, sample_rate=16000
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
                optimizers=[
                    opti and opti.state_dict() for opti in self.optimizers
                ]
            ),
            checkpoint_path
        )
        if self.use_cuda:
            self.cuda()
        print(f"{datetime.now()}: Saved model and optimizer state at iteration "
              f"{self.iteration} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path)
        checkpoint_dict = torch.load(str(checkpoint_path), map_location='cpu')
        [m.load_state_dict(d) for m, d in zip(
            self.models, checkpoint_dict['models'])]
        iteration = checkpoint_dict['iteration']
        self.iteration = iteration + 1
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

    def cuda(self):
        self.models = [m.cuda() for m in self.models]
        for opti in self.optimizers:
            if opti is None:
                continue
            for state in opti.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()


class BatchIterator(BaseIterator):
    def __init__(self, input_generator, batchsize, collate_fn=default_collate):
        self.input_generator = input_generator
        self.batchsize = batchsize
        self.collate_fn = collate_fn

    def __iter__(self):
        current_batch = list()
        for element in self.input_generator():
            current_batch.append(element)
            if len(current_batch) >= self.batchsize:
                yield self.collate_fn(current_batch)
                current_batch = list()
        if len(current_batch) > 0:
            yield self.collate_fn(current_batch)

    def __getitem__(self, index):
        input_index = index * self.batchsize
        return self.collate_fn(
            [self.input_generator[i]
             for i in range(input_index, input_index + self.batchsize)])

    def __len__(self):
        return int(np.ceil(len(self.input_generator) / self.batchsize))


class RepeatIterator(BaseIterator):
    """
    Equal to BaseIterator.tile(n_epochs) expect for the print.
    Note: np.repeat does something different.
    """
    def __init__(self, input_generator, n_epochs):
        self.input_generator = input_generator
        self.n_epochs = n_epochs

    def __iter__(self):
        i = 0
        while self.n_epochs is None or i < self.n_epochs:
            i += 1
            print(f'Epoch: {i}')
            for element in self.input_generator():
                yield element

    def __getitem__(self, index):
        return self.input_generator[index % len(self.input_generator)]

    def __len__(self):
        return self.n_epochs * len(self.input_generator)
