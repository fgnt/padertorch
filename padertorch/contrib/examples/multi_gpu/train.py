"""
A basic example to demonstrate, how DataParallel can be used in padertorch.
We recommend to use the build in support for data parallel, because it uses the
virtual mini batch instead of the real mini batch to distribute the work on
multiple GPUs.

The code is inspired from the examples in
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
Authors: Sung Kim and Jenny Kang

Use the following command:

    python train.py with storage_root=/path/to/dir

to start the experiment an take a look at the

    forward shape: torch.Size([11, 28])
    forward shape: torch.Size([12, 28])
    review shape: torch.Size([23, 28])

in the output. You should see, that the shape in the forward is different
from the shape in the review. Also the forward is more offten called than the
review.
"""
from pathlib import Path
import numpy as np

import torch
import torch.nn

import sacred
import sacred.commands

import paderbox as pb
import padertorch as pt
from padertorch.modules import fully_connected_stack

from padertorch.contrib.cb.io import get_new_folder

experiment = sacred.Experiment()


class MyModel(pt.base.Model):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, example):
        print("\tforward shape:", example['feature'].size())
        out = self.net(example['feature'])
        return out

    def review(self, example, prediction):
        print("\treview shape:", example['feature'].size())
        loss = torch.nn.CrossEntropyLoss()(prediction, example['label'])

        with torch.no_grad():
            _, predicted = torch.max(prediction, 1)
            acc = (predicted == example['label']).sum().item() / len(example['label'])

        return dict(loss=loss, scalars={'Accuracy': acc})


@experiment.config
def config():
    epochs = 2
    storage_root = None

    features = 28
    classes = 10
    
    trainer = pb.utils.nested.deflatten({
        'model.factory': MyModel,
        'model.net.factory': fully_connected_stack,
        'model.net.input_size': features,
        'model.net.hidden_size': [10],
        'model.net.output_size': classes,
        'model.net.activation': 'relu',
        'model.net.dropout': 0.,

        'optimizer.factory': pt.optimizer.SGD,
        'summary_trigger': (1, 'epoch'),
        'checkpoint_trigger': (1, 'epoch'),
        'stop_trigger': (2, 'epoch'),
        'storage_dir': str(get_new_folder(storage_root, mkdir=False)),
    })
    pt.Trainer.get_config(trainer)

    experiment.observers.append(sacred.observers.FileStorageObserver.create(
        Path(trainer['storage_dir']) / 'sacred'
    ))


def get_random_dataset(length, features=28, classes=10):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> np.random.seed(0)
    >>> pprint(get_random_dataset(10)[0], max_array_length=10)
    {'feature': array(shape=(25, 28), dtype=float32),
     'label': array(shape=(25,), dtype=int64)}
    """

    dataset = []
    for i in range(length):
        frames = np.random.randint(20, 30)
        label = np.random.randint(0, classes)
        label = np.array([label] * frames, dtype=np.int64)
        
        dataset.append({
            'feature': (label[:, None] + np.random.normal(size=(frames, features))).astype(np.float32),
            'label': label,
        })
    
    return dataset


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, attr):
        """
        Overwrite __getattr__ so the user does not recognize if the
        model is wrapped by DataParallel, i.e. forward attribute access
        to the model.
        """
        try:
            # torch.nn.module overwrites __getattr__,
            # hence first call super.
            return super().__getattr__(attr)
        except AttributeError:
            pass
        return getattr(self.module, attr)


@experiment.automain
def main(_run, _config, features, classes):
    # Get Datasets
    train_ds = get_random_dataset(10, features=features, classes=classes)
    dev_ds = get_random_dataset(1, features=features, classes=classes)

    sacred.commands.print_config(_run)
    trainer: pt.Trainer = pt.Trainer.from_config(_config['trainer'])

    pb.io.dump(_config, trainer.storage_dir / 'config.yaml')
    pb.io.dump(_config, trainer.storage_dir / 'config.json')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        trainer.model = DataParallel(trainer.model)
    else:
        print('#' * 79)
        print('WARNING: Could not find multiple GPUs !!!')
        print('#' * 79)
    print('torch.cuda.device_count()', torch.cuda.device_count())
    print('torch.cuda.is_available()', torch.cuda.is_available())

    trainer.register_validation_hook(validation_iterator=dev_ds)
    trainer.train(train_ds)
