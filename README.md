# padertorch
[![Build Status](https://dev.azure.com/fgnt/fgnt/_apis/build/status/fgnt.padertorch?branchName=master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![Azure DevOps tests](https://img.shields.io/azure-devops/tests/fgnt/fgnt/3/master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/fgnt/fgnt/3/master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/lazy_dataset/blob/master/LICENSE)

Padertorch is designed to simplify the training of deep learning models written with PyTorch.
While focusing on speech and audio processing, it is not limited to these application areas.

This repository is currently under construction.

[//]: <> (The examples in contrib/examples are only working in the Paderborn NT environment)

# Highlights

- **Fast prototyping**: The trainer and models are all compatible with [sacred](https://github.com/IDSIA/sacred) and easily allow hyperparameter changes over the command line. Check out the [configurable](padertorch/configurable.py) module and the [examples](padertorch/contrib/examples) for how it works.
- **Easily extensible**: Write your own network modules and models based on `padertorch.Module` and `padertorch.Model`.
- **Seamless integration**: You provide your own data and model. We provide the trainer and optimizer to wrap around your model and data. One simple command starts the training: `padertorch.Trainer.train`.
- **Hiding the bulk**: Our `padertorch.Trainer` takes care of the nasty training loop and allows you to focus on network tuning. Included are features such as:
  - Periodically executed validation runs
  - Automatic checkpointing: The parameters of the model and the state of the trainer are periodically saved. The checkpoint interval and number of total checkpoints are customizable: Keep one, some or all checkpoints. We also keep track of the best checkpoint on the validation data given some metric.
  - Learning rate scheduling
  - Backoff: Restore the best checkpoint and change the learning rate if the loss is not decreasing.
  - [Averaging multiple checkpoints]():
  - [Hooks](padertorch/train/hooks.py): Extend the basic features of the trainer with your own functionality.
- **Logging**: As logging backend, we use `TensorboardX` to generate a `tfevents` file that can be visualized from a `tensorboard`. Custom values to be logged can be defined in subclasses of `padertorch.Model`.
- **Test run**: The trainer has a `test_run` function to train the model for few iterations and test if
    - the model is executable (burn test)
    - the validation is deterministic/reproducable
    - the model changes the parameter in the training
- **Virtual minibatch**:
  - The `Trainer` usually does not know if the model is trained with a single example or multiple examples (minibatch), because the examples that are yielded from the dataset are directly forwarded to the model.
  - When the `virtual_minibatch_size` option is larger than one, the trainer calls the forward and backward step `virtual_minibatch_size` times before applying the gradients. This increases the minibatch size, while the memory consumption stays similar.
- **Multi-GPU training**: Easily deploy your model onto multiple GPUs to increase the total batch size and speed up the training. See [here](doc/virtual_batch_size_multi_gpu.md) for implementation details and the [example](padertorch/contrib/examples/multi_gpu) for how to enable it.
- **Support for lazy data preparation**: Ever tired of pre-computing features, taking up time and space on the hard disk? Padertorch works with lazy dataloaders (e.g., [lazy_dataset](https://github.com/fgnt/lazy_dataset)) which compute the features on the fly!

# Implemented Network Architectures

Padertorch provides a selection of frequently used network architectures, ready for you to integrate into your own models.

- [Multi-layer Feed-Forward](padertorch/modules/fully_connected): Multiple fully-connected layers with non-linearity and dropout.
- [CNN](padertorch/contrib/je/modules/conv.py) (currently subject to breaking changes and hardly any documentation): 1d- and 2d-CNNs with residual connections, dropout, gated activations, batch and sequence norm and correct handling of down- and upsampling.
- [Normalization](paderotrch/modules/normalization.py): Perform normalization of arbitrary axes/dimensions of your features, keep track of running statistics and apply learnable affine transformation.
- [Dual-Path RNN (DPRNN)](padertorch/modules/dual_path_rnn.py): See the [paper](https://arxiv.org/abs/1910.06379).
- [Collection of activation functions](padertorch/ops/mappings.py): Fast access of various activation functions with just a string.

## Support for sequential and speech data

Padertorch offers extensive support for sequential data such as:
- [Masking](padertorch/ops/sequence/mask.py): Calculate a mask which has non-zero entries for non-padded positions in the sequence.
- [Visualization in tensorboard](padertorch/summary/tbx_utils.py): Prepare spectrograms and speech masks for visualization and audio for playback in tensorboard.

# Installation

**Requirements**
- Python 3
- torch >= 1.0

```bash
$ git clone https://github.com/fgnt/padertorch.git
$ cd padertorch && pip install -e .
```

# Getting Started
```python
import torch
import padertorch as pt

train_dataset = ...
validation_dataset = ...

class MyModel(pt.Model):
    def __init__(self):
        self.net = torch.nn.Sequential(...)

    def forward(self, example):
        output = self.net(example['observation'])
        return output

    def review(self, example, output):
        loss = ...  # calculate loss
        with torch.no_grad():
            ...  # calculate general metrics
            if self.training:
                ...  # calculate training specific metrics
            else:
                ...  # calculate validation specific metrics
        return {
            'loss': loss,
            'scalars': {
                'accuracy': ...,
                ...
            },
        }  # Furthers keys: 'images', 'audios', 'histograms', 'texts', 'figures'


trainer = padertorch.Trainer(
    model=MyModel(),
    storage_dir='path/where/to/save/the/trained/model',
    optimizer=pt.train.optimizer.Adam(),
    loss_weights=None,
    summary_trigger=(1, 'epoch'),
    checkpoint_trigger=(1, 'epoch'),
    stop_trigger=(1, 'epoch'),
    virtual_minibatch_size=1,
)
trainer.test_run(train_dataset, validation_dataset)
trainer.register_validation_hook(validation_dataset)
trainer.train(train_dataset)
```

## padertorch.Configurable

ToDo
