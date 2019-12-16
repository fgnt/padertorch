# padertorch
[![Build Status](https://dev.azure.com/fgnt/fgnt/_apis/build/status/fgnt.padertorch?branchName=master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![Azure DevOps tests](https://img.shields.io/azure-devops/tests/fgnt/fgnt/3)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/fgnt/fgnt/3)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=3&branchName=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgnt/lazy_dataset/blob/master/LICENSE)

Padertorch is designed to simplify the training of deep learning models written with PyTorch.
While focusing on speech and audio processing, it is not limited to these application areas.

This repository is currently under construction.

The examples in contrib/examples are only working in the Paderborn NT environment


# Highlights

## padertorch.Trainer

- Logging:
  - The `review` of the model returns a dictionary that will be logged and visualized via 'tensorboard'. The keys define the logging type (e.g. `scalars`).
  - As logging backend we use `TensorboardX` to generate a `tfevents` file that can be visualized from a `tensorboard`.
- Dataset type: 
  - lazy_dataset.Dataset, torch.utils.data.DataLoader and other iterables...
- Validation:
  - The `ValidationHook` runs periodically and and logs the validations results.
- Learning rate decay with backoff:
  - The `ValidationHook` has also parameters to do a learning rate with backoff.
- Test run: 
  - The trainer has a test run function to train the model for few iterations and test if
    - the model is executable (burn test)
    - the validation is deterministic/reproducable
    - the model changes the parameter in the training
- Hooks:
  - The hooks are used to extend the basic features of the trainer. Usually the user dont rearly care about the hooks. By default a `SummaryHook`, a `CheckpointHook` and a `StopTrainingHook` is registerd. So the user only need to register a `ValidationHook`
- Checkpointing:
  - The parameters of the model and the state of the trainer are periodically saved. The intervall can be specified with the `checkpoint_trigger` (The units are `epoch` and `iteration`)
- Virtual minibatch:
  - The `Trainer` usually do not know if the model is trained with a single example or multiple examples (minibatch), because the exaples that are yielded from the dataset are directly forwarded to the model. 
  - When the `virtual_minibatch_size` option is larger than one, the trainer calls the forward and backward step `virtual_minibatch_size` times before applying the gradients. This increases the minibatch size, while the memory consumption stays similar.


```python
import torch
import padertorch as pt

train_dataset = ...
validation_dataset = ...

class MyModel(pt.base.Model):
    def __init__(self):
        self.net = torch.nn.Sequential(...)

    def forward(inputs):
        outputs = self.net(inputs['observation'])
        return outputs

    def review(self, inputs, outputs):
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
