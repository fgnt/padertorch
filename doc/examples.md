# Examples: Getting Started using Padertorch

Padertorch's main purpose is to provide an open interface that simplifies tedious work and works with as many different 
workflows as possible. Due to the open design of Padertorch, there is no "right" or "wrong" way to train a neural network.
However, to make getting started easier, we have provided several examples with our recommended workflow when using Padertorch. 

We recommend the usage of the following components when using Padertorch:
  - [sacred](): Experiment Organization / Ensuring reproducability
  - [lazy_dataset](https://github.com/fgnt/lazy_dataset): Data preparation and on-the-fly data generation
  - [Padertorch Configurable](): Adjustment of training hyperparameters
  
The examples mostly use all of these components, whereas the simple examples  (See [Getting Started]()) omit some of these components to reduce the overall complexity of the example as much as possible.


## Basic Structure of a Training:

Each example several has several steps:
  - Experiment setup (directory creation, reproducability, logging)
  - Data preprocessing
  - Training
  - (Optional:) Evaluation

### Experiment setup
During the experiment setup, the experiment directory is created. We recommend using [Sacred]() to monitor each training.
Inside this directory, the checkpoints of the training as well as the event files for visualization are saved. 
The examples using Padertorch Configurable and Sacred also save the model hyperparameters in this folder and log the 
changes over multiple starts of the same training.

In its most basic form this step looks as simple as:
``` python
import os

storage_dir = pt.io.get_new_storage_dir('my_experiment')
```
For more information regarding the usage of Configurable or Sacred 
see [Configurable Introduction]() and [Sacred Introduction]()

### Data preprocessing
Data preprocessing describes the creation of an iterable object (e.g. a list of examples)for the Padertorch Trainer.
In our examples, we use [lazy_dataset](https://github.com/fgnt/lazy_dataset) to obtain an iterable object and map all necessary transformations onto this iterable.

``` python
import lazy_dataset.database

def mapping_function(example):
    example['new_keys'] = some_transform_function(example)
    return example

iterable = lazy_dataset.database.DictDatabase(data)
iterable = iterable.map(mapping_function)
...
iterable = iterable.batch(batch_size)
```

However, all other approaches that create an iterable object can be used for this step as well.

### Training
The training itself is mostly handled by the [Padertorch Trainer](). 
Therefore, the training consists of the initialization of the Padertorch model and the preparation of the iterable object.
These two properties are then fed into the Padertorch Trainer.
``` python
from padertorch import Trainer, optimizer
model = MyModel()
my_optimizer = optimizer.Adam() 

trainer = Trainer(
    model=model,
    optimizer=my_optimizer,
    storage_dir=storage_dir,
)
trainer.register_validation_hook(validation_iterable)
trainer.test_run(iterable, validation_iterable)
trainer.train(iterable)
```

Optionally, many different hooks can be registered before starting the training to use e.g. learning rate scheduling.
For more information how to use, register and write hooks, See [hooks: customizing your training]()  

### Evaluation
In essence, the evaluation can be done in the same way as the training. Nevertheless, some examples
also have a dedicated evaluation file. In these evaluation examples, concepts like parallelization with MPI
are used to speed up the time necessary for an evaluation.


## Getting Started:
To get started, the "lean" examples are best suited.
Therefore, the examples for mask estimation and audio tagging are a good starting point to quickly get the first training running.
The PIT example furthermore uses Sacred and Configurable, providing a basic structure of our recommended workflow. 

For most purposes, the structure of the examples should also serve as a good template when constructing your own training. 



    
## List of current examples:
  - Source Separation:
    - PIT: Implementation of <>
    - OR-PIT:
    - TasNet/ConvTasNet: See <>
  - Speech Enhancement:
    - Mask Estimator: 
  - Acoustic Model:
  - Audio Tagging:
  - Wavenet: See <>
  - MNIST: Small toy example showing the application of Padertorch to image data
  - Functions:
    - Multi-GPU: Basic description and code example of Padertorch's Multi-GPU support
    - Configurable: Introduction into the usage of the Configurable class to setup trainings
