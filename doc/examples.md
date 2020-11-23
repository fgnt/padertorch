# Examples: Getting Started using Padertorch

Padertorch's main purpose is the simplification of neural network training.
To ensure this goal for a multitude of applications it has a modular structure and the single components are easily exchangeable.
In this examples section we have provided several examples with our recommended workflow for using Padertorch.
These examples are meant to convey the inherent logic and structure of a training and serve as a blueprint for new trainings at the same time.
Due to the modular structure of Padertorch, there are many more workflows that are compatible with Padertorch as well.
However, getting started will likely become easier by basing your trainings on the examples provided, here.

We recommend the usage of the following components when using Padertorch:
  - [sacred](sacred.md) / [Padertorch Configurable](configurable.md): Experiment organization / Ensuring reproducability / Adjustment of training hyperparameters
  - [lazy_dataset](https://github.com/fgnt/lazy_dataset): Data preparation and on-the-fly data generation
  
The examples mostly use all of these components, whereas the simple examples  (See Getting Started omit some of these components to reduce the overall complexity of the example as much as possible.


## Basic Structure of a Training:

Each example has several steps:
  - Experiment setup (directory creation, reproducability, logging)
  - Data preprocessing
  - Training
  - Evaluation

### Experiment setup
During the experiment setup, the experiment directory is created. We recommend using [Sacred](https://github.com/IDSIA/sacred) to monitor each training.
Inside this directory, the checkpoints of the training as well as the event files for visualization are saved. 
The examples using Padertorch Configurable and Sacred also save the model hyperparameters in this folder and log the 
changes over multiple starts of the same training.

In its most basic form this step looks as simple as:
``` python
import padertorch as pt
import os

# This function requires the environment variable STORAGE_ROOT to be set.
storage_dir = pt.io.get_new_storage_dir('my_experiment')
# You can also directly use the following function to get a unique experiment folder for each training without specifying STORAGE ROOT: 
# storage_dir = pt.io.get_new_subdir('/my/experiment/path')
```
For more information regarding the usage of Configurable or Sacred 
see [Configurable Introduction](configurable.md) and [Sacred Introduction](sacred.md)

### Data preprocessing
Data preprocessing describes the creation of an iterable object (e.g. a list of examples) for the Padertorch Trainer.
In our examples, we use [lazy_dataset](https://github.com/fgnt/lazy_dataset) to obtain an iterable object and map all necessary transformations onto this iterable. A lazy_dataset database allows for several additional helper functions like combining multiple datasets. The specifics of the Database object can be found at https://github.com/fgnt/lazy_dataset/blob/master/lazy_dataset/database.py.
``` python
import lazy_dataset.database
import padertorch as pt

def mapping_function(example):
    example['new_keys'] = some_transform_function(example)
    return example

iterable = lazy_dataset.database.DictDatabase(data)
iterable = iterable.map(mapping_function)
...
iterable = iterable.batch(batch_size)
iterable = iterable.map(pt.data.utils.collate_fn)

```

However, all other approaches that create an iterable object can be used for this step as well.

### Training
The training itself is mostly handled by the [Padertorch Trainer](trainer.md). 
Therefore, the training consists of the initialization of the Padertorch model and
the preparation of the iterable object.
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
For more information how to use, register and write hooks, See [Hooks: customizing your training](hooks.md)  

### Evaluation
Usually, a "light" evaluation (i.e. validation) is included in the training.
Often the actual evaluation requires more computation power or the 
trained model should be tested with varying evaluation hyperparameters.
To allow this, these examples have a dedicated evaluation file. 
In these evaluation examples, concepts like parallelization with MPI
are used to speed up the time necessary for an evaluation.


## Getting Started:
To get started, the "lean" examples are best suited.
Therefore, the examples for mask estimation and audio tagging are a good starting point to quickly get the first training running.
The PIT example furthermore uses Sacred and Configurable, providing a basic structure of our recommended workflow. 

For most purposes, the structure of the examples should also serve as a good template when constructing your own training. 



    
## List of current examples:
  - Source Separation:
    - [PIT](../padertorch/contrib/examples/source_separation/pit/README.md): Implementation of https://arxiv.org/abs/1703.06284
    - [OR-PIT](../padertorch/contrib/examples/source_separation/or_pit/README.md): Implementation of https://arxiv.org/abs/1904.03065
    - [TasNet/ConvTasNet](../padertorch/contrib/examples/source_separation/tasnet/README.md): Implementation of https://arxiv.org/abs/1711.00541
  - Speech Enhancement:
    - [Mask Estimator](../padertorch/contrib/examples/speech_enhancement/mask_estimator/README.md): 
  - Audio Tagging:
  - [Speaker Classification](../padertorch/contrib/examples/speaker_classification/supervised/README.md):
  - Audio Synthesis
    - [Wavenet](../padertorch/contrib/examples/audio_synthesis/wavenet/README.md): See <>
  - Toy Examples:
    - MNIST: Small toy example showing the application of Padertorch to image data
    - Multi-GPU: Basic description and code example of Padertorch's Multi-GPU support
    - Configurable: Introduction into the usage of the Configurable class to setup trainings
