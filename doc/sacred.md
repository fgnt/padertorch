# Running Experiments with sacred

[sacred](https://github.com/IDSIA/sacred) is a package that simplifies running experiments by

 - collecting important information about the experiment, such as host information, configuration, current git
  state, and storing the information about the experiment on disk or in a database with observers,
 - building configuration, and
 - providing an easy-to-use command line interface.

The documentation for sacred can be found on [readthedocs](https://sacred.readthedocs.io/en/stable/).
The [quickstart section](https://sacred.readthedocs.io/en/stable/quickstart.html) gives a good introduction into sacred.

## Sacred, the Configuration and Configurable

The [`padertorch.Configurable`](doc/configurable.md) class is built to work seamlessly with sacred, and all relevant
 classes in `padertorch` inherit from it.
While the `Configurable` allows creation of objects from a configuration dictionary, sacred is there to assist you in 
constructing such a configuration dictionary.
The configuration in sacred can be set and modified in different ways:

 - Using **[dicts](https://sacred.readthedocs.io/en/stable/configuration.html#dictionaries)**: Set the configuration
  directly as a dictionary.
 - Using **[Configuration files](https://sacred.readthedocs.io/en/stable/configuration.html#config-files)**: Sacred
  supports loading configuration from JSON and yaml files.
 - Using **[ConfigScopes](https://sacred.readthedocs.io/en/stable/configuration.html#config-scopes)**: ConfigScopes are 
  functions wrapped by sacred with the `config` decorator. All local
  variables in these functions are interpreted as configuration values.
  This enables complex computations, dependencies and automatic filling of missing configuration values.
  This is the recommended way of setting configuration values in the code.
 - Using **[named ConfigScopes](https://sacred.readthedocs.io/en/stable/configuration.html#named-configurations)**: 
  You can bundle frequently used non-default configuration in config scopes and give
  them a name with the `named_config` decorator. This way you can call your script `with named_config_name` and
  sacred inserts the config.
 - Via the **[command line interface](https://sacred.readthedocs.io/en/stable/command_line.html#configuration-updates)**:
  You can call your script with the arguments `with configuration_key=value` and
  it automatically updates all configuration values that depend on `configuration_key` and stores information about
  what was modified.
For more information about how sacred handles configuration, have a look at the documentation about 
[the config](https://sacred.readthedocs.io/en/stable/configuration.html) and 
[the command line interface](https://sacred.readthedocs.io/en/stable/command_line.html).
 
Consider the following example for a trainings script using sacred:

```python
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
import padertorch as pt
from pathlib import Path

# Define the experiment. Experiment is the main class for managing experiments.
ex = Experiment('my-experiment-name')


class MyModel(pt.Model):
    # Implement your model here
    def __init__(self, num_layers=2):
        super().__init__()
        self.num_layers = num_layers


# Create a ConfigScope with the config decorator
@ex.config
def config():
    """
    This function contains the configuration of the experiment. All local 
    variables in this function will be captured by sacred and used as 
    configuration values.
    """
    # padertorch.Trainer inherits from Configurable, so we can easily construct 
    # the config dictionary here and instantiate the trainer from the config 
    # later
    trainer = {
        # The 'factory' key tells the Configurable which class to instantiate
        'factory': pt.Trainer,
        # The next line creates a new experiment folder for this experiment
        'storage_dir': None,
        'model': {
            'factory': MyModel, # <-- insert your model class here
            # You can put additional configuration of your model here
        },
        # Here goes additional configuration of the trainer
    }   
    if not trainer['storage_dir'] is None:
        # This finds and creates an unused storage dir relative to the 
        # environment variable $STORAGE_ROOT. It is enclosed in the if statement
        # because otherwise it would create unused folders on resume
        trainer['storage_dir'] = pt.io.get_new_storage_dir(ex.path)
    
    # The following line fills the trainer config with the default arguments of
    # the configurable classes. In this example, it will, for example, insert 
    # the "num_layers" argument from the model.
    pt.Trainer.get_config(trainer)
    
    # This adds an observer to the experiment that stores the collected 
    # information about the experiment. In this case, we use a 
    # FileStorageObserver that writes the information to the filesystem to a 
    # sub-folder of the experiment's storage directory.
    ex.observers.append(FileStorageObserver(
        Path(trainer['storage_dir']) / 'sacred')
    )


@ex.automain
def main(trainer, _config, _run):
    """This is the main function of your experiment. It receives all 
    configuration values set in `config` as parameters. `_config` contains the  
    whole configuration as a dictionary."""

    # Print the configuration with marked modifications
    print_config(_run)

    # The following line creates the trainer from the configuration using 
    # classmethods provided by the Configurable class
    trainer = pt.Trainer.from_config(trainer)

    # This stores the configuration in the way it is expected by the model 
    # loading functions in padertorch (pt.Model.from_storage_dir)
    pt.io.dump_config(_config, trainer.storage_dir / 'config.json')
    
    # Run your experiment, start the training or do whatever you want here...
    # For example, start training
    train_dataset = ...
    trainer.train(
        train_dataset,
        # If the checkpoint directory already exists, we probably want to resume
        # instead of starting a new training 
        resume=(trainer.storage_dir / 'checkpoints').exists()
    )
```

You can run the script with the argument `print_config` to print the configuration it is going to use.
Note how `pt.io.get_new_storage_dir` created a new unused experiment dir and how `pt.Trainer.get_config` filled in the
 configuration values that we didn't specify. In a colored terminal you can also see which values were added and
  modified compared to the default configuration (i.e., defined by the config scope).

```bash
$ python experiment.py print_config
INFO - my-experiment-name - Running command 'print_config'
INFO - my-experiment-name - Started
Configuration (modified, added, typechanged, doc):
  """
  This function contains the configuration of the experiment. All local 
  variables in this function will be captured by sacred and used as 
  configuration values.
  """
  seed = 547702770                   # the random seed for this experiment
  trainer:                           # later
    checkpoint_trigger = [1, 'epoch']
    factory = 'padertorch.train.trainer.Trainer'
    loss_weights = None
    stop_trigger = [1, 'epoch']
    storage_dir = '/your/storage/root/my-experiment-name/1'
    summary_trigger = [1, 'epoch']
    virtual_minibatch_size = 1
    model:
      factory = 'MyModel'
      num_layers = 2
    optimizer:
      amsgrad = False
      betas = [0.9, 0.999]
      eps = 1e-08
      factory = 'padertorch.train.optimizer.Adam'
      gradient_clipping = 10000000000.0
      lr = 0.001
      weight_decay = 0
INFO - my-experiment-name - Completed after 0:00:00
```

## Accessing information stored by sacred

The `FileStorageObserver` stores experiment information in the storage dir with the following structure:

```
.                       # The storage dir
├── config.json         # Config saved by ourselves for pt.Model.from_storage_dir
├── checkpoints         # This directory contains the model and trainer checkpoints
│   ├── ckpt_<iteration_number>.pth
│   ├── ckpt_best_loss.pth  # Symlink to best checkpoint judged by validation loss
│   └── ckpt_latest.pth     # Symlink to latest checkpoint 
└── sacred              # Sub-folder for the FileStorageObserver
    ├── 1               # A Unique ID created by the observer to handle multipe runs in the same storage dir (e.g., resume)
    │   ├── config.json     # Configuration of this run
    │   ├── cout.txt        # Captured stdout and stderr streams
    │   ├── metrics.json    # Reported metrics (not used by the trainer, so this file is typically empty)
    │   └── run.json        # Information about the run, such as time, host info, git status, imported packages ...
    └── _sources                  # The _sources folder can contain source files that are not checked into git
        └── experiment_544d3ddcc21032c781ad70f6aa728f18.py
```

Once you created and trained a model with a script similar to the above example, you can easily access the trained
 model with:
 
```python
import padertorch as pt
model = pt.Model.from_storage_dir('/path/to/the/storage/dir')
```