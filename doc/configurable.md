# Configurable
The `Configurable` class allows for a simple initialization of a class from a
dictionary.
```python
from torch import nn
import padertorch as pt
config = dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
)
feed_forward_layer = pt.Configurable.from_config(config)
```
Here `Configurable` takes the class specified with the `factory` key and
initializes it using the other items in the dictionary as kwargs.
Therefore, all parameters not mentioned in the dictionary fallback to the
defaults.
In case of nested dictionaries all nested classes with a `factory` key are 
first initialized by `Configurable` before using them as input to the upper-level 
class. The value for `factory` may either be a `Callable` (class, a function) 
or a string. If `factory` is string it must be a python path to a class
which can be initialized from the dictionary 
(all inputs without defaults are specified in the dictionary):
```python
import padertorch as pt
config = dict(
        factory='torch.nn.Linear',
        in_features=120,
        out_features=84
)
feed_forward_layer = pt.Configurable.from_config(config)
```
Supporting strings allows for human-readable serializations like yaml or json.

## Configurable: What is it good for?

The main idea of the `Configurable` object is to simplify the initialization
of a neural network model from a config dictionary.
A neural network usually consists of multiple modules with inter-dependent
configurations which are combined in the main model.
A simple model inspired by the model in
[pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
looks like this:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
```
Instantiating the modules inside of the model makes it more difficult to
change parameter in the module or leads to an unnecessary high amount of
input parameters for the model init.
Therefore, we assume that the already initialized modules are the input
to the model:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, fc1, fc2):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(nn.Linear(120, 84), nn.Linear(84, 10))
```
Using`Configurable` we can initialize `Net` from a dictionary:
 
```python
config = dict(
    factory=Net,
    fc1=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
net = pt.Configurable.from_config(config)
```
This initialization looks more complicated than the two possiblities above,
but it has major upsites.
The main benefit is that we can now easily adjust the parameters from outside
of the model.
Another benefit is, that this initialization allows to easily save the used
configuration for example by writting the config dictionary to a json/yaml file.
However, the current config does not include all parameters used in the model
since there are some default values in the modules which are not specified in
the config dictionary.
We can get a config with all parameters in the model by
calling `pt.Configurable.get_config`:
```python
update = dict(
    factory=Net,
    fc1=dict(
        factory=nn.Linear,
        in_features=120,
        out_features=84
    ),
    fc2=dict(
        factory=nn.Linear,
        in_features=84,
        out_features=10
    ),
)
full_config = pt.Configurable.get_config(update)
```
The full config now includes all input parameters to the sub-modules:
```
{'factory': 'Net',
 'fc1': {'factory': 'torch.nn.modules.linear.Linear',
  'in_features': 120,
  'out_features': 84,
  'bias': True},
 'fc2': {'factory': 'torch.nn.modules.linear.Linear',
  'in_features': 84,
  'out_features': 10,
  'bias': True}}
```
The key ```factory``` is a fixed signifier for a class representer and should
 not be used as a parameter name in combination with `Configurable`. 


## Additonal perks of using Configurable 
Configurable is designed to simplify class initialization and config 
generation. One feature to allow interdependents bertween parameters is the 
finalize_dogmatic_config classmethod. Here parameters from submodules can
be changed dependent on the previous layer. For example one can define the
input size as the output size of the previous layer:
```python
import padertorch as pt
import torch.nn as nn
import torch.nn.functional as F

class Net(pt.Module):
    @classmethod
    def finalize_dogmatic_config(cls, config):
       config['fc1'] = {
            'factory': nn.Linear,
            'in_features':120,
            'out_features': 84
        }
       config['fc2'] = {
            'factory': nn.Linear,
            'in_features':config['fc1']['out_features'],
            'out_features': 10
       }

    def __init__(self, fc1, fc2):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net.from_config(Net.get_config({}))
```
This allows us to initialize net without specificying any parameters,
if the default parameters are chosen.
The resulting network looks like this:
```
Net(
  (fc1): Linear(in_features=120, out_features=84, bias=True)
  (fc2): Linear(in_features=84, out_features=84, bias=True)
)
```
Additionally, we can easily change the
layer size of the first layer and the second layer will be adjusted 
accordingly:
```
net = Net.from_config(Net.get_config({'fc1': {'out_features': 320}}))
```
which results in the following network:
```
Net(
  (fc1): Linear(in_features=120, out_features=320, bias=True)
  (fc2): Linear(in_features=320, out_features=84, bias=True)
)
```

Some modules need functions or non-initialized classes as an input.
Therefore, one can use the partial key instead of the factory key.
All defined kwargs will overwrite the defaults without calling the function
or initializing the class using partial.
One example is SpeechBrain which requires the activity to be not
initialized at the class input.

```python
import padertorch as pt
import torch.nn as nn

class Net(pt.Module):
    @classmethod
    def finalize_dogmatic_config(cls, config):
       config['fc1'] = {
            'factory': nn.Linear,
            'in_features':120,
            'out_features': 84
        }
       config['fc2'] = {
            'factory': nn.Linear,
            'in_features':config['fc1']['out_features'],
            'out_features': 10
       }
       config['activation'] = {
            'partial': nn.ReLU
       }

    def __init__(self, fc1, fc2, activation):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.activation=activation()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net.from_config(Net.get_config({}))
```
To overwrite the activation, we can just define the new activation in the 
update for the config.
```
net = Net.from_config(Net.get_config({'activation': {'partial': torch.nn.LeakyReLU}}))
```
which results in the following network:
```
Net(
  (fc1): Linear(in_features=120, out_features=84, bias=True)
  (fc2): Linear(in_features=84, out_features=10, bias=True)
  (activation): LeakyReLU(negative_slope=0.01)
)
```
For more example on how to use finalize_dogmatic_config please refer to the
class doctest, our toy example or our example models.

Configurable is designed to work with sacred to simplify the config generation
and allow for changes in the configuration from the training script call.
For more information on how to work with sacred visit our sacred documentation
[here](sacred.md).

Furthermore, Configurable allows for easily reproducable evaluations by 
initializing the model from a model directory without knowledge about the
specific model architecture: 
`model = pt.Module.from_config_and_checkpoint()`.
For further code examples please refer to our example evaluation scripts.